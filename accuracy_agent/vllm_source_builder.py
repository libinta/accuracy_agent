"""Build comparison peers from one vllm-project/vllm commit.

Both peer setup modes end up here, because both need the same thing: a container
running a KNOWN vLLM commit, installed from source into a vendor PyTorch image.

  * ``vllm.commit`` set -- build BOTH peers from that commit
    (``autoconfigure_from_commit``). Answers "does commit <sha> diverge?".
  * ``vllm.commit`` unset, XPU container given -- ask that container which commit
    it actually runs and build the CUDA peer from it
    (``autoconfigure_gpu_from_xpu_commit``). Answers "does the vLLM my XPU
    container runs diverge from the same code on CUDA?".

The second mode used to pair the XPU container with the closest published
``vllm/vllm-openai`` release instead. That was cheap but weak: a dev build maps
to a release hundreds of commits away, so a "divergence" could just as well have
been a version difference. Building the reported commit costs one install and
removes that doubt entirely.

The pipeline is the same either way:

  1. resolve the commit in a local vllm-project/vllm clone (fetching it from
     ``origin`` when the clone does not have it yet)
  2. export it to a per-commit source tree, via a tool-managed branch, so the
     tree is a real git checkout -- vLLM's build needs git metadata
  3. per device, start a container from the vendor PyTorch base image, install
     that tree *editable* into it, then ``docker commit`` the result so later
     runs skip the install
  4. hand back the container plus the source path, which is also the path the
     patcher rewrites: with an editable install, patches land on the checked-out
     files and take effect without reinstalling

CUDA kernels: compiling vLLM from source takes 1-2 h, so the default is vLLM's
own precompiled fast path (``VLLM_USE_PRECOMPILED=1``) -- the commit's Python
code with binaries from the nearest nightly wheel. That is the right trade-off
while bisecting Python-level divergence and the WRONG one when the commit
touches C++/CUDA, so ``build_kernels=True`` forces a real compile. The two are
cached under different image tags and ``VLLMBuild.note`` says which produced the
peer, so a report can never silently claim kernel coverage it does not have.

The XPU side always builds from source (there is no precompiled XPU wheel), but
its heavy kernels come from the pinned ``vllm_xpu_kernels`` wheel in
``requirements/xpu.txt``, so that build is minutes rather than hours.
"""

import json
import logging
import re
import shlex
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from accuracy_agent.docker_probe import (
    CommandRunner,
    DockerProbeError,
    VLLMCommit,
    _container_state,
    detect_vllm_commit,
    detect_vllm_path,
    docker_available,
    ensure_gpu_container,
    host_has_nvidia_gpu,
    is_local_host,
    running_inside_container,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from accuracy_agent.config import DebugConfig

logger = logging.getLogger(__name__)

VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"

#: Vendor PyTorch base images. Both are resolved to their newest tag at runtime
#: when the network allows; these pins are the offline fallback and were the
#: newest tags when this was written.
NGC_PYTORCH_REPO = "nvcr.io/nvidia/pytorch"
DEFAULT_CUDA_BASE_IMAGE = f"{NGC_PYTORCH_REPO}:26.07-py3"
IPEX_REPO = "intel/intel-extension-for-pytorch"
DEFAULT_XPU_BASE_IMAGE = f"{IPEX_REPO}:2.8.10-xpu"

#: Alternative XPU base: what vLLM's own docker/Dockerfile.xpu builds on. Has
#: the oneAPI *devel* toolchain but no vendor torch (that comes from
#: requirements/xpu.txt), so it is the safer base when an IPEX build fails.
DLE_XPU_BASE_IMAGE = "intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04"

#: nvcr.io serves anonymous pulls of public images behind a token endpoint, so
#: the standard registry v2 tag list needs that token first.
_NGC_AUTH_URL = "https://nvcr.io/proxy_auth?scope=repository:{repo}:pull"
_NGC_TAGS_URL = "https://nvcr.io/v2/{repo}/tags/list"
_DOCKER_HUB_TAGS_URL = "https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100"

#: NGC PyTorch release tags look like ``26.07-py3``. ``-igpu``/``-devel`` and
#: friends are variants we do not want to pick automatically.
_NGC_TAG_RE = re.compile(r"^(\d{2})\.(\d{2})-py3$")

#: IPEX XPU tags look like ``2.8.10-xpu``; the ``-idp-*``/``-pip-*``/serving
#: variants are deliberately excluded.
_IPEX_TAG_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)-xpu$")

#: Container flags per device. Both are idle containers we docker exec into.
_DEVICE_RUN_ARGS = {
    "cuda": ("--gpus", "all", "--ipc=host", "--shm-size=16g"),
    "xpu": ("--device", "/dev/dri", "--ipc=host", "--shm-size=16g", "--cap-add=SYS_PTRACE"),
}

#: Where the per-commit source tree is mounted inside every peer container.
CONTAINER_SOURCE_DIR = "/workspace/vllm"

_BUILD_TIMEOUT_FAST = 5400       # precompiled CUDA / XPU-with-prebuilt-kernels
_BUILD_TIMEOUT_COMPILE = 21600   # full CUDA kernel compile


class VLLMBuildError(RuntimeError):
    """Raised when a commit could not be resolved, exported, or installed."""


@dataclass
class VLLMBuild:
    """One built peer: a container running vLLM at a specific commit."""
    device: str                 # "cuda" or "xpu"
    commit: str                 # full 40-char sha
    base_image: str             # vendor PyTorch image it was installed into
    image: str                  # committed image, e.g. accuracy_agent/vllm:cuda-<short>
    container: str
    vllm_path: str              # editable source dir inside the container
    source_dir: str             # host path of that source tree
    built: bool = False         # True if we ran the install (vs reused an image)
    precompiled: bool = False   # CUDA fast path: kernels from a nightly wheel
    note: str = ""

    @property
    def short(self) -> str:
        return self.commit[:12]

    def summary(self) -> str:
        how = "reused" if not self.built else "built"
        bits = [f"{self.device}: vLLM @ {self.short} {how} on {self.base_image}"]
        if self.precompiled:
            bits.append("(kernels from nearest nightly wheel, not this commit)")
        bits.append(f"-> {self.container}")
        return " ".join(bits)


@dataclass
class GPUPeerFromXPU:
    """A GPU peer derived from what the existing XPU container runs."""
    xpu_commit: Optional[VLLMCommit] = None   # commit detected in the XPU container
    commit: str = ""                          # full sha it resolved to upstream
    build: Optional[VLLMBuild] = None         # the CUDA peer built from it
    container: str = ""                       # container started from a pinned gpu.image
    image: str = ""                           # that pinned image
    note: str = ""


@dataclass
class PeerSetup:
    """What automatic peer configuration produced, for reporting."""
    commit: Optional[str] = None
    builds: Dict[str, VLLMBuild] = field(default_factory=dict)
    xpu_commit: Optional[VLLMCommit] = None    # set when the commit came from the XPU peer
    gpu_container: str = ""                    # set when a pinned gpu.image was launched
    gpu_image: str = ""
    notes: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def configured(self) -> bool:
        return bool(self.builds) or bool(self.gpu_container)

    def summary_lines(self) -> List[str]:
        lines = []
        if self.xpu_commit is not None:
            lines.append(self.xpu_commit.summary())
            if self.xpu_commit.dirty:
                lines.append(
                    f"note: {self.xpu_commit.path} has uncommitted changes; the GPU "
                    f"peer is built from the committed sha only"
                )
        if self.gpu_container:
            lines.append(f"gpu: {self.gpu_image} -> {self.gpu_container} (pinned image, not built)")
        for build in self.builds.values():
            lines.append(build.summary())
            if build.note:
                lines.append(f"note: {build.note}")
        lines.extend(f"note: {note}" for note in self.notes)
        return lines


# --------------------------------------------------------------------------
# base image selection
# --------------------------------------------------------------------------

def fetch_ngc_pytorch_tags(repo: str = "nvidia/pytorch", timeout: int = 15) -> List[str]:
    """Newest-last list of ``NN.NN-py3`` tags of an NGC image.

    Raises:
        OSError / ValueError / KeyError on network or payload problems --
        callers degrade to the pinned default.
    """
    auth_url = _NGC_AUTH_URL.format(repo=repo)
    with urllib.request.urlopen(auth_url, timeout=timeout) as response:  # nosec B310 - fixed https URL
        token = json.load(response)["token"]

    request = urllib.request.Request(
        _NGC_TAGS_URL.format(repo=repo), headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - fixed https URL
        payload = json.load(response)

    matched = [(m.group(1), m.group(2), t) for t in payload.get("tags", [])
               for m in [_NGC_TAG_RE.match((t or "").strip())] if m]
    matched.sort()
    return [tag for _y, _m, tag in matched]


def fetch_ipex_xpu_tags(repo: str = IPEX_REPO, max_pages: int = 5, timeout: int = 15) -> List[str]:
    """Newest-last list of ``X.Y.Z-xpu`` tags of the IPEX image on Docker Hub."""
    url = _DOCKER_HUB_TAGS_URL.format(repo=repo)
    matched: List[Tuple[Tuple[int, int, int], str]] = []
    pages = 0

    while url and pages < max_pages:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310 - fixed https URL
            payload = json.load(response)
        for entry in payload.get("results", []):
            name = (entry.get("name") or "").strip()
            match = _IPEX_TAG_RE.match(name)
            if match:
                key = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                matched.append((key, name))
        url = payload.get("next")
        pages += 1

    matched.sort()
    return [name for _key, name in matched]


def resolve_base_image(device: str, explicit: str = "", allow_network: bool = True) -> Tuple[str, str]:
    """Pick the vendor PyTorch base image for `device`.

    An explicit reference is taken verbatim (that is the escape hatch when the
    newest vendor image does not build). Otherwise the newest published tag is
    used when the registry is reachable, and the pinned default otherwise.

    Returns:
        (image reference, note explaining where it came from).
    """
    if explicit:
        return explicit, ""

    default = DEFAULT_CUDA_BASE_IMAGE if device == "cuda" else DEFAULT_XPU_BASE_IMAGE
    if not allow_network:
        return default, f"offline: using pinned base image {default}"

    repo = NGC_PYTORCH_REPO if device == "cuda" else IPEX_REPO
    try:
        tags = fetch_ngc_pytorch_tags() if device == "cuda" else fetch_ipex_xpu_tags()
    except Exception as e:
        logger.warning(f"Could not list {repo} tags ({e}); using pinned {default}")
        return default, f"registry unreachable ({e}); using pinned base image {default}"

    if not tags:
        return default, f"no usable tags found for {repo}; using pinned {default}"
    return f"{repo}:{tags[-1]}", ""


# --------------------------------------------------------------------------
# commit resolution and source export
# --------------------------------------------------------------------------

def _git(repo: Path, args: str, runner: CommandRunner, timeout: int = 600):
    return runner.run(f"git -C {shlex.quote(str(repo))} {args}", timeout=timeout)


def find_vllm_repo(explicit: str = "", runner: Optional[CommandRunner] = None) -> Path:
    """Locate a local vllm-project/vllm clone, cloning one if there is none.

    Search order: the explicit path, then ``~/vllm``, then a tool-managed clone
    under ``~/.cache/accuracy_agent``. A clone is only made as a last resort --
    it is a large download, and most users already have one.
    """
    runner = runner or CommandRunner()
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.append(Path.home() / "vllm")
    cache_clone = Path.home() / ".cache" / "accuracy_agent" / "vllm"
    candidates.append(cache_clone)

    for candidate in candidates:
        if (candidate / ".git").exists():
            logger.info(f"Using vLLM clone at {candidate}")
            return candidate

    if explicit:
        raise VLLMBuildError(
            f"{explicit} is not a git clone of vllm-project/vllm (no .git directory)"
        )

    logger.info(f"Cloning {VLLM_REPO_URL} into {cache_clone} (one-time, several hundred MB)")
    cache_clone.parent.mkdir(parents=True, exist_ok=True)
    result = runner.run(
        f"git clone {VLLM_REPO_URL} {shlex.quote(str(cache_clone))}", timeout=3600
    )
    if not result.ok:
        raise VLLMBuildError(
            f"Could not clone {VLLM_REPO_URL}: {result.stderr.strip()[:400]}. "
            "Point vllm.repo_path at an existing clone instead."
        )
    return cache_clone


def resolve_commit(
    ref: str,
    repo: Path,
    runner: Optional[CommandRunner] = None,
    allow_fetch: bool = True,
) -> str:
    """Resolve `ref` to a full commit sha in `repo`, fetching it if needed.

    Accepts anything git accepts (short sha, full sha, tag, branch), so a user
    can paste a hash straight out of a GitHub URL.

    Raises:
        VLLMBuildError: If the ref cannot be resolved even after a fetch.
    """
    runner = runner or CommandRunner()
    result = _git(repo, f"rev-parse --verify --quiet {shlex.quote(ref)}^{{commit}}", runner, timeout=120)
    if result.ok and result.stdout.strip():
        return result.stdout.strip()

    if not allow_fetch:
        raise VLLMBuildError(f"{ref!r} is not a commit in {repo} (fetching disabled)")

    # A hash from a PR or a branch the clone has never fetched: ask origin for it
    # directly. Fetching a bare sha needs a server that allows it, so fall back
    # to a full fetch of all branches.
    logger.info(f"{ref} not in {repo}; fetching from origin")
    fetch = _git(repo, f"fetch origin {shlex.quote(ref)}", runner, timeout=1800)
    if not fetch.ok:
        fetch = _git(repo, "fetch --tags origin", runner, timeout=3600)
        if not fetch.ok:
            raise VLLMBuildError(
                f"Could not fetch {ref!r} from origin in {repo}: {fetch.stderr.strip()[:300]}"
            )

    for candidate in (f"{ref}^{{commit}}", "FETCH_HEAD^{commit}"):
        result = _git(repo, f"rev-parse --verify --quiet {shlex.quote(candidate)}", runner, timeout=120)
        if result.ok and result.stdout.strip():
            return result.stdout.strip()

    raise VLLMBuildError(
        f"{ref!r} could not be resolved to a commit in {repo} even after fetching. "
        "Is it a commit of vllm-project/vllm?"
    )


def ensure_source_tree(
    repo: Path,
    commit: str,
    build_root: Path,
    runner: Optional[CommandRunner] = None,
) -> Path:
    """Export `commit` to its own checkout under `build_root`, reusing it if present.

    The export is a real clone rather than an archive because vLLM's build reads
    git metadata (setuptools-scm). It is created from a tool-managed branch in
    `repo`, which is what makes an arbitrary fetched sha clonable.

    The tree is deliberately reused even when dirty: the install rewrites
    requirement files in place (``use_existing_torch.py``), and a rebuilt tree
    would throw away compiled extensions.
    """
    runner = runner or CommandRunner()
    short = commit[:12]
    dest = build_root / f"vllm-{short}"

    if (dest / ".git").exists():
        head = _git(dest, "rev-parse HEAD", runner, timeout=120)
        if head.ok and head.stdout.strip() == commit:
            logger.info(f"Reusing source tree {dest}")
            return dest
        raise VLLMBuildError(
            f"{dest} exists but is not at {commit} (HEAD={head.stdout.strip() or 'unknown'}). "
            "Remove it or choose another vllm.build_root."
        )

    branch = f"accuracy-agent/{short}"
    mark = _git(repo, f"branch --force {shlex.quote(branch)} {shlex.quote(commit)}", runner, timeout=300)
    if not mark.ok:
        raise VLLMBuildError(
            f"Could not mark {commit} with branch {branch} in {repo}: {mark.stderr.strip()[:300]}"
        )

    build_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting vLLM @ {short} to {dest}")
    clone = runner.run(
        f"git clone --no-hardlinks --single-branch --branch {shlex.quote(branch)} "
        f"{shlex.quote(str(repo))} {shlex.quote(str(dest))}",
        timeout=3600,
    )
    if not clone.ok:
        raise VLLMBuildError(
            f"Could not export {commit} to {dest}: {clone.stderr.strip()[:400]}"
        )
    return dest


# --------------------------------------------------------------------------
# building a peer container
# --------------------------------------------------------------------------

def build_image_tag(device: str, commit: str, build_kernels: bool) -> str:
    """Image tag for a built peer.

    A precompiled CUDA build and a fully compiled one are NOT interchangeable,
    so they get separate tags and can coexist in the cache.
    """
    suffix = "-src" if (build_kernels and device == "cuda") else ""
    return f"accuracy_agent/vllm:{device}-{commit[:12]}{suffix}"


def peer_container_name(device: str, commit: str, build_kernels: bool) -> str:
    suffix = "_src" if (build_kernels and device == "cuda") else ""
    return f"accuracy_agent_vllm_{device}_{commit[:12]}{suffix}"


def _install_script(device: str, commit: str, build_kernels: bool) -> str:
    """Bash that installs the mounted source tree editable into the container.

    ``use_existing_torch.py`` strips the torch pins from vLLM's requirements so
    the vendor image's own torch build is kept -- that is the whole point of
    installing into the vendor PyTorch image rather than a generic one.
    """
    lines = [
        "set -euo pipefail",
        f"cd {CONTAINER_SOURCE_DIR}",
        # The tree is bind-mounted from the host, so it belongs to the host user
        # while this script runs as the container's root. Without an exception git
        # refuses it ("detected dubious ownership") and vLLM's setuptools-scm
        # introspection fails, taking `pip install -e .` down with it.
        f"git config --global --add safe.directory {CONTAINER_SOURCE_DIR}",
        # oneAPI images put the compiler env behind setvars.sh. `|| true` covers a
        # setvars that *returns* non-zero; one that calls `exit` still aborts the
        # script, which is the right outcome -- without a working oneAPI env the
        # XPU build cannot succeed anyway.
        "if [ -f /opt/intel/oneapi/setvars.sh ]; then "
        "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; fi",
        "python3 -m pip install --upgrade pip",
        # Keep the vendor torch: strip torch/torchvision/torchaudio pins. Guarded
        # because the commit may predate this helper, and `set -e` would abort.
        "if [ -f use_existing_torch.py ]; then python3 use_existing_torch.py; fi",
    ]

    if device == "cuda":
        lines += [
            # `if` rather than `[ -f ... ] && ...`: a false test as the last
            # command of the loop body would fail the whole script under `set -e`.
            "for f in requirements/build/cuda.txt requirements/build.txt; do "
            "if [ -f \"$f\" ]; then python3 -m pip install -r \"$f\"; fi; done",
        ]
        if build_kernels:
            lines += [
                "export VLLM_TARGET_DEVICE=cuda",
                "echo '>>> compiling CUDA kernels from source (this takes 1-2 hours)'",
            ]
        else:
            lines += [
                "export VLLM_USE_PRECOMPILED=1",
                # Prefer this commit's own nightly wheel; setup.py falls back to
                # the nearest base commit in main when it does not exist.
                f"export VLLM_PRECOMPILED_WHEEL_COMMIT={commit}",
                "echo '>>> installing with precompiled kernels (VLLM_USE_PRECOMPILED=1)'",
            ]
    else:
        lines += [
            # The heavy XPU kernels are a pinned prebuilt wheel in this file, so
            # this stays minutes rather than hours.
            "python3 -m pip install -r requirements/xpu.txt",
            "python3 -m pip install grpcio-tools protobuf nanobind",
            "export VLLM_TARGET_DEVICE=xpu",
            "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
        ]

    lines += [
        "python3 -m pip install --no-build-isolation -e .",
        # find_spec does not import vllm, so this works without a live device.
        "python3 -c \"import importlib.util as u; print('vllm at', u.find_spec('vllm').origin)\"",
    ]
    return "\n".join(lines)


def _assert_container_has_source(name: str, source_dir: str, runner: CommandRunner) -> None:
    """Fail if an existing container does not have `source_dir` mounted.

    The generated container name encodes the commit, so reuse is normally safe.
    An explicit ``{gpu,xpu}.container_name`` does not, and reusing one that still
    has a previous commit's tree mounted would compare the wrong code.
    """
    result = runner.run(
        "docker inspect -f '{{range .Mounts}}{{.Source}}{{\"\\n\"}}{{end}}' "
        + shlex.quote(name),
        timeout=120,
    )
    if not result.ok:
        return  # cannot tell; the caller's own checks still apply
    sources = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if sources and source_dir not in sources:
        raise VLLMBuildError(
            f"Container {name} exists but has {sorted(sources)} mounted, not the "
            f"source tree {source_dir} for this commit. It was probably built for "
            f"another commit -- remove it (docker rm -f {name}) or pick another "
            "container_name."
        )


def _start_peer_container(
    name: str,
    image: str,
    device: str,
    source_dir: str,
    runner: CommandRunner,
    mounts: Tuple[str, ...] = (),
    extra_run_args: str = "",
) -> None:
    """Start an idle container with the source tree bind-mounted."""
    mount_args = [f"-v {shlex.quote(source_dir)}:{CONTAINER_SOURCE_DIR}"]
    mount_args += [f"-v {shlex.quote(m)}:{shlex.quote(m)}" for m in mounts if m]

    run_cmd = (
        f"docker run -d --name {shlex.quote(name)} "
        f"{' '.join(_DEVICE_RUN_ARGS[device])} "
        f"{' '.join(mount_args)} "
        + (f"{extra_run_args} " if extra_run_args else "")
        # Vendor images may start a notebook or a server; we want a plain idle
        # container the rest of the tool can docker exec into.
        + f"--entrypoint /bin/bash {shlex.quote(image)} -c 'sleep infinity'"
    )
    result = runner.run(run_cmd, timeout=900)
    if not result.ok:
        raise VLLMBuildError(
            f"Could not start container {name} from {image}: {result.stderr.strip()[:400]}"
        )
    if _container_state(name, runner) != "running":
        raise VLLMBuildError(
            f"Container {name} did not stay running. Check `docker logs {name}`."
        )


def build_peer(
    device: str,
    commit: str,
    source_dir: str,
    runner: CommandRunner,
    base_image: str = "",
    container: str = "",
    mounts: Tuple[str, ...] = (),
    extra_run_args: str = "",
    build_kernels: bool = False,
    rebuild: bool = False,
    allow_network: bool = True,
) -> VLLMBuild:
    """Produce a container running vLLM at `commit` for `device`.

    Reuse is layered so a rerun is cheap: a running container is used as is, a
    stopped one is restarted, an already-built image is started fresh, and only a
    cold cache pays for the install.

    Args:
        device: "cuda" or "xpu".
        source_dir: Host path of the per-commit checkout (bind-mounted at
            ``/workspace/vllm``); must be visible to `runner`'s host.
        base_image: Vendor PyTorch image; resolved to the newest tag when empty.
        build_kernels: CUDA only -- compile kernels instead of using the
            precompiled fast path.
        rebuild: Ignore a cached image and install again.

    Raises:
        VLLMBuildError: On any failure to pull, start, install, or locate vLLM.
    """
    if device not in _DEVICE_RUN_ARGS:
        raise VLLMBuildError(f"Unknown device {device!r} (expected 'cuda' or 'xpu')")

    name = container or peer_container_name(device, commit, build_kernels)
    target_image = build_image_tag(device, commit, build_kernels)
    precompiled = device == "cuda" and not build_kernels

    if not rebuild:
        state = _container_state(name, runner)
        if state is not None:
            _assert_container_has_source(name, source_dir, runner)
            if state == "running":
                logger.info(f"Reusing running {device} peer {name}")
                note = "reused the running container; not rebuilt"
            else:
                logger.info(f"Restarting {device} peer {name} (state={state})")
                start = runner.run(f"docker start {shlex.quote(name)}", timeout=600)
                if not start.ok:
                    raise VLLMBuildError(
                        f"Could not start {name}: {start.stderr.strip()[:300]}"
                    )
                note = "restarted an existing container; not rebuilt"

            # A failed install leaves the container behind on purpose, so it can
            # be inspected -- but it has the source mounted and no vLLM in it.
            # Reusing that would dead-end every later run on "vLLM is not
            # installed", so treat it as a cold cache and install again.
            if detect_vllm_path(name, runner):
                return _finish_build(
                    device, commit, base_image or "(cached)", target_image, name,
                    source_dir, runner, built=False, precompiled=precompiled,
                    note=note,
                )
            logger.warning(
                f"{name} exists but has no vLLM installed (leftover from a failed "
                "install); removing it and installing again"
            )
            runner.run(f"docker rm -f {shlex.quote(name)}", timeout=300)

        cached = runner.run(f"docker image inspect {shlex.quote(target_image)}", timeout=120)
        if cached.ok:
            logger.info(f"Starting {device} peer from cached image {target_image}")
            _start_peer_container(
                name, target_image, device, source_dir, runner, mounts, extra_run_args
            )
            return _finish_build(
                device, commit, base_image or "(cached)", target_image, name,
                source_dir, runner, built=False, precompiled=precompiled,
                note=f"reused cached image {target_image}",
            )

    if rebuild:
        runner.run(f"docker rm -f {shlex.quote(name)}", timeout=300)

    resolved_base, base_note = resolve_base_image(device, base_image, allow_network)
    logger.info(f"Pulling {resolved_base} for the {device} peer")
    pull = runner.run(f"docker pull {shlex.quote(resolved_base)}", timeout=7200)
    if not pull.ok:
        local = runner.run(f"docker image inspect {shlex.quote(resolved_base)}", timeout=120)
        if not local.ok:
            raise VLLMBuildError(
                f"Could not pull {resolved_base} and it is not present locally: "
                f"{pull.stderr.strip()[:300]}. Set "
                f"{'gpu' if device == 'cuda' else 'xpu'}.base_image to an image you can pull."
            )
        logger.warning(f"Pull of {resolved_base} failed; using the local copy")

    _start_peer_container(
        name, resolved_base, device, source_dir, runner, mounts, extra_run_args
    )

    script = _install_script(device, commit, build_kernels)
    timeout = _BUILD_TIMEOUT_COMPILE if (build_kernels and device == "cuda") else _BUILD_TIMEOUT_FAST
    logger.info(
        f"Installing vLLM @ {commit[:12]} into {name} "
        f"({'compiling kernels' if not precompiled else 'precompiled kernels'}); "
        f"up to {timeout // 60} min"
    )
    install = runner.run(
        f"docker exec {shlex.quote(name)} bash -lc {shlex.quote(script)}", timeout=timeout
    )
    if not install.ok:
        raise VLLMBuildError(
            f"Installing vLLM @ {commit[:12]} into {name} failed (exit {install.exit_code}).\n"
            f"stderr tail:\n{install.stderr.strip()[-2000:]}\n"
            f"The container is left running for inspection: docker exec -it {name} bash"
        )

    logger.info(f"Committing {name} to {target_image} so later runs skip the install")
    commit_result = runner.run(
        f"docker commit {shlex.quote(name)} {shlex.quote(target_image)}", timeout=1800
    )
    if not commit_result.ok:
        # Not fatal: the running container is usable, only the cache is lost.
        logger.warning(
            f"Could not commit {name} to {target_image}: {commit_result.stderr.strip()[:200]}"
        )

    note = base_note
    if precompiled:
        note = "; ".join(filter(None, [
            note,
            "CUDA kernels come from the nearest nightly wheel, not this commit "
            "(use --build-kernels for a real compile)",
        ]))
    return _finish_build(
        device, commit, resolved_base, target_image, name, source_dir, runner,
        built=True, precompiled=precompiled, note=note,
    )


def _finish_build(
    device: str,
    commit: str,
    base_image: str,
    image: str,
    container: str,
    source_dir: str,
    runner: CommandRunner,
    built: bool,
    precompiled: bool,
    note: str,
) -> VLLMBuild:
    """Confirm vLLM is importable in `container` and package up the result."""
    vllm_path = detect_vllm_path(container, runner)
    if not vllm_path:
        raise VLLMBuildError(
            f"vLLM is not installed in {container} after the build. "
            f"Inspect it with: docker exec -it {container} bash"
        )
    if vllm_path != CONTAINER_SOURCE_DIR:
        # A non-editable install would leave the patcher rewriting files nobody
        # imports, which looks like "the patch had no effect".
        note = "; ".join(filter(None, [
            note,
            f"vLLM resolves to {vllm_path}, not the mounted source tree "
            f"{CONTAINER_SOURCE_DIR} -- patches apply to {vllm_path}",
        ]))

    return VLLMBuild(
        device=device,
        commit=commit,
        base_image=base_image,
        image=image,
        container=container,
        vllm_path=vllm_path,
        source_dir=source_dir,
        built=built,
        precompiled=precompiled,
        note=note,
    )


# --------------------------------------------------------------------------
# wiring into DebugConfig
# --------------------------------------------------------------------------

def host_has_xpu(runner: CommandRunner) -> bool:
    """True if the host reachable through `runner` exposes an Intel GPU."""
    if runner.run("xpu-smi discovery", timeout=120).ok:
        return True
    return runner.run("test -e /dev/dri/renderD128", timeout=60).ok


def default_build_root(config: "DebugConfig") -> Path:
    """Where per-commit source trees live.

    The shared filesystem is preferred because a remote peer host can only see
    the tree if it is shared; a local-only fallback keeps the feature usable on
    a single box with no shared mount.
    """
    if config.vllm_build_root:
        return Path(config.vllm_build_root).expanduser()
    shared = Path(config.shared_fs) if config.shared_fs else None
    if shared and shared.is_dir():
        return shared / "accuracy_agent_builds"
    return Path.home() / ".cache" / "accuracy_agent" / "builds"


def _resolve_and_export(
    config: "DebugConfig",
    ref: str,
    build_root: Path,
    runner: CommandRunner,
) -> Tuple[str, Path]:
    """Resolve `ref` in a local vLLM clone and export it to its own checkout.

    Raises:
        VLLMBuildError: If no clone is available, or `ref` is not a commit of
            vllm-project/vllm even after fetching.
    """
    repo = find_vllm_repo(config.vllm_repo_path, runner)
    commit = resolve_commit(ref, repo, runner)
    source_dir = ensure_source_tree(repo, commit, build_root, runner)
    logger.info(f"vLLM commit {commit[:12]} exported to {source_dir}")
    return commit, source_dir


def autoconfigure_from_commit(
    config: "DebugConfig",
    launch: bool = True,
    allow_network: bool = True,
    on_skip: Optional[Callable[[str], None]] = None,
) -> Dict[str, VLLMBuild]:
    """Build both peers from ``config.vllm_commit`` and point `config` at them.

    Mutates, per side that was built: ``{gpu,xpu}_docker``, ``{gpu,xpu}_image``,
    ``{gpu,xpu}_vllm_path`` and ``{gpu,xpu}_inside_container`` (pinned False --
    a freshly launched container is never the one we run in).

    A side is skipped, not failed, when it is already configured explicitly or
    its hardware is absent, so a one-sided box still gets its own peer built.

    Returns:
        {"gpu": VLLMBuild, "xpu": VLLMBuild} for the sides that were built.

    Raises:
        VLLMBuildError: If the commit itself cannot be resolved or exported, or
            if a side that should have been built failed to build.
    """
    def skip(reason: str) -> None:
        logger.info(f"Skipping commit build: {reason}")
        if on_skip:
            on_skip(reason)

    builds: Dict[str, VLLMBuild] = {}
    if not config.vllm_commit:
        return builds
    if config.backend != "vllm":
        skip(f"backend {config.backend!r} is not vllm; vllm.commit needs the vLLM backend")
        return builds

    build_root = default_build_root(config)
    mounts = tuple(m for m in (config.shared_fs,) if m and Path(m).is_dir())
    sides = (
        ("gpu", "cuda", config.gpu_host, config.gpu_user, config.gpu_ssh_key_path,
         config.gpu_docker, config.gpu_base_image, config.gpu_container_name,
         config.gpu_docker_run_args, host_has_nvidia_gpu, "NVIDIA GPU"),
        ("xpu", "xpu", config.xpu_host, config.xpu_user, config.xpu_ssh_key_path,
         config.xpu_docker, config.xpu_base_image, config.xpu_container_name,
         config.xpu_docker_run_args, host_has_xpu, "Intel GPU"),
    )

    # Decide what is buildable BEFORE touching the repo: resolving and exporting
    # a commit is a multi-GB clone, and a box where neither side can be built
    # (no docker, no devices, both sides configured by hand) must not pay for it.
    plan = []
    runners = []
    try:
        for (side, device, host, user, key, existing, base_image, container_name,
             run_args, has_device, device_label) in sides:
            if existing:
                skip(f"{side}.docker={existing!r} is set explicitly; not building that side")
                continue
            if not is_local_host(host) and not str(build_root).startswith(str(config.shared_fs)):
                skip(
                    f"{side} host {host!r} is remote but the build root {build_root} is not "
                    f"under the shared filesystem {config.shared_fs!r}, so it cannot see the "
                    "source tree -- set vllm.build_root to a shared path"
                )
                continue

            runner = CommandRunner(host=host, user=user, ssh_key_path=key)
            runners.append(runner)
            if not docker_available(runner):
                skip(f"no usable docker CLI on the {side} host {host or 'localhost'}")
                continue
            if not has_device(runner):
                skip(
                    f"no {device_label} visible on {host or 'localhost'}; "
                    f"not building the {side} peer"
                )
                continue
            plan.append((side, device, runner, base_image, container_name, run_args))

        if not plan and launch:
            skip(f"no side left to build vLLM @ {config.vllm_commit} for")
            return builds

        local_runner = CommandRunner()
        runners.append(local_runner)
        commit, source_dir = _resolve_and_export(
            config, config.vllm_commit, build_root, local_runner
        )

        if not launch:
            return builds

        for side, device, runner, base_image, container_name, run_args in plan:
            build = build_peer(
                device,
                commit,
                str(source_dir),
                runner,
                base_image=base_image,
                container=container_name,
                mounts=mounts,
                extra_run_args=run_args,
                build_kernels=config.vllm_build_kernels,
                rebuild=config.vllm_build_rebuild,
                allow_network=allow_network,
            )
            setattr(config, f"{side}_docker", build.container)
            setattr(config, f"{side}_image", build.image)
            setattr(config, f"{side}_vllm_path", build.vllm_path)
            setattr(config, f"{side}_inside_container", False)
            builds[side] = build
            logger.info(f"{side} peer ready: {build.summary()}")
    finally:
        for runner in runners:
            runner.close()

    return builds


def autoconfigure_gpu_from_xpu_commit(
    config: "DebugConfig",
    launch: bool = True,
    allow_network: bool = True,
    on_skip: Optional[Callable[[str], None]] = None,
) -> GPUPeerFromXPU:
    """Build a CUDA peer running the same vLLM commit as the existing XPU container.

    This is the "I already have an XPU container, give me something to compare it
    against" path. The XPU side is left exactly as it is -- it is the subject of
    the comparison -- and only the GPU side is created:

      1. ask the XPU container which commit its vLLM is (``detect_vllm_commit``),
      2. resolve and export that commit from a local vllm-project/vllm clone,
      3. install it into the newest ``nvcr.io/nvidia/pytorch`` image (``gpu.base_image``
         to pin one), the same base the ``vllm.commit`` mode uses for CUDA.

    Mutates on success: ``gpu_docker``, ``gpu_image``, ``gpu_vllm_path``,
    ``gpu_inside_container`` (pinned False -- a freshly launched container is
    never the one we run in) and ``vllm_commit`` (so reports name the commit that
    was compared).

    Returns an empty result when the mode does not apply:
      - backend is not vLLM (this is vLLM-specific)
      - ``gpu_docker`` is already set (explicit config always wins)
      - ``gpu_auto_image`` is disabled
      - the XPU docker is not reachable locally (a remote XPU container cannot be
        interrogated cheaply, so its GPU peer is the user's to configure)
      - no docker CLI, or no NVIDIA GPU on the GPU host -- e.g. an XPU-only box,
        where the run should stay XPU-only
      - the GPU host is remote but the build root is not on the shared filesystem,
        so that host could not see the exported source tree

    An explicit ``gpu.image`` short-circuits all of it: that image is started as
    is, with nothing detected and nothing built.

    Args:
        on_skip: Called with the human-readable reason for each of those skips.

    Raises:
        VLLMBuildError: If the mode applies but fails -- commit undetectable, not
            a commit of vllm-project/vllm, or the build/launch failed.
    """
    def skip(reason: str) -> None:
        logger.info(f"Skipping GPU peer setup from the XPU container: {reason}")
        if on_skip:
            on_skip(reason)

    peer = GPUPeerFromXPU()

    if config.backend != "vllm":
        skip(f"backend {config.backend!r} is not vllm")
        return peer
    if config.gpu_docker:
        logger.debug(f"gpu_docker={config.gpu_docker!r} already set; not deriving a GPU peer")
        return peer
    if not getattr(config, "gpu_auto_image", False):
        skip("gpu_auto_image is disabled")
        return peer

    inside_xpu = running_inside_container(config.xpu_vllm_path)
    if not inside_xpu and not is_local_host(config.xpu_host):
        skip(
            f"the XPU docker runs on remote host {config.xpu_host!r}, so the vLLM "
            "commit it runs cannot be auto-detected -- set gpu.docker/gpu.image, or "
            "vllm.commit to build both sides from a known commit"
        )
        return peer
    if not inside_xpu and not config.xpu_docker:
        skip("no xpu.docker is configured")
        return peer

    xpu_runner = CommandRunner(host="", user=config.xpu_user)
    gpu_runner = CommandRunner(
        host=config.gpu_host,
        user=config.gpu_user,
        ssh_key_path=config.gpu_ssh_key_path,
    )
    local_runner = CommandRunner()

    try:
        if not inside_xpu and not docker_available(xpu_runner):
            skip("no usable docker CLI on this machine")
            return peer

        # Both checks come before any repo work: exporting a commit is a
        # multi-GB clone, and a box with no GPU to build for must not pay for it.
        if launch:
            if not docker_available(gpu_runner):
                skip(f"no usable docker CLI on GPU host {config.gpu_host or 'localhost'}")
                return peer
            if not host_has_nvidia_gpu(gpu_runner):
                skip(
                    f"no NVIDIA GPU visible on {config.gpu_host or 'localhost'} -- "
                    "leaving the GPU side unconfigured (XPU-only run)"
                )
                return peer

        mounts = tuple(m for m in (config.shared_fs,) if m and Path(m).is_dir())

        # Escape hatch: a ready-made GPU image is used as is. Nothing is detected
        # and nothing is built, so the two sides' vLLM code is the user's to vouch
        # for.
        if config.gpu_image:
            peer.image = config.gpu_image
            if not launch:
                return peer
            container, _launched = ensure_gpu_container(
                config.gpu_image,
                gpu_runner,
                container=config.gpu_container_name or None,
                mounts=mounts,
                extra_run_args=config.gpu_docker_run_args,
            )
            vllm_path = detect_vllm_path(container, gpu_runner)
            if not vllm_path:
                raise VLLMBuildError(
                    f"Could not locate the vllm package inside container {container}. "
                    f"Is {config.gpu_image} a vLLM image?"
                )
            config.gpu_docker = container
            config.gpu_vllm_path = vllm_path
            config.gpu_inside_container = False
            peer.container = container
            peer.note = "gpu.image was pinned, so the GPU peer's vLLM was not verified"
            logger.info(f"GPU side configured from gpu.image: container={container}")
            return peer

        try:
            xpu_commit = detect_vllm_commit(
                config.xpu_docker,
                xpu_runner,
                inside_container=inside_xpu,
                vllm_path=config.xpu_vllm_path,
            )
        except DockerProbeError as e:
            raise VLLMBuildError(str(e)) from e
        peer.xpu_commit = xpu_commit

        build_root = default_build_root(config)
        if not is_local_host(config.gpu_host) and not str(build_root).startswith(str(config.shared_fs)):
            skip(
                f"GPU host {config.gpu_host!r} is remote but the build root {build_root} "
                f"is not under the shared filesystem {config.shared_fs!r}, so it cannot "
                "see the source tree -- set vllm.build_root to a shared path"
            )
            return peer

        commit, source_dir = _resolve_and_export(
            config, xpu_commit.sha, build_root, local_runner
        )
        peer.commit = commit
        # Record it so every later report names the commit that was compared,
        # exactly as if it had been passed as vllm.commit.
        config.vllm_commit = commit

        if not launch:
            return peer

        build = build_peer(
            "cuda",
            commit,
            str(source_dir),
            gpu_runner,
            base_image=config.gpu_base_image,
            container=config.gpu_container_name,
            mounts=mounts,
            extra_run_args=config.gpu_docker_run_args,
            build_kernels=config.vllm_build_kernels,
            rebuild=config.vllm_build_rebuild,
            allow_network=allow_network,
        )
        config.gpu_docker = build.container
        config.gpu_image = build.image
        config.gpu_vllm_path = build.vllm_path
        config.gpu_inside_container = False
        peer.build = build
        logger.info(f"GPU peer ready: {build.summary()}")
        return peer

    finally:
        xpu_runner.close()
        gpu_runner.close()
        local_runner.close()


def maybe_autoconfigure_peers(
    config: "DebugConfig",
    launch: bool = True,
    allow_network: bool = True,
    on_skip: Optional[Callable[[str], None]] = None,
) -> PeerSetup:
    """Best-effort automatic peer configuration -- the single entry point.

    Dispatches on how the peers were requested:
      - ``vllm.commit`` set: build BOTH sides from that commit
        (``autoconfigure_from_commit``)
      - otherwise: keep the existing XPU container and build a CUDA peer running
        the same commit it does (``autoconfigure_gpu_from_xpu_commit``)

    Never raises: automatic setup is a convenience, so a failure must not abort a
    run that could still proceed (XPU-only, or explicitly configured peers).
    Safe to call from several entry points; it runs at most once per config.
    """
    setup = PeerSetup()
    if getattr(config, "_peer_autoconfig_attempted", False):
        return setup
    config._peer_autoconfig_attempted = True

    if getattr(config, "vllm_commit", ""):
        setup.commit = config.vllm_commit
        try:
            setup.builds = autoconfigure_from_commit(
                config, launch=launch, allow_network=allow_network,
                on_skip=setup.skipped.append,
            )
        except Exception as e:
            message = f"Building vLLM @ {config.vllm_commit} failed: {e}"
            logger.warning(f"{message}. Continuing without built peers.")
            setup.skipped.append(message)
    else:
        try:
            peer = autoconfigure_gpu_from_xpu_commit(
                config, launch=launch, allow_network=allow_network,
                on_skip=setup.skipped.append,
            )
            setup.xpu_commit = peer.xpu_commit
            setup.commit = peer.commit or None
            setup.gpu_container = peer.container
            setup.gpu_image = peer.image
            if peer.build is not None:
                setup.builds["gpu"] = peer.build
            if peer.note:
                setup.notes.append(peer.note)
        except Exception as e:
            message = f"Building a GPU peer for the XPU container failed: {e}"
            logger.warning(f"{message}. Continuing without a GPU peer.")
            setup.skipped.append(message)

    if on_skip:
        for reason in setup.skipped:
            on_skip(reason)
    return setup
