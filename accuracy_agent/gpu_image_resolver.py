"""Derive the GPU docker image/container from the vLLM version in the XPU docker.

A GPU-vs-XPU comparison is only meaningful when both sides run the SAME vLLM
version -- otherwise a "divergence" may just be a version difference. Looking up
the XPU container's vLLM version by hand and then finding the matching
vllm/vllm-openai release is the most error-prone part of the setup, so when the
XPU docker is reachable locally we do it automatically:

  1. query the XPU container for its installed vLLM version
  2. normalize that to a RELEASE version (drop dev/rc/+local parts)
  3. pick the matching ``vllm/vllm-openai`` release tag -- never a nightly build
     -- verified against Docker Hub when the network allows, constructed offline
     otherwise
  4. start (or reuse) a GPU container from that image and detect the vLLM
     package path inside it, so the rest of the tool works unchanged

Only the *local* case is automated: if the XPU docker lives on a remote host we
cannot cheaply interrogate it, so the caller is expected to configure the GPU
side by hand (see ``autoconfigure_gpu_docker`` for every skip condition).

Nothing here imports paramiko at module level -- a purely local setup should not
need it.
"""

import json
import logging
import re
import shlex
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from accuracy_agent.config import DebugConfig

logger = logging.getLogger(__name__)

#: Official vLLM CUDA server image. Release tags look like ``v0.11.0`` /
#: ``v0.8.5.post1``; nightly/dev tags (``nightly``, ``nightly-<sha>``, ``latest``)
#: are deliberately excluded by _RELEASE_TAG_RE.
VLLM_OPENAI_REPO = "vllm/vllm-openai"

_DOCKER_HUB_TAGS_URL = "https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100"

#: Hostnames that mean "this machine", so the docker CLI can be used directly.
_LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1", "::1"}

#: Matches only published RELEASE tags: vX.Y.Z with an optional .postN.
_RELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?$")

#: Matches a PEP 440-ish version, e.g. 0.11.0, 0.11.1rc2.dev0, 0.8.5.post1.
_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?(?P<pre>.*)$")

_PRERELEASE_RE = re.compile(r"(rc|dev|a|b|alpha|beta)", re.IGNORECASE)

#: Probes tried in order inside the XPU container. The importlib.metadata ones
#: come first because they read package metadata only -- importing vllm pulls in
#: torch and the whole device runtime, which is slow and can fail on a busy card.
_VERSION_PROBES = (
    "python3 -c \"import importlib.metadata as m; print(m.version('vllm'))\"",
    "python -c \"import importlib.metadata as m; print(m.version('vllm'))\"",
    "pip show vllm",
    'python3 -c "import vllm; print(vllm.__version__)"',
)

#: Probes for the directory that CONTAINS the vllm package (what vllm_path means
#: everywhere else in this tool, i.e. site-packages for a wheel install).
#: find_spec resolves the location without executing the module.
_VLLM_PATH_PROBES = (
    "python3 -c \"import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec('vllm').origin)))\"",
    'python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))"',
)

#: Flags for the auto-launched GPU container. It is started as an idle shell
#: (the vllm-openai ENTRYPOINT is the API server, which we do not want) so the
#: existing `docker exec` code paths can patch and drive it.
_DEFAULT_RUN_ARGS = ("--gpus", "all", "--ipc=host", "--shm-size=16g")


class GPUImageResolutionError(RuntimeError):
    """Raised when the GPU image/container could not be determined or started."""


@dataclass
class CommandResult:
    """Result of a shell command run on a docker host."""
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class GPUImageResolution:
    """Outcome of resolving (and optionally launching) the GPU docker."""
    xpu_vllm_version: str          # raw version string reported by the XPU container
    release_version: str           # normalized release, e.g. "0.11.0"
    tag: str                       # chosen release tag, e.g. "v0.11.0"
    image: str                     # full image ref, e.g. "vllm/vllm-openai:v0.11.0"
    verified: bool = False         # tag confirmed to exist on Docker Hub
    exact: bool = True             # tag matches the XPU release version exactly
    note: str = ""
    container: Optional[str] = None    # GPU container name (once launched/reused)
    vllm_path: Optional[str] = None    # vLLM location inside that container
    launched: bool = False             # True if we created the container
    mounts: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"XPU vLLM {self.xpu_vllm_version} -> {self.image}"]
        if not self.exact:
            bits.append("(nearest release)")
        if not self.verified:
            bits.append("(unverified: Docker Hub not consulted)")
        if self.container:
            bits.append(f"container={self.container}")
        return " ".join(bits)


@dataclass(frozen=True)
class VLLMVersion:
    """A parsed vLLM version, reduced to what release matching needs."""
    raw: str
    major: int
    minor: int
    patch: int
    post: Optional[int] = None
    is_prerelease: bool = False

    @property
    def release(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    @property
    def release_str(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}.post{self.post}" if self.post is not None else base

    @property
    def tag(self) -> str:
        """Tag this version would carry if published as-is."""
        return f"v{self.release_str}"


def is_local_host(host: str) -> bool:
    """True if `host` refers to the machine we are running on.

    An empty host counts as local: that is how the config expresses "no separate
    host", and it is the case this module automates.
    """
    candidate = (host or "").strip().lower()
    if candidate in _LOCAL_HOSTS:
        return True
    hostname = socket.gethostname().lower()
    return candidate in {hostname, hostname.split(".")[0]}


class CommandRunner:
    """Run shell commands on the host owning a docker daemon (local or via SSH).

    The local path uses subprocess, so a fully local setup never needs paramiko.
    The SSH path is lazy for the same reason.
    """

    def __init__(self, host: str = "", user: str = "root", ssh_key_path: Optional[str] = None):
        self.host = host or ""
        self.user = user or "root"
        self.ssh_key_path = ssh_key_path
        self.is_local = is_local_host(self.host)
        self._ssh = None

    def run(self, cmd: str, timeout: int = 300) -> CommandResult:
        if self.is_local:
            return self._run_local(cmd, timeout)
        return self._run_ssh(cmd, timeout)

    def _run_local(self, cmd: str, timeout: int) -> CommandResult:
        logger.debug(f"Running locally: {cmd}")
        try:
            proc = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return CommandResult(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", f"Command timed out after {timeout}s: {cmd}")
        except Exception as e:  # pragma: no cover - defensive
            return CommandResult(-1, "", f"Local execution failed: {e}")

    def _run_ssh(self, cmd: str, timeout: int) -> CommandResult:
        if self._ssh is None:
            import paramiko  # imported lazily: only remote hosts need it

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                self.host,
                username=self.user,
                key_filename=self.ssh_key_path or None,
                timeout=30,
            )
            self._ssh = client
            logger.info(f"Connected to {self.user}@{self.host} for docker control")

        logger.debug(f"Running on {self.host}: {cmd}")
        _stdin, stdout, stderr = self._ssh.exec_command(cmd, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        return CommandResult(exit_code, stdout.read().decode(), stderr.read().decode())

    def close(self) -> None:
        if self._ssh is not None:
            self._ssh.close()
            self._ssh = None


def docker_available(runner: CommandRunner) -> bool:
    """True if a usable docker CLI + daemon is reachable through `runner`."""
    return runner.run("docker version --format '{{.Server.Version}}'", timeout=60).ok


def host_has_nvidia_gpu(runner: CommandRunner) -> bool:
    """True if the host reachable through `runner` exposes an NVIDIA GPU.

    Used as a guard: on an XPU-only machine there is nothing to compare against,
    so auto-configuration must stay out of the way and leave the run XPU-only.
    """
    result = runner.run("nvidia-smi -L", timeout=60)
    return result.ok and "GPU" in result.stdout


def running_inside_container(vllm_path: str) -> bool:
    """True if this process is running inside the container it wants to query.

    Mirrors VLLMPatcher._detect_local: a docker environment plus the vllm package
    present under `vllm_path` means we can probe without any docker exec.
    """
    try:
        return Path("/.dockerenv").exists() and (Path(vllm_path) / "vllm").exists()
    except OSError:  # pragma: no cover - defensive
        return False


def _docker_exec(container: str, command: str) -> str:
    return f"docker exec {shlex.quote(container)} bash -lc {shlex.quote(command)}"


def _parse_version_output(output: str) -> Optional[str]:
    """Pull a version string out of probe output (bare version or `pip show`)."""
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("version:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                return candidate
        elif re.match(r"^v?\d+\.\d+", line):
            return line
    return None


def detect_vllm_version(
    container: str,
    runner: CommandRunner,
    inside_container: bool = False,
) -> str:
    """Query a container for the vLLM version it is running.

    Args:
        container: Container name (ignored when `inside_container`).
        runner: Runner for the host owning that container.
        inside_container: True when this process runs inside `container`, in
            which case the probe is executed directly instead of via docker exec.

    Returns:
        Raw version string as reported by the container, e.g. "0.11.1rc2.dev0+xpu".

    Raises:
        GPUImageResolutionError: If no probe yielded a version.
    """
    errors = []
    for probe in _VERSION_PROBES:
        cmd = probe if inside_container else _docker_exec(container, probe)
        result = runner.run(cmd, timeout=300)
        if not result.ok:
            errors.append(f"{probe!r} -> exit {result.exit_code}: {result.stderr.strip()[:200]}")
            continue
        version = _parse_version_output(result.stdout)
        if version:
            logger.info(f"XPU container {container or '(local)'} runs vLLM {version}")
            return version
        errors.append(f"{probe!r} -> unparseable output: {result.stdout.strip()[:200]}")

    raise GPUImageResolutionError(
        "Could not determine the vLLM version inside "
        f"{container or 'the local environment'}. Tried:\n  " + "\n  ".join(errors)
    )


def detect_vllm_path(container: str, runner: CommandRunner) -> Optional[str]:
    """Find the directory containing the vllm package inside `container`."""
    for probe in _VLLM_PATH_PROBES:
        result = runner.run(_docker_exec(container, probe), timeout=300)
        if result.ok:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("/"):
                    return line
    return None


def parse_vllm_version(raw: str) -> VLLMVersion:
    """Parse a vLLM version string down to its release identity.

    The local part (everything after "+", e.g. "+xpu", "+g1a2b3c4") is dropped:
    it identifies a build, never a release. A dev/rc suffix marks the version as
    a PRE-release of that number, which matters for tag selection -- a
    0.11.1.dev build predates the v0.11.1 image.

    Raises:
        ValueError: If `raw` is not a recognizable version.
    """
    text = (raw or "").strip()
    base = text.split("+", 1)[0]
    match = _VERSION_RE.match(base)
    if not match:
        raise ValueError(f"Unrecognized vLLM version string: {raw!r}")

    pre = match.group("pre") or ""
    post = match.group(4)
    return VLLMVersion(
        raw=text,
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
        post=int(post) if post is not None else None,
        is_prerelease=bool(_PRERELEASE_RE.search(pre)),
    )


def _tag_sort_key(tag_info: Tuple[int, int, int, Optional[int], str]):
    major, minor, patch, post, _name = tag_info
    return (major, minor, patch, -1 if post is None else post)


def fetch_release_tags(
    repo: str = VLLM_OPENAI_REPO,
    max_pages: int = 10,
    timeout: int = 10,
) -> List[Tuple[int, int, int, Optional[int], str]]:
    """List published RELEASE tags of `repo` from the Docker Hub API.

    Nightly, dev, rc and architecture-suffixed tags are filtered out by
    _RELEASE_TAG_RE, so only real releases can ever be selected.

    Returns:
        Tags as (major, minor, patch, post, name), ascending by version.

    Raises:
        urllib.error.URLError / OSError / ValueError on network or payload
        problems -- callers are expected to degrade to offline construction.
    """
    url = _DOCKER_HUB_TAGS_URL.format(repo=repo)
    tags: List[Tuple[int, int, int, Optional[int], str]] = []
    pages = 0

    while url and pages < max_pages:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310 - fixed https URL
            payload = json.load(response)

        for entry in payload.get("results", []):
            name = (entry.get("name") or "").strip()
            match = _RELEASE_TAG_RE.match(name)
            if not match:
                continue
            post = match.group(4)
            tags.append((
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(post) if post is not None else None,
                name,
            ))

        url = payload.get("next")
        pages += 1

    tags.sort(key=_tag_sort_key)
    logger.debug(f"Found {len(tags)} release tags for {repo}")
    return tags


def select_release_tag(
    version: VLLMVersion,
    available_tags: Sequence[Tuple[int, int, int, Optional[int], str]],
) -> Tuple[Optional[str], bool, str]:
    """Pick the release tag matching `version` from `available_tags`.

    Selection rules:
      - final release: exact (release, post) match wins; otherwise the highest
        release <= this version (a newer patch may simply not be published as an
        image yet).
      - pre-release (dev/rc): the build predates its own version number, so the
        highest release STRICTLY BELOW it is the closest published match.

    Returns:
        (tag or None if nothing suitable, exact-match flag, explanatory note).
    """
    if not available_tags:
        return None, False, "no release tags available"

    if version.is_prerelease:
        lower = [t for t in available_tags if (t[0], t[1], t[2]) < version.release]
        if lower:
            tag = max(lower, key=_tag_sort_key)[4]
            return tag, False, (
                f"XPU runs pre-release build {version.raw}, which predates "
                f"v{version.release_str}; using closest published release {tag}"
            )
        return None, False, (
            f"no published release below pre-release build {version.raw}"
        )

    for major, minor, patch, post, name in available_tags:
        if (major, minor, patch) == version.release and post == version.post:
            return name, True, ""

    lower = [
        t for t in available_tags
        if _tag_sort_key(t) <= _tag_sort_key(
            (version.major, version.minor, version.patch, version.post, "")
        )
    ]
    if lower:
        tag = max(lower, key=_tag_sort_key)[4]
        return tag, False, (
            f"no {VLLM_OPENAI_REPO}:{version.tag} release image is published; "
            f"using closest published release {tag}"
        )

    return None, False, (
        f"no published release at or below v{version.release_str}"
    )


def image_tag(image: str) -> str:
    """Tag part of an image reference ("latest" when untagged).

    Splits on the last path segment first so a registry port (myreg:5000/vllm)
    is not mistaken for a tag.
    """
    last_segment = image.rsplit("/", 1)[-1]
    return last_segment.rsplit(":", 1)[-1] if ":" in last_segment else "latest"


def resolve_gpu_image(
    version: VLLMVersion,
    repo: str = VLLM_OPENAI_REPO,
    allow_network: bool = True,
    tag_fetcher=None,
) -> GPUImageResolution:
    """Map an XPU vLLM version onto a vllm/vllm-openai RELEASE image.

    Consults Docker Hub when `allow_network` is set so the chosen tag is known
    to exist; on any network failure it falls back to constructing the tag from
    the version (flagged ``verified=False``) rather than failing the run.

    Args:
        tag_fetcher: Override for the tag source (defaults to fetch_release_tags).
    """
    tags: Sequence = ()
    fetch_error = ""

    if allow_network:
        try:
            tags = (tag_fetcher or fetch_release_tags)(repo)
        except Exception as e:  # network down, proxy, API change, ...
            fetch_error = str(e)
            logger.warning(
                f"Could not list {repo} tags on Docker Hub ({e}); "
                "falling back to constructing the tag from the vLLM version"
            )

    if tags:
        tag, exact, note = select_release_tag(version, tags)
        if tag:
            if note:
                logger.info(note)
            return GPUImageResolution(
                xpu_vllm_version=version.raw,
                release_version=version.release_str,
                tag=tag,
                image=f"{repo}:{tag}",
                verified=True,
                exact=exact,
                note=note,
            )
        note = f"{note}; constructed {version.tag} from the vLLM version instead"
    elif fetch_error:
        note = f"Docker Hub unreachable ({fetch_error}); constructed tag from the vLLM version"
    elif allow_network:
        note = "Docker Hub returned no release tags; constructed tag from the vLLM version"
    else:
        note = "Docker Hub lookup disabled; constructed tag from the vLLM version"

    logger.warning(note)
    return GPUImageResolution(
        xpu_vllm_version=version.raw,
        release_version=version.release_str,
        tag=version.tag,
        image=f"{repo}:{version.tag}",
        verified=False,
        exact=not version.is_prerelease,
        note=note,
    )


def container_name_for(tag: str, prefix: str = "accuracy_agent_gpu") -> str:
    """Deterministic container name for a tag, so reruns reuse one container."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
    return f"{prefix}_{safe}"


def _container_state(container: str, runner: CommandRunner) -> Optional[str]:
    """Return "running"/"exited"/... for `container`, or None if absent."""
    result = runner.run(
        f"docker inspect -f '{{{{.State.Status}}}}' {shlex.quote(container)}",
        timeout=60,
    )
    if not result.ok:
        return None
    status = result.stdout.strip()
    return status or None


def ensure_gpu_container(
    image: str,
    runner: CommandRunner,
    container: Optional[str] = None,
    mounts: Iterable[str] = (),
    extra_run_args: str = "",
    pull: bool = True,
) -> Tuple[str, bool]:
    """Make sure a usable GPU container exists for `image`, creating it if needed.

    The container is deterministically named after the image tag and started as
    an idle shell, so repeated runs reuse it instead of re-pulling a multi-GB
    image. An existing-but-stopped container is restarted.

    Args:
        image: Image reference, e.g. "vllm/vllm-openai:v0.11.0".
        runner: Runner for the GPU host.
        container: Container name override (default derived from the tag).
        mounts: Host paths to bind-mount at the same path inside the container
            (the shared filesystem holding the model and outputs).
        extra_run_args: Additional raw `docker run` flags.
        pull: Pull the image first; a pull failure is tolerated when the image
            is already present locally.

    Returns:
        (container name, True if we created it).

    Raises:
        GPUImageResolutionError: If the image is unavailable or the container
            could not be started.
    """
    name = container or container_name_for(image_tag(image))

    state = _container_state(name, runner)
    if state == "running":
        logger.info(f"Reusing running GPU container {name} ({image})")
        return name, False
    if state is not None:
        logger.info(f"Starting existing GPU container {name} (state={state})")
        result = runner.run(f"docker start {shlex.quote(name)}", timeout=300)
        if not result.ok:
            raise GPUImageResolutionError(
                f"Could not start existing container {name}: {result.stderr.strip()}"
            )
        return name, False

    if pull:
        logger.info(f"Pulling {image} (this can take a while on first use)...")
        pull_result = runner.run(f"docker pull {shlex.quote(image)}", timeout=3600)
        if not pull_result.ok:
            local = runner.run(f"docker image inspect {shlex.quote(image)}", timeout=60)
            if not local.ok:
                raise GPUImageResolutionError(
                    f"Could not pull {image} and it is not present locally: "
                    f"{pull_result.stderr.strip()}"
                )
            logger.warning(f"Pull of {image} failed; using the local copy")

    mount_args = " ".join(
        f"-v {shlex.quote(m)}:{shlex.quote(m)}" for m in mounts if m
    )
    run_cmd = (
        f"docker run -d --name {shlex.quote(name)} "
        f"{' '.join(_DEFAULT_RUN_ARGS)} "
        + (f"{mount_args} " if mount_args else "")
        + (f"{extra_run_args} " if extra_run_args else "")
        # The vllm-openai ENTRYPOINT is the OpenAI API server; we need an idle
        # container to docker exec into instead.
        + f"--entrypoint /bin/bash {shlex.quote(image)} -c 'sleep infinity'"
    )

    logger.info(f"Launching GPU container {name} from {image}")
    result = runner.run(run_cmd, timeout=600)
    if not result.ok:
        raise GPUImageResolutionError(
            f"Failed to launch GPU container from {image}: {result.stderr.strip()}"
        )

    if _container_state(name, runner) != "running":
        raise GPUImageResolutionError(
            f"GPU container {name} did not stay running after launch. "
            f"Check `docker logs {name}`."
        )

    logger.info(f"GPU container {name} is running")
    return name, True


def autoconfigure_gpu_docker(
    config: "DebugConfig",
    launch: bool = True,
    allow_network: bool = True,
    on_skip: Optional[Callable[[str], None]] = None,
) -> Optional[GPUImageResolution]:
    """Fill in the GPU docker side of `config` from the XPU container's vLLM version.

    Mutates `config` on success: ``gpu_image``, ``gpu_docker``, ``gpu_vllm_path``
    and ``gpu_inside_container`` (pinned to False -- the auto-launched container
    is by definition a different container than the one we run in).

    Returns None when auto-configuration does not apply:
      - backend is not vLLM (image/version matching is vLLM-specific)
      - ``gpu_docker`` is already set (explicit config always wins)
      - ``gpu_auto_image`` is disabled
      - the XPU docker is not reachable locally (remote XPU hosts are the user's
        to configure)
      - no docker CLI, or no NVIDIA GPU on the GPU host -- e.g. an XPU-only box,
        where the run should stay XPU-only

    Args:
        on_skip: Called with the human-readable reason for each of those skips,
            so a caller can report it without depending on logging config.

    Raises:
        GPUImageResolutionError: If auto-configuration applies but fails (version
            undetectable, image unavailable, container will not start).
    """
    def skip(reason: str) -> None:
        logger.info(f"Skipping GPU image resolution: {reason}")
        if on_skip:
            on_skip(reason)

    if config.backend != "vllm":
        skip(f"backend {config.backend!r} is not vllm")
        return None

    if config.gpu_docker:
        logger.debug(f"gpu_docker={config.gpu_docker!r} already set; skipping GPU image resolution")
        return None

    if not getattr(config, "gpu_auto_image", False):
        skip("gpu_auto_image is disabled")
        return None

    inside_xpu = running_inside_container(config.xpu_vllm_path)
    if not inside_xpu and not is_local_host(config.xpu_host):
        skip(
            f"the XPU docker runs on remote host {config.xpu_host!r}, so its vLLM "
            "version cannot be auto-detected -- set gpu.docker or gpu.image explicitly"
        )
        return None

    if not inside_xpu and not config.xpu_docker:
        skip("no xpu.docker is configured")
        return None

    xpu_runner = CommandRunner(host="", user=config.xpu_user)
    gpu_runner = CommandRunner(
        host=config.gpu_host,
        user=config.gpu_user,
        ssh_key_path=config.gpu_ssh_key_path,
    )

    try:
        if not inside_xpu and not docker_available(xpu_runner):
            skip("no usable docker CLI on this machine")
            return None

        if launch:
            if not docker_available(gpu_runner):
                skip(f"no usable docker CLI on GPU host {config.gpu_host or 'localhost'}")
                return None
            if not host_has_nvidia_gpu(gpu_runner):
                skip(
                    f"no NVIDIA GPU visible on {config.gpu_host or 'localhost'} -- "
                    "leaving the GPU side unconfigured (XPU-only run)"
                )
                return None

        # An explicit gpu.image short-circuits version detection: the user has
        # already chosen the image, we only need a container for it.
        if config.gpu_image:
            resolution = GPUImageResolution(
                xpu_vllm_version="(not queried)",
                release_version="",
                tag=image_tag(config.gpu_image),
                image=config.gpu_image,
                verified=False,
                note="using the image from config (gpu.image)",
            )
        else:
            raw_version = detect_vllm_version(
                config.xpu_docker, xpu_runner, inside_container=inside_xpu
            )
            try:
                version = parse_vllm_version(raw_version)
            except ValueError as e:
                raise GPUImageResolutionError(str(e)) from e
            resolution = resolve_gpu_image(version, allow_network=allow_network)

        config.gpu_image = resolution.image
        logger.info(f"Matched GPU image for XPU vLLM: {resolution.image}")

        if not launch:
            return resolution

        mounts = [m for m in (config.shared_fs,) if m]
        resolution.mounts = mounts
        container, launched = ensure_gpu_container(
            resolution.image,
            gpu_runner,
            container=config.gpu_container_name or None,
            mounts=mounts,
            extra_run_args=config.gpu_docker_run_args,
        )
        resolution.container = container
        resolution.launched = launched

        vllm_path = detect_vllm_path(container, gpu_runner)
        if not vllm_path:
            raise GPUImageResolutionError(
                f"Could not locate the vllm package inside container {container}. "
                f"Is {resolution.image} a vLLM image?"
            )
        resolution.vllm_path = vllm_path

        config.gpu_docker = container
        config.gpu_vllm_path = vllm_path
        # The GPU container is never the container this process runs in, and its
        # vllm_path (site-packages) may coincidentally exist here too -- pin the
        # flag so the patcher cannot mis-detect it as local execution.
        config.gpu_inside_container = False

        logger.info(
            f"GPU side auto-configured: container={container}, vllm_path={vllm_path}"
        )
        return resolution

    finally:
        xpu_runner.close()
        gpu_runner.close()


def maybe_autoconfigure_gpu_docker(
    config: "DebugConfig",
    launch: bool = True,
    allow_network: bool = True,
    on_skip: Optional[Callable[[str], None]] = None,
) -> Optional[GPUImageResolution]:
    """Best-effort ``autoconfigure_gpu_docker``: warn instead of raising.

    Auto-configuration is a convenience, so a failure must not abort a run that
    would otherwise proceed (XPU-only extraction, or an explicitly configured
    GPU side). Returns None on failure, after logging why.

    Safe to call from several entry points (CLI, Bisector): it runs at most once
    per config, so a failed attempt is not retried (and a multi-GB image pull is
    not repeated) later in the same run.
    """
    if getattr(config, "_gpu_autoconfig_attempted", False):
        return None
    config._gpu_autoconfig_attempted = True

    try:
        return autoconfigure_gpu_docker(
            config, launch=launch, allow_network=allow_network, on_skip=on_skip
        )
    except Exception as e:
        message = f"GPU docker auto-configuration failed: {e}"
        logger.warning(f"{message}. Continuing without an auto-configured GPU peer.")
        if on_skip:
            on_skip(message)
        return None
