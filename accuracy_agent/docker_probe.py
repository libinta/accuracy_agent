"""Reach the docker hosts that own the comparison peers, and probe them.

Everything here is plumbing shared by the peer setup paths in
``vllm_source_builder``:

  * ``CommandRunner`` runs a shell command on the host owning a docker daemon --
    locally via subprocess, remotely over SSH. Nothing imports paramiko at
    module level, so a purely local setup does not need it.
  * host probes (``docker_available``, ``host_has_nvidia_gpu``) decide whether a
    side can be configured at all.
  * container probes say what a *running* peer contains: which vLLM version
    (``detect_vllm_version``), which exact commit (``detect_vllm_commit``), and
    where the package lives (``detect_vllm_path``).
  * ``ensure_gpu_container`` starts an idle container from a ready-made image --
    the ``gpu.image`` escape hatch, when the GPU peer should not be built.

``detect_vllm_commit`` is what makes an existing XPU container comparable. A
GPU-vs-XPU divergence only means something when both sides run the SAME vLLM
code, so the GPU peer is built from the commit the XPU container reports -- not
from whichever published release happens to sit closest to its version number,
which would leave hundreds of commits of difference in the comparison.
"""

import logging
import re
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Hostnames that mean "this machine", so the docker CLI can be used directly.
_LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1", "::1"}

#: Prefix for every in-container python probe. `python -c` prepends the working
#: directory to sys.path, and a peer container has the vLLM *source root* mounted
#: at /workspace/vllm while NGC images set WORKDIR=/workspace. From there `vllm`
#: resolves to that directory as a namespace package -- origin and __file__ are
#: both None -- hiding the real install, so a freshly built peer reports "vLLM is
#: not installed". Probing from ``/`` cannot be shadowed by a mount.
_CWD_NEUTRAL = "cd / && "

#: Probes tried in order inside the container. The importlib.metadata ones come
#: first because they read package metadata only -- importing vllm pulls in torch
#: and the whole device runtime, which is slow and can fail on a busy card.
_VERSION_PROBES = (
    _CWD_NEUTRAL + "python3 -c \"import importlib.metadata as m; print(m.version('vllm'))\"",
    _CWD_NEUTRAL + "python -c \"import importlib.metadata as m; print(m.version('vllm'))\"",
    _CWD_NEUTRAL + "pip show vllm",
    _CWD_NEUTRAL + 'python3 -c "import vllm; print(vllm.__version__)"',
)

#: Probes for the directory that CONTAINS the vllm package (what vllm_path means
#: everywhere else in this tool, i.e. site-packages for a wheel install, and the
#: source root for an editable one). find_spec resolves the location without
#: executing the module.
_VLLM_PATH_PROBES = (
    _CWD_NEUTRAL + "python3 -c \"import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec('vllm').origin)))\"",
    _CWD_NEUTRAL + 'python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))"',
)

#: A git sha, abbreviated (as stamped into a version) or full.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

#: setuptools-scm stamps the abbreviated sha into the local part of a dev
#: version: the "g7794b1e08" of "0.26.1rc1.dev353+g7794b1e08.xpu". vLLM's own
#: collect_env reports the commit from exactly this.
_LOCAL_SHA_RE = re.compile(r"[+.]g(?P<sha>[0-9a-f]{7,40})(?=$|[.+])")

#: Flags for a container started from a ready-made image. It is an idle shell
#: (the vllm-openai ENTRYPOINT is the API server, which we do not want) so the
#: existing `docker exec` code paths can patch and drive it.
_DEFAULT_RUN_ARGS = ("--gpus", "all", "--ipc=host", "--shm-size=16g")


class DockerProbeError(RuntimeError):
    """Raised when a host or container could not be probed or started."""


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
class VLLMCommit:
    """The exact vLLM commit a container runs, and how we know."""
    sha: str                # full 40-char sha from git, abbreviated from a version
    source: str             # human-readable origin, e.g. "git checkout /workspace/vllm"
    version: str = ""       # raw vLLM version string, when it was read
    dirty: bool = False     # the git checkout has uncommitted changes on top
    path: str = ""          # directory the commit was read from (git source only)

    @property
    def abbreviated(self) -> bool:
        return len(self.sha) < 40

    def summary(self) -> str:
        bits = [f"XPU container runs vLLM @ {self.sha}"]
        if self.version:
            bits.append(f"(version {self.version})")
        bits.append(f"[{self.source}]")
        return " ".join(bits)


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
    so peer configuration must stay out of the way and leave the run XPU-only.
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


def _in_container(container: str, command: str, inside_container: bool) -> str:
    """The command to run, wrapped in a docker exec unless we are already inside."""
    return command if inside_container else _docker_exec(container, command)


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
        DockerProbeError: If no probe yielded a version.
    """
    errors = []
    for probe in _VERSION_PROBES:
        result = runner.run(_in_container(container, probe, inside_container), timeout=300)
        if not result.ok:
            errors.append(f"{probe!r} -> exit {result.exit_code}: {result.stderr.strip()[:200]}")
            continue
        version = _parse_version_output(result.stdout)
        if version:
            logger.info(f"Container {container or '(local)'} runs vLLM {version}")
            return version
        errors.append(f"{probe!r} -> unparseable output: {result.stdout.strip()[:200]}")

    raise DockerProbeError(
        "Could not determine the vLLM version inside "
        f"{container or 'the local environment'}. Tried:\n  " + "\n  ".join(errors)
    )


def detect_vllm_path(
    container: str,
    runner: CommandRunner,
    inside_container: bool = False,
) -> Optional[str]:
    """Find the directory containing the vllm package inside `container`."""
    for probe in _VLLM_PATH_PROBES:
        result = runner.run(_in_container(container, probe, inside_container), timeout=300)
        if result.ok:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("/"):
                    return line
    return None


def _git_probe(directory: str, args: str) -> str:
    """A git command against a checkout inside a container.

    Guarded by a ``.git`` test so a directory that is merely *inside* somebody
    else's repository (site-packages under a checked-out /workspace, say) cannot
    answer with that repository's HEAD. ``safe.directory=*`` covers the usual
    case of the checkout being owned by another uid than the exec user.
    """
    quoted = shlex.quote(directory)
    return f"test -e {quoted}/.git && git -c safe.directory='*' -C {quoted} {args}"


def _commit_from_git(
    directory: str,
    container: str,
    runner: CommandRunner,
    inside_container: bool,
) -> Optional[Tuple[str, bool]]:
    """(sha, dirty) of the checkout at `directory`, or None if it is not one."""
    head = runner.run(
        _in_container(container, _git_probe(directory, "rev-parse HEAD"), inside_container),
        timeout=300,
    )
    if not head.ok:
        return None
    sha = head.stdout.strip().splitlines()[-1].strip() if head.stdout.strip() else ""
    if not _SHA_RE.match(sha):
        return None

    # Uncommitted changes are not fatal -- vLLM's own install rewrites
    # requirement files -- but the peer is built from the committed sha only, so
    # the caller has to be able to say so.
    status = runner.run(
        _in_container(
            container,
            _git_probe(directory, "status --porcelain --untracked-files=no"),
            inside_container,
        ),
        timeout=300,
    )
    return sha, bool(status.ok and status.stdout.strip())


def detect_vllm_commit(
    container: str,
    runner: CommandRunner,
    inside_container: bool = False,
    vllm_path: str = "",
) -> VLLMCommit:
    """Determine the exact vLLM commit a container is running.

    Two sources, most exact first:

      1. the git checkout vLLM is imported from -- a full 40-char sha, plus
         whether that tree carries uncommitted changes on top of it. The
         *imported* location is probed before the configured ``vllm_path``, so a
         stale checkout lying around cannot outvote the code actually in use.
      2. the ``+g<sha>`` local part of the installed version, which
         setuptools-scm stamps into every dev build (and which vLLM's own
         collect_env reports as "git sha"). Abbreviated, but enough for git to
         resolve in a clone.

    Args:
        container: Container name (ignored when `inside_container`).
        runner: Runner for the host owning that container.
        inside_container: True when this process runs inside `container`.
        vllm_path: Configured vLLM location, tried as a second git candidate.

    Returns:
        The commit, with the source it came from for reporting.

    Raises:
        DockerProbeError: If neither source yields a commit -- a plain release
            wheel records none, and then there is nothing to build a peer from.
    """
    errors: List[str] = []

    candidates: List[str] = []
    imported = detect_vllm_path(container, runner, inside_container=inside_container)
    for candidate in (imported, vllm_path):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if not imported:
        errors.append("could not locate the vllm package (is vLLM installed?)")

    for directory in candidates:
        found = _commit_from_git(directory, container, runner, inside_container)
        if found is None:
            errors.append(f"{directory} is not a git checkout")
            continue
        sha, dirty = found
        logger.info(f"Container {container or '(local)'} runs vLLM @ {sha} (git {directory})")
        if dirty:
            logger.warning(
                f"The vLLM checkout {directory} has uncommitted changes; only the "
                f"committed sha {sha[:12]} can be reproduced on the other side"
            )
        return VLLMCommit(
            sha=sha,
            source=f"git checkout {directory}",
            dirty=dirty,
            path=directory,
        )

    version = ""
    try:
        version = detect_vllm_version(container, runner, inside_container=inside_container)
    except DockerProbeError as e:
        errors.append(str(e).splitlines()[0])

    if version:
        match = _LOCAL_SHA_RE.search(version)
        if match:
            sha = match.group("sha")
            logger.info(f"Container {container or '(local)'} runs vLLM @ {sha} (from {version})")
            return VLLMCommit(sha=sha, source="version string", version=version)
        errors.append(
            f"version {version!r} carries no +g<sha> build part (a release wheel "
            "does not record its commit)"
        )

    raise DockerProbeError(
        "Could not determine the exact vLLM commit installed in "
        f"{container or 'the local environment'}, so no matching CUDA peer can be "
        "built. Tried:\n  " + "\n  ".join(errors) + "\n"
        "Set vllm.commit (--vllm-commit) to build both sides from a known commit, "
        "or gpu.docker/gpu.image to configure the GPU side yourself."
    )


def image_tag(image: str) -> str:
    """Tag part of an image reference ("latest" when untagged).

    Splits on the last path segment first so a registry port (myreg:5000/vllm)
    is not mistaken for a tag.
    """
    last_segment = image.rsplit("/", 1)[-1]
    return last_segment.rsplit(":", 1)[-1] if ":" in last_segment else "latest"


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

    Used for a ready-made image (``gpu.image``), where nothing has to be built:
    the container is deterministically named after the image tag and started as
    an idle shell, so repeated runs reuse it instead of re-pulling a multi-GB
    image. An existing-but-stopped container is restarted.

    Args:
        image: Image reference, e.g. "my-registry/vllm:my-build".
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
        DockerProbeError: If the image is unavailable or the container could not
            be started.
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
            raise DockerProbeError(
                f"Could not start existing container {name}: {result.stderr.strip()}"
            )
        return name, False

    if pull:
        logger.info(f"Pulling {image} (this can take a while on first use)...")
        pull_result = runner.run(f"docker pull {shlex.quote(image)}", timeout=3600)
        if not pull_result.ok:
            local = runner.run(f"docker image inspect {shlex.quote(image)}", timeout=60)
            if not local.ok:
                raise DockerProbeError(
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
        # A vLLM server image's ENTRYPOINT is the OpenAI API server; we need an
        # idle container to docker exec into instead.
        + f"--entrypoint /bin/bash {shlex.quote(image)} -c 'sleep infinity'"
    )

    logger.info(f"Launching GPU container {name} from {image}")
    result = runner.run(run_cmd, timeout=600)
    if not result.ok:
        raise DockerProbeError(
            f"Failed to launch GPU container from {image}: {result.stderr.strip()}"
        )

    if _container_state(name, runner) != "running":
        raise DockerProbeError(
            f"GPU container {name} did not stay running after launch. "
            f"Check `docker logs {name}`."
        )

    logger.info(f"GPU container {name} is running")
    return name, True
