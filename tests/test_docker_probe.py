"""Tests for probing docker hosts and the containers holding the peers.

Everything here is offline: the docker CLI is replaced by a scripted
CommandRunner, so no network and no daemon is touched.
"""

import pytest

from accuracy_agent.docker_probe import (
    CommandResult,
    DockerProbeError,
    container_name_for,
    detect_vllm_commit,
    detect_vllm_path,
    detect_vllm_version,
    ensure_gpu_container,
    is_local_host,
)

SHA = "7794b1e08bf505ff28664515ffaaeeec955ab796"
IMAGE = "my-registry/vllm:my-build"


class FakeRunner:
    """CommandRunner stand-in that answers commands from substring rules."""

    def __init__(self, rules, default=CommandResult(1, "", "no rule")):
        self.rules = rules
        self.default = default
        self.commands = []
        self.is_local = True

    def run(self, cmd, timeout=300):
        self.commands.append(cmd)
        for needle, result in self.rules:
            if needle in cmd:
                return result() if callable(result) else result
        return self.default

    def close(self):
        pass

    def ran(self, needle):
        return [c for c in self.commands if needle in c]


# --------------------------------------------------------------------------
# version detection in a container
# --------------------------------------------------------------------------

def test_detect_vllm_version_via_docker_exec():
    runner = FakeRunner([("importlib.metadata", CommandResult(0, "0.11.0\n", ""))])
    assert detect_vllm_version("xpu_container", runner) == "0.11.0"
    assert "docker exec" in runner.commands[0]
    assert "xpu_container" in runner.commands[0]


def test_detect_vllm_version_inside_container_skips_docker_exec():
    runner = FakeRunner([("importlib.metadata", CommandResult(0, "0.11.0\n", ""))])
    assert detect_vllm_version("xpu_container", runner, inside_container=True) == "0.11.0"
    assert "docker exec" not in runner.commands[0]


def test_detect_vllm_version_falls_back_to_pip_show():
    runner = FakeRunner([
        ("importlib.metadata", CommandResult(1, "", "ModuleNotFoundError")),
        ("pip show vllm", CommandResult(0, "Name: vllm\nVersion: 0.11.1rc2.dev0+xpu\n", "")),
    ])
    assert detect_vllm_version("xpu_container", runner) == "0.11.1rc2.dev0+xpu"


def test_detect_vllm_version_raises_when_all_probes_fail():
    runner = FakeRunner([], default=CommandResult(1, "", "No such container"))
    with pytest.raises(DockerProbeError, match="Could not determine the vLLM version"):
        detect_vllm_version("xpu_container", runner)


# --------------------------------------------------------------------------
# vLLM location
# --------------------------------------------------------------------------

def test_detect_vllm_path_uses_find_spec():
    runner = FakeRunner([("find_spec", CommandResult(0, "/workspace/vllm\n", ""))])
    assert detect_vllm_path("xpu_container", runner) == "/workspace/vllm"


def test_detect_vllm_path_inside_container_skips_docker_exec():
    runner = FakeRunner([("find_spec", CommandResult(0, "/workspace/vllm\n", ""))])
    assert detect_vllm_path("c", runner, inside_container=True) == "/workspace/vllm"
    assert "docker exec" not in runner.commands[0]


def test_probes_do_not_run_from_the_working_directory():
    """A mounted source root must not shadow the installed package.

    A peer container has the vLLM source root bind-mounted at /workspace/vllm and
    NGC images set WORKDIR=/workspace. `python -c` prepends the cwd to sys.path,
    so from there `vllm` resolves to that *directory* as a namespace package with
    origin/__file__ None -- the real install becomes invisible and a freshly
    built peer reports "vLLM is not installed". Observed on nvcr.io/nvidia/pytorch.
    """
    runner = FakeRunner([], default=CommandResult(1, "", "boom"))
    detect_vllm_path("c", runner)
    with pytest.raises(DockerProbeError):   # every probe fails; we only want the commands
        detect_vllm_version("c", runner)

    assert runner.commands, "no probe ran"
    for cmd in runner.commands:
        assert "cd / &&" in cmd, f"probe runs in the container's cwd: {cmd}"


def test_detect_vllm_path_returns_none_when_absent():
    runner = FakeRunner([], default=CommandResult(1, "", "ModuleNotFoundError"))
    assert detect_vllm_path("xpu_container", runner) is None


# --------------------------------------------------------------------------
# commit detection: the GPU peer is built from whatever this returns
# --------------------------------------------------------------------------

def _git_runner(rules=()):
    """Runner whose vLLM lives in an editable checkout at /workspace/vllm."""
    return FakeRunner(list(rules) + [
        ("find_spec", CommandResult(0, "/workspace/vllm\n", "")),
    ])


def test_commit_comes_from_the_git_checkout_vllm_is_imported_from():
    runner = _git_runner([
        ("rev-parse HEAD", CommandResult(0, SHA + "\n", "")),
        ("status --porcelain", CommandResult(0, "", "")),
    ])
    commit = detect_vllm_commit("xpu_container", runner, vllm_path="/workspace/vllm")

    assert commit.sha == SHA
    assert commit.abbreviated is False
    assert commit.dirty is False
    assert "/workspace/vllm" in commit.source


def test_git_probe_is_guarded_by_a_dot_git_test_and_safe_directory():
    """A dir merely inside another repo must not answer with that repo's HEAD,
    and a checkout owned by another uid must still be readable."""
    runner = _git_runner([
        ("rev-parse HEAD", CommandResult(0, SHA + "\n", "")),
        ("status --porcelain", CommandResult(0, "", "")),
    ])
    detect_vllm_commit("xpu_container", runner)

    probe = runner.ran("rev-parse HEAD")[0]
    assert "test -e" in probe and "/.git" in probe
    assert "safe.directory" in probe


def test_uncommitted_changes_are_reported_not_fatal():
    runner = _git_runner([
        ("rev-parse HEAD", CommandResult(0, SHA + "\n", "")),
        ("status --porcelain", CommandResult(0, " M requirements/xpu.txt\n", "")),
    ])
    commit = detect_vllm_commit("xpu_container", runner)

    assert commit.sha == SHA
    assert commit.dirty is True


def test_commit_falls_back_to_the_version_local_part():
    """A wheel build has no checkout, but setuptools-scm stamped the sha in."""
    runner = FakeRunner([
        ("find_spec", CommandResult(0, "/usr/lib/python3/dist-packages\n", "")),
        ("rev-parse HEAD", CommandResult(1, "", "not a git repository")),
        ("importlib.metadata", CommandResult(0, "0.26.1rc1.dev353+g7794b1e08.xpu\n", "")),
    ])
    commit = detect_vllm_commit("xpu_container", runner)

    assert commit.sha == "7794b1e08"
    assert commit.abbreviated is True          # git resolves it in the clone
    assert commit.version == "0.26.1rc1.dev353+g7794b1e08.xpu"
    assert commit.source == "version string"


@pytest.mark.parametrize("version,expected", [
    ("0.6.5.dev258+g6c5af09b", "6c5af09b"),
    ("0.6.5.dev258+g6c5af09b.d20241119", "6c5af09b"),
    ("0.11.0+xpu.g1a2b3c4d", "1a2b3c4d"),
    ("0.26.1rc1.dev353+g7794b1e08.xpu", "7794b1e08"),
])
def test_sha_is_read_out_of_every_scm_version_shape(version, expected):
    runner = FakeRunner([
        ("find_spec", CommandResult(0, "/usr/lib/python3/dist-packages\n", "")),
        ("importlib.metadata", CommandResult(0, version + "\n", "")),
    ])
    assert detect_vllm_commit("xpu_container", runner).sha == expected


def test_git_checkout_wins_over_the_version_string():
    """The checkout is exact; the stamped sha is abbreviated and can be stale."""
    runner = _git_runner([
        ("rev-parse HEAD", CommandResult(0, SHA + "\n", "")),
        ("status --porcelain", CommandResult(0, "", "")),
        ("importlib.metadata", CommandResult(0, "0.11.0+g0000000\n", "")),
    ])
    assert detect_vllm_commit("xpu_container", runner).sha == SHA


def test_configured_vllm_path_is_a_second_git_candidate():
    """The imported path may be site-packages while the tool patches a checkout."""
    heads = iter([
        CommandResult(1, "", "not a git repository"),   # site-packages
        CommandResult(0, SHA + "\n", ""),               # /workspace/vllm
    ])
    runner = FakeRunner([
        ("find_spec", CommandResult(0, "/usr/lib/python3/dist-packages\n", "")),
        ("rev-parse HEAD", lambda: next(heads)),
        ("status --porcelain", CommandResult(0, "", "")),
    ])
    commit = detect_vllm_commit("xpu_container", runner, vllm_path="/workspace/vllm")

    assert commit.sha == SHA
    assert "/workspace/vllm" in commit.source


def test_release_wheel_without_a_commit_raises_with_the_way_out():
    """No checkout and no +g<sha>: there is nothing to build a peer from."""
    runner = FakeRunner([
        ("find_spec", CommandResult(0, "/usr/lib/python3/dist-packages\n", "")),
        ("rev-parse HEAD", CommandResult(1, "", "not a git repository")),
        ("importlib.metadata", CommandResult(0, "0.11.0\n", "")),
    ])
    with pytest.raises(DockerProbeError) as excinfo:
        detect_vllm_commit("xpu_container", runner)

    message = str(excinfo.value)
    assert "Could not determine the exact vLLM commit" in message
    assert "--vllm-commit" in message and "gpu.docker" in message


def test_commit_detection_inside_the_container_skips_docker_exec():
    runner = _git_runner([
        ("rev-parse HEAD", CommandResult(0, SHA + "\n", "")),
        ("status --porcelain", CommandResult(0, "", "")),
    ])
    detect_vllm_commit("xpu_container", runner, inside_container=True)
    assert not runner.ran("docker exec")


# --------------------------------------------------------------------------
# container lifecycle for a ready-made image (gpu.image)
# --------------------------------------------------------------------------

def test_container_name_is_deterministic_and_docker_safe():
    assert container_name_for("v0.11.0") == "accuracy_agent_gpu_v0.11.0"
    assert container_name_for("v0.11.0-cu126") == "accuracy_agent_gpu_v0.11.0-cu126"


def test_ensure_gpu_container_reuses_running_container():
    runner = FakeRunner([("docker inspect", CommandResult(0, "running\n", ""))])
    name, launched = ensure_gpu_container(IMAGE, runner)
    assert name == "accuracy_agent_gpu_my-build"
    assert launched is False
    assert not runner.ran("docker run")


def test_ensure_gpu_container_starts_stopped_container():
    runner = FakeRunner([
        ("docker inspect", CommandResult(0, "exited\n", "")),
        ("docker start", CommandResult(0, "", "")),
    ])
    _name, launched = ensure_gpu_container(IMAGE, runner)
    assert launched is False
    assert runner.ran("docker start")


def test_ensure_gpu_container_launches_with_mounts_and_idle_entrypoint():
    inspect_results = iter([
        CommandResult(1, "", "No such object"),   # first check: absent
        CommandResult(0, "running\n", ""),        # post-launch check
    ])

    class LaunchRunner(FakeRunner):
        def run(self, cmd, timeout=300):
            self.commands.append(cmd)
            if "docker inspect" in cmd:
                return next(inspect_results)
            return CommandResult(0, "", "")

    runner = LaunchRunner([])
    name, launched = ensure_gpu_container(IMAGE, runner, mounts=["/mnt/weka"])
    run_cmd = next(c for c in runner.commands if c.startswith("docker run"))

    assert launched is True
    assert f"--name {name}" in run_cmd
    assert "--gpus all" in run_cmd
    assert "-v /mnt/weka:/mnt/weka" in run_cmd
    # A vLLM server image's entrypoint is the API server; we need an idle container.
    assert "--entrypoint /bin/bash" in run_cmd
    assert "sleep infinity" in run_cmd
    assert any(c.startswith("docker pull") for c in runner.commands)


def test_ensure_gpu_container_raises_when_image_unavailable():
    runner = FakeRunner([
        ("docker inspect -f", CommandResult(1, "", "No such object")),
        ("docker pull", CommandResult(1, "", "manifest unknown")),
        ("docker image inspect", CommandResult(1, "", "No such image")),
    ])
    with pytest.raises(DockerProbeError, match="Could not pull"):
        ensure_gpu_container(IMAGE, runner)


# --------------------------------------------------------------------------
# image reference parsing / host classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("image,expected", [
    ("vllm/vllm-openai:v0.11.0", "v0.11.0"),
    ("vllm/vllm-openai", "latest"),
    ("myregistry:5000/vllm/vllm-openai", "latest"),   # port is not a tag
    ("myregistry:5000/vllm/vllm-openai:v0.11.0", "v0.11.0"),
])
def test_image_tag(image, expected):
    from accuracy_agent.docker_probe import image_tag

    assert image_tag(image) == expected


@pytest.mark.parametrize("host", ["", "localhost", "127.0.0.1", "LOCALHOST"])
def test_local_hosts(host):
    assert is_local_host(host)


def test_remote_host():
    assert not is_local_host("gpu-host.example.com")
