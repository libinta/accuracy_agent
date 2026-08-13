"""Tests for deriving the GPU docker image from the XPU container's vLLM version.

Everything here is offline: Docker Hub is replaced by a fake tag list and the
docker CLI by a scripted CommandRunner, so no network or daemon is touched.
"""

import pytest

from accuracy_agent.config import DebugConfig
from accuracy_agent.gpu_image_resolver import (
    CommandResult,
    GPUImageResolutionError,
    VLLM_OPENAI_REPO,
    autoconfigure_gpu_docker,
    container_name_for,
    detect_vllm_version,
    ensure_gpu_container,
    is_local_host,
    parse_vllm_version,
    resolve_gpu_image,
    select_release_tag,
)

# (major, minor, patch, post, tag) as fetch_release_tags returns them.
FAKE_TAGS = [
    (0, 8, 5, None, "v0.8.5"),
    (0, 8, 5, 1, "v0.8.5.post1"),
    (0, 10, 0, None, "v0.10.0"),
    (0, 11, 0, None, "v0.11.0"),
]

#: A release that exists nowhere upstream, so autoconfiguration tests cannot
#: pass by accidentally reaching the real Docker Hub.
FANTASY_TAGS = [(7, 7, 7, None, "v7.7.7")]


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
                return result
        return self.default

    def close(self):
        pass


# --------------------------------------------------------------------------
# version parsing
# --------------------------------------------------------------------------

def test_parse_plain_release():
    version = parse_vllm_version("0.11.0")
    assert version.release == (0, 11, 0)
    assert version.post is None
    assert not version.is_prerelease
    assert version.tag == "v0.11.0"


def test_parse_strips_local_build_part():
    """+xpu / +gitsha identify a build, not a release."""
    version = parse_vllm_version("0.11.0+xpu.g1a2b3c4")
    assert version.release_str == "0.11.0"
    assert not version.is_prerelease


def test_parse_post_release():
    version = parse_vllm_version("0.8.5.post1")
    assert version.post == 1
    assert version.tag == "v0.8.5.post1"


@pytest.mark.parametrize("raw", ["0.11.1rc2", "0.11.1.dev123+g1a2b3c4", "v0.11.1rc2.dev0"])
def test_parse_detects_prerelease(raw):
    version = parse_vllm_version(raw)
    assert version.release == (0, 11, 1)
    assert version.is_prerelease


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_vllm_version("not-a-version")


# --------------------------------------------------------------------------
# release tag selection
# --------------------------------------------------------------------------

def test_select_exact_release():
    tag, exact, _note = select_release_tag(parse_vllm_version("0.11.0"), FAKE_TAGS)
    assert (tag, exact) == ("v0.11.0", True)


def test_select_exact_post_release():
    tag, exact, _note = select_release_tag(parse_vllm_version("0.8.5.post1"), FAKE_TAGS)
    assert (tag, exact) == ("v0.8.5.post1", True)


def test_select_prerelease_falls_back_to_previous_release():
    """A 0.11.1 dev build predates v0.11.1, so v0.11.0 is the closest release."""
    tag, exact, note = select_release_tag(
        parse_vllm_version("0.11.1rc2.dev0+xpu"), FAKE_TAGS
    )
    assert tag == "v0.11.0"
    assert exact is False
    assert "pre-release" in note


def test_select_unpublished_release_uses_nearest_lower():
    tag, exact, note = select_release_tag(parse_vllm_version("0.11.2"), FAKE_TAGS)
    assert tag == "v0.11.0"
    assert exact is False
    assert "closest published release" in note


def test_select_below_all_tags_returns_none():
    tag, _exact, note = select_release_tag(parse_vllm_version("0.1.0"), FAKE_TAGS)
    assert tag is None
    assert note


def test_select_with_no_tags_returns_none():
    tag, _exact, _note = select_release_tag(parse_vllm_version("0.11.0"), [])
    assert tag is None


# --------------------------------------------------------------------------
# image resolution
# --------------------------------------------------------------------------

def test_resolve_verified_image():
    resolution = resolve_gpu_image(
        parse_vllm_version("0.11.0"), tag_fetcher=lambda repo: FAKE_TAGS
    )
    assert resolution.image == f"{VLLM_OPENAI_REPO}:v0.11.0"
    assert resolution.verified and resolution.exact


def test_resolve_falls_back_when_docker_hub_unreachable():
    def boom(repo):
        raise OSError("network unreachable")

    resolution = resolve_gpu_image(parse_vllm_version("0.11.0"), tag_fetcher=boom)
    assert resolution.image == f"{VLLM_OPENAI_REPO}:v0.11.0"
    assert resolution.verified is False
    assert "unreachable" in resolution.note


def test_resolve_offline_never_queries_docker_hub():
    def boom(repo):  # pragma: no cover - must not be called
        raise AssertionError("network was consulted despite allow_network=False")

    resolution = resolve_gpu_image(
        parse_vllm_version("0.11.0"), allow_network=False, tag_fetcher=boom
    )
    assert resolution.tag == "v0.11.0"
    assert resolution.verified is False


def test_resolve_never_selects_a_nightly_tag():
    """Only vX.Y.Z(.postN) tags are candidates, so nightlies cannot be chosen."""
    tags_with_nightly = FAKE_TAGS + [(9, 9, 9, None, "nightly")]
    resolution = resolve_gpu_image(
        parse_vllm_version("0.11.1.dev0"), tag_fetcher=lambda repo: tags_with_nightly
    )
    assert "nightly" not in resolution.image


# --------------------------------------------------------------------------
# version detection in the XPU container
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
    with pytest.raises(GPUImageResolutionError, match="Could not determine the vLLM version"):
        detect_vllm_version("xpu_container", runner)


# --------------------------------------------------------------------------
# container lifecycle
# --------------------------------------------------------------------------

def test_container_name_is_deterministic_and_docker_safe():
    assert container_name_for("v0.11.0") == "accuracy_agent_gpu_v0.11.0"
    assert container_name_for("v0.11.0-cu126") == "accuracy_agent_gpu_v0.11.0-cu126"


def test_ensure_gpu_container_reuses_running_container():
    runner = FakeRunner([("docker inspect", CommandResult(0, "running\n", ""))])
    name, launched = ensure_gpu_container(f"{VLLM_OPENAI_REPO}:v0.11.0", runner)
    assert name == "accuracy_agent_gpu_v0.11.0"
    assert launched is False
    assert not any("docker run" in c for c in runner.commands)


def test_ensure_gpu_container_starts_stopped_container():
    runner = FakeRunner([
        ("docker inspect", CommandResult(0, "exited\n", "")),
        ("docker start", CommandResult(0, "", "")),
    ])
    _name, launched = ensure_gpu_container(f"{VLLM_OPENAI_REPO}:v0.11.0", runner)
    assert launched is False
    assert any("docker start" in c for c in runner.commands)


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
    name, launched = ensure_gpu_container(
        f"{VLLM_OPENAI_REPO}:v0.11.0", runner, mounts=["/mnt/weka"]
    )
    run_cmd = next(c for c in runner.commands if c.startswith("docker run"))

    assert launched is True
    assert f"--name {name}" in run_cmd
    assert "--gpus all" in run_cmd
    assert "-v /mnt/weka:/mnt/weka" in run_cmd
    # The vllm-openai entrypoint is the API server; we need an idle container.
    assert "--entrypoint /bin/bash" in run_cmd
    assert "sleep infinity" in run_cmd
    assert any(c.startswith("docker pull") for c in runner.commands)


def test_ensure_gpu_container_raises_when_image_unavailable():
    runner = FakeRunner([
        ("docker inspect -f", CommandResult(1, "", "No such object")),
        ("docker pull", CommandResult(1, "", "manifest unknown")),
        ("docker image inspect", CommandResult(1, "", "No such image")),
    ])
    with pytest.raises(GPUImageResolutionError, match="Could not pull"):
        ensure_gpu_container(f"{VLLM_OPENAI_REPO}:v9.9.9", runner)


# --------------------------------------------------------------------------
# end-to-end autoconfiguration against DebugConfig
# --------------------------------------------------------------------------

def _local_xpu_config(**overrides):
    kwargs = dict(
        backend="vllm",
        model_path="/mnt/weka/model",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/out",
        xpu_host="localhost",
        xpu_docker="xpu_container",
        xpu_vllm_path="/workspace/vllm",
    )
    kwargs.update(overrides)
    return DebugConfig(**kwargs)


def _install_fakes(monkeypatch, module, *, version="7.7.8.dev1+xpu", gpu=True):
    """Point the resolver at fake docker/GPU/Docker Hub facilities.

    The fantasy version/tag pair means a test would fail loudly (rather than
    coincidentally pass) if the real Docker Hub were ever consulted.
    """
    monkeypatch.setattr(module, "docker_available", lambda runner: True)
    monkeypatch.setattr(module, "host_has_nvidia_gpu", lambda runner: gpu)
    monkeypatch.setattr(module, "running_inside_container", lambda path: False)
    monkeypatch.setattr(module, "detect_vllm_version", lambda c, r, inside_container=False: version)
    monkeypatch.setattr(module, "fetch_release_tags", lambda repo=VLLM_OPENAI_REPO: FANTASY_TAGS)
    monkeypatch.setattr(
        module, "ensure_gpu_container",
        lambda image, runner, container=None, mounts=(), extra_run_args="", pull=True:
            (container or container_name_for(image.rsplit(":", 1)[-1]), True),
    )
    monkeypatch.setattr(
        module, "detect_vllm_path",
        lambda container, runner: "/usr/local/lib/python3.12/dist-packages",
    )


def test_autoconfigure_fills_in_gpu_side(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module)
    config = _local_xpu_config()

    resolution = autoconfigure_gpu_docker(config)

    assert resolution is not None
    # 7.7.8 dev build -> closest published release image.
    assert config.gpu_image == f"{VLLM_OPENAI_REPO}:v7.7.7"
    assert config.gpu_docker == "accuracy_agent_gpu_v7.7.7"
    assert config.gpu_vllm_path == "/usr/local/lib/python3.12/dist-packages"
    # Must never be mistaken for the container we are running in.
    assert config.gpu_inside_container is False


def test_autoconfigure_respects_explicit_gpu_docker(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("explicit gpu_docker must win")

    monkeypatch.setattr(module, "detect_vllm_version", fail)
    config = _local_xpu_config(gpu_host="gpu.example.com", gpu_docker="my_gpu_container")

    assert autoconfigure_gpu_docker(config) is None
    assert config.gpu_docker == "my_gpu_container"


def test_autoconfigure_skipped_when_disabled(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module)
    config = _local_xpu_config(gpu_auto_image=False)

    assert autoconfigure_gpu_docker(config) is None
    assert config.gpu_docker == ""


def test_autoconfigure_skipped_for_remote_xpu(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module)
    config = _local_xpu_config(xpu_host="xpu-host.example.com")

    assert autoconfigure_gpu_docker(config) is None
    assert config.gpu_docker == ""


def test_autoconfigure_skipped_without_nvidia_gpu(monkeypatch):
    """On an XPU-only box the run must stay XPU-only."""
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module, gpu=False)
    config = _local_xpu_config()

    assert autoconfigure_gpu_docker(config) is None
    assert config.gpu_docker == ""


def test_autoconfigure_skipped_for_non_vllm_backend(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module)
    config = _local_xpu_config(backend="pytorch")

    assert autoconfigure_gpu_docker(config) is None


def test_autoconfigure_uses_explicit_gpu_image_without_probing(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module)

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("gpu.image should skip version detection")

    monkeypatch.setattr(module, "detect_vllm_version", fail)
    config = _local_xpu_config(gpu_image=f"{VLLM_OPENAI_REPO}:v0.10.0")

    resolution = autoconfigure_gpu_docker(config)

    assert resolution.image == f"{VLLM_OPENAI_REPO}:v0.10.0"
    assert config.gpu_docker == "accuracy_agent_gpu_v0.10.0"


def test_autoconfigure_resolve_only_does_not_launch(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module)

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("launch=False must not touch containers")

    monkeypatch.setattr(module, "ensure_gpu_container", fail)
    config = _local_xpu_config()

    resolution = autoconfigure_gpu_docker(config, launch=False)

    assert resolution.image == f"{VLLM_OPENAI_REPO}:v7.7.7"
    assert config.gpu_image == f"{VLLM_OPENAI_REPO}:v7.7.7"
    assert config.gpu_docker == ""  # nothing to exec into yet


def test_maybe_autoconfigure_runs_at_most_once(monkeypatch):
    """CLI and Bisector both call it; a failed (slow) attempt must not repeat."""
    import accuracy_agent.gpu_image_resolver as module

    calls = []

    def failing(config, launch=True, allow_network=True, on_skip=None):
        calls.append(config)
        raise GPUImageResolutionError("nope")

    monkeypatch.setattr(module, "autoconfigure_gpu_docker", failing)
    config = _local_xpu_config()

    assert module.maybe_autoconfigure_gpu_docker(config) is None
    assert module.maybe_autoconfigure_gpu_docker(config) is None
    assert len(calls) == 1


def test_on_skip_reports_reason_when_skipped(monkeypatch):
    """The CLI has no logging configured, so the reason must be reported back."""
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module, gpu=False)
    reasons = []

    assert autoconfigure_gpu_docker(_local_xpu_config(), on_skip=reasons.append) is None
    assert reasons and "GPU" in reasons[0]


def test_on_skip_reports_reason_when_resolution_fails(monkeypatch):
    import accuracy_agent.gpu_image_resolver as module

    _install_fakes(monkeypatch, module)

    def boom(*a, **k):
        raise GPUImageResolutionError("no vllm in there")

    monkeypatch.setattr(module, "detect_vllm_version", boom)
    reasons = []

    result = module.maybe_autoconfigure_gpu_docker(
        _local_xpu_config(), on_skip=reasons.append
    )

    assert result is None
    assert any("no vllm in there" in r for r in reasons)


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
    from accuracy_agent.gpu_image_resolver import image_tag

    assert image_tag(image) == expected

@pytest.mark.parametrize("host", ["", "localhost", "127.0.0.1", "LOCALHOST"])
def test_local_hosts(host):
    assert is_local_host(host)


def test_remote_host():
    assert not is_local_host("gpu-host.example.com")
