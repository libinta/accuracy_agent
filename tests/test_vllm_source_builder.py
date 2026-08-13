"""Tests for building both peers from one vllm-project/vllm commit.

Everything here is offline: git, docker and both registries are replaced by
scripted fakes, so no network, no daemon, and no clone is touched.
"""

import json
import pytest

from accuracy_agent.config import DebugConfig
from accuracy_agent.gpu_image_resolver import CommandResult
from accuracy_agent.vllm_source_builder import (
    DEFAULT_CUDA_BASE_IMAGE,
    DEFAULT_XPU_BASE_IMAGE,
    IPEX_REPO,
    NGC_PYTORCH_REPO,
    VLLMBuild,
    VLLMBuildError,
    autoconfigure_from_commit,
    build_image_tag,
    build_peer,
    ensure_source_tree,
    fetch_ipex_xpu_tags,
    fetch_ngc_pytorch_tags,
    maybe_autoconfigure_peers,
    peer_container_name,
    resolve_base_image,
    resolve_commit,
    _install_script,
)

SHA = "7794b1e08bf505ff28664515ffaaeeec955ab796"
SHORT = SHA[:12]


class FakeRunner:
    """CommandRunner stand-in answering commands from substring rules."""

    def __init__(self, rules=(), default=CommandResult(0, "", "")):
        self.rules = list(rules)
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


def _fake_urlopen(payloads):
    """Fake urllib.request.urlopen returning `payloads` in order."""
    remaining = list(payloads)

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=None):
        return FakeResponse(remaining.pop(0))

    return urlopen


# --------------------------------------------------------------------------
# cache identity: a precompiled build must never masquerade as a compiled one
# --------------------------------------------------------------------------

def test_image_tag_and_container_name_per_device():
    assert build_image_tag("cuda", SHA, False) == f"accuracy_agent/vllm:cuda-{SHORT}"
    assert build_image_tag("xpu", SHA, False) == f"accuracy_agent/vllm:xpu-{SHORT}"
    assert peer_container_name("xpu", SHA, False) == f"accuracy_agent_vllm_xpu_{SHORT}"


def test_compiled_cuda_build_does_not_share_a_tag_with_precompiled():
    assert build_image_tag("cuda", SHA, True) != build_image_tag("cuda", SHA, False)
    assert peer_container_name("cuda", SHA, True) != peer_container_name("cuda", SHA, False)


def test_build_kernels_does_not_split_the_xpu_cache():
    """XPU has no precompiled path, so the flag must not fork its cache."""
    assert build_image_tag("xpu", SHA, True) == build_image_tag("xpu", SHA, False)


# --------------------------------------------------------------------------
# install scripts
# --------------------------------------------------------------------------

def test_cuda_install_script_uses_precompiled_fast_path():
    script = _install_script("cuda", SHA, build_kernels=False)
    assert "VLLM_USE_PRECOMPILED=1" in script
    assert f"VLLM_PRECOMPILED_WHEEL_COMMIT={SHA}" in script
    # The vendor torch is the reason for using the NGC image at all.
    assert "use_existing_torch.py" in script
    assert "pip install --no-build-isolation -e ." in script


def test_cuda_install_script_compiles_when_asked():
    script = _install_script("cuda", SHA, build_kernels=True)
    assert "VLLM_USE_PRECOMPILED" not in script
    assert "VLLM_TARGET_DEVICE=cuda" in script


def test_oneapi_env_is_optional():
    """Non-oneAPI bases (IPEX, NGC) must not fail on a missing setvars.sh."""
    for device in ("cuda", "xpu"):
        script = _install_script(device, SHA, False)
        assert "if [ -f /opt/intel/oneapi/setvars.sh ]; then" in script
        assert "|| true; fi" in script


def test_xpu_install_script_builds_from_source_with_pinned_kernels():
    script = _install_script("xpu", SHA, build_kernels=False)
    assert "VLLM_TARGET_DEVICE=xpu" in script
    assert "requirements/xpu.txt" in script      # brings the vllm_xpu_kernels wheel
    assert "VLLM_USE_PRECOMPILED" not in script  # no such thing for XPU
    assert "setvars.sh" in script                # oneAPI env for the build


def test_install_scripts_fail_loudly():
    for device in ("cuda", "xpu"):
        assert _install_script(device, SHA, False).startswith("set -euo pipefail")


@pytest.mark.parametrize("device,kernels", [("cuda", False), ("cuda", True), ("xpu", False)])
def test_install_scripts_are_valid_bash(device, kernels, tmp_path):
    import subprocess

    script = tmp_path / "install.sh"
    script.write_text(_install_script(device, SHA, kernels))
    check = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_optional_file_probes_survive_set_e(device, tmp_path):
    """`[ -f x ] && cmd` as a trailing command aborts the whole script under set -e.

    Runs the real conditionals against an empty tree with pip/python stubbed out,
    so a regression here fails as a test rather than as a dead build.
    """
    import subprocess

    lines = [
        line for line in _install_script(device, SHA, False).splitlines()
        # The oneAPI line is excluded on purpose: on an XPU host it would source
        # the *host's* real setvars.sh, which exits the shell it is sourced into.
        if line.startswith(("set -", "if [ -f", "for f in")) and "oneapi" not in line
    ]
    script = tmp_path / "probe.sh"
    script.write_text("\n".join(lines) + "\necho REACHED_END\n")

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    for stub in ("python3", "pip"):
        (stub_dir / stub).write_text("#!/bin/sh\nexit 0\n")
        (stub_dir / stub).chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, capture_output=True, text=True,
        env={"PATH": f"{stub_dir}:/usr/bin:/bin"},
    )
    assert "REACHED_END" in result.stdout, f"script aborted early: {result.stderr}"


# --------------------------------------------------------------------------
# base image selection
# --------------------------------------------------------------------------

def test_explicit_base_image_wins():
    image, note = resolve_base_image("cuda", explicit="my/own:image")
    assert (image, note) == ("my/own:image", "")


def test_offline_base_images_are_the_pinned_defaults():
    assert resolve_base_image("cuda", allow_network=False)[0] == DEFAULT_CUDA_BASE_IMAGE
    assert resolve_base_image("xpu", allow_network=False)[0] == DEFAULT_XPU_BASE_IMAGE


def test_base_image_falls_back_when_registry_unreachable(monkeypatch):
    import accuracy_agent.vllm_source_builder as module

    def boom(*a, **k):
        raise OSError("registry unreachable")

    monkeypatch.setattr(module, "fetch_ngc_pytorch_tags", boom)
    image, note = resolve_base_image("cuda")
    assert image == DEFAULT_CUDA_BASE_IMAGE
    assert "unreachable" in note


def test_base_image_uses_newest_published_tag(monkeypatch):
    import accuracy_agent.vllm_source_builder as module

    monkeypatch.setattr(module, "fetch_ngc_pytorch_tags", lambda *a, **k: ["26.06-py3", "26.07-py3"])
    monkeypatch.setattr(module, "fetch_ipex_xpu_tags", lambda *a, **k: ["2.7.10-xpu", "2.8.10-xpu"])

    assert resolve_base_image("cuda")[0] == f"{NGC_PYTORCH_REPO}:26.07-py3"
    assert resolve_base_image("xpu")[0] == f"{IPEX_REPO}:2.8.10-xpu"


def test_ngc_tag_listing_keeps_only_py3_releases_newest_last(monkeypatch):
    import accuracy_agent.vllm_source_builder as module

    monkeypatch.setattr(module.urllib.request, "urlopen", _fake_urlopen([
        {"token": "t"},
        {"tags": ["26.07-py3", "25.12-py3", "26.07-py3-igpu", "latest", "26.07-py3-devel"]},
    ]))
    assert fetch_ngc_pytorch_tags() == ["25.12-py3", "26.07-py3"]


def test_ipex_tag_listing_excludes_variants(monkeypatch):
    import accuracy_agent.vllm_source_builder as module

    monkeypatch.setattr(module.urllib.request, "urlopen", _fake_urlopen([
        {"results": [
            {"name": "2.8.10-xpu"},
            {"name": "2.8.10-xpu-idp-jupyter"},
            {"name": "2.8.10-serving-xpu"},
            {"name": "2.7.10-xpu"},
            {"name": "2.8.0-idp-base"},
        ], "next": None},
    ]))
    assert fetch_ipex_xpu_tags() == ["2.7.10-xpu", "2.8.10-xpu"]


# --------------------------------------------------------------------------
# commit resolution and source export
# --------------------------------------------------------------------------

def test_resolve_commit_from_local_clone(tmp_path):
    runner = FakeRunner([("rev-parse --verify", CommandResult(0, SHA + "\n", ""))])
    assert resolve_commit("7794b1e", tmp_path, runner) == SHA
    assert not runner.ran("fetch")


def test_resolve_commit_fetches_unknown_sha(tmp_path):
    """A hash from a PR branch is not in the clone yet."""
    seen = {"n": 0}

    def rev_parse():
        seen["n"] += 1
        return CommandResult(0, SHA + "\n", "") if seen["n"] > 1 else CommandResult(1, "", "")

    runner = FakeRunner([
        ("rev-parse --verify", rev_parse),
        ("fetch origin", CommandResult(0, "", "")),
    ])
    assert resolve_commit(SHA, tmp_path, runner) == SHA
    assert runner.ran("fetch origin")


def test_resolve_commit_raises_when_unresolvable(tmp_path):
    runner = FakeRunner([
        ("rev-parse --verify", CommandResult(1, "", "")),
        ("fetch", CommandResult(1, "", "couldn't find remote ref")),
    ])
    with pytest.raises(VLLMBuildError, match="Could not fetch"):
        resolve_commit("deadbeef", tmp_path, runner)


def test_resolve_commit_can_refuse_to_fetch(tmp_path):
    runner = FakeRunner([("rev-parse --verify", CommandResult(1, "", ""))])
    with pytest.raises(VLLMBuildError, match="fetching disabled"):
        resolve_commit("deadbeef", tmp_path, runner, allow_fetch=False)


def test_ensure_source_tree_reuses_matching_checkout(tmp_path):
    dest = tmp_path / "builds" / f"vllm-{SHORT}"
    (dest / ".git").mkdir(parents=True)
    runner = FakeRunner([("rev-parse HEAD", CommandResult(0, SHA + "\n", ""))])

    assert ensure_source_tree(tmp_path / "repo", SHA, tmp_path / "builds", runner) == dest
    assert not runner.ran("git clone")


def test_ensure_source_tree_refuses_a_tree_at_another_commit(tmp_path):
    dest = tmp_path / "builds" / f"vllm-{SHORT}"
    (dest / ".git").mkdir(parents=True)
    runner = FakeRunner([("rev-parse HEAD", CommandResult(0, "0" * 40 + "\n", ""))])

    with pytest.raises(VLLMBuildError, match="is not at"):
        ensure_source_tree(tmp_path / "repo", SHA, tmp_path / "builds", runner)


def test_ensure_source_tree_clones_via_a_tool_managed_branch(tmp_path):
    """A fetched sha is only clonable once a ref points at it."""
    runner = FakeRunner()
    dest = ensure_source_tree(tmp_path / "repo", SHA, tmp_path / "builds", runner)

    assert dest == tmp_path / "builds" / f"vllm-{SHORT}"
    assert runner.ran(f"branch --force accuracy-agent/{SHORT} {SHA}")
    clone = runner.ran("git clone")[0]
    assert f"--branch accuracy-agent/{SHORT}" in clone
    assert "--no-hardlinks" in clone


# --------------------------------------------------------------------------
# building a peer
# --------------------------------------------------------------------------

#: Source dirs the tests below bind-mount; the mount check must accept them.
_TEST_SOURCE_DIRS = "/src\n/host/src\n/builds/vllm-new\n"


def _build_runner(rules=(), vllm_path="/workspace/vllm", mounts=_TEST_SOURCE_DIRS):
    # The Mounts query is a `docker inspect -f` too, so it must be matched by a
    # more specific rule before the container-state one.
    return FakeRunner([("{{range .Mounts}}", CommandResult(0, mounts, ""))]
                      + list(rules)
                      + [("find_spec", CommandResult(0, vllm_path + "\n", ""))])


def test_build_peer_reuses_running_container():
    runner = _build_runner([("docker inspect -f", CommandResult(0, "running\n", ""))])
    build = build_peer("xpu", SHA, "/src", runner, allow_network=False)

    assert build.built is False
    assert build.container == peer_container_name("xpu", SHA, False)
    assert not runner.ran("docker run")
    assert not runner.ran("docker pull")


def test_build_peer_refuses_a_container_holding_another_commits_tree():
    """An explicit container_name does not encode the commit, so verify the mount."""
    runner = _build_runner([("docker inspect -f", CommandResult(0, "running\n", ""))],
                           mounts="/builds/vllm-oldoldoldold\n")
    with pytest.raises(VLLMBuildError, match="not the source tree"):
        build_peer("xpu", SHA, "/builds/vllm-new", runner,
                   container="my_peer", allow_network=False)


def test_build_peer_accepts_a_container_with_the_right_tree():
    runner = _build_runner([("docker inspect -f", CommandResult(0, "running\n", ""))],
                           mounts="/mnt/weka\n/builds/vllm-new\n")
    build = build_peer("xpu", SHA, "/builds/vllm-new", runner,
                       container="my_peer", allow_network=False)
    assert build.container == "my_peer"


def test_build_peer_restarts_stopped_container():
    runner = _build_runner([("docker inspect -f", CommandResult(0, "exited\n", ""))])
    build = build_peer("xpu", SHA, "/src", runner, allow_network=False)

    assert build.built is False
    assert runner.ran("docker start")
    assert not runner.ran("pip install")


def test_build_peer_starts_from_cached_image_without_reinstalling():
    states = iter([
        CommandResult(1, "", "No such object"),  # container absent
        CommandResult(0, "running\n", ""),       # after launch
    ])
    runner = _build_runner([
        ("docker inspect -f", lambda: next(states)),
        ("docker image inspect", CommandResult(0, "[]", "")),
    ])
    build = build_peer("cuda", SHA, "/src", runner, allow_network=False)

    assert build.built is False
    assert build.image == build_image_tag("cuda", SHA, False)
    assert runner.ran("docker run")
    assert not runner.ran("pip install")
    assert not runner.ran("docker pull")


def test_build_peer_cold_cache_pulls_installs_and_commits():
    states = iter([
        CommandResult(1, "", "No such object"),  # container absent
        CommandResult(0, "running\n", ""),       # after launch
    ])
    runner = _build_runner([
        ("docker inspect -f", lambda: next(states)),
        ("docker image inspect", CommandResult(1, "", "No such image")),
    ])
    build = build_peer(
        "xpu", SHA, "/host/src", runner, mounts=("/mnt/weka",), allow_network=False
    )

    assert build.built is True
    assert build.base_image == DEFAULT_XPU_BASE_IMAGE
    run_cmd = runner.ran("docker run")[0]
    assert "-v /host/src:/workspace/vllm" in run_cmd
    assert "-v /mnt/weka:/mnt/weka" in run_cmd
    assert "--device /dev/dri" in run_cmd
    # Vendor images may auto-start a server; we need an idle container.
    assert "--entrypoint /bin/bash" in run_cmd and "sleep infinity" in run_cmd
    assert runner.ran("pip install --no-build-isolation -e .")
    assert runner.ran(f"docker commit") and build_image_tag("xpu", SHA, False) in runner.ran("docker commit")[0]


def test_build_peer_cuda_container_gets_gpu_flags():
    states = iter([CommandResult(1, "", ""), CommandResult(0, "running\n", "")])
    runner = _build_runner([
        ("docker inspect -f", lambda: next(states)),
        ("docker image inspect", CommandResult(1, "", "")),
    ])
    build_peer("cuda", SHA, "/src", runner, allow_network=False)
    assert "--gpus all" in runner.ran("docker run")[0]


def test_build_peer_reports_precompiled_kernels_in_the_note():
    """A report must never imply kernel coverage the build does not have."""
    states = iter([CommandResult(1, "", ""), CommandResult(0, "running\n", "")])
    runner = _build_runner([
        ("docker inspect -f", lambda: next(states)),
        ("docker image inspect", CommandResult(1, "", "")),
    ])
    build = build_peer("cuda", SHA, "/src", runner, allow_network=False)

    assert build.precompiled is True
    assert "not this commit" in build.note
    assert "not this commit" in build.summary()


def test_build_peer_raises_with_inspection_hint_when_install_fails():
    states = iter([CommandResult(1, "", ""), CommandResult(0, "running\n", "")])
    runner = _build_runner([
        ("docker inspect -f", lambda: next(states)),
        ("docker image inspect", CommandResult(1, "", "")),
        ("pip install --no-build-isolation", CommandResult(1, "", "error: cmake failed")),
    ])
    with pytest.raises(VLLMBuildError, match="cmake failed"):
        build_peer("xpu", SHA, "/src", runner, allow_network=False)


def test_build_peer_raises_when_base_image_unavailable():
    states = iter([CommandResult(1, "", "")])
    runner = _build_runner([
        ("docker inspect -f", lambda: next(states)),
        ("docker image inspect", CommandResult(1, "", "No such image")),
        ("docker pull", CommandResult(1, "", "unauthorized")),
    ])
    with pytest.raises(VLLMBuildError, match="Could not pull"):
        build_peer("cuda", SHA, "/src", runner, allow_network=False)


def test_build_peer_warns_when_install_is_not_editable():
    """A non-editable install would make patches land on unimported files."""
    runner = _build_runner(
        [("docker inspect -f", CommandResult(0, "running\n", ""))],
        vllm_path="/usr/lib/python3/dist-packages",
    )
    build = build_peer("xpu", SHA, "/src", runner, allow_network=False)
    assert "not the mounted source tree" in build.note


def test_build_peer_rejects_unknown_device():
    with pytest.raises(VLLMBuildError, match="Unknown device"):
        build_peer("tpu", SHA, "/src", FakeRunner(), allow_network=False)


# --------------------------------------------------------------------------
# end-to-end wiring into DebugConfig
# --------------------------------------------------------------------------

def _commit_config(tmp_path, **overrides):
    kwargs = dict(
        backend="vllm",
        model_path=str(tmp_path / "model"),
        shared_fs=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        vllm_commit=SHA,
        vllm_repo_path=str(tmp_path / "vllm"),
        vllm_build_root=str(tmp_path / "builds"),
    )
    kwargs.update(overrides)
    return DebugConfig(**kwargs)


def _install_fakes(monkeypatch, module, *, nvidia=True, xpu=True, docker=True):
    monkeypatch.setattr(module, "find_vllm_repo", lambda p="", r=None: __import__("pathlib").Path(p or "/repo"))
    monkeypatch.setattr(module, "resolve_commit", lambda ref, repo, runner=None, allow_fetch=True: SHA)
    monkeypatch.setattr(
        module, "ensure_source_tree",
        lambda repo, commit, root, runner=None: __import__("pathlib").Path(root) / f"vllm-{SHORT}",
    )
    monkeypatch.setattr(module, "docker_available", lambda runner: docker)
    monkeypatch.setattr(module, "host_has_nvidia_gpu", lambda runner: nvidia)
    monkeypatch.setattr(module, "host_has_xpu", lambda runner: xpu)
    monkeypatch.setattr(module, "CommandRunner", lambda **kwargs: FakeRunner())

    def fake_build(device, commit, source_dir, runner, **kwargs):
        return VLLMBuild(
            device=device, commit=commit,
            base_image=f"base/{device}", image=build_image_tag(device, commit, False),
            container=peer_container_name(device, commit, False),
            vllm_path="/workspace/vllm", source_dir=source_dir, built=True,
            precompiled=(device == "cuda"),
        )

    monkeypatch.setattr(module, "build_peer", fake_build)


def test_autoconfigure_builds_both_peers(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module)
    config = _commit_config(tmp_path)

    builds = autoconfigure_from_commit(config)

    assert set(builds) == {"gpu", "xpu"}
    assert config.gpu_docker == peer_container_name("cuda", SHA, False)
    assert config.xpu_docker == peer_container_name("xpu", SHA, False)
    assert config.gpu_vllm_path == config.xpu_vllm_path == "/workspace/vllm"
    assert config.gpu_image == build_image_tag("cuda", SHA, False)
    assert config.xpu_image == build_image_tag("xpu", SHA, False)
    # Freshly launched containers are never the one this process runs in.
    assert config.gpu_inside_container is False
    assert config.xpu_inside_container is False


def test_autoconfigure_builds_only_the_present_device(monkeypatch, tmp_path):
    """On an XPU-only box the XPU peer is still built; the run stays XPU-only."""
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module, nvidia=False)
    config = _commit_config(tmp_path)
    reasons = []

    builds = autoconfigure_from_commit(config, on_skip=reasons.append)

    assert set(builds) == {"xpu"}
    assert config.gpu_docker == ""
    assert config.xpu_docker == peer_container_name("xpu", SHA, False)
    assert any("NVIDIA GPU" in r for r in reasons)


def test_autoconfigure_leaves_an_explicitly_configured_side_alone(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module)
    config = _commit_config(tmp_path, xpu_docker="my_xpu_container")
    reasons = []

    builds = autoconfigure_from_commit(config, on_skip=reasons.append)

    assert set(builds) == {"gpu"}
    assert config.xpu_docker == "my_xpu_container"
    assert any("explicitly" in r for r in reasons)


def test_autoconfigure_does_not_clone_when_nothing_is_buildable(monkeypatch, tmp_path):
    """Exporting a commit is a multi-GB clone; a no-op run must not pay for it."""
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module, nvidia=False, xpu=False)

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("no buildable side means no repo work")

    monkeypatch.setattr(module, "find_vllm_repo", fail)
    monkeypatch.setattr(module, "ensure_source_tree", fail)
    reasons = []

    assert autoconfigure_from_commit(_commit_config(tmp_path), on_skip=reasons.append) == {}
    assert any("no side left to build" in r for r in reasons)


def test_autoconfigure_skips_without_docker(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module, docker=False)
    assert autoconfigure_from_commit(_commit_config(tmp_path)) == {}


def test_autoconfigure_requires_the_vllm_backend(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module)
    config = _commit_config(tmp_path, backend="pytorch")
    reasons = []

    assert autoconfigure_from_commit(config, on_skip=reasons.append) == {}
    assert any("not vllm" in r for r in reasons)


def test_autoconfigure_is_a_noop_without_a_commit(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("no commit means no repo work")

    monkeypatch.setattr(module, "find_vllm_repo", fail)
    assert autoconfigure_from_commit(_commit_config(tmp_path, vllm_commit="")) == {}


def test_autoconfigure_export_only_does_not_build(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module)

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("launch=False must not build containers")

    monkeypatch.setattr(module, "build_peer", fail)
    config = _commit_config(tmp_path)

    assert autoconfigure_from_commit(config, launch=False) == {}
    assert config.gpu_docker == ""


# --------------------------------------------------------------------------
# the single entry point both the CLI and the Bisector use
# --------------------------------------------------------------------------

def test_maybe_autoconfigure_dispatches_to_the_commit_path(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    _install_fakes(monkeypatch, module)

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("a commit build must not also match release images")

    monkeypatch.setattr(module, "maybe_autoconfigure_gpu_docker", fail)
    setup = maybe_autoconfigure_peers(_commit_config(tmp_path))

    assert setup.commit == SHA
    assert set(setup.builds) == {"gpu", "xpu"}
    assert setup.configured
    assert any(SHORT in line for line in setup.summary_lines())


def test_maybe_autoconfigure_falls_back_to_release_matching(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    called = []
    monkeypatch.setattr(
        module, "maybe_autoconfigure_gpu_docker",
        lambda config, launch=True, allow_network=True, on_skip=None: called.append(config) or None,
    )
    setup = maybe_autoconfigure_peers(_commit_config(tmp_path, vllm_commit=""))

    assert len(called) == 1
    assert setup.builds == {}


def test_maybe_autoconfigure_reports_failure_without_raising(monkeypatch, tmp_path):
    import accuracy_agent.vllm_source_builder as module

    def boom(*a, **k):
        raise VLLMBuildError("cmake exploded")

    monkeypatch.setattr(module, "autoconfigure_from_commit", boom)
    reasons = []

    setup = maybe_autoconfigure_peers(_commit_config(tmp_path), on_skip=reasons.append)

    assert setup.configured is False
    assert any("cmake exploded" in r for r in reasons)


def test_maybe_autoconfigure_runs_at_most_once(monkeypatch, tmp_path):
    """The CLI and the Bisector both call it; a multi-hour build must not repeat."""
    import accuracy_agent.vllm_source_builder as module

    calls = []
    monkeypatch.setattr(
        module, "autoconfigure_from_commit",
        lambda config, **kwargs: calls.append(config) or {},
    )
    config = _commit_config(tmp_path)

    maybe_autoconfigure_peers(config)
    maybe_autoconfigure_peers(config)

    assert len(calls) == 1
