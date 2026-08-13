# accuracy_agent/cli.py
import click
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import load_model_info
from accuracy_agent.bisector import Bisector
from accuracy_agent.vllm_source_builder import maybe_autoconfigure_peers

console = Console()

@click.command()
@click.option('--config', type=click.Path(exists=True), help='Config YAML file')
@click.option('--model', type=str, help='Model path (overrides config)')
@click.option('--backend', type=click.Choice(['vllm', 'pytorch', 'sglang']), default=None,
              help='Inference backend (default: pytorch, or the value in --config)')
@click.option('--gpu-host', type=str, help='GPU host')
@click.option('--gpu-docker', type=str, help='GPU docker container')
@click.option('--xpu-host', type=str, help='XPU host')
@click.option('--xpu-docker', type=str, help='XPU docker container')
@click.option('--gpu-image', type=str, help='GPU docker image (default: matched to the XPU vLLM version)')
@click.option('--no-auto-gpu-image', is_flag=True, default=False,
              help='Do not derive/launch the GPU docker from the XPU vLLM version')
@click.option('--vllm-commit', type=str, default=None,
              help='vllm-project/vllm commit (sha/tag/branch) to install from source '
                   'into the vendor PyTorch images on both sides')
@click.option('--vllm-repo', type=str, default=None,
              help='Local vllm-project/vllm clone to resolve --vllm-commit in (default: ~/vllm)')
@click.option('--build-kernels', is_flag=True, default=False,
              help='With --vllm-commit: compile CUDA kernels at that commit (1-2h) instead '
                   'of using the precompiled nightly wheel')
@click.option('--rebuild-vllm', is_flag=True, default=False,
              help='With --vllm-commit: ignore cached built images and install again')
@click.option('--gpu-base-image', type=str, default=None,
              help='Base image for the built GPU peer (default: newest nvcr.io/nvidia/pytorch)')
@click.option('--xpu-base-image', type=str, default=None,
              help='Base image for the built XPU peer (default: newest intel/intel-extension-for-pytorch)')
@click.option('--shared-fs', type=str, default='/mnt/weka', help='Shared filesystem path')
@click.option('--output-dir', type=str, help='Output directory on shared FS')
@click.option('--layer-start', type=int, default=None, help='First layer to test')
@click.option('--layer-end', type=int, default=None, help='Last layer to test (exclusive)')
def main(config, model, backend, gpu_host, gpu_docker, xpu_host, xpu_docker, gpu_image,
         no_auto_gpu_image, vllm_commit, vllm_repo, build_kernels, rebuild_vllm,
         gpu_base_image, xpu_base_image, shared_fs, output_dir, layer_start, layer_end):
    """XPU Accuracy Debugger - Find GPU/XPU divergences automatically."""

    console.print("[bold cyan]XPU Accuracy Debugger POC[/bold cyan]\n")

    # Load config
    if config:
        # Use DebugConfig.from_yaml() to properly load all fields including backend
        debug_config = DebugConfig.from_yaml(config)

        # Override with CLI arguments if provided
        if model:
            debug_config.model_path = model
        if backend:
            debug_config.backend = backend
        if gpu_host:
            debug_config.gpu_host = gpu_host
        if gpu_docker:
            debug_config.gpu_docker = gpu_docker
        if xpu_host:
            debug_config.xpu_host = xpu_host
        if xpu_docker:
            debug_config.xpu_docker = xpu_docker
        if gpu_image:
            debug_config.gpu_image = gpu_image
        if no_auto_gpu_image:
            debug_config.gpu_auto_image = False
        if vllm_commit:
            debug_config.vllm_commit = vllm_commit
        if vllm_repo:
            debug_config.vllm_repo_path = vllm_repo
        if build_kernels:
            debug_config.vllm_build_kernels = True
        if rebuild_vllm:
            debug_config.vllm_build_rebuild = True
        if gpu_base_image:
            debug_config.gpu_base_image = gpu_base_image
        if xpu_base_image:
            debug_config.xpu_base_image = xpu_base_image
        if shared_fs != '/mnt/weka':  # Check if non-default
            debug_config.shared_fs = shared_fs
        if output_dir:
            debug_config.output_dir = output_dir
        if layer_start is not None:
            debug_config.layer_start = layer_start
        if layer_end is not None:
            debug_config.layer_end = layer_end
    else:
        # CLI args only. Both docker sides are optional: --vllm-commit builds
        # them, and otherwise a local XPU docker is enough to derive the GPU side
        # from its vLLM version.
        if not model or not (xpu_docker or vllm_commit):
            console.print(
                "[red]Error: Must provide either --config, or --model plus "
                "--xpu-docker (or --vllm-commit to build both peers)[/red]"
            )
            return

        debug_config = DebugConfig(
            model_path=model,
            backend=backend or "pytorch",
            gpu_host=gpu_host or "",
            gpu_docker=gpu_docker or "",
            xpu_host=xpu_host or "",
            xpu_docker=xpu_docker or "",
            gpu_image=gpu_image or "",
            gpu_auto_image=not no_auto_gpu_image,
            vllm_commit=vllm_commit or "",
            vllm_repo_path=vllm_repo or "",
            vllm_build_kernels=build_kernels,
            vllm_build_rebuild=rebuild_vllm,
            gpu_base_image=gpu_base_image or "",
            xpu_base_image=xpu_base_image or "",
            shared_fs=shared_fs,
            output_dir=output_dir or f"{shared_fs}/accuracy_debug_output",
            layer_start=layer_start if layer_start is not None else 0,
            layer_end=layer_end if layer_end is not None else 3
        )

    # Fill in the docker peers automatically (build them from --vllm-commit, or
    # match a release image to the XPU container's version) before the config is
    # printed, so the table below reflects what the run will actually use.
    needs_peers = debug_config.backend == "vllm" and (
        debug_config.vllm_commit
        or (not debug_config.gpu_docker and debug_config.gpu_auto_image)
    )
    if needs_peers:
        if debug_config.vllm_commit:
            console.print(
                f"[yellow]Building vLLM @ {debug_config.vllm_commit} for both peers "
                f"({'compiling CUDA kernels' if debug_config.vllm_build_kernels else 'precompiled kernels'})"
                "...[/yellow]"
            )
        else:
            console.print("[yellow]Matching GPU docker image to the XPU vLLM version...[/yellow]")

        setup = maybe_autoconfigure_peers(debug_config)
        for line in setup.summary_lines():
            console.print(f"[green]✓ {line}[/green]" if not line.startswith("note:")
                          else f"[yellow]  {line}[/yellow]")
        if not setup.configured:
            # Report why, since the CLI does not configure logging.
            for reason in setup.skipped:
                console.print(f"[yellow]  {reason}[/yellow]")
            console.print(
                "[yellow]No peer auto-configured; continuing "
                "(XPU-only unless --gpu-docker is given).[/yellow]"
            )
        console.print()

    # Print config
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Backend", debug_config.backend)
    table.add_row("Model", debug_config.model_path)
    if debug_config.vllm_commit:
        table.add_row("vLLM commit", debug_config.vllm_commit)
    table.add_row("GPU", f"{debug_config.gpu_host or 'localhost'} / {debug_config.gpu_docker or '(none)'}")
    if debug_config.gpu_image:
        table.add_row("GPU image", debug_config.gpu_image)
    table.add_row("XPU", f"{debug_config.xpu_host or 'localhost'} / {debug_config.xpu_docker or '(none)'}")
    if debug_config.xpu_image:
        table.add_row("XPU image", debug_config.xpu_image)
    table.add_row("Layers", f"{debug_config.layer_start}-{debug_config.layer_end}")
    table.add_row("Output", debug_config.output_dir)

    console.print(table)
    console.print()

    # Load model info
    console.print("[yellow]Loading model config...[/yellow]")
    model_info = load_model_info(debug_config.model_path)

    console.print(f"✓ Model: {model_info.num_layers} layers, {model_info.layer_type} architecture\n")

    # Run bisection
    bisector = Bisector(debug_config, model_info)

    try:
        if debug_config.layer_select == "auto":
            groups = model_info.layer_groups or [("standard", debug_config.layer_start)]
            reps = ", ".join(f"{n}@{i}" for n, i in groups)
            console.print(f"[yellow]Auto-selected representative layers: {reps}[/yellow]\n")
            result = bisector.bisect_layer_set(groups)
        else:
            result = bisector.bisect_layers(
                debug_config.layer_start,
                debug_config.layer_end
            )

        # Print results
        console.print("\n" + "="*60)
        console.print("[bold]Bisection Results[/bold]")
        console.print("="*60 + "\n")

        if getattr(result, "extracted_only", False):
            console.print("[green]✓ XPU hidden states extracted (no GPU peer to compare)[/green]")
        elif result.divergent_layer is not None:
            console.print(f"[red]✗ Divergence found in layer {result.divergent_layer}[/red]")
        else:
            console.print("[green]✓ All layers match![/green]")

        console.print(f"\n{result.report}\n")

        # Print detailed comparison results
        if result.comparison_results:
            comp_table = Table(title="Layer Comparisons")
            comp_table.add_column("Layer Range", style="cyan")
            comp_table.add_column("Status", style="white")
            comp_table.add_column("Cosine Sim", style="white")
            comp_table.add_column("Rel Error", style="white")

            for i, comp in enumerate(result.comparison_results):
                status = "✓ Match" if comp.match else "✗ Diverge"
                style = "green" if comp.match else "red"

                # Label by layer KIND when representative layers were tested;
                # otherwise fall back to the contiguous-range index.
                if result.tested_layers and i < len(result.tested_layers):
                    name, idx = result.tested_layers[i]
                    layer_label = f"Layer {idx} ({name})"
                else:
                    layer_label = f"Layer {debug_config.layer_start + i}"

                comp_table.add_row(
                    layer_label,
                    f"[{style}]{status}[/{style}]",
                    f"{comp.cosine_similarity:.6f}",
                    f"{comp.max_rel_error:.6f}"
                )

            console.print(comp_table)

    except Exception as e:
        console.print(f"[red]Error during bisection: {e}[/red]")
        raise

if __name__ == "__main__":
    main()
