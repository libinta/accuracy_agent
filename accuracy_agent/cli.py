# accuracy_agent/cli.py
import click
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import load_model_info
from accuracy_agent.bisector import Bisector

console = Console()

@click.command()
@click.option('--config', type=click.Path(exists=True), help='Config YAML file')
@click.option('--model', type=str, help='Model path (overrides config)')
@click.option('--gpu-host', type=str, help='GPU host')
@click.option('--gpu-docker', type=str, help='GPU docker container')
@click.option('--xpu-host', type=str, help='XPU host')
@click.option('--xpu-docker', type=str, help='XPU docker container')
@click.option('--shared-fs', type=str, default='/mnt/weka', help='Shared filesystem path')
@click.option('--output-dir', type=str, help='Output directory on shared FS')
@click.option('--layer-start', type=int, default=None, help='First layer to test')
@click.option('--layer-end', type=int, default=None, help='Last layer to test (exclusive)')
def main(config, model, gpu_host, gpu_docker, xpu_host, xpu_docker, shared_fs, output_dir, layer_start, layer_end):
    """XPU Accuracy Debugger - Find GPU/XPU divergences automatically."""

    console.print("[bold cyan]XPU Accuracy Debugger POC[/bold cyan]\n")

    # Load config
    if config:
        # Use DebugConfig.from_yaml() to properly load all fields including backend
        debug_config = DebugConfig.from_yaml(config)

        # Override with CLI arguments if provided
        if model:
            debug_config.model_path = model
        if gpu_host:
            debug_config.gpu_host = gpu_host
        if gpu_docker:
            debug_config.gpu_docker = gpu_docker
        # CLI flags stay named --xpu-* (legacy, CLI-only path); they map onto the
        # renamed device-under-test config fields.
        if xpu_host:
            debug_config.dut_host = xpu_host
        if xpu_docker:
            debug_config.dut_docker = xpu_docker
        if shared_fs != '/mnt/weka':  # Check if non-default
            debug_config.shared_fs = shared_fs
        if output_dir:
            debug_config.output_dir = output_dir
        if layer_start is not None:
            debug_config.layer_start = layer_start
        if layer_end is not None:
            debug_config.layer_end = layer_end
    else:
        # CLI args only
        if not all([model, gpu_host, gpu_docker, xpu_host, xpu_docker]):
            console.print("[red]Error: Must provide either --config or all required arguments[/red]")
            return

        debug_config = DebugConfig(
            model_path=model,
            gpu_host=gpu_host,
            gpu_docker=gpu_docker,
            dut_host=xpu_host,
            dut_docker=xpu_docker,
            shared_fs=shared_fs,
            output_dir=output_dir or f"{shared_fs}/accuracy_debug_output",
            layer_start=layer_start if layer_start is not None else 0,
            layer_end=layer_end if layer_end is not None else 3
        )

    # Print config
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model", debug_config.model_path)
    # Reference (peer) slot is always CUDA GPU; always show it.
    table.add_row("GPU", f"{debug_config.gpu_host} / {debug_config.gpu_docker}")
    # Device-under-test slot: label by its real type (XPU for Intel GPU, HPU for
    # Intel Gaudi), not a hardcoded "XPU" -- a Gaudi run sets xpu.device_type=hpu.
    dut_label = (debug_config.dut_device_type or "xpu").upper()
    table.add_row(dut_label, f"{debug_config.dut_host} / {debug_config.dut_docker}")
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
            console.print(
                f"[green]✓ {dut_label} hidden states extracted "
                f"(no GPU peer to compare)[/green]")
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
                elif (len(result.comparison_results) == 1
                      and debug_config.layer_end - debug_config.layer_start > 1):
                    # Whole-window match: bisection never split into per-layer
                    # results (the window compared equal), so this single row is
                    # the range [start, end), NOT layer 0.
                    layer_label = (f"Layers {debug_config.layer_start}-"
                                   f"{debug_config.layer_end}")
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
