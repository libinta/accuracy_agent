#!/usr/bin/env python3
"""Basic CLI validation tests."""

import ast
import yaml
from pathlib import Path

def test_cli_syntax():
    """Verify CLI file has valid Python syntax."""
    cli_path = Path(__file__).parent / 'accuracy_agent' / 'cli.py'
    with open(cli_path) as f:
        code = f.read()
    ast.parse(code)
    print("✓ CLI syntax is valid")

def test_cli_has_click_command():
    """Verify CLI uses click and has main command."""
    cli_path = Path(__file__).parent / 'accuracy_agent' / 'cli.py'
    with open(cli_path) as f:
        code = f.read()

    assert 'import click' in code
    assert '@click.command()' in code
    assert 'def main(' in code
    assert 'if __name__ == "__main__":' in code
    print("✓ CLI has click command structure")

def test_cli_has_required_options():
    """Verify CLI has all required options."""
    cli_path = Path(__file__).parent / 'accuracy_agent' / 'cli.py'
    with open(cli_path) as f:
        code = f.read()

    required_options = [
        '--config',
        '--model',
        '--gpu-host',
        '--gpu-docker',
        '--xpu-host',
        '--xpu-docker',
        '--shared-fs',
        '--output-dir',
        '--layer-start',
        '--layer-end'
    ]

    for option in required_options:
        assert option in code, f"Missing option: {option}"
    print(f"✓ CLI has all {len(required_options)} required options")

def test_cli_imports():
    """Verify CLI imports required modules."""
    cli_path = Path(__file__).parent / 'accuracy_agent' / 'cli.py'
    with open(cli_path) as f:
        code = f.read()

    required_imports = [
        'import click',
        'import yaml',
        'from rich.console import Console',
        'from rich.table import Table',
        'from accuracy_agent.config import DebugConfig',
        'from accuracy_agent.model_loader import load_model_info',
        'from accuracy_agent.bisector import Bisector',
    ]

    for imp in required_imports:
        assert imp in code, f"Missing import: {imp}"
    print(f"✓ CLI has all required imports")

def test_example_config_exists():
    """Verify example config file exists."""
    config_path = Path(__file__).parent / 'examples' / 'flash_tp2_config.yaml'
    assert config_path.exists(), f"Example config not found: {config_path}"
    print(f"✓ Example config exists: {config_path}")

def test_example_config_valid():
    """Test that example config file is valid YAML with required keys."""
    config_path = Path(__file__).parent / 'examples' / 'flash_tp2_config.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check required keys
    assert 'model' in config
    assert 'path' in config['model']
    assert 'gpu' in config
    assert 'host' in config['gpu']
    assert 'docker' in config['gpu']
    assert 'xpu' in config
    assert 'host' in config['xpu']
    assert 'docker' in config['xpu']
    assert 'shared_fs' in config
    assert 'output_dir' in config
    assert 'test' in config
    assert 'layer_start' in config['test']
    assert 'layer_end' in config['test']

    # Validate paths
    assert config['model']['path'].startswith(config['shared_fs'])
    assert config['output_dir'].startswith(config['shared_fs'])

    print(f"✓ Example config has valid structure")

def test_setup_entry_point():
    """Verify setup.py has CLI entry point."""
    setup_path = Path(__file__).parent / 'setup.py'
    with open(setup_path) as f:
        content = f.read()

    assert 'accuracy-debug=accuracy_agent.cli:main' in content
    print("✓ setup.py has CLI entry point")

if __name__ == '__main__':
    test_cli_syntax()
    test_cli_has_click_command()
    test_cli_has_required_options()
    test_cli_imports()
    test_example_config_exists()
    test_example_config_valid()
    test_setup_entry_point()
    print("\n✅ All CLI basic tests passed!")
