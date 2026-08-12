#!/usr/bin/env python3
"""Test CLI structure and configuration loading."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml

# Mock all heavy dependencies before importing
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['safetensors'] = Mock()
sys.modules['paramiko'] = Mock()

# Now we can import
from accuracy_agent.cli import main
from click.testing import CliRunner

def test_cli_help():
    """Test that CLI shows help message."""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])

    assert result.exit_code == 0
    assert 'XPU Accuracy Debugger' in result.output
    assert '--config' in result.output
    assert '--model' in result.output
    assert '--gpu-host' in result.output
    assert '--xpu-host' in result.output
    print("✓ CLI help message works")

def test_cli_with_config():
    """Test CLI with config file."""
    runner = CliRunner()

    # Create temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'model': {
                'path': '/mnt/weka/model'
            },
            'gpu': {
                'host': 'gpu.example.com',
                'docker': 'gpu_container'
            },
            'xpu': {
                'host': 'xpu.example.com',
                'docker': 'xpu_container'
            },
            'shared_fs': '/mnt/weka',
            'output_dir': '/mnt/weka/output',
            'test': {
                'layer_start': 0,
                'layer_end': 3
            }
        }
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Mock the heavy lifting parts
        with patch('accuracy_agent.cli.load_model_info') as mock_load:
            with patch('accuracy_agent.cli.Bisector') as mock_bisector:
                # Setup mocks
                mock_model_info = Mock()
                mock_model_info.num_layers = 60
                mock_model_info.layer_type = 'standard'
                mock_load.return_value = mock_model_info

                mock_result = Mock()
                mock_result.divergent_layer = None
                mock_result.report = "All layers match"
                mock_result.comparison_results = []

                mock_bisector_instance = Mock()
                mock_bisector_instance.bisect_layers.return_value = mock_result
                mock_bisector.return_value = mock_bisector_instance

                # Run CLI
                result = runner.invoke(main, ['--config', config_path])

                # Check that it ran
                assert result.exit_code == 0
                assert 'XPU Accuracy Debugger' in result.output
                print("✓ CLI with config file works")

                # Verify load_model_info was called with correct path
                mock_load.assert_called_once_with('/mnt/weka/model')

                # Verify bisector was created and called
                assert mock_bisector.called
                mock_bisector_instance.bisect_layers.assert_called_once_with(0, 3)

    finally:
        Path(config_path).unlink()

def test_cli_missing_required_args():
    """Test CLI fails gracefully without required args."""
    runner = CliRunner()

    # Try to run without config or required args
    result = runner.invoke(main, ['--model', '/mnt/weka/model'])

    # Should exit with error message
    assert 'Error: Must provide either --config or all required arguments' in result.output
    print("✓ CLI validates required arguments")

def test_example_config_valid():
    """Test that example config file is valid."""
    config_path = Path(__file__).parent / 'examples' / 'flash_tp2_config.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check required keys
    assert 'model' in config
    assert 'path' in config['model']
    assert 'gpu' in config
    assert 'xpu' in config
    assert 'shared_fs' in config
    assert 'output_dir' in config
    assert 'test' in config

    print(f"✓ Example config is valid: {config_path}")

if __name__ == '__main__':
    test_cli_help()
    test_cli_missing_required_args()
    test_cli_with_config()
    test_example_config_valid()
    print("\nAll CLI tests passed!")
