from setuptools import setup, find_packages

setup(
    name="accuracy_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "paramiko>=3.0.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "accuracy-debug=accuracy_agent.cli:main",
        ],
    },
)
