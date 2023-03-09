from setuptools import find_packages, setup

setup(
    name="attasr",
    version="1.1.0",
    package_dir=find_packages(),
    include_package_data=True,
    install_requires=[
        "datasets>=2.10.1",
        "pandas>=1.5.3",
        "setuptools>=65.5.0",
        "tqdm>=4.64.1",
        "transformers>=4.26.1",
        "torch>=1.12.1",
        "torchaudio>=0.12.1",
    ],
    entry_points={
        "console_scripts": [],
    },
)
