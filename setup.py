"""Install repository tooling; model frameworks have separate environments."""
from setuptools import find_packages, setup

setup(
    name="prose-tools",
    version="0.1.0",
    description="Experiment entry points and shared utilities for the PROSE checkout",
    python_requires=">=3.10",
    packages=find_packages(include=["prose", "prose.*", "scripts", "scripts.*", "act", "act.*"]),
    py_modules=["hf_datasets_compat"],
)
