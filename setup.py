from setuptools import setup, find_packages

__version__ = "0.1.0"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="interval_diff",
    version=__version__,
    description="Implementation of a vectorised interval set difference operation in numpy",
    long_description=long_description,
    author="BlakeJC94",
    author_email="blakejamescook@gmail.com",
    url="https://github.com/BlakeJC94/BlakeJC94/interval-diff",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "plotly",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "black",
            "pip-tools",
            "pylint",
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": ["interval-diff=interval_diff.__main__:main"],
    },
)
