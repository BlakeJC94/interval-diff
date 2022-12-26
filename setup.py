from setuptools import setup, find_packages

__version__ = "0.0.0"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PLACEHOLDER",
    version=__version__,
    description="PLACEHOLDER",
    long_description=long_description,
    author="PLACEHOLDER",
    author_email="PLACEHOLDER@PLACEHOLDER.PLACEHOLDER",
    url="https://github.com/BlakeJC94/PLACEHOLDER",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "dev": [
            "black",
            "pip-tools",
            "pre-commit",
            "pylint",
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": ["PLACEHOLDER=PLACEHOLDER.__main__:main"],
    },
)
