from pathlib import Path
from pkg_resources import get_distribution

from .log import setup_logging

setup_logging()

__all__ = [
    "__version__",
]

__version__ = get_distribution(Path(__file__).parent.name).version
