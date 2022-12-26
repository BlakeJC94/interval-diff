from pathlib import Path

import pytest


# Add modules for doctest namespaces
@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["Path"] = Path

