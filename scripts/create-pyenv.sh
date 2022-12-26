#!/usr/bin/env bash
# Generate pyenv
#
# Usage:
#     $ ./scripts/create-pyenv.sh
#
pyenv virtualenv 3.9.7 interval-diff
pyenv local interval-diff
pip install pip-tools
