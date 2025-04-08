#!/bin/bash -xe

ruff check "$@"
pylint "$@"
flake8 "$@"
pytest "$@"
