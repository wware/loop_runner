#!/bin/bash -xe

ruff check "$@"
pylint "$@"
flake8 "$@"
for file in "$@"; do
    if [[ $(basename "$file") =~ ^test_ ]]; then
        pytest "$file"
    fi
done
