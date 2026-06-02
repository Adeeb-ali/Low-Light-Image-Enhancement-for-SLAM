#!/bin/bash

if [ ! -z "$EXTRA_PACKAGE" ]; then
    pip install --no-cache-dir $EXTRA_PACKAGE
fi

python run_inference.py "$@"