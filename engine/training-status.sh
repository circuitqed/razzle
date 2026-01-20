#!/bin/bash
# Training status dashboard - run from another terminal to monitor distributed training
cd "$(dirname "$0")"
.venv/bin/python scripts/training_status.py --output output/distributed --watch "$@"
