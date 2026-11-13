#!/usr/bin/env bash
# The HF weights of a model for the types experiment (the pipeline deletes its own copy after converting).
set -euo pipefail
"$PANTHER_HF" download "$1" --revision "$2" --local-dir "$3"
