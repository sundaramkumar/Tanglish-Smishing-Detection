#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper.
set -euo pipefail
cd "$(dirname "$0")"
python3 src/audit.py
python3 src/experiments.py
python3 src/make_figure.py
echo "Done. See the results/ folder."
