#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

python3 tests/run_public.py
python3 tests/run_extended.py
python3 tests/run_crash.py
