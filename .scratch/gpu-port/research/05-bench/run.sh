#!/bin/bash
# Usage: ./run.sh <stage> args...  -- one process under an 8 GB host cgroup; appends RESULT lines to results.jsonl
cd "$(dirname "$0")"
export PYTHONPATH=../../../../src XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.9}
systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-8G} -p MemorySwapMax=0 \
  ~/.venvs/mjsoccer-gpu/bin/python bench.py "$@" 2>&1 | tee -a log.txt | grep --line-buffered '^RESULT' | sed -u 's/^RESULT //' | tee -a results.jsonl
