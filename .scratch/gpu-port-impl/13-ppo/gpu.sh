#!/usr/bin/env bash
# Run a GPU job the ticket-13 way: no XLA preallocation, under a 10 GB memory cap.
#   .scratch/gpu-port-impl/13-ppo/gpu.sh <python args...>
cd "$(dirname "$0")/../../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec systemd-run --user --scope -q -p MemoryMax=10G .venv/bin/python "$@"
