"""The one way a test reaches the GPU: a fresh subprocess, so the device is let go when it
ends and the JAX-CPU session (tests/conftest.py) never sees it.

The child runs without the session's `JAX_PLATFORMS`, does not preallocate GPU memory, and
exits `SKIP` when it cannot run here (no CUDA device, no gpu group).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

SKIP = 77  # a child's exit code for "cannot run here" (pytest's own "skipped" code)


def run(*argv, timeout: float | None = 1800) -> subprocess.CompletedProcess:
    """`python *argv` with the GPU visible; the exit code is the caller's to judge."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return subprocess.run([sys.executable, *map(str, argv)], env=env, capture_output=True,
                          text=True, timeout=timeout, check=False)


def run_json(script, *args, timeout: float | None = 1800) -> dict:
    """`run(script, *args)`: skipped on `SKIP`, failed on any other non-zero exit, else the
    child's last stdout line as JSON."""
    p = run(script, *args, timeout=timeout)
    if p.returncode == SKIP:
        pytest.skip("no CUDA device")
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])
