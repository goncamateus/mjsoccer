"""Session-wide test setup.

JAX picks its platform once, when its first backend is made, so whichever test module
happens to be collected first must not decide it. The in-process JAX tests (task-logic
parity, the DR logic) are a CPU-only harness (ADR 0002): the session is pinned to the CPU
here, before any test module imports jax. GPU runs go through `tests/_gpu.py`, in a
subprocess whose environment drops this.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
