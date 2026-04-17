from __future__ import annotations

import os


def pytest_configure(config):
    """
    Configure environment for unit tests.

    This runs before test collection/execution, ensuring that:
    - Numba JIT is disabled globally
    - Tests remain deterministic
    """
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
