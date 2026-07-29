from __future__ import annotations

import os
import random
import runpy

import numpy as np


def main() -> None:
    """Seed supported random generators and start CodeEntropy."""
    seed = int(os.environ.get("CODEENTROPY_RANDOM_SEED", "0"))

    random.seed(seed)
    np.random.seed(seed)

    runpy.run_module("CodeEntropy", run_name="__main__")


if __name__ == "__main__":
    main()
