from __future__ import annotations

import random

from nvf.benchmark.schema import BenchmarkItem


def split_calibration_eval(
    items: list[BenchmarkItem],
    calibration_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[BenchmarkItem], list[BenchmarkItem]]:
    """Split benchmark into calibration and evaluation sets deterministically."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_cal = int(len(shuffled) * calibration_fraction)
    return shuffled[:n_cal], shuffled[n_cal:]
