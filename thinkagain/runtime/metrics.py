"""Shared helpers for lightweight metrics and profiling."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def sample_stats(
    samples: Iterable[float],
    *,
    percentiles: tuple[int, ...] = (50, 95, 99),
) -> dict[str, float]:
    """Compute basic stats for a sample set.

    Returns mean, pXX percentiles, and count. Percentiles use linear
    interpolation over the sorted sample list.
    """
    data = list(samples)
    count = len(data)
    if count == 0:
        stats = {"mean": 0.0, "count": 0}
        for p in percentiles:
            stats[f"p{p}"] = 0.0
        return stats

    sorted_data = sorted(data)
    n = len(sorted_data)

    def pct(p: float) -> float:
        k = (n - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

    stats = {"mean": statistics.fmean(data), "count": count}
    for p in percentiles:
        stats[f"p{p}"] = pct(p)
    return stats
