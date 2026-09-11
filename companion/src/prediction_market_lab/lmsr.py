"""Classroom LMSR; liquidity B is distinct from an executable order-book bid."""

from math import isfinite
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, softmax


def _inputs(q: Sequence[float], liquidity_B: float) -> tuple[NDArray, float]:
    positions = np.asarray(q, dtype=float)
    B = float(liquidity_B)
    if (positions.ndim != 1 or positions.size < 2 or not np.all(np.isfinite(positions))
            or not isfinite(B) or B <= 0):
        raise ValueError("finite outcome quantities and positive finite liquidity B required")
    return positions, B


def cost(q: Sequence[float], liquidity_B: float) -> float:
    positions, B = _inputs(q, liquidity_B)
    anchor = float(positions.max())
    with np.errstate(over="ignore"):
        result = anchor + B * float(logsumexp((positions - anchor) / B))
    if not isfinite(result):
        raise ValueError("LMSR cost exceeds numeric range")
    return result


def prices(q: Sequence[float], liquidity_B: float) -> NDArray:
    positions, B = _inputs(q, liquidity_B)
    with np.errstate(over="ignore"):
        return softmax((positions - positions.max()) / B)


def trade_cost(q: Sequence[float], delta: Sequence[float], liquidity_B: float) -> float:
    positions, B = _inputs(q, liquidity_B)
    change = np.asarray(delta, dtype=float)
    if change.shape != positions.shape or not np.all(np.isfinite(change)):
        raise ValueError("trade vector must match finite outcome positions")
    # Remove a shared offset before subtraction to avoid catastrophic cancellation.
    shifted = positions - positions.max()
    return cost(shifted + change, B) - cost(shifted, B)
