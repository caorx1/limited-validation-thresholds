"""Deterministic threshold selectors for protocol v1.1 budget experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

SCORE_TIE_TOLERANCE = 1e-12
ALL_NEGATIVE_THRESHOLD = float(np.nextafter(1.0, np.inf))


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    macro_f1: float
    candidate_count: int


def _validate(y_true: Iterable[int], probability: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and probability must be non-empty 1D arrays of equal length")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    return y.astype(np.int8, copy=False), p


def _macro_f1_for_thresholds(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    prefix_positive = np.concatenate(([0], np.cumsum(sorted_y, dtype=np.int64)))
    n = len(y)
    total_positive = int(prefix_positive[-1])
    left = np.searchsorted(sorted_p, thresholds, side="left")
    predicted_positive = n - left
    true_positive = total_positive - prefix_positive[left]
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    true_negative = n - total_positive - false_positive
    pos_denominator = 2 * true_positive + false_positive + false_negative
    neg_denominator = 2 * true_negative + false_negative + false_positive
    positive_f1 = np.divide(
        2 * true_positive, pos_denominator,
        out=np.zeros_like(thresholds, dtype=np.float64), where=pos_denominator != 0,
    )
    negative_f1 = np.divide(
        2 * true_negative, neg_denominator,
        out=np.zeros_like(thresholds, dtype=np.float64), where=neg_denominator != 0,
    )
    return 0.5 * (positive_f1 + negative_f1)


def choose_threshold(y_true: Iterable[int], probability: Iterable[float], *, grid: bool = False) -> ThresholdFit:
    y, p = _validate(y_true, probability)
    if grid:
        candidates = np.unique(np.concatenate((np.arange(101, dtype=np.float64) / 100.0, [ALL_NEGATIVE_THRESHOLD])))
    else:
        candidates = np.unique(np.concatenate((p, [0.0, 0.5, ALL_NEGATIVE_THRESHOLD])))
    scores = _macro_f1_for_thresholds(y, p, candidates)
    best_score = float(np.max(scores))
    tied = np.flatnonzero(scores >= best_score - SCORE_TIE_TOLERANCE)
    chosen = min(tied, key=lambda i: (abs(float(candidates[i]) - 0.5), -float(candidates[i])))
    return ThresholdFit(float(candidates[chosen]), best_score, len(candidates))


def choose_bootstrap_median_threshold(
    y_true: Iterable[int], probability: Iterable[float], *, rng: np.random.Generator,
    bootstrap_replicates: int = 100,
) -> tuple[float, np.ndarray]:
    y, p = _validate(y_true, probability)
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    thresholds = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, len(y), size=len(y))
        thresholds[replicate] = choose_threshold(y[indices], p[indices]).threshold
    return float(np.median(thresholds)), thresholds
