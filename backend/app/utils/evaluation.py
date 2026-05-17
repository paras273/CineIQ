from typing import Iterable, List, Sequence

import numpy as np


def precision_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    predicted_k = predicted[:k]
    if not predicted_k:
        return 0.0
    hits = len(set(actual) & set(predicted_k))
    return hits / float(k)


def recall_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    hits = len(set(actual) & set(predicted_k))
    return hits / float(len(actual))


def average_precision_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    if not actual:
        return 0.0
    score = 0.0
    hits = 0.0
    for i, p in enumerate(predicted[:k], start=1):
        if p in actual:
            hits += 1
            score += hits / i
    return score / min(len(actual), k)


def map_at_k(actual_list: Iterable[Sequence[int]], predicted_list: Iterable[Sequence[int]], k: int) -> float:
    scores = [average_precision_at_k(a, p, k) for a, p in zip(actual_list, predicted_list)]
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(predicted_k, start=1):
        if p in actual:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(actual), k) + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0
