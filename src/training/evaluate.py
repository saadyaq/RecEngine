import math


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    if k == 0:
        return 0.0
    rec_at_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in rec_at_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    rec_at_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in rec_at_k if item in relevant_set)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because positions are 1-indexed

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(recommended: list, relevant: list) -> float:
    relevant_set = set(relevant)
    for i, item in enumerate(recommended):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0
