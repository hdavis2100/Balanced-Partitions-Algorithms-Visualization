from __future__ import annotations
from typing import Dict, List
import random
import networkx as nx

def random_zero_sum_with_anchor_signs(G: nx.Graph, anchors: List[str], *, seed: int | None = None,
                                      anchor_sign: str = "negative") -> Dict[str, float]:
    rng = random.Random(seed)
    w = {n: rng.gauss(0.0, 1.0) for n in G.nodes()}
    mu = sum(w.values()) / G.number_of_nodes()
    for n in w: w[n] -= mu
    if anchor_sign not in ("negative", "positive"):
        raise ValueError("anchor_sign must be 'negative' or 'positive'")
    def force(x: float) -> float:
        return (-abs(x) if anchor_sign == "negative" else abs(x)) or ((-1.0) if anchor_sign=="negative" else 1.0)
    for a in anchors: w[a] = force(w[a])
    total = sum(w.values())
    others = [n for n in G.nodes() if n not in anchors]
    if others:
        delta = total / len(others)
        for n in others: w[n] -= delta
    drift = sum(w.values())
    if abs(drift) > 1e-10 and others:
        w[others[0]] -= drift
    return w

def default_anchor_equal_zero_sum(G: nx.Graph, anchors: List[str], *, anchor_sign: str = "negative",
                                  anchor_value: float = 1.0, seed: int | None = None) -> Dict[str, float]:
    if anchor_sign not in ("negative", "positive"):
        raise ValueError("anchor_sign must be 'negative' or 'positive'")
    sgn = -1.0 if anchor_sign == "negative" else +1.0
    w: Dict[str, float] = {}
    for a in anchors: w[a] = sgn * float(anchor_value)
    others = [n for n in G.nodes() if n not in anchors]
    if not others:
        for a in anchors: w[a] = 0.0
        return w
    rng = random.Random(seed)
    for x in others:
        w[x] = rng.gauss(0.0, 1.0)
    total = sum(w.values())
    delta = total / len(others)
    for x in others: w[x] -= delta
    drift = sum(w.values())
    if abs(drift) > 1e-10:
        w[others[0]] -= drift
    return w

def binary_pm_one_zero_sum_k(G: nx.Graph, anchors: List[str], *, seed: int | None = None,
                             anchor_sign: str = "negative") -> Dict[str, float]:
    n = G.number_of_nodes()
    k = len(anchors)
    if n % 2 != 0:
        raise ValueError("binary ±1 zero-sum requires an even number of nodes (|V| must be even).")
    if anchor_sign not in ("negative", "positive"):
        raise ValueError("anchor_sign must be 'negative' or 'positive'")
    if 2*k > n:
        raise ValueError(f"binary ±1 zero-sum with all {k} anchors forced {anchor_sign} is infeasible for |V|={n}. "
                         f"Requires 2*|anchors| <= |V|. Try larger graphs, 'random', or 'yaml/default'.")
    total_pos = n // 2; total_neg = n // 2
    w: Dict[str, float] = {}
    if anchor_sign == "negative":
        for a in anchors: w[a] = -1.0
        pos_rem = total_pos; neg_rem = total_neg - k
    else:
        for a in anchors: w[a] = +1.0
        pos_rem = total_pos - k; neg_rem = total_neg
    others: List[str] = [x for x in G.nodes() if x not in anchors]
    if pos_rem < 0 or neg_rem < 0:
        raise ValueError("Internal guard: negative remaining slot count in binary assignment.")
    rng = random.Random(seed); rng.shuffle(others)
    for i, x in enumerate(others):
        if i < pos_rem: w[x] = +1.0
        else:           w[x] = -1.0
    s = sum(w.values())
    if abs(s) > 1e-9:
        raise RuntimeError(f"internal error: binary assignment not zero-sum (sum={s})")
    return w
