from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import networkx as nx

class STNumberingError(Exception): pass

# Doubly-linked list helper for st-numbering insertion
class _LL:
    def __init__(self):
        self.prev: Dict[str, Optional[str]] = {}
        self.next: Dict[str, Optional[str]] = {}
        self.head: Optional[str] = None
        self.tail: Optional[str] = None
    def append(self, v: str) -> None:
        if self.head is None:
            self.head = self.tail = v
            self.prev[v] = None; self.next[v] = None
        else:
            t = self.tail
            self.next[t] = v
            self.prev[v] = t
            self.next[v] = None
            self.tail = v
    def insert_after(self, x: str, v: str) -> None:
        nx_ = self.next.get(x)
        self.next[x] = v
        self.prev[v] = x
        self.next[v] = nx_
        if nx_ is not None: self.prev[nx_] = v
        else: self.tail = v
    def insert_before(self, x: str, v: str) -> None:
        px = self.prev.get(x)
        self.prev[x] = v; self.next[v] = x; self.prev[v] = px
        if px is not None: self.next[px] = v
        else: self.head = v
    def to_list(self) -> List[str]:
        out: List[str] = []; cur = self.head
        while cur is not None: out.append(cur); cur = self.next.get(cur)
        return out

# st-numbering via Tarjan's algorithm
def st_numbering_tarjan(G: nx.Graph, s: str, t: str) -> List[str]:
    if s == t: raise STNumberingError("s and t must be distinct")
    if not nx.is_biconnected(G): raise STNumberingError("Graph must be biconnected")
    H = G.copy(); added = False
    if not H.has_edge(s,t): H.add_edge(s,t); added = True
    if not nx.is_biconnected(H): raise STNumberingError("Internal error: G+(s,t) not biconnected")
    pre: Dict[str,int] = {}; low: Dict[str,int] = {}; parent: Dict[str,Optional[str]] = {s: None}
    order: List[str] = []; tstamp=0

    # Ensure DFS starts at (s,t)
    neigh_s = list(H.neighbors(s))
    if t in neigh_s: neigh_s.remove(t); neigh_s=[t]+neigh_s
    stack = [(s, iter(neigh_s))]; pre[s]=tstamp; low[s]=tstamp; order.append(s); tstamp+=1

    # DFS traversal with lowlink computation
    while stack:
        v,it = stack[-1]
        try:
            w = next(it)
            if parent.get(v) == w: continue
            if w not in pre:
                parent[w]=v; pre[w]=tstamp; low[w]=tstamp; order.append(w); tstamp+=1
                stack.append((w, iter(H.neighbors(w))))
            else:
                if pre[w] < pre[v]:
                    low[v] = min(low[v], pre[w])
        except StopIteration:
            stack.pop(); pv = parent.get(v)
            if pv is not None:
                low[pv] = min(low[pv], low[v])
    inv_pre = {pre[v]: v for v in pre}
    lowvert = {v: inv_pre[low[v]] for v in pre}

    # Build st-numbering using lowlink info
    L = _LL(); L.append(s); L.append(t); sign = {s:-1, t:+1}
    for v in order:
        if v==s or v==t: continue
        a = lowvert[v]; p = parent[v]
        if p is None: continue
        if sign.get(a,-1) == +1: L.insert_after(p, v); sign[v] = -1
        else: L.insert_before(p, v); sign[v] = +1
    st = L.to_list()
    if st and st[0] != s and s in st: i = st.index(s); st = st[i:] + st[:i]
    if st and st[-1] != t: st = list(reversed(st)); 
    if st and st[0] != s and s in st: i = st.index(s); st = st[i:] + st[:i]
    if not st or st[0] != s or st[-1] != t or len(st) != H.number_of_nodes(): raise STNumberingError("Invalid st-order produced")
    if added: H.remove_edge(s,t)
    return st
