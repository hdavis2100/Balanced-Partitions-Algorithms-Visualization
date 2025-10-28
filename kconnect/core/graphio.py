from __future__ import annotations
from typing import Tuple, Dict, List
import re
import networkx as nx
import yaml

ROOT = "__ROOT__"

def extreme_tree_k2(G: nx.Graph, u: str, v: str) -> nx.Graph:
    if u == v: raise ValueError("Anchors u and v must be distinct")
    if v not in G or u not in G: raise ValueError("Anchors must be in G")
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    seen = {u}
    parent = {u: None}
    q = [u]
    while q:
        a = q.pop(0)
        for b in G.neighbors(a):
            if b == v: continue
            if b not in seen:
                seen.add(b); parent[b] = a; q.append(b)
                T.add_edge(a, b)
    T.add_edge(ROOT, u); T.add_edge(ROOT, v)
    return T

def _name_nodes_like(prefix: str, nodes):
    mapping = {}
    for i, n in enumerate(nodes):
        mapping[n] = f"{prefix}{i}"
    return mapping

# ---------- k=2 (biconnected) ----------

def _k2_cycle(n: int):
    G = nx.cycle_graph(n)
    mapping = {i: f"R{i}" for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    u = "R0"; v = f"R{n//2}"
    return G, (u, v)

def _k2_ladder(n: int):
    G = nx.ladder_graph(n)
    mapping = _name_nodes_like("L", G.nodes())
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())
    u = nodes[0]; v = nodes[-1]
    return G, (u, v)

def _k2_circ_ladder(n: int):
    G = nx.circular_ladder_graph(n)
    mapping = _name_nodes_like("CL", G.nodes())
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())
    u = nodes[0]; v = nodes[len(nodes)//2]
    return G, (u, v)

def _k2_torus(m: int, n: int):
    # Manual torus (m x n) with wrap-around; labels T{i}_{j}
    G = nx.Graph()
    for i in range(m):
        for j in range(n):
            G.add_node(f"T{i}_{j}")
    for i in range(m):
        for j in range(n):
            a = f"T{i}_{j}"
            b = f"T{(i+1)%m}_{j}"
            c = f"T{i}_{(j+1)%n}"
            G.add_edge(a, b); G.add_edge(a, c)
    u = "T0_0"; v = f"T{m//2}_{n//2}"
    return G, (u, v)

def _k2_hubring(n: int):
    R = [f"R{i}" for i in range(n)]
    G = nx.Graph(); G.add_nodes_from(R + ["X", "V"])
    for i in range(n): G.add_edge(R[i], R[(i+1)%n])
    for s in [2, n//2, n-3]: G.add_edge("X", R[s])
    u = R[0]; v = "V"
    G.add_edge(v, u); G.add_edge(v, "X"); G.add_edge(u, "X")
    return G, (u, v)

# ---------- k=3 (node-connectivity >= 3) ----------

def _k3_complete(n: int):
    if n < 4: raise ValueError("completeN requires N>=4")
    G = nx.complete_graph(n)
    mapping = {i: f"K{n}_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())
    return G, (nodes[0], nodes[1], nodes[2])

def _k3_wheel(n: int):
    G = nx.wheel_graph(n)
    mapping = {i: ("Hub0" if i==0 else f"C{i}") for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    rim = [x for x in G.nodes() if x.startswith("C")]
    rim.sort(key=lambda s: int(s[1:]))
    center = "Hub0"; a=rim[0]; b=rim[len(rim)//2]
    return G, (center, a, b)

def _k3_prism(n: int):
    if n < 3: raise ValueError("prismN requires N>=3")
    G = nx.Graph()
    top = [f"T{i}" for i in range(n)]
    bot = [f"B{i}" for i in range(n)]
    G.add_nodes_from(top + bot)
    for i in range(n):
        G.add_edge(top[i], top[(i+1)%n])
        G.add_edge(bot[i], bot[(i+1)%n])
        G.add_edge(top[i], bot[i])
    a0 = top[0]; a1 = bot[(n//3)%n]; a2 = top[(2*n)//3 % n]
    return G, (a0, a1, a2)

def _k3_hypercube(d: int):
    if d < 3: raise ValueError("hypercube_dN requires N>=3")
    G = nx.hypercube_graph(d)
    mapping = {tuple(bits): "".join(map(str,bits)) for bits in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    nodes = sorted(G.nodes())
    return G, (nodes[0], nodes[len(nodes)//3], nodes[(2*len(nodes))//3])

def _k3_complete_bipartite(m: int, n: int):
    if min(m,n) < 3: raise ValueError("Kmxn requires m,n >= 3")
    G = nx.complete_bipartite_graph(m, n)
    L = [f"A{i}" for i in range(m)]; R = [f"B{j}" for j in range(n)]
    H = nx.Graph(); H.add_nodes_from(L+R)
    for i in range(m):
        for j in range(n):
            H.add_edge(L[i], R[j])
    return H, (L[0], R[0], L[-1])



def _k3_octahedron():
    G = nx.octahedral_graph()
    mapping = _name_nodes_like("O", G.nodes())
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())
    return G, (nodes[0], nodes[2], nodes[4])

def _k3_icosahedron():
    G = nx.icosahedral_graph()
    mapping = _name_nodes_like("I", G.nodes())
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())
    return G, (nodes[0], nodes[5], nodes[9])

def _k3_dodecahedron():
    G = nx.dodecahedral_graph()
    mapping = _name_nodes_like("D", G.nodes())
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())
    return G, (nodes[0], nodes[6], nodes[12])

def _k3_cubical():
    G = nx.cubical_graph()
    mapping = _name_nodes_like("Q3_", G.nodes())
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())
    return G, (nodes[0], nodes[3], nodes[6])


def load_graph_from_yaml(path: str) -> Tuple[nx.Graph, Dict[str, float], Tuple[str, ...]]:
    with open(path, "r") as f: data = yaml.safe_load(f)
    G = nx.Graph(); weights: Dict[str, float] = {}
    for n in data.get("nodes", []):
        nid = str(n["id"]); G.add_node(nid); weights[nid] = float(n.get("weight", 0.0))
    for a,b in data.get("edges", []): G.add_edge(str(a), str(b))
    anchors = tuple(map(str, data.get("anchors", [])))
    if len(anchors) < 2: raise ValueError("YAML must include 'anchors' with at least 2 nodes")
    for a in anchors:
        if a not in G: raise ValueError(f"anchor '{a}' not found in nodes")
    return G, weights, anchors

def catalog_graph(k: int, catalog: str, graph_id: str):
    if catalog not in ("k2","k3"):
        raise ValueError("catalog must be 'k2' or 'k3'")
    if catalog == "k2":
        if m := re.fullmatch(r"cycle(\d+)", graph_id):
            G, anchors = _k2_cycle(int(m.group(1)))
        elif m := re.fullmatch(r"ladder(\d+)", graph_id):
            G, anchors = _k2_ladder(int(m.group(1)))
        elif m := re.fullmatch(r"circ_ladder(\d+)", graph_id):
            G, anchors = _k2_circ_ladder(int(m.group(1)))
        elif m := re.fullmatch(r"torus(\d+)x(\d+)", graph_id):
            G, anchors = _k2_torus(int(m.group(1)), int(m.group(2)))
        elif m := re.fullmatch(r"hubring(\d+)", graph_id):
            G, anchors = _k2_hubring(int(m.group(1)))
        else:
            raise ValueError("Unknown k2 graph_id. Try: cycle10, ladder6, circ_ladder8, hubring12, torus4x5")
        if not nx.is_biconnected(G):
            raise ValueError(f"Requested graph_id='{graph_id}' is NOT 2-connected")
    else:
        if graph_id == "K4":
            G, anchors = _k3_complete(4)
        elif m := re.fullmatch(r"complete(\d+)", graph_id):
            G, anchors = _k3_complete(int(m.group(1)))
        elif m := re.fullmatch(r"wheel(\d+)", graph_id):
            G, anchors = _k3_wheel(int(m.group(1)))
        elif m := re.fullmatch(r"prism(\d+)", graph_id):
            G, anchors = _k3_prism(int(m.group(1)))
        elif graph_id == "tri_prism":
            G, anchors = _k3_prism(3)
        elif m := re.fullmatch(r"hypercube_d(\d+)", graph_id):
            G, anchors = _k3_hypercube(int(m.group(1)))
        elif m := re.fullmatch(r"K(\d+)x(\d+)", graph_id):
            G, anchors = _k3_complete_bipartite(int(m.group(1)), int(m.group(2)))
        elif graph_id == "octahedron":
            G, anchors = _k3_octahedron()
        elif graph_id == "icosahedron":
            G, anchors = _k3_icosahedron()
        elif graph_id == "dodecahedron":
            G, anchors = _k3_dodecahedron()
        elif graph_id == "cubical":
            G, anchors = _k3_cubical()
        else:
            raise ValueError("Unknown k3 graph_id. Try: K4, completeN, wheelN, prismN, hypercube_dN, Kmxn, gp(n,k), octahedron, icosahedron, dodecahedron, cubical, theta343")
        if nx.node_connectivity(G) < 3:
            raise ValueError(f"Requested graph_id='{graph_id}' is NOT 3-connected")
    weights: Dict[str,float] = {}
    return G, weights, anchors
