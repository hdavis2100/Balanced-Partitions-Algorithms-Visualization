from __future__ import annotations
from typing import Dict, Set, Tuple, List, Optional, Any
import os, shutil
import networkx as nx
import yaml
from .graphio import extreme_tree_k2
from ..viz import save_partition_png

class K2Result:
    def __init__(self, parts, bound, phi_val, tree, iterations, dump_dir: Optional[str] = None,
                 timeline_yaml: Optional[str] = None, best_iter: Optional[int] = None,
                 best_L1: Optional[float] = None, best_SU: Optional[float] = None,
                 best_snapshot_yaml: Optional[str] = None, best_snapshot_png: Optional[str] = None):
        self.parts = parts
        self.bound = bound
        self.phi_val = phi_val
        self.tree = tree
        self.iterations = iterations
        self.dump_dir = dump_dir
        self.timeline_yaml = timeline_yaml
        self.best_iter = best_iter
        self.best_L1 = best_L1
        self.best_SU = best_SU
        self.best_snapshot_yaml = best_snapshot_yaml
        self.best_snapshot_png = best_snapshot_png

# Store parent and children relationships in U tree
def _parent_children_U(T: nx.Graph, U: Set[str], u: str):
    parent: Dict[str, Optional[str]] = {u: None}
    children: Dict[str, List[str]] = {u: []}
    stack = [u]
    Tu = T.subgraph(U)
    while stack:
        a = stack.pop()
        for b in Tu.neighbors(a):
            if parent.get(a) == b:
                continue
            if b in parent:
                continue
            parent[b] = a
            children.setdefault(a, []).append(b)
            children.setdefault(b, [])
            stack.append(b)
    return parent, children

# Collect subtree under 'root' to be reintegrated in U tree
def _subtree_nodes(children: Dict[str, List[str]], root: str) -> Set[str]:
    S: Set[str] = set()
    st = [root]
    while st:
        z = st.pop()
        if z in S:
            continue
        S.add(z)
        st.extend(children.get(z, []))
    return S

# Get children of x in U tree
def _u_children_of(T: nx.Graph, U: Set[str], x: str, parent: Dict[str, Optional[str]]) -> List[str]:
    nbrsU = [y for y in T.neighbors(x) if y in U]
    px = parent.get(x)
    return [y for y in nbrsU if y != px]

class _DSU:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, rb, False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra, rb, True

# Merge child branches of frontier vertex and reattach within U tree
def _expose_x_union_components(G: nx.Graph, T: nx.Graph, U: Set[str], u: str, x: str, trace: bool = False, debug: bool = False) -> Tuple[int, int, Dict[str, Any]]:
    
    # Collect children of x in U tree
    parent, children = _parent_children_U(T, U, u)
    child_list = _u_children_of(T, U, x, parent)
    report: Dict[str, Any] = {"x": x, "child_list_before": list(child_list), "phase1_events": [], "phase2_events": [], "Sx": [], "components": []}
    if not child_list:
        return 0, 0, report

    # Collect the subtree of each child of x
    owner: Dict[str, str] = {}
    nodes_per_child: Dict[str, Set[str]] = {}
    Sx: Set[str] = set()

    for c in child_list:
        Sc = _subtree_nodes(children, c)
        nodes_per_child[c] = Sc 
        Sx.update(Sc)
        for z in Sc:
            owner[z] = c
    report["Sx"] = sorted(Sx)

    # Use DSU to merge components within Sx
    dsu_build = _DSU(child_list)
    unions: List[Tuple[str, str, str, str]] = []
    for a, b in G.edges():
        if a in Sx and b in Sx:
            ca, cb = owner[a], owner[b]
            if ca == cb: continue
            ra, rb = dsu_build.find(ca), dsu_build.find(cb)
            if ra != rb:
                unions.append((a, b, ca, cb))
                dsu_build.union(ra, rb)
    # Apply unions to T, removing child edges from x as needed
    dsu_apply = _DSU(child_list)
    rep_child: Dict[str, str] = {c: c for c in child_list}
    phase1 = 0
    for p, q, ca, cb in unions:
        ra, rb = dsu_apply.find(ca), dsu_apply.find(cb)
        if ra == rb: continue
        T.add_edge(p, q)
        c_loser = rep_child[rb]
        removed = None
        if T.has_edge(x, c_loser):
            T.remove_edge(x, c_loser); removed = c_loser
        else:
            c_winner = rep_child[ra]
            if T.has_edge(x, c_winner):
                T.remove_edge(x, c_winner); removed = c_winner
        if removed is not None:
            phase1 += 1
            report["phase1_events"].append({"add_internal": [p, q], "removed_child_edge": [x, removed]})
            if trace: print(f"[worst:trace] ADD_INT ({p},{q})  REM_CHILD ({x},{removed})")
        new_rep, old_rep, _ = dsu_apply.union(ra, rb)
        if new_rep == ra: rep_child[new_rep] = rep_child[ra]
        else:             rep_child[new_rep] = rep_child[rb]

    comp_nodes: Dict[str, Set[str]] = {}
    for c in child_list:
        r = dsu_apply.find(c)
        comp_nodes.setdefault(r, set()).update(nodes_per_child[c])

    parent, _ = _parent_children_U(T, U, u)
    remaining_children = [nbr for nbr in T.neighbors(x) if nbr in U and nbr != parent.get(x)]
    rep_to_child_edge: Dict[str, str] = {}
    for c in remaining_children:
        rep_to_child_edge[dsu_apply.find(c)] = c

    
    U_out = (U - Sx) - {x}
    phase2 = 0

    # Connect each component to U outside Sx
    for r, nodes in comp_nodes.items():
        s = a = None
        for s_candidate in nodes:
            for a_candidate in G.neighbors(s_candidate):
                if a_candidate in U_out:
                    s, a = s_candidate, a_candidate; break
            if s is not None: break
        
        T.add_edge(s, a)
        c_edge = rep_to_child_edge.get(r)
        if c_edge and T.has_edge(x, c_edge):
            T.remove_edge(x, c_edge); phase2 += 1
            report["phase2_events"].append({"add_external": [s, a], "removed_child_edge": [x, c_edge], "component_nodes": sorted(nodes)})
            if trace: print(f"[worst:trace] ADD_OUT ({s},{a})  REM_CHILD ({x},{c_edge})")

    for r, nodes in comp_nodes.items():
        report["components"].append({"nodes": sorted(nodes)})
    return phase1, phase2, report

def balanced_partition_worst(
    G: nx.Graph, p: Dict[str, float], u: str, v: str, debug: bool = False,
    debug_check: bool = False, trace: bool = False, max_iters: Optional[int] = None,
    dump_yaml_dir: Optional[str] = None, dump_plots: bool = False
) -> K2Result:
    if dump_yaml_dir:
        os.makedirs(dump_yaml_dir, exist_ok=True)
    
    T = extreme_tree_k2(G, u, v)
    U: Set[str] = set(n for n in G.nodes() if n != v)
    Vset: Set[str] = {v}

    def sums() -> Tuple[float, float]:
        su = float(sum(p.get(x, 0.0) for x in U))
        sv = float(sum(p.get(x, 0.0) for x in Vset))
        return su, sv

    M = max((abs(float(w)) for w in p.values()), default=0.0)
    frame = 0
    iters = 0
    timeline_entries: List[Dict[str, Any]] = []

    su0, sv0 = sums()
    init_L1 = abs(su0) + abs(sv0)
    init_yaml = init_png = None
    if dump_yaml_dir:
        init = {
            "frame": frame, "initial": True,
            "U": sorted(U), "V": sorted(Vset),
            "U_components": [sorted(list(c)) for c in nx.connected_components(G.subgraph(U))] if len(U) else [],
            "V_components": [sorted(list(c)) for c in nx.connected_components(G.subgraph(Vset))] if len(Vset) else [],
            "sum_U": float(su0), "sum_V": float(sv0), "L1": float(init_L1), "M": float(M),
        }
        init_yaml = os.path.join(dump_yaml_dir, f"iter_{frame:03d}.yaml")
        with open(init_yaml, "w") as f:
            yaml.safe_dump(init, f, sort_keys=False)
        if dump_plots:
            init_png = os.path.join(dump_yaml_dir, f"iter_{frame:03d}.png")
            save_partition_png(G, U, Vset, p, (u, v), init_png,
                               title=f"iter {frame:03d}: initial  L1={init_L1:.3f} (M={M:.3f})",
                               tree=T, draw_tree_only=True,
                               highlight_subtree=None, highlight_edges=None)

    best_L1 = init_L1
    best_iter = frame
    best_U = set(U); best_V = set(Vset)
    best_yaml = init_yaml; best_png = init_png

    cap = len(G.nodes()) - 1 if max_iters is None else max_iters

    while iters < cap:
        parent, children = _parent_children_U(T, U, u)
        depth: Dict[str, int] = {u: 0}
        st = [u]
        while st:
            a = st.pop()
            for b in children.get(a, []):
                depth[b] = depth[a] + 1; st.append(b)

        F: List[str] = [x for x in U if any((y in Vset) for y in G.neighbors(x))]
        if not F:
            if debug: print("[worst] frontier empty; stopping" )
            break

        x = max((z for z in F if z != u), key=lambda z: depth.get(z, -1), default=None)
        if x is None:
            if debug: print("[worst] only frontier candidate is u; stopping")
            break

        phase1, phase2, report = _expose_x_union_components(G, T, U, u, x, trace=trace, debug=debug)

        y = next((y for y in G.neighbors(x) if y in Vset), None)
        if y is None:
            iters += 1; frame += 1; continue

        parent, _ = _parent_children_U(T, U, u)
        px = parent.get(x)
        if px is not None and T.has_edge(x, px):
            T.remove_edge(x, px)
        T.add_edge(x, y)

        U.remove(x); Vset.add(x)
        iters += 1; frame += 1

        
        if debug_check:
            Gu = G.subgraph(U); Gv = G.subgraph(Vset)
            if len(U) > 0 and not nx.is_connected(Gu):
                raise RuntimeError(f"[worst] G[U] disconnected at frame {frame} after moving {x}")
            if len(Vset) > 0 and not nx.is_connected(Gv):
                raise RuntimeError(f"[worst] G[V] disconnected at frame {frame} after moving {x}")
            Tu = T.subgraph(U); Tv = T.subgraph(Vset)
            if len(U) > 0 and not nx.is_connected(Tu):
                raise RuntimeError(f"[worst] T[U] disconnected at frame {frame} after moving {x} (tree maintenance bug)")
            if len(Vset) > 0 and not nx.is_connected(Tv):
                raise RuntimeError(f"[worst] T[V] disconnected at frame {frame} after moving {x} (tree maintenance bug)")

        su, sv = sums(); L1 = abs(su) + abs(sv)

        U_comps = [sorted(list(c)) for c in nx.connected_components(G.subgraph(U))] if len(U) else []
        V_comps = [sorted(list(c)) for c in nx.connected_components(G.subgraph(Vset))] if len(Vset) else []

        
        internal_edges = [(ev["add_internal"][0], ev["add_internal"][1])
                          for ev in report.get("phase1_events", []) if "add_internal" in ev]
        external_edges = [(ev["add_external"][0], ev["add_external"][1])
                          for ev in report.get("phase2_events", []) if "add_external" in ev]
        attach_edges = internal_edges + external_edges + [(x, y)]
        seen = set()
        attach_edges = [(a,b) for (a,b) in attach_edges if (a,b) not in seen and not seen.add((a,b))]

        snap_path = None
        if dump_yaml_dir:
            snap = {
                "frame": frame,
                "iter": iters,
                "move": {"vertex": x, "attached_to_v": y, "depth": int(depth.get(x, -1))},
                "phase1": report.get("phase1_events", []),
                "phase2": report.get("phase2_events", []),
                "Sx": report.get("Sx", []),
                "components": report.get("components", []),
                "U": sorted(U), "V": sorted(Vset),
                "U_components": U_comps, "V_components": V_comps,
                "sum_U": float(su), "sum_V": float(sv),
                "L1": float(L1), "M": float(M),
            }
            snap_path = os.path.join(dump_yaml_dir, f"iter_{frame:03d}.yaml")
            with open(snap_path, "w") as f:
                yaml.safe_dump(snap, f, sort_keys=False)

        png_path = None
        if dump_yaml_dir and dump_plots:
            png_path = os.path.join(dump_yaml_dir, f"iter_{frame:03d}.png")
            save_partition_png(
                G, U, Vset, p, (u, v), png_path,
                highlight=x,
                title=f"iter {frame:03d}: moved {x}  L1={L1:.3f} (M={M:.3f})",
                tree=T, draw_tree_only=True,
                highlight_subtree=set(report.get("Sx", [])),
                highlight_edges=attach_edges,
                moved_node_color="red", subtree_color="gold", attach_edge_color="tab:blue"
            )

        if L1 < best_L1 - 1e-15:
            best_L1 = L1
            best_iter = frame
            best_U = set(U); best_V = set(Vset)
            best_yaml = snap_path
            best_png = png_path

        timeline_entries.append({
            "frame": frame,
            "iter": iters,
            "moved": x,
            "attached_to": y,
            "sizes": {"U": len(U), "V": len(Vset)},
            "sums": {"U": float(su), "V": float(sv)},
            "L1": float(L1),
            "M": float(M),
            "snapshot_yaml": os.path.basename(snap_path) if snap_path else None,
            "snapshot_png": os.path.basename(png_path) if png_path else None,
            "U_components_count": len(U_comps),
            "V_components_count": len(V_comps)
        })

    best_yaml_out = None
    best_png_out = None
    if dump_yaml_dir:
        if best_yaml and os.path.exists(best_yaml):
            best_yaml_out = os.path.join(dump_yaml_dir, "best.yaml")
            shutil.copyfile(best_yaml, best_yaml_out)
        if best_png and os.path.exists(best_png):
            best_png_out = os.path.join(dump_yaml_dir, "best.png")
            shutil.copyfile(best_png, best_png_out)

    timeline_yaml_path = None
    if dump_yaml_dir:
        final_su = float(sum(p.get(x, 0.0) for x in U))
        final_sv = float(sum(p.get(x, 0.0) for x in Vset))
        index = {
            "anchors": {"u": u, "v": v},
            "initial": {"frame": 0, "S_U": float(su0), "S_V": float(sv0), "L1": float(init_L1), "M": float(M)},
            "best": {
                "frame": int(best_iter),
                "sizes": {"U": len(best_U), "V": len(best_V)},
                "sums": {"U": float(sum(p.get(x,0.0) for x in best_U)), "V": float(sum(p.get(x,0.0) for x in best_V))},
                "L1": float(best_L1),
                "snapshot_yaml": os.path.basename(best_yaml_out) if best_yaml_out else None,
                "snapshot_png": os.path.basename(best_png_out) if best_png_out else None,
            },
            "final_after_sweep": {"frame": frame, "S_U": final_su, "S_V": final_sv, "L1": abs(final_su)+abs(final_sv)},
            "iterations": timeline_entries,
        }
        timeline_yaml_path = os.path.join(dump_yaml_dir, "timeline.yaml")
        with open(timeline_yaml_path, "w") as f:
            yaml.safe_dump(index, f, sort_keys=False)

    parts = {u: best_U, v: best_V}
    return K2Result(parts=parts, bound=M, phi_val=float(sum(p.get(x,0.0) for x in best_U)), tree=T,
                    iterations=iters, dump_dir=dump_yaml_dir, timeline_yaml=timeline_yaml_path,
                    best_iter=best_iter, best_L1=best_L1, best_SU=float(sum(p.get(x,0.0) for x in best_U)),
                    best_snapshot_yaml=best_yaml_out, best_snapshot_png=best_png_out)
