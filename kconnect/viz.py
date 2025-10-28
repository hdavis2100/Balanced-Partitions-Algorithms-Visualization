from __future__ import annotations
from typing import Dict, Set, Tuple, Optional, Iterable, List
import networkx as nx
import matplotlib.pyplot as plt

ROOT = "__ROOT__"

def _sumw(nodes, w):
    return sum(w.get(n, 0.0) for n in nodes)

def plot_partition(G: nx.Graph, U: Set[str], Vset: Set[str], weights: Dict[str, float], anchors: Tuple[str,str]):
    u, v = anchors
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axU, axV = axes[0], axes[1]

    Gu = G.subgraph(U).copy()
    Gv = G.subgraph(Vset).copy()

    posU = nx.spring_layout(Gu, seed=2)
    posV = nx.spring_layout(Gv, seed=3)

    nx.draw_networkx_edges(Gu, posU, ax=axU, alpha=0.6)
    nx.draw_networkx_nodes(Gu, posU, ax=axU, node_size=600)
    labU = {n: f"{n}\n{weights.get(n,0):.2f}" + ("\n(ANCHOR)" if n == u else "") for n in Gu.nodes()}
    nx.draw_networkx_labels(Gu, posU, labels=labU, ax=axU, font_size=9)
    axU.set_title(f"U (|U|={len(U)}, sum={_sumw(U, weights):.3f})")
    axU.axis('off')

    nx.draw_networkx_edges(Gv, posV, ax=axV, alpha=0.6)
    nx.draw_networkx_nodes(Gv, posV, ax=axV, node_size=600)
    labV = {n: f"{n}\n{weights.get(n,0):.2f}" + ("\n(ANCHOR)" if n == v else "") for n in Gv.nodes()}
    nx.draw_networkx_labels(Gv, posV, labels=labV, ax=axV, font_size=9)
    axV.set_title(f"V (|V|={len(Vset)}, sum={_sumw(Vset, weights):.3f})")
    axV.axis('off')

    fig.suptitle("k=2 partition (L1 bound): separated connected components", y=0.98)
    plt.tight_layout()
    plt.show()

def save_partition_png(
    G: nx.Graph,
    U: Set[str],
    Vset: Set[str],
    weights: Dict[str, float],
    anchors: Tuple[str, str],
    out_path: str,
    highlight: Optional[str] = None,
    title: Optional[str] = None,
    tree: Optional[nx.Graph] = None,
    draw_tree_only: bool = False,
    highlight_subtree: Optional[Set[str]] = None,
    highlight_edges: Optional[List[Tuple[str,str]]] = None,
    moved_node_color: str = "red",
    subtree_color: str = "gold",
    attach_edge_color: str = "tab:blue"
):
    u, v = anchors
    Gu = G.subgraph(U).copy()
    Gv = G.subgraph(Vset).copy()

    posU = nx.spring_layout(Gu, seed=2)
    posV = nx.spring_layout(Gv, seed=3)

    fig, (axU, axV) = plt.subplots(1, 2, figsize=(12, 6))

    # U panel
    if draw_tree_only and tree is not None:
        eU = [(a,b) for (a,b) in tree.edges() if a in U and b in U and a != ROOT and b != ROOT]
        nx.draw_networkx_edges(Gu, posU, edgelist=eU, ax=axU, alpha=0.9)
    else:
        nx.draw_networkx_edges(Gu, posU, ax=axU, alpha=0.6)
    if highlight_edges:
        eU_h = [(a,b) for (a,b) in highlight_edges if a in U and b in U]
        if eU_h:
            nx.draw_networkx_edges(Gu, posU, edgelist=eU_h, ax=axU, width=2.8, edge_color=attach_edge_color)
    sizesU = []
    colorsU = []
    for n in Gu.nodes():
        sizesU.append(900 if n == highlight else 600)
        if n == u:
            colorsU.append("lightgray")
        elif highlight_subtree and n in highlight_subtree:
            colorsU.append(subtree_color)
        else:
            colorsU.append("lightgray")
    nx.draw_networkx_nodes(Gu, posU, ax=axU, node_size=sizesU, node_color=colorsU)
    labU = {n: f"{n}\n{weights.get(n,0):.2f}" + ("\n(ANCHOR)" if n == u else "") for n in Gu.nodes()}
    nx.draw_networkx_labels(Gu, posU, labels=labU, ax=axU, font_size=9)
    axU.set_title(f"U (|U|={len(U)}, sum={_sumw(U, weights):.3f})")
    axU.axis('off')

    # V panel
    if draw_tree_only and tree is not None:
        eV = [(a,b) for (a,b) in tree.edges() if a in Vset and b in Vset and a != ROOT and b != ROOT]
        nx.draw_networkx_edges(Gv, posV, edgelist=eV, ax=axV, alpha=0.9)
    else:
        nx.draw_networkx_edges(Gv, posV, ax=axV, alpha=0.6)
    if highlight_edges:
        eV_h = [(a,b) for (a,b) in highlight_edges if a in Vset and b in Vset]
        if eV_h:
            nx.draw_networkx_edges(Gv, posV, edgelist=eV_h, ax=axV, width=2.8, edge_color=attach_edge_color)
    sizesV = []
    colorsV = []
    for n in Gv.nodes():
        sizesV.append(900 if n == highlight else 600)
        if n == v:
            colorsV.append("lightgray")
        elif n == highlight:
            colorsV.append(moved_node_color)
        else:
            colorsV.append("lightgray")
    nx.draw_networkx_nodes(Gv, posV, ax=axV, node_size=sizesV, node_color=colorsV)
    labV = {n: f"{n}\n{weights.get(n,0):.2f}" + ("\n(ANCHOR)" if n == v else "") for n in Gv.nodes()}
    nx.draw_networkx_labels(Gv, posV, labels=labV, ax=axV, font_size=9)
    axV.set_title(f"V (|V|={len(Vset)}, sum={_sumw(Vset, weights):.3f})")
    axV.axis('off')

    if title:
        fig.suptitle(title, y=0.98)
    else:
        fig.suptitle("k=2 partition snapshot", y=0.98)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


from typing import Optional, Tuple, Set, Dict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def save_partition_png_k3(
    G: nx.Graph,
    U: Set[str],
    Vset: Set[str],
    Wset: Set[str],
    weights: Dict[str, float],
    anchors: Tuple[str, str, str],
    out_path: str,
    highlight: Optional[str] = None,
    title: Optional[str] = None,
    draw_tree_only: bool = True,
    attach_edge: Optional[Tuple[str, str]] = None,
):
    """Strict forest-only renderer:
    - For each partition H, build F = spanning forest (BFS per component) and draw **only** F's edges.
    - Overlay a single purple attach-edge if both endpoints are in the same panel.
    """
    u, v, w = anchors

    GU = G.subgraph(U).copy()
    GV = G.subgraph(Vset).copy()
    GW = G.subgraph(Wset).copy()

    # Independent layouts per panel
    posU = nx.spring_layout(GU, seed=2)
    posV = nx.spring_layout(GV, seed=3)
    posW = nx.spring_layout(GW, seed=4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axU, axV, axW = axes

    def component_forest_edges(H: nx.Graph, anchor: str):
        """Return a BFS-tree edge list per connected component of H (forest)."""
        forest = []
        for comp in nx.connected_components(H):
            comp_nodes = set(comp)
            root = anchor if anchor in comp_nodes else next(iter(comp_nodes))
            visited = {root}
            q = [root]
            while q:
                a = q.pop(0)
                for b in H.neighbors(a):
                    if b in comp_nodes and b not in visited:
                        visited.add(b)
                        q.append(b)
                        forest.append((a, b))
        return forest

    def draw_panel(H, pos, ax, anchor, part_nodes, label):
        # Build explicit forest graph F with only the tree edges
        F = nx.Graph()
        F.add_nodes_from(H.nodes())
        if H.number_of_nodes() > 0:
            edges = component_forest_edges(H, anchor)
            F.add_edges_from(edges)
            ecoll = nx.draw_networkx_edges(F, pos, edgelist=list(F.edges()), ax=ax, alpha=0.9)
            if ecoll is not None:
                try:
                    ecoll.set_zorder(1)
                except Exception:
                    pass

        # Nodes (on top of edges)
        sizes = []
        colors = []
        for n in H.nodes():
            sizes.append(900 if n == highlight else 600)
            if n == anchor:
                colors.append("lightgray")
            elif n == highlight:
                colors.append("red")
            else:
                colors.append("lightgray")
        ncoll = nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes, node_color=colors)
        if ncoll is not None:
            try:
                ncoll.set_zorder(2)
            except Exception:
                pass

        # Labels (on top of nodes)
        labels = {}
        for n in H.nodes():
            lbl = f"{n}\n{weights.get(n, 0):.2f}"
            if n == anchor:
                lbl += "\n(ANCHOR)"
            labels[n] = lbl
        texts = nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, font_size=9)
        if isinstance(texts, dict):
            for t in texts.values():
                try:
                    t.set_zorder(3)
                except Exception:
                    pass

        ax.set_title(
            f"{label} (|{label}|={len(part_nodes)}, sum={sum(weights.get(x, 0.0) for x in part_nodes):.3f})"
        )
        ax.axis("off")

    draw_panel(GU, posU, axU, u, U, "U")
    draw_panel(GV, posV, axV, v, Vset, "V")
    draw_panel(GW, posW, axW, w, Wset, "W")

    # Overlay the attach edge (purple) only if both endpoints are in the same panel
    if attach_edge:
        a, b = attach_edge
        for H, pos, ax in ((GU, posU, axU), (GV, posV, axV), (GW, posW, axW)):
            if a in H and b in H:
                one = nx.Graph()
                one.add_edge(a, b)
                ec2 = nx.draw_networkx_edges(one, pos, edgelist=[(a, b)], ax=ax, width=3.0, edge_color="tab:purple")
                if ec2 is not None:
                    try:
                        ec2.set_zorder(4)
                    except Exception:
                        pass

    if title:
        fig.suptitle(title, y=0.98)
    handles = [
        mpatches.Patch(color="red", label="moved node (this frame)"),
        mlines.Line2D([], [], color="tab:purple", lw=3, label="attach edge to destination"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)