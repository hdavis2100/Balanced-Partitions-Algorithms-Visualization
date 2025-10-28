# kconnect — Balanced Connected Graph Partitioning (k=2,3)

Algorithms, visualizations, and a tiny web UI for exploring **balanced connected partitions** of weighted graphs with fixed **anchors**. The toolkit supports:

- **k=2** (2‑vertex‑connected graphs):  
  - `k2-sim` — an iterative “extreme‑tree” sweep that moves subtrees across the cut while *preserving connectivity* on both sides.  
  - `k2-st` — a fast baseline that selects the best L1 cut along a Tarjan **st‑numbering** ordering.
- **k=3** (3‑connected graphs):  
  - `k3-local` — frontier‑based 1‑opt local moves that keep all three parts connected.

Outputs include per‑iteration **YAML logs**, **PNG** snapshots, and an optional **animated GIF** timeline.

> **Objective (L1)**  
> **k=2:** minimize `|Σ_U w| + |Σ_V w|`  
> **k=3:** minimize `|Σ_U w| + |Σ_V w| + |Σ_W w|`  
> where `w` are node weights and `U,V,(W)` are connected parts that contain specified anchors.

---


## Quick start

### 1) Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

**Requires** Python 3.10+ (tested on 3.11). Dependencies are pinned in `requirements.txt`: `networkx`, `pyyaml`, `matplotlib`, `Pillow`, `flask`.

---

### 2) CLI usage

From `Balanced_Partitions/Balanced_Partitions/`:

```bash
# k=2 (iterative extreme‑tree sweep) on a 12‑cycle; dump frames + GIF
python -m kconnect.service.main \
  --algo k2-sim --catalog k2 --graph-id cycle12 \
  --weights default --anchor-sign negative \
  --viz --gif --outdir runs

# k=2 (st‑numbering cut) on a ladder
python -m kconnect.service.main \
  --algo k2-st --catalog k2 --graph-id ladder10 \
  --weights random --rand-seed 7 --viz --outdir runs

# k=3 (local 1‑opt moves) on an octahedron
python -m kconnect.service.main \
  --algo k3-local --catalog k3 --graph-id octahedron \
  --weights default --anchor-sign negative \
  --viz --gif --outdir runs
```

**Key flags**

- `--algo`: `k2-sim`, `k2-st`, or `k3-local`  
- Graph source — choose exactly one:
  - **Catalog**: `--catalog k2|k3 --graph-id <ID>`  
    - Examples  
      - k=2: `cycle12`, `ladder8`, `circ_ladder10`, `hubring12`, `torus4x5`  
      - k=3: `K4`, `complete12`, `wheel10`, `prism8`, `hypercube_d4`, `K3x8`, `octahedron`, `icosahedron`, `dodecahedron`, `cubical`
  - **YAML**: `--graph path/to/graph.yaml` (overrides catalog)  
- Weights: `--weights yaml|default|random|binary`  
  - `default`: anchors get fixed sign (±1); the rest are adjusted to make the total sum zero  
  - `random`: zero‑sum random weights with anchor sign enforced  
  - `binary`: ±1 assignment with zero sum (requires even |V| and ≥ k anchors)  
  - `yaml`: use weights from the YAML file (if provided)  
- Extras: `--viz` dumps `iter_###.{yaml,png}`; `--gif` also emits `timeline.gif`; `--debug` for verbose logs  
- Reproducibility: `--rand-seed` seeds weight generation; `--anchor-sign` controls anchor polarity

**Outputs** (under `runs/<algo>_<graph-id>/`):
- `iter_000.yaml/png, iter_001.yaml/png, …`
- `best.yaml` and `best.png` (best L1 snapshot)
- `timeline.yaml` (index of frames) and optional `timeline.gif`

---

### 3) Minimal web UI

```bash
# from Balanced_Partitions
python webui/app.py
# then open http://127.0.0.1:5000
```

Pick a catalog graph or upload a YAML, choose the algorithm and weights, and download `best.png` / `timeline.gif` from the results page. The UI also shows stdout/stderr logs from the run.

---

## YAML graph format

Provide nodes, edges, and anchors. Weights are optional unless `--weights yaml` is selected.

```yaml
# example.yaml
nodes:
  - {id: A, weight: -1.0}
  - {id: B, weight:  0.6}
  - {id: C, weight:  0.4}
  - {id: D, weight:  0.0}

edges:
  - [A, B]
  - [B, C]
  - [C, D]
  - [D, A]

anchors: [A, C]     # for k=2; use exactly three nodes for k=3 (e.g., [A, B, C])
```

> Anchors must be present in `nodes`. For `k=2`, algorithms use the first two anchors; for `k=3`, the first three.

---

## Python API examples

```python
# k=2 sweep (extreme‑tree)
import networkx as nx
from kconnect.core.graphio import catalog_graph
from kconnect.core.weights import default_anchor_equal_zero_sum
from kconnect.core.k2_sim import balanced_partition_worst

G, _, anchors = catalog_graph(k=2, catalog="k2", graph_id="cycle12")
w = default_anchor_equal_zero_sum(G, anchors=list(anchors), anchor_sign="negative", seed=7)
res = balanced_partition_worst(
    G, w, anchors[0], anchors[1],
    debug=False, debug_check=False, trace=False,
    dump_yaml_dir="runs/demo_k2", dump_plots=True
)
print("best L1:", res.best_L1, "iterations:", res.iterations, "bound:", res.bound)
```

```python
# k=3 local search
from kconnect.core.graphio import catalog_graph
from kconnect.core.weights import default_anchor_equal_zero_sum
from kconnect.core.k3_local import balanced_partition_k3_local

G, _, anchors = catalog_graph(k=3, catalog="k3", graph_id="octahedron")
w = default_anchor_equal_zero_sum(G, anchors=list(anchors), anchor_sign="negative", seed=7)
res = balanced_partition_k3_local(G, w, anchors, viz=True, dump_dir="runs/demo_k3", make_gif=True)
print("best L1:", res.best_L1, "best frame:", res.best_iter)
```

---

## Notes on the methods

- **k2‑sim (extreme‑tree sweep).** Constructs an “extreme” spanning tree anchored at `(u,v)` and repeatedly moves a *whole subtree* across the cut when it reduces L1, keeping both sides connected. Each move is logged; PNGs highlight the moved node, transfer subtree, and attach edge.
- **k2‑st (Tarjan st‑numbering).** Computes an st‑numbering for a biconnected graph and scans prefix/suffix cuts to pick the best L1 split. Useful as a fast baseline.
- **k3‑local.** Starts from a one‑branch state and performs connectivity‑preserving frontier moves among `U,V,W`, tracking and snapshotting the best L1.

---

## Troubleshooting

- **“Graph is not 2‑vertex‑connected” (k=2) or connectivity \< 3 (k=3)** — choose a different catalog graph or supply a YAML whose connectivity matches the algorithm.  
- **`binary` weights disabled in the UI** — requires even `|V|` and that `2 * anchors ≤ |V|` (k=2 uses two anchors; k=3 uses three).  
- **No `timeline.gif`** — include `--gif` or use `tools/make_gif.py` after producing PNG frames with `--viz`.

---

## License

TBD.

---


