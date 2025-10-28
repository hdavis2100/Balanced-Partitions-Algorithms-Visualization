from __future__ import annotations
import argparse, os, sys
import networkx as nx
import yaml

from ..core.graphio import load_graph_from_yaml, catalog_graph
from ..core.k2_sim import balanced_partition_worst
from ..core.k2_st import st_numbering_tarjan, STNumberingError
from ..core.k3_local import balanced_partition_k3_local
from ..core.weights import random_zero_sum_with_anchor_signs, default_anchor_equal_zero_sum, binary_pm_one_zero_sum_k
from ..viz import save_partition_png, save_partition_png_k3

def _load_bundle(args):
    if args.graph:
        G, weights, anchors = load_graph_from_yaml(args.graph)
    else:
        k = 2 if args.algo in ("k2-sim","k2-st") else 3
        G, weights, anchors = catalog_graph(k, args.catalog, args.graph_id)
    # weights selection
    try:
        if args.weights == "yaml":
            if not weights:
                weights = default_anchor_equal_zero_sum(G, list(anchors), anchor_sign=args.anchor_sign, seed=args.rand_seed)
        elif args.weights == "default":
            weights = default_anchor_equal_zero_sum(G, list(anchors), anchor_sign=args.anchor_sign, seed=args.rand_seed)
        elif args.weights == "random":
            weights = random_zero_sum_with_anchor_signs(G, list(anchors), seed=args.rand_seed, anchor_sign=args.anchor_sign)
        elif args.weights == "binary":
            weights = binary_pm_one_zero_sum_k(G, list(anchors), seed=args.rand_seed, anchor_sign=args.anchor_sign)
        else:
            raise SystemExit("Unsupported --weights")
    except Exception as e:
        print(f"[weights] {e}", file=sys.stderr)
        raise
    return G, weights, anchors

def _ensure_dir(path): os.makedirs(path, exist_ok=True); return path

def _maybe_make_gif(dump_dir: str, do_gif: bool, fps: float = 2.0):
    if not do_gif or not dump_dir: return
    try:
        import glob
        from PIL import Image
        frames = sorted(glob.glob(os.path.join(dump_dir, "iter_*.png")))
        if not frames: return
        out_gif = os.path.join(dump_dir, "timeline.gif")
        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frames]
        duration_ms = int(1000.0 / max(fps, 0.1))
        imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], loop=0, duration=duration_ms, optimize=False, disposal=2)
        print(f"[gif] wrote {out_gif}")
    except Exception as e:
        print(f"[gif] skipped ({e})")

def main():
    ap = argparse.ArgumentParser("kconnect v21 (k3 forest; default weights; GUI tweaks)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--algo", choices=["k2-sim","k2-st","k3-local"], default="k3-local")
    ap.add_argument("--graph", type=str, default=None, help="YAML graph path; overrides catalog/graph-id")
    ap.add_argument("--catalog", type=str, default="k3", help="catalog namespace (k2|k3)")
    ap.add_argument("--graph-id", type=str, default="K4", help="graph identifier (also used as output label)")
    ap.add_argument("--weights", choices=["yaml","default","random","binary"], default="default")
    ap.add_argument("--rand-seed", type=int, default=7)
    ap.add_argument("--anchor-sign", choices=["negative","positive"], default="negative")
    ap.add_argument("--viz", action="store_true", help="dump per-iteration YAML+PNG (where applicable).")
    ap.add_argument("--gif", action="store_true", help="if --viz, also build a GIF; otherwise try if frames exist.")
    ap.add_argument("--debug", action="store_true", help="verbose logging (merged)")
    ap.add_argument("--outdir", type=str, default="runs", help="base output directory for dumps")
    args = ap.parse_args()

    try:
        G, weights, anchors = _load_bundle(args)
    except Exception as e:
        print(f"[ERROR] failed to load/assign weights: {e}", file=sys.stderr)
        sys.exit(1)

    dump_dir = os.path.join(args.outdir, f"{args.algo}_{args.graph_id}")
    _ensure_dir(dump_dir)

    if args.algo in ("k2-sim","k2-st"):
        if not nx.is_biconnected(G):
            print("[ERROR] Graph is not 2-vertex-connected; aborting for k=2 algorithm.", file=sys.stderr)
            sys.exit(2)

    if args.algo == "k3-local":
        try:
            kappa = nx.node_connectivity(G)
        except Exception:
            kappa = 0
        if kappa < 3:
            print(f"[ERROR] Graph node connectivity is {kappa} (<3); aborting for k=3 algorithm.", file=sys.stderr)
            sys.exit(3)

    if args.algo == "k3-local":
        res = balanced_partition_k3_local(G, weights, anchors, debug=args.debug, viz=args.viz, dump_dir=dump_dir, make_gif=args.gif)
        U,V,W = res.parts["U"], res.parts["V"], res.parts["W"]
        best_png = os.path.join(dump_dir, "best.png")
        save_partition_png_k3(G, U, V, W, weights, anchors, best_png, title=f"k3-local best L1={res.best_L1:.3f}")
        print(f"[k3] best: frame={res.best_iter} |U|={len(U)} |V|={len(V)} |W|={len(W)} L1={res.best_L1:.6f}")
        _maybe_make_gif(dump_dir, args.gif)

    elif args.algo == "k2-sim":
        u, v = anchors[:2]
        res = balanced_partition_worst(
            G, weights, u, v,
            debug=args.debug, debug_check=args.debug, trace=args.debug,
            max_iters=None, dump_yaml_dir=(dump_dir if args.viz else dump_dir), dump_plots=bool(args.viz)
        )
        U = set(res.parts[u]) if hasattr(res, 'parts') else set(res.U)
        Vset = set(res.parts[v]) if hasattr(res, 'parts') else set(res.V)
        best_val = getattr(res, 'best_L1', None)
        best_png = os.path.join(dump_dir, "best.png")
        save_partition_png(G, U, Vset, weights, (u, v), best_png, title=(f"k2-sim best L1={best_val:.3f}" if best_val is not None else "k2-sim best"))
        print(f"[k2-sim] iters={getattr(res,'iterations',None)} bound={getattr(res,'bound',None)} L1_best={best_val}")
        _maybe_make_gif(dump_dir, args.gif)

    elif args.algo == "k2-st":
        s, t = anchors[:2]
        try:
            order = st_numbering_tarjan(G, s, t)
        except STNumberingError as e:
            print(f"[k2-st] cannot run: {e}")
            return
        w = {v: float(weights.get(v,0.0)) for v in G.nodes()}
        Tsum = sum(w.values()); pref=0.0; best_i=None; best=1e100
        for i,vv in enumerate(order):
            pref += w[vv]
            if 0<i<len(order)-1:
                L1 = abs(pref)+abs(Tsum-pref)
                if L1 < best: best=L1; best_i=i
        Uset = set(order[:best_i+1]); Vset = set(order[best_i+1:])
        best_png = os.path.join(dump_dir, "best.png")
        save_partition_png(G, Uset, Vset, weights, (s,t), best_png, title=f"k2-st best L1={best:.3f}")
        print(f"[k2-st] |U|={len(Uset)} |V|={len(Vset)} L1={best:.6f}")
    else:
        print("Unknown --algo")

if __name__ == "__main__":
    main()
