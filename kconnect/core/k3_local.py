from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
import os, sys
import networkx as nx
import yaml
from .proofcheck import max_abs_weight
from ..viz import save_partition_png_k3

class K3Result:
    def __init__(self, parts, M, best_L1, best_iter, iters, dump_dir, best_yaml, best_png, timeline_yaml):
        self.parts = parts; self.M = M; self.best_L1 = best_L1; self.best_iter = best_iter
        self.iters = iters; self.dump_dir = dump_dir; self.best_yaml = best_yaml
        self.best_png = best_png; self.timeline_yaml = timeline_yaml

# Frontier vertices: those with neighbors in other parts
def _frontier(G, parts):
    U,V,W=parts; S=set()
    for X,Y,Z in [(U,V,W),(V,U,W),(W,U,V)]:
        for x in X:
            for y in G.neighbors(x):
                if y in Y or y in Z: S.add(x); break
    return S

def balanced_partition_k3_local(G: nx.Graph, weights: Dict[str,float], anchors: Tuple[str,str,str],
                                *, debug: bool=False, viz: bool=False, dump_dir: Optional[str]=None, make_gif: bool=False) -> K3Result:
    u,v,w = anchors
    if dump_dir: os.makedirs(dump_dir, exist_ok=True)

    # One-branch initialization
    U = set(G.nodes()) - {v, w}
    V = {v}
    W = {w}
    assert u in U

    def sums_dict(U,V,W): return {"U": float(sum(weights.get(x,0.0) for x in U)),
                                  "V": float(sum(weights.get(x,0.0) for x in V)),
                                  "W": float(sum(weights.get(x,0.0) for x in W))}
    sums = sums_dict(U,V,W); M=max_abs_weight(weights); L1=abs(sums["U"])+abs(sums["V"])+abs(sums["W"])
    frame=0; iters=0; best_L1=L1; best_iter=0; best_state=(set(U),set(V),set(W)); best_yaml=best_png=timeline_yaml=None

    if viz and dump_dir:
        Uc=[sorted(list(c)) for c in nx.connected_components(G.subgraph(U))] if len(U) else []
        Vc=[sorted(list(c)) for c in nx.connected_components(G.subgraph(V))] if len(V) else []
        Wc=[sorted(list(c)) for c in nx.connected_components(G.subgraph(W))] if len(W) else []
        snap={"frame":frame,"initial":True,"U":sorted(U),"V":sorted(V),"W":sorted(W),
              "U_components":Uc,"V_components":Vc,"W_components":Wc,"sum_U":sums["U"],"sum_V":sums["V"],"sum_W":sums["W"],
              "L1":L1,"bound2M":2.0*M}
        with open(os.path.join(dump_dir,f"iter_{frame:03d}.yaml"),"w") as f: yaml.safe_dump(snap,f,sort_keys=False)
        save_partition_png_k3(G,U,V,W,weights,anchors,os.path.join(dump_dir,f"iter_{frame:03d}.png"),
                              highlight=None,title=f"iter {frame:03d}: initial L1={L1:.3f} (2M={2*M:.3f})")
        frame+=1

    while True:
        F=_frontier(G,(U,V,W))
        if debug: print(f"[k3] iter={iters} |F|={len(F)} L1={L1:.6f} (2M={2*M:.6f})", file=sys.stderr)
        artU=set(nx.articulation_points(G.subgraph(U))) if len(U)>2 else set()
        artV=set(nx.articulation_points(G.subgraph(V))) if len(V)>2 else set()
        artW=set(nx.articulation_points(G.subgraph(W))) if len(W)>2 else set()

        best_move=None; best_delta=0.0
        for x in F:
            src = "U" if x in U else ("V" if x in V else "W")
            if (src=="U" and x==u) or (src=="V" and x==v) or (src=="W" and x==w): continue
            nbrs=set(G.neighbors(x))
            for dst,Sdst in (("U",U),("V",V),("W",W)):
                if dst==src: continue
                if not (nbrs & Sdst): continue
                if (src=="U" and x in artU) or (src=="V" and x in artV) or (src=="W" and x in artW): continue
                w_x=float(weights.get(x,0.0))
                S=dict({"U":sums["U"],"V":sums["V"],"W":sums["W"]})
                S[src]-=w_x; S[dst]+=w_x
                new_L1=abs(S["U"])+abs(S["V"])+abs(S["W"])
                delta=new_L1 - L1
                if delta < best_delta - 1e-15:
                    y=next(iter(nbrs & Sdst))
                    best_delta=delta; best_move=(x,src,dst,y)
        if best_move is None:
            if debug: print("[k3] no improving 1-opt move; stopping", file=sys.stderr)
            break
        x,src,dst,y = best_move
        if src=="U": U.remove(x)
        elif src=="V": V.remove(x)
        else: W.remove(x)
        if dst=="U": U.add(x)
        elif dst=="V": V.add(x)
        else: W.add(x)
        w_x=float(weights.get(x,0.0)); sums[src]-=w_x; sums[dst]+=w_x
        L1=abs(sums["U"])+abs(sums["V"])+abs(sums["W"]); iters+=1
        if viz and dump_dir:
            Uc=[sorted(list(c)) for c in nx.connected_components(G.subgraph(U))] if len(U) else []
            Vc=[sorted(list(c)) for c in nx.connected_components(G.subgraph(V))] if len(V) else []
            Wc=[sorted(list(c)) for c in nx.connected_components(G.subgraph(W))] if len(W) else []
            snap={"frame":frame,"iter":iters,"move":{"vertex":x,"src":src,"dst":dst,"attach_to":y},
                  "U":sorted(U),"V":sorted(V),"W":sorted(W),
                  "U_components":Uc,"V_components":Vc,"W_components":Wc,
                  "sum_U":sums["U"],"sum_V":sums["V"],"sum_W":sums["W"],"L1":L1,"bound2M":2.0*M}
            with open(os.path.join(dump_dir,f"iter_{frame:03d}.yaml"),"w") as f: yaml.safe_dump(snap,f,sort_keys=False)
            save_partition_png_k3(G,U,V,W,weights,anchors,os.path.join(dump_dir,f"iter_{frame:03d}.png"),
                                  highlight=x,title=f"iter {frame:03d}: move {x} {src}->{dst}  L1={L1:.3f} (2M={2*M:.3f})",
                                  attach_edge=(x,y))
            frame+=1
        if L1 < best_L1 - 1e-15:
            best_L1=L1; best_iter=frame-1; best_state=(set(U),set(V),set(W))
            if viz and dump_dir:
                by=os.path.join(dump_dir,"best.yaml"); bp=os.path.join(dump_dir,"best.png")
                with open(by,"w") as f: yaml.safe_dump({"frame":best_iter,"U":sorted(best_state[0]),"V":sorted(best_state[1]),"W":sorted(best_state[2]),
                    "sum_U":sum(weights.get(x,0.0) for x in best_state[0]),"sum_V":sum(weights.get(x,0.0) for x in best_state[1]),
                    "sum_W":sum(weights.get(x,0.0) for x in best_state[2]),"L1":best_L1,"bound2M":2.0*M}, f, sort_keys=False)
                save_partition_png_k3(G,best_state[0],best_state[1],best_state[2],weights,anchors,bp,highlight=None,
                                      title=f"best L1={best_L1:.3f} (2M={2*M:.3f})")
    final_png=None
    if dump_dir:
        final_png=os.path.join(dump_dir,"final.png")
        save_partition_png_k3(G,best_state[0],best_state[1],best_state[2],weights,anchors,final_png,
                              highlight=None,title=f"final (best) L1={best_L1:.3f} (2M={2*M:.3f})")
    return K3Result(parts={"U":best_state[0],"V":best_state[1],"W":best_state[2]}, M=M, best_L1=best_L1,
                    best_iter=best_iter, iters=iters, dump_dir=dump_dir, best_yaml=None, best_png=final_png, timeline_yaml=None)
