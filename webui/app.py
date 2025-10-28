from __future__ import annotations
import os, sys, subprocess, uuid, re
from flask import Flask, render_template, request, redirect, url_for, flash

BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE, ".."))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
UPLOAD_DIR = os.path.join(BASE, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "kconnect-webui-demo"
app.static_folder = RUNS_DIR
app.static_url_path = "/runs"

def sanitize_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)[:80] or "yaml"

def build_command(params: dict, label: str):
    exe = sys.executable
    cmd = [exe, "-m", "kconnect.service.main",
           "--algo", params["algo"],
           "--weights", params["weights"],
           "--rand-seed", str(params.get("seed", 7)),
           "--anchor-sign", params.get("anchor_sign", "negative")]
    if params["source"] == "catalog":
        cmd += ["--catalog", params["catalog"], "--graph-id", params["graph_id"]]
    else:
        cmd += ["--graph", params["graph_path"], "--graph-id", label]
    if params.get("viz"): cmd.append("--viz")
    if params.get("gif"): cmd.append("--gif")
    if params.get("debug"): cmd.append("--debug")
    return cmd

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        algo = request.form.get("algo", "k3-local")
        source = request.form.get("source", "catalog")
        catalog = request.form.get("catalog", "k3")
        graph_id = request.form.get("graph_id", "K4")
        weights = request.form.get("weights", "default")
        seed = int(request.form.get("seed", "7") or "7")
        anchor_sign = request.form.get("anchor_sign", "negative")
        viz = bool(request.form.get("viz"))
        gif = bool(request.form.get("gif"))
        debug = bool(request.form.get("debug"))

        params = {"algo": algo, "source": source, "catalog": catalog, "graph_id": graph_id,
                  "weights": weights, "seed": seed, "anchor_sign": anchor_sign,
                  "viz": viz, "gif": gif, "debug": debug}

        label = graph_id
        if source == "yaml":
            f = request.files.get("yamlfile")
            if not f or not f.filename:
                flash("Please choose a YAML file when source=YAML.")
                return redirect(url_for("index"))
            fname = sanitize_label(f.filename)
            save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{fname}")
            f.save(save_path)
            params["graph_path"] = save_path
            label = os.path.splitext(fname)[0]

        cmd = build_command(params, label=sanitize_label(label))
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        stdout = proc.stdout; stderr = proc.stderr; rc = proc.returncode

        outdir = os.path.join(RUNS_DIR, f"{algo}_{label}")
        best_png = os.path.join(outdir, "best.png")
        gif_path = os.path.join(outdir, "timeline.gif")
        has_best = os.path.exists(best_png); has_gif = os.path.exists(gif_path)

        return render_template("result.html",
                               algo=algo, label=label, cmd=" ".join(cmd),
                               rc=rc, stdout=stdout, stderr=stderr,
                               has_best=has_best, best_url=(url_for("static", filename=f"{algo}_{label}/best.png") if has_best else None),
                               has_gif=has_gif, gif_url=(url_for("static", filename=f"{algo}_{label}/timeline.gif") if has_gif else None))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
