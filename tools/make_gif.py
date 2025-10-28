from __future__ import annotations
import argparse, os, glob
from PIL import Image

def main():
    ap = argparse.ArgumentParser(description="Create an animated GIF from iter_XXX.png snapshots")
    ap.add_argument("--dir", required=True, help="Directory containing iter_*.png files")
    ap.add_argument("--out", required=True, help="Output GIF path")
    ap.add_argument("--fps", type=float, default=2.0, help="Frames per second (default 2.0)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "iter_*.png")))
    if not files:
        print(f"No PNGs found under {args.dir}. Did you run with --viz?")
        return

    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in files]
    duration_ms = int(1000.0 / max(args.fps, 0.1))

    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
        disposal=2,
    )
    print(f"Wrote {args.out} ({len(files)} frames at {args.fps} fps)")

if __name__ == "__main__":
    main()
