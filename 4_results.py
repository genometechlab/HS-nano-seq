#!/usr/bin/env python3
"""
Confusion-matrix figures + accuracy summary for every run's predictions.

Reads the run list from experiments.sh (respects ONLY=...), loads each run's
Data/processed/<exp>/<run>/preds.pkl, and writes one figure per run to --out-dir.

    training run (TRAIN_RUN) -> train / val / test panels from split_dict.pkl
    any other run            -> a single 'test' panel over all its reads

Matrices are row-normalized (diagonal = per-class recall) and annotated with
recall and raw count per cell. Axes show the class as its PH009 barcode name
(class 0 -> BC01 ... class 15 -> BC16); the title gives the run's own barcodes.

Also importable (4_results.ipynb uses it; the leading digit needs importlib):
    results = importlib.import_module("4_results")
"""

import argparse
import os
import os.path as osp
import pickle
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

NUM_CLASSES = 16
LABELS = list(range(NUM_CLASSES))
TICK_LABELS = [f"BC{c + 1:02d}" for c in LABELS]   # class 0 -> BC01 ... 15 -> BC16

REPO = osp.dirname(osp.abspath(__file__))


# --------------------------------------------------------------------------- #
# Config / data loading
# --------------------------------------------------------------------------- #
def load_config(data=None):
    """
    Source experiments.sh and return
        {"data": ..., "processed": ..., "split_dict": ..., "train_run": ...,
         "runs": [{"key": "PH009/9fea1263", "first_bc": 1}, ...]}
    """
    env = dict(os.environ)
    if data:
        env["DATA"] = data
    script = (
        'source ./experiments.sh; '
        'echo "$DATA"; echo "$PROCESSED"; echo "$SPLIT_DICT"; echo "$TRAIN_RUN"; '
        'for l in "${RUNS[@]}"; do parse_run "$l"; echo "$EXP/$RUN $FIRST_BC"; done'
    )
    out = subprocess.run(["bash", "-c", script], cwd=REPO, env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr.strip() or "failed to source experiments.sh")
    lines = out.stdout.splitlines()

    def resolve(p):
        return p if osp.isabs(p) else osp.normpath(osp.join(REPO, p))

    runs = []
    for line in lines[4:]:
        key, first_bc = line.split()
        runs.append({"key": key, "first_bc": int(first_bc)})
    return {
        "data": resolve(lines[0]),
        "processed": resolve(lines[1]),
        "split_dict": resolve(lines[2]),
        "train_run": lines[3],
        "runs": runs,
    }


def load_split(path):
    """train/val/test read-id lists of the training run, or None if absent."""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        print(f"[warn] {path} not found; the training run falls back to all reads")
        return None


def load_preds(processed, key):
    """{read_id: {"gt", "pred"}} for one run, or None if not produced yet."""
    path = osp.join(processed, key, "preds.pkl")
    if not osp.exists(path):
        print(f"[warn] {key}: {path} not found; skipping")
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Panels + plotting
# --------------------------------------------------------------------------- #
def panels(key, preds, split_rids, train_run):
    """[(name, gt, pred, missing)] -- one per split for the training run, else one."""
    use_split = key == train_run and split_rids is not None
    names = ["train", "val", "test"] if use_split else ["test"]
    out = []
    for name in names:
        rids = split_rids[name] if use_split else preds.keys()
        gts, prs, missing = [], [], 0
        for rid in rids:
            rec = preds.get(rid)
            if rec is None:
                missing += 1     # e.g. reverse-strand reads, now skipped at trimming
                continue
            gts.append(rec["gt"]); prs.append(rec["pred"])
        out.append((name, np.array(gts), np.array(prs), missing))
    return out


def draw(ax, gt, pr, title):
    cm = confusion_matrix(gt, pr, labels=LABELS)
    row_sums = cm.sum(axis=1, keepdims=True)
    recall = np.divide(cm, row_sums, out=np.zeros(cm.shape), where=row_sums != 0)

    im = ax.imshow(recall, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(LABELS); ax.set_yticks(LABELS)
    ax.set_xticklabels(TICK_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(TICK_LABELS)
    ax.set_xlabel("pred"); ax.set_ylabel("gt")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{recall[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7,
                    color="white" if recall[i, j] > 0.5 else "black")
    return im


def plot_run(run, preds, split_rids, train_run):
    """Figure for one run + summary rows [(key, panel, n, acc, missing)]."""
    key, first_bc = run["key"], run["first_bc"]
    bcs = f"barcode{first_bc:02d}-{first_bc + NUM_CLASSES - 1:02d}"
    ps = panels(key, preds, split_rids, train_run)

    fig, axes = plt.subplots(len(ps), 1, figsize=(13, 13 * len(ps)), squeeze=False)
    rows = []
    for ax, (name, gt, pr, missing) in zip(axes[:, 0], ps):
        acc = (gt == pr).mean() if len(gt) else float("nan")
        title = (f"{key} {name}  ({bcs})  n={len(gt)}  acc={acc:.3f}"
                 + (f"  (missing {missing})" if missing else ""))
        im = draw(ax, gt, pr, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        rows.append((key, name, len(gt), acc, missing))
    fig.tight_layout()
    return fig, rows


def print_summary(rows):
    print(f"\n{'run':<16} {'panel':<6} {'n':>10} {'acc':>7} {'missing':>8}")
    for key, name, n, acc, missing in rows:
        print(f"{key:<16} {name:<6} {n:>10} {acc:>7.4f} {missing:>8}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Write a confusion-matrix figure per run (runs from experiments.sh)."
    )
    p.add_argument("--data", default=None,
                   help="Data root (default: DATA from experiments.sh, i.e. ./Data).")
    p.add_argument("--out-dir", default="figures",
                   help="Where to write confusion_matrix_<exp>_<run>.png. Default: figures/")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")          # headless: write files, never open a window

    args = parse_args()
    cfg = load_config(args.data)
    split_rids = load_split(cfg["split_dict"])
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for run in cfg["runs"]:
        preds = load_preds(cfg["processed"], run["key"])
        if preds is None:
            continue
        fig, rows = plot_run(run, preds, split_rids, cfg["train_run"])
        out = osp.join(args.out_dir, f"confusion_matrix_{run['key'].replace('/', '_')}.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
        all_rows += rows

    if not all_rows:
        sys.exit("No predictions found; run inference first.")
    print_summary(all_rows)


if __name__ == "__main__":
    main()
