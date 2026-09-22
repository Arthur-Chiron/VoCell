"""Control for latent_probe.py: the same scores from an UNTRAINED ResNet18.

A randomly initialised convnet is already a serviceable random projection of an
image, and a k-NN on it scores well above chance. Without this control, "the
SimCLR latent reaches x2.9 the chance level" says nothing about what the
contrastive pre-training actually bought.

Both models are run on the same crops, in the same preprocessing, and scored
the same way as measurements B and E of latent_probe.py. Also reports how many
of the 512 backbone dimensions are identically zero, which is what bounds the
checkpoint: a collapsed dimension carries nothing, whatever the metric.

Needs torch (requirements-analysis.txt) and a checkpoint from cellf-supervised.

    python scripts/control_random.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIBLING = os.path.join(ROOT, "..", "cellf-supervised")
sys.path.insert(0, SIBLING)

from Lib.models.simclr_resnet18 import SimCLRResNet18  # noqa: E402

CLOUD = os.path.join(ROOT, "data", "cloud")
CROPS = os.path.join(ROOT, "data", "crops")
RESTORE = os.path.join(ROOT, "data", "RESTORE", "nuclei.npy")
DEFAULT_CKPT = os.path.join(SIBLING, "simclr_resnet18_outHPA_augcell.pt")

K = 10
PER_CLASS = 350
PER_SOURCE = 400
UNLABELED = "non étiqueté"

RNG = np.random.default_rng(0)

d = np.load(os.path.join(CLOUD, "features.npz"))
CLASS_IDX = d["class_idx"]
CLASS_NAMES = [str(x) for x in d["class_names"]]
MANIFEST = json.loads(str(d["manifest"]))
LABELED = np.array([n != UNLABELED for n in CLASS_NAMES])
N = sum(s["count"] for s in MANIFEST)


def crops_at(idx: np.ndarray) -> np.ndarray:
    """The raw 64x64 uint8 crops for sorted global indices.

    Same conventions as scripts/extract_embeddings.py: BBBC051 padded rather
    than resized, RESTORE max-projected along Z.
    """
    out = np.zeros((len(idx), 64, 64), np.uint8)
    for src in MANIFEST:
        lo, hi = src["start"], src["start"] + src["count"]
        here = (idx >= lo) & (idx < hi)
        if not here.any():
            continue
        local = idx[here] - lo
        if src["name"] == "RESTORE":
            vol = np.load(RESTORE, mmap_mode="r")
            out[here] = np.asarray(vol[local, ..., 0]).max(axis=1)
            continue
        arr = np.load(os.path.join(CROPS, src["name"], "crops.npy"), mmap_mode="r")
        b = np.asarray(arr[local])
        if b.shape[1] < 64:
            o = (64 - b.shape[1]) // 2
            out[here, o:o + b.shape[1], o:o + b.shape[2]] = b
        else:
            out[here] = b
    return out


def embed(model, device, crops, batch=512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i:i + batch].astype(np.float32) / 127.5 - 1.0)
            out.append(model.backbone(x.unsqueeze(1).to(device)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def cosine_agreement(x, y, k=K):
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    s = u @ u.T
    np.fill_diagonal(s, -np.inf)
    return float((y[np.argpartition(-s, k, axis=1)[:, :k]] == y[:, None]).mean())


def chance_level(y):
    _, cnt = np.unique(y, return_counts=True)
    return float(((cnt / cnt.sum()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    untrained = SimCLRResNet18().eval().to(device)      # random init, never trained
    trained = SimCLRResNet18()
    trained.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                       weights_only=False))
    trained.eval().to(device)

    print(f"{os.path.basename(args.checkpoint)} · device={device}")
    print(f"\nAccord des {K} plus proches voisins, cosinus sur h (512)")
    print(f"   {'':<9s} {'hasard':>7s} {'NON ENTRAÎNÉ':>18s} {'ENTRAÎNÉ':>16s}")

    for src in MANIFEST:
        if not src["labeled"]:
            continue
        lo = src["start"]
        cls = CLASS_IDX[lo:lo + src["count"]]
        pool = np.flatnonzero(LABELED[cls])
        picks = [RNG.choice(pool[cls[pool] == u],
                            min(PER_CLASS, int((cls[pool] == u).sum())),
                            replace=False)
                 for u in np.unique(cls[pool]) if (cls[pool] == u).sum() >= 40]
        local = np.sort(np.concatenate(picks))
        y, crops = cls[local], crops_at(lo + local)
        chance = chance_level(y)
        a_r = cosine_agreement(embed(untrained, device, crops), y)
        a_t = cosine_agreement(embed(trained, device, crops), y)
        print(f"   {src['name']:<9s} {chance:7.3f} {a_r:11.3f} (x{a_r / chance:.2f})"
              f" {a_t:9.3f} (x{a_t / chance:.2f})")

    idx = np.sort(np.concatenate([
        RNG.choice(np.arange(s["start"], s["start"] + s["count"]),
                   PER_SOURCE, replace=False) for s in MANIFEST]))
    source_of = np.zeros(N, np.int16)
    for i, s in enumerate(MANIFEST):
        source_of[s["start"]:s["start"] + s["count"]] = i
    y, crops = source_of[idx], crops_at(idx)
    chance = chance_level(y)
    h_r, h_t = embed(untrained, device, crops), embed(trained, device, crops)
    a_r, a_t = cosine_agreement(h_r, y), cosine_agreement(h_t, y)
    print(f"   {'SOURCE':<9s} {chance:7.3f} {a_r:11.3f} (x{a_r / chance:.2f})"
          f" {a_t:9.3f} (x{a_t / chance:.2f})")

    print(f"\n   dimensions identiquement nulles sur h :"
          f" {(h_r.std(0) < 1e-6).sum()}/512 non entraîné,"
          f" {(h_t.std(0) < 1e-6).sum()}/512 entraîné")


if __name__ == "__main__":
    main()
