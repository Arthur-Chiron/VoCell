"""Run every nucleus of the cloud through the SimCLR ResNet18 of cellf-supervised.

Writes two memory-mapped float16 arrays in the SAME global order as
data/cloud/features.npz -- the manifest's source order, then crops.npy order:

    data/cloud/emb_h.npy   (N, 512)  backbone output, what the linear probe uses
    data/cloud/emb_z.npy   (N, 128)  projector output, what InfoNCE is computed on

Both are kept because the question they answer is which one cosine belongs in:
the contrastive loss lives on z, downstream practice lives on h.

Preprocessing reproduces cellf-supervised's eval transform exactly
(ToTensor -> Normalize([0.5], [0.5]) on a uint8 PIL 'L' image), i.e. x/127.5 - 1.
BBBC051 is zero-padded 32 -> 64 rather than resized, like data.crop_2d, and
RESTORE is max-projected along Z, like data.get_source_2d: the same conventions
the rest of VoCell uses, so a nucleus is fed to the network as the explorer
shows it.

Needs torch (requirements-analysis.txt) and a checkpoint from cellf-supervised;
neither is required by the app. ~13 min for 2.6 M nuclei on an M-series GPU.

    python scripts/extract_embeddings.py
    python scripts/extract_embeddings.py --limit 3072     # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIBLING = os.path.join(ROOT, "..", "cellf-supervised")
sys.path.insert(0, SIBLING)

from Lib.models.simclr_resnet18 import SimCLRResNet18  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "cloud")
FEATURES = os.path.join(OUT_DIR, "features.npz")
CROPS = os.path.join(ROOT, "data", "crops")
RESTORE = os.path.join(ROOT, "data", "RESTORE", "nuclei.npy")
DEFAULT_CKPT = os.path.join(SIBLING, "simclr_resnet18_outHPA_augcell.pt")

BATCH = 1024


def batches(src: dict, limit: int | None):
    """Yield (offset, uint8 array (B, 64, 64)) for one source, in crops order."""
    name, count = src["name"], src["count"]
    n = min(count, limit) if limit else count

    if name == "RESTORE":
        vol = np.load(RESTORE, mmap_mode="r")
        for i in range(0, n, BATCH):
            b = np.asarray(vol[i:i + BATCH, ..., 0])    # (B, 64, 64, 64)
            yield i, b.max(axis=1)                      # project along Z
        return

    arr = np.load(os.path.join(CROPS, name, "crops.npy"), mmap_mode="r")
    for i in range(0, n, BATCH):
        b = np.asarray(arr[i:i + BATCH])
        if b.shape[1] < 64:                             # BBBC051: pad, never resize
            o = (64 - b.shape[1]) // 2
            p = np.zeros((len(b), 64, 64), dtype=b.dtype)
            p[:, o:o + b.shape[1], o:o + b.shape[2]] = b
            b = p
        yield i, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap every source at N nuclei (smoke test). Writes to "
                         "a separate suffixed file.")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLRResNet18()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                     weights_only=False))
    model.eval().to(device)

    manifest = json.loads(str(np.load(FEATURES)["manifest"]))
    n_total = sum(s["count"] for s in manifest)
    print(f"{os.path.basename(args.checkpoint)} · device={device} · "
          f"{n_total:,} noyaux", flush=True)

    suffix = f".limit{args.limit}" if args.limit else ""
    emb_h = np.lib.format.open_memmap(
        os.path.join(OUT_DIR, f"emb_h{suffix}.npy"), mode="w+",
        dtype=np.float16, shape=(n_total, model.feat_dim))
    emb_z = np.lib.format.open_memmap(
        os.path.join(OUT_DIR, f"emb_z{suffix}.npy"), mode="w+",
        dtype=np.float16, shape=(n_total, 128))

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for src in manifest:
            ts = time.time()
            for off, b in batches(src, args.limit):
                x = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0)
                h = model.backbone(x.unsqueeze(1).to(device))
                z = model.projector(h)
                lo = src["start"] + off
                emb_h[lo:lo + len(b)] = h.float().cpu().numpy().astype(np.float16)
                emb_z[lo:lo + len(b)] = z.float().cpu().numpy().astype(np.float16)
                done += len(b)
            n = min(src["count"], args.limit) if args.limit else src["count"]
            dt = max(time.time() - ts, 1e-9)
            print(f"  {src['name']:<20s} {n:>9,} en {dt:6.1f}s "
                  f"({n / dt:,.0f}/s)   {done:,} / {n_total:,}", flush=True)

    emb_h.flush()
    emb_z.flush()
    print(f"\n{done:,} embeddings · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
