"""Run every nucleus of the cloud through the SimCLR ResNet18s of cellf-supervised.

Every `simclr_resnet18_*.pt` at the root of the sibling repo is a model; each
one writes two memory-mapped float16 arrays in the SAME global order as
data/cloud/features.npz -- the manifest's source order, then crops.npy order:

    data/cloud/emb/<key>_h.npy   (N, 512)  backbone output, what the linear probe uses
    data/cloud/emb/<key>_z.npy   (N, 128)  projector output, what InfoNCE is computed on

`<key>` is what follows `simclr_resnet18_` in the checkpoint's name
(cloud.model_key), and it is the name the cloud's layout picker files the
model under. A model whose arrays already exist is skipped unless --force.

Both are kept because the question they answer is which one cosine belongs in:
the contrastive loss lives on z, downstream practice lives on h.

Preprocessing reproduces cellf-supervised's eval transform exactly
(ToTensor -> Normalize([0.5], [0.5]) on a uint8 PIL 'L' image), i.e. x/127.5 - 1.
BBBC051 is zero-padded 32 -> 64 rather than resized, like data.crop_2d, and
RESTORE is max-projected along Z, like data.get_source_2d: the same conventions
the rest of VoCell uses, so a nucleus is fed to the network as the explorer
shows it.

Needs torch (requirements-analysis.txt) and a checkpoint from cellf-supervised;
neither is required by the app. ~13 min per model for 2.6 M nuclei on an
M-series GPU.

    python scripts/extract_embeddings.py                  # every checkpoint
    python scripts/extract_embeddings.py --checkpoint path/to/model.pt
    python scripts/extract_embeddings.py --limit 3072     # smoke test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIBLING = os.path.join(ROOT, "..", "cellf-supervised")
sys.path.insert(0, SIBLING)

sys.path.insert(0, os.path.join(ROOT, "src"))

from Lib.models.simclr_resnet18 import SimCLRResNet18  # noqa: E402
import cloud  # noqa: E402

FEATURES = os.path.join(ROOT, "data", "cloud", "features.npz")
EMB_DIR = os.path.join(ROOT, "data", "cloud", "emb")
CROPS = os.path.join(ROOT, "data", "crops")
RESTORE = os.path.join(ROOT, "data", "RESTORE", "nuclei.npy")

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


def checkpoints() -> list:
    """Every SimCLR ResNet18 checkpoint at the root of cellf-supervised."""
    found = glob.glob(os.path.join(SIBLING, cloud.CKPT_PREFIX + "*.pt"))
    by_key = {cloud.model_key(p): p for p in found}
    return [by_key[k] for k in cloud.model_order(list(by_key))]


def extract(ckpt: str, manifest: list, device: str, limit, force: bool) -> None:
    key = cloud.model_key(ckpt)
    suffix = f".limit{limit}" if limit else ""
    out_h = os.path.join(EMB_DIR, f"{key}_h{suffix}.npy")
    out_z = os.path.join(EMB_DIR, f"{key}_z{suffix}.npy")
    n_total = sum(s["count"] for s in manifest)
    if not force and os.path.exists(out_h) and os.path.exists(out_z) \
            and np.load(out_h, mmap_mode="r").shape[0] == n_total:
        print(f"{key} : déjà extrait, ignoré (--force pour refaire)")
        return

    model = SimCLRResNet18()
    model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=False))
    model.eval().to(device)
    print(f"{os.path.basename(ckpt)} · device={device} · "
          f"{n_total:,} noyaux", flush=True)

    # Written under a temporary name and renamed at the end: an interrupted
    # run must not leave a full-size, half-zero array that build_cloud.py
    # would take for a finished one.
    tmp_h, tmp_z = out_h + ".part", out_z + ".part"
    emb_h = np.lib.format.open_memmap(
        tmp_h, mode="w+", dtype=np.float16, shape=(n_total, model.feat_dim))
    emb_z = np.lib.format.open_memmap(
        tmp_z, mode="w+", dtype=np.float16, shape=(n_total, 128))

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for src in manifest:
            ts = time.time()
            for off, b in batches(src, limit):
                x = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0)
                h = model.backbone(x.unsqueeze(1).to(device))
                z = model.projector(h)
                lo = src["start"] + off
                emb_h[lo:lo + len(b)] = h.float().cpu().numpy().astype(np.float16)
                emb_z[lo:lo + len(b)] = z.float().cpu().numpy().astype(np.float16)
                done += len(b)
            n = min(src["count"], limit) if limit else src["count"]
            dt = max(time.time() - ts, 1e-9)
            print(f"  {src['name']:<20s} {n:>9,} en {dt:6.1f}s "
                  f"({n / dt:,.0f}/s)   {done:,} / {n_total:,}", flush=True)

    emb_h.flush()
    emb_z.flush()
    del emb_h, emb_z
    os.replace(tmp_h, out_h)
    os.replace(tmp_z, out_z)
    print(f"{key} : {done:,} embeddings · {time.time() - t0:.0f}s\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", nargs="*", default=None,
                    help="Only these checkpoints (default: every "
                         "simclr_resnet18_*.pt of cellf-supervised).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap every source at N nuclei (smoke test). Writes to "
                         "a separate suffixed file.")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract models whose arrays already exist.")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(str(np.load(FEATURES)["manifest"]))
    os.makedirs(EMB_DIR, exist_ok=True)

    ckpts = args.checkpoint or checkpoints()
    if not ckpts:
        sys.exit(f"Aucun {cloud.CKPT_PREFIX}*.pt dans {SIBLING}")
    for ckpt in ckpts:
        extract(ckpt, manifest, device, args.limit, args.force)


if __name__ == "__main__":
    main()
