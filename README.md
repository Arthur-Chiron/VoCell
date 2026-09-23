# VoCell — 3D augmentations of cell nuclei

VoCell generates **3D volumetric augmentations from 2D images of cell nuclei**.
The founding idea: rebuild a pseudo-volume from a single 2D crop, then slice it
along an arbitrary plane. Each slice is a view of the nucleus that was never
acquired, which is what makes it an augmentation.

The repository has two parts:

1. **An interactive volumetric explorer** (Streamlit + Plotly). Its entry page
   is a point cloud of **2,633,390 nuclei from eleven sources**: hover a point
   to see the nucleus and its segmentation mask, click it to open it in the
   explorer. There, the nucleus is reconstructed in 3D, cut along a plane you
   choose, and shown as a voxel ("Minecraft") render next to the matching 2D
   slice.
2. **A SAM3D pipeline**: AI-based 3D reconstruction (Segment Anything 3D
   Objects), with a fine-tuning script on real confocal volumes. It needs a
   remote GPU machine and does not run locally (see [below](#sam3d-pipeline)).

> The user interface is in French. Everything needed to run it is in this
> README.

---

## Quick start

For people who already have access to the data (see
[Access](#access-two-private-dependencies)). Around 10 minutes and 300 MB of
data. This gives the point cloud, the class colouring and the explorer:

```bash
git clone https://github.com/Arthur-Chiron/VoCell.git
cd VoCell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt huggingface_hub

# A small subset of the data: one labelled source + three tiny ones
hf auth login
hf download CellfSup/CellfSup --repo-type dataset --local-dir data \
  --include "crops/BBBC051/*" --include "crops/AitslabBioimaging1/*" \
  --include "crops/NucleusSegData/*" --include "crops/S-BSST265/*"

python scripts/build_cloud.py      # ~20 s for this subset
streamlit run src/app.py           # opens http://localhost:8501
```

Each step is explained below.

---

## Requirements

- **Python 3.9 or newer.** Tested on 3.9.6 under macOS.
- **No GPU needed.** The app and the data preparation run on CPU. `torch` is
  only needed for optional analysis scripts and is kept out of
  `requirements.txt` on purpose.
- **Disk space:** from ~300 MB for the quick-start subset up to ~13 GB for all
  eleven sources plus the generated cloud assets (see the table below).
- **git**, to install `cellaug` straight from GitHub.

### Access: two private dependencies

Two things this app needs are currently **private**. Without access to them,
`pip install` or the data download fails:

| What | Where | Needed for | Symptom without access |
|---|---|---|---|
| `cellaug` (augmentation library) | GitHub `Arthur-Chiron/cellaug` | `pip install -r requirements.txt` | `fatal: could not read Username for 'https://github.com'` |
| The nucleus crops | Hugging Face dataset `CellfSup/CellfSup` | everything except the logo | `401` / `RepositoryNotFoundError` from `hf download` |

Ask the maintainer for access to both. For GitHub, pip calls `git` in a
subprocess, so git itself must be able to authenticate (a credential helper,
`gh auth setup-git`, or an SSH URL). For Hugging Face, run `hf auth login`
with an account that belongs to the `CellfSup` organisation.

---

## Installation

```bash
git clone https://github.com/Arthur-Chiron/VoCell.git
cd VoCell
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers the app and the data-preparation scripts. There are
two other requirement files, and the app needs neither:

- `requirements-analysis.txt`: `torch`, for the latent-space scripts;
- `requirements-sam3d.txt`: the GPU pipeline, in its own conda environment.

**Contributors to `cellaug`:** install a local editable clone instead, shared
with `cellf-supervised`, so an edit shows up in both projects without
reinstalling:

```bash
git clone https://github.com/Arthur-Chiron/cellaug.git ../cellaug
pip install -e ../cellaug
```

---

## Data

Nothing under `data/` is versioned. Each machine has to fill it in. The app
expects this layout:

```
data/
  crops/<SOURCE>/crops.npy            # (N, 64, 64) uint8, BBBC051 is (N, 32, 32)
  crops/<SOURCE>/crop_metadata.csv    # labels, for CODEX / HPA / BBBC051
  RESTORE/nuclei.npy                  # (N, 64, 64, 64, 1) uint8, optional
```

**Any subset works.** The explorer only lists the sources it finds, and the
cloud builds from whatever is there. With no source at all, the app says so in
the sidebar instead of crashing.

### The eleven sources

| Source | Nuclei | Labels | Size | Notes |
|---|---:|---|---:|---|
| CODEX | 168,992 | 14 cell families | 678 MB | colorectal cancer, Hoechst, ~0.376 µm/px |
| HPA | 673,936 | 17 cell lines + unlabelled | 5.3 GB | Human Protein Atlas |
| BBBC051 | 232,429 | 11 kidney cell types | 236 MB | native **32²** crops, ~0.5 µm/px |
| TissueNet | 1,335,905 | — | 5.2 GB | DAPI, multi-tissue |
| HelaCytoNuc | 146,910 | — | 583 MB | HeLa, DAPI |
| DSB2018 | 32,155 | — | 129 MB | Data Science Bowl 2018 |
| NuInSeg | 29,332 | — | 117 MB | H&E, multi-organ |
| S-BSST265 | 5,380 | — | 21 MB | DAPI |
| NucleusSegData | 3,250 | — | 13 MB | Huh7 / HepG2 |
| AitslabBioimaging1 | 1,735 | — | 7 MB | U2OS, Hoechst |
| RESTORE | 3,366 | — | 842 MB | **real** 3D confocal volumes, DAPI, ~0.15 µm/px |

The first ten are 2D crops, already segmented and cut out. Their volume is
**synthesised** (Gaussian or linear depth profile, or SAM3D). RESTORE is the
only source with a real volume, with sparse Z slices.

Colouring the cloud by class needs at least one of the three labelled sources
(CODEX, HPA, BBBC051). BBBC051 is the smallest of them.

### Option A: download from Hugging Face

The Hugging Face dataset already uses the `crops/<SOURCE>/` layout, so it can
be downloaded straight into `data/`:

```bash
pip install huggingface_hub
hf auth login
hf download CellfSup/CellfSup --repo-type dataset --local-dir data \
  --include "crops/CODEX/*" --include "crops/BBBC051/*"   # pick any sources
```

Repeat `--include` once per source. With `hf` 1.x, `--include a b` does
**not** mean two patterns: the second one is silently read as a file name, and
only the first source is downloaded. Use `--include "crops/*"` to get all ten
(~12 GB).

### Option B: link a local `cellf-supervised` clone

If the sibling repository `cellf-supervised` is already cloned next to this
one, with its data fetched (`python fetch_data.py` over there), link it
instead of copying 12 GB:

```bash
mkdir -p data
ln -s ../../cellf-supervised/Data/crops data/crops
```

The link is relative: it stays valid as long as both repositories sit in the
same parent directory.

### RESTORE (optional)

RESTORE is not distributed. It is rebuilt from raw Imaris confocal
acquisitions (`.ims` files and `_mask.npy` instance masks), which only the
project members have. With those files in `data/2021/`:

```bash
python scripts/preprocess_restore.py      # ~1 min, writes data/RESTORE/nuclei.npy
```

The script isolates each nucleus and puts its slices back at their
**physical** Z position in a 64³ grid, using the axial and lateral
resolutions read from the file's metadata. It finds the DAPI channel **by
name**, because its index changes from one series to the next.

Without RESTORE, everything works except the one source with real volumes.

---

## Build the point cloud

The cloud reads precomputed assets: descriptors, thumbnail atlases and masks.
Generate them from whatever is in `data/`:

```bash
python scripts/build_cloud.py
```

| Data present | Time | Output |
|---|---:|---|
| the quick-start subset | ~20 s | ~60 MB |
| all eleven sources | ~4 min | ~1 GB |

It writes `data/cloud/features.npz` and, under
`src/components/nuclei_cloud/`, the `cols/`, `atlas/`, `masks/`,
`classes.bin` and `meta.json` that the browser component fetches. All of it is
gitignored, because it describes the data present on *this* machine.

**Re-run it whenever you add or remove a source.** Without these assets the
app skips the cloud and opens directly on the explorer.

Useful flags:

```bash
python scripts/build_cloud.py --limit 5000        # 5,000 nuclei per source, smoke test
python scripts/build_cloud.py --only CODEX HPA    # restrict to some sources
python scripts/build_cloud.py --meta-only         # rewrite metadata/colours only (~20 s)
```

### Optional: the latent layout

A third cloud layout, "ACP latente", places each nucleus by its embedding in
the SimCLR ResNet18 of `cellf-supervised`, instead of by its descriptors. It
needs `torch`, the sibling repository and one of its checkpoints:

```bash
pip install -r requirements-analysis.txt
python scripts/extract_embeddings.py      # ~13 min on an Apple GPU, 3.4 GB
python scripts/build_cloud.py --meta-only
```

The app never needs it. The layout only appears once the embeddings exist.

---

## Run

```bash
streamlit run src/app.py
```

Run it **from the repository root**. The data paths are relative, and the
modules in `src/` import each other by name. The app opens at
<http://localhost:8501>.

The dark theme is forced on purpose (`.streamlit/config.toml`): the cloud and
the logo paint on a fixed dark background.

### What you can do

- **Point cloud** (entry page): each nucleus is placed by two morphological
  descriptors of your choice, by a PCA, or by the latent layout.
  - Scroll to zoom, drag to pan. Once you are close enough, the points turn into the
    nuclei's thumbnails. Zoom out all the way to get back to the overview.
  - Hover a point to see the nucleus and its mask.
  - Click a point to open it in the explorer.
  - The **Sources** legend (bottom left) hides or shows whole datasets. Its
    `classe` mode colours one labelled dataset by cell type. Similar-looking
    types get close colours; this similarity comes from what is known of each
    cell type, not from the data.
- **Logo**: each of the six letters is a real nucleus. Hover a letter to find
  it in the cloud, click it to open it.
- **Explorer**:
  - Choose a source and a nucleus (or roll a random one).
  - Choose how the 3D volume is rebuilt: none, Gaussian or linear depth
    profile, or SAM3D; for RESTORE, the real slices or an interpolation.
  - Cut the volume with an arbitrary plane (azimuth, elevation, offset) and
    hide either half.
  - Compare with `cellaug`'s ObliqueSection, which simulates the same cut from
    the 2D crop alone.

> **Scales are not harmonised across sources.** Each one kept its native pixel
> size, and most of those sizes are unknown. The median nucleus diameter goes
> from 13 px (HelaCytoNuc) to 42 px (AitslabBioimaging1). This is deliberate:
> the gap between clusters in the cloud is the domain gap between
> acquisitions, not a display artefact.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pip install` fails on `cellaug` with `could not read Username` | No GitHub access to the private `cellaug`. See [Access](#access-two-private-dependencies). |
| Sidebar says *Aucune source de noyaux trouvée dans `data/`*, with no logo and no cloud, although the data is there | Streamlit was started from another directory, so the relative paths point nowhere. Run `streamlit run src/app.py` from the repository root. |
| Same message, run from the root | `data/crops/<SOURCE>/crops.npy` is missing. See [Data](#data). |
| The app opens on the explorer, with no cloud | The cloud assets were not built. Run `python scripts/build_cloud.py`. |
| The cloud shows a source you deleted, or misses one you added | The assets are stale. Re-run `python scripts/build_cloud.py`. |
| Clicking a logo letter shows *… n'est pas installé sur cette machine* | That letter's nucleus comes from a source you did not download. This is expected. |
| `ImportError` mentioning `pyarrow` | Custom Streamlit components need it: `pip install -r requirements.txt`. |

---

## SAM3D pipeline

> This part **does not run on a standard development machine**. It needs a
> GPU machine with a dedicated conda environment (`sam3d-engine`), the sibling
> repository `sam-3d-objects`, and paths currently hard-coded
> (`/home/arthur.chiron/…` in `src/sam3d_engine.py` and
> `scripts/train_sam3d.py`). Without it, the app works fully; only the
> "SAM3D" reconstruction profile is unavailable.

```bash
pip install -r requirements-sam3d.txt
```

**Inference.** The app's SAM3D button hands off to `src/run_inference.py`,
executed through `subprocess` in the separate conda environment (PyTorch3D
and Streamlit do not coexist). The script normalises the crop, derives a
binary mask, runs SAM3D, voxelises the resulting mesh to 64³, fills and
recentres it, then carries vertex intensities over to voxels with a KD-tree.

**Fine-tuning.** `scripts/train_sam3d.py` specialises SAM3D's `ss_generator`
on RESTORE (decoder and embedder frozen, Stage 0 dropped to free VRAM,
DataParallel, bfloat16 autocast, BCE loss on occupancy).

```bash
python scripts/train_sam3d.py --epochs 10 --batch_size 1 --lr 1e-5 --wandb
```

⚠️ `run_inference.py` currently loads the **base** SAM3D pipeline, not a
fine-tuned checkpoint: the two are not wired together yet.

---

## Repository layout

```
src/
  app.py              # Entry point: routes cloud / explorer, then orchestrates
  ui_components.py    # Streamlit widgets, volume selection, cloud view
  data.py             # Dataset loading and volume generation
  geometry.py         # Pure maths: cutting plane, clipping, voxel mesh, 2D slice
  cloud.py            # Pure maths: normalisation, PCA, class palette (build-time)
  augment.py          # Bridge to cellaug: app plane -> ObliqueSection
  visualization.py    # Plotly figure
  sam3d_engine.py     # Subprocess wrapper around the SAM3D conda env
  run_inference.py    # Standalone script run INSIDE the SAM3D conda env
  components/
    nuclei_cloud/     # Cloud component (plain HTML/canvas, no npm build);
                      #   its generated assets are gitignored
    logo/             # Wordmark component + its sprites (versioned, 15 kB)
scripts/
  preprocess_restore.py   # .ims + masks -> data/RESTORE/nuclei.npy
  build_cloud.py          # Descriptors, columns, atlases for the cloud
  extract_embeddings.py   # All nuclei through the SimCLR ResNet18 (optional)
  latent_probe.py         # Does cosine similarity separate nuclei, and where
  control_random.py       # Same test on an untrained ResNet18 (control)
  build_logo.py           # The "VoCell" wordmark, one real nucleus per letter
  train_sam3d.py          # SAM3D fine-tuning (GPU machine)
assets/                   # Logo renders (PNG, vector PDF)
```

Volumes are always `(Z, Y, X)`, `float32` normalised to `[0, 1]`, 64³.

See [CLAUDE.md](CLAUDE.md) for internal conventions and known pitfalls, and
[journal.md](journal.md) for the project history. Both are in French.
