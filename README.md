# VoCell — Augmentations 3D de noyaux cellulaires

Générer des **augmentations volumétriques 3D à partir d'images 2D de noyaux**.
L'idée fondatrice : reconstruire un pseudo-volume à partir d'une coupe 2D, puis le
trancher selon un plan arbitraire — ce qui produit des coupes inédites, et donc
enrichit un jeu de données biologique avec des vues qui n'existaient pas.

Le dépôt contient deux volets :

1. **Un explorateur volumétrique interactif** (Streamlit + Plotly) : chargement
   d'un noyau, reconstruction 3D, coupe selon un plan arbitraire, rendu voxel
   « Minecraft » et coupe 2D correspondante côte à côte. L'entrée se fait par un
   **nuage de points de tous les noyaux des deux datasets**, survolable et
   cliquable.
2. **Un pipeline SAM3D** : reconstruction 3D par IA à partir du crop 2D, avec un
   script de fine-tuning sur des volumes confocaux réels.

## Datasets

| Dataset | Emplacement | Nature | Échelle |
|---|---|---|---|
| **CODEX** | `data/CODEX/crops.npy`, `data/CODEX/crop_metadata.csv` | crops 2D ; le volume 3D est **synthétisé** (profil gaussien ou linéaire en profondeur) | ~0.377 µm/px |
| **RESTORE** | `data/RESTORE/nuclei.npy` | volumes 3D **réels** (confocal, canal DAPI), coupes Z éparses | ~0.15 µm/px |

`data/` n'est pas versionné. Le dossier doit être peuplé avant de lancer l'app.

### CODEX — lien vers `cellf-supervised`

Les crops CODEX sont déjà présents dans le dépôt voisin `cellf-supervised`, qui
les synchronise depuis Hugging Face. Plutôt que d'en dupliquer 692 Mo, on pointe
dessus par lien symbolique :

```bash
mkdir -p data && ln -s ../../cellf-supervised/Data/crops/CODEX data/CODEX
```

Le lien est relatif : il reste valide tant que les deux dépôts sont voisins dans
le même dossier parent. `data/` étant gitignoré, le lien n'est pas versionné —
c'est une étape d'installation à refaire sur chaque machine.

### RESTORE — à régénérer

RESTORE n'est pas distribué avec `cellf-supervised` : il se reconstruit depuis les
acquisitions confocales brutes, qui vivent dans `~/2021`.

```bash
ln -s ../../../2021 data/2021
python scripts/preprocess_restore.py
```

```bash
python scripts/preprocess_restore.py
```

Le script lit les paires `.ims` (Imaris/HDF5) + `_mask.npy`, isole chaque noyau,
et replace ses coupes à leur position **physique** en Z dans une grille 64³ — le
pas dépend du rapport entre résolution axiale et latérale lue dans les métadonnées
du fichier. Sortie : `(N, 64, 64, 64, 1)` en `uint8`, soit 842 Mo.

Le canal DAPI est repéré **par son nom** dans les métadonnées Imaris, car son
index varie d'une série à l'autre (voir [journal.md](journal.md)).

Sur le jeu actuel : 22 `.ims`, dont 20 avec masque, dont 18 exploitables
(un `.ims` et un masque sont corrompus, le script les saute) → **3366 noyaux**.

### Nuage de noyaux — à construire

Le nuage de points a besoin d'une table de descripteurs et d'atlas de vignettes,
tous deux dérivés des deux datasets ci-dessus :

```bash
python scripts/build_cloud.py
```

~20 s. Produit `data/cloud/features.npz` (6 Mo) et 2694 PNG dans
`src/components/nuclei_cloud/atlas/` (44 Mo, gitignorés). Sans eux l'application
démarre directement sur l'explorateur, sans le nuage.

## Lancer l'explorateur

```bash
pip install -r requirements.txt
```

```bash
streamlit run src/app.py
```

À lancer **depuis la racine du dépôt** (les chemins de données sont relatifs, et
les modules de `src/` s'importent à plat). L'interface s'ouvre sur
`http://localhost:8501`.

### Ce que l'interface permet

- **Nuage de noyaux** : les 172 358 noyaux des deux datasets placés par leurs
  descripteurs morphologiques (une paire d'axes au choix, ou les deux premières
  composantes d'une ACP). Survoler affiche le noyau, cliquer l'ouvre dans
  l'explorateur ci-dessous ; en zoomant, les points deviennent les vignettes.
- **Choix du noyau** : dataset, index, tirage aléatoire.
- **Profil de reconstruction 3D** — sur CODEX : aucun (coupe centrale seule),
  gaussien (σ réglable), linéaire (épaisseur réglable), ou SAM3D ; sur RESTORE :
  coupes réelles brutes, ou interpolation linéaire pour combler les vides en Z.
- **Plan de coupe arbitraire** : azimut, élévation, position, avec masquage
  optionnel de la moitié du volume au-dessus ou au-dessous du plan.
- **Rendu** : seuil d'intensité et opacité globale ; axes gradués en micromètres.

## Pipeline SAM3D

> Ce volet **ne tourne pas sur une machine de développement standard**. Il dépend
> d'une machine GPU avec un environnement conda dédié (`sam3d-engine`), du dépôt
> voisin `sam-3d-objects`, et de chemins actuellement codés en dur
> (`/home/arthur.chiron/…` dans `src/sam3d_engine.py` et
> `scripts/train_sam3d.py`).

```bash
pip install -r requirements-sam3d.txt
```

**Inférence.** Le bouton « Lancer la sculpture SAM3D » de l'interface délègue à
`src/run_inference.py`, exécuté par `subprocess` dans l'environnement conda séparé
— PyTorch3D et Streamlit ne cohabitant pas. Le script normalise le crop, en tire
un masque binaire, lance SAM3D, voxelise le mesh obtenu à 64³, le remplit et le
recentre, puis reporte l'intensité des sommets sur les voxels par KD-Tree.

**Fine-tuning.** `scripts/train_sam3d.py` spécialise le `ss_generator` de SAM3D
sur RESTORE (décodeur et embedder gelés, Stage 0 supprimé pour libérer de la
VRAM, DataParallel, autocast bfloat16, perte BCE sur l'occupancy).

```bash
python scripts/train_sam3d.py --epochs 10 --batch_size 1 --lr 1e-5 --wandb
```

⚠️ `run_inference.py` charge aujourd'hui le pipeline SAM3D **de base**, pas un
checkpoint fine-tuné : les deux ne sont pas encore reliés.

## Structure

```
src/
  app.py              # Point d'entrée : route nuage / explorateur, puis orchestre
  ui_components.py    # Widgets Streamlit + logique de choix du volume + vue nuage
  data.py             # Chargement des datasets et génération des volumes
  geometry.py         # Maths pures : plan, clipping, maillage voxel, coupe 2D
  cloud.py            # Maths pures : descripteurs → positions 2D du nuage
  augment.py          # Pont vers cellaug : plan de l'app → ObliqueSection
  visualization.py    # Construction de la figure Plotly
  sam3d_engine.py     # Wrapper subprocess vers l'env conda SAM3D
  run_inference.py    # Script autonome exécuté DANS l'env conda SAM3D
  components/nuclei_cloud/
    index.html        # Composant Streamlit du nuage (canvas, sans build npm)
    atlas/            # Vignettes générées, gitignorées
scripts/
  preprocess_restore.py
  build_cloud.py      # Descripteurs + atlas de vignettes du nuage
  train_sam3d.py
```

Les volumes circulent partout en `(Z, Y, X)`, `float32` normalisé `[0, 1]`, 64³.

Voir [CLAUDE.md](CLAUDE.md) pour les conventions internes et les pièges connus, et
[journal.md](journal.md) pour l'historique du projet.
