# VoCell — Augmentations 3D de noyaux cellulaires

Générer des **augmentations volumétriques 3D à partir d'images 2D de noyaux**.
L'idée fondatrice : reconstruire un pseudo-volume à partir d'une coupe 2D, puis le
trancher selon un plan arbitraire — ce qui produit des coupes inédites, et donc
enrichit un jeu de données biologique avec des vues qui n'existaient pas.

Le dépôt contient deux volets :

1. **Un explorateur volumétrique interactif** (Streamlit + Plotly) : chargement
   d'un noyau, reconstruction 3D, coupe selon un plan arbitraire, rendu voxel
   « Minecraft » et coupe 2D correspondante côte à côte. L'entrée se fait par un
   **nuage de points des 2 633 390 noyaux des onze sources**, survolable et
   cliquable.
2. **Un pipeline SAM3D** : reconstruction 3D par IA à partir du crop 2D, avec un
   script de fine-tuning sur des volumes confocaux réels.

## Datasets

Onze sources, **2 633 390 noyaux**. Dix viennent du dépôt voisin
`cellf-supervised` (crops 2D déjà segmentés et découpés), la onzième est propre
à VoCell.

| Source | N | Étiquettes | Nature |
|---|---:|---|---|
| **CODEX** | 168 992 | 14 familles cellulaires | CRC, Hoechst, ~0,376 µm/px |
| **HPA** | 673 936 | 17 lignées + non étiqueté | Human Protein Atlas |
| **BBBC051** | 232 429 | 11 types rénaux | crops natifs **32²**, ~0,5 µm/px |
| **TissueNet** | 1 335 905 | — | DAPI, multi-tissus |
| **HelaCytoNuc** | 146 910 | — | HeLa, DAPI |
| **DSB2018** | 32 155 | — | Data Science Bowl 2018 |
| **NuInSeg** | 29 332 | — | H&E, multi-organes |
| **S-BSST265** | 5 380 | — | DAPI |
| **NucleusSegData** | 3 250 | — | Huh7 / HepG2 |
| **AitslabBioimaging1** | 1 735 | — | U2OS, Hoechst |
| **RESTORE** | 3 366 | — | volumes 3D **réels** (confocal, DAPI), ~0,15 µm/px |

Les dix premières sont natives 2D : leur volume est **synthétisé** (profil
gaussien ou linéaire en profondeur, ou SAM3D). RESTORE est la seule à apporter
un vrai volume confocal, à coupes Z éparses.

> **Les échelles ne sont pas harmonisées.** `rescale_images()` est commenté
> dans le notebook de build de `cellf-supervised`, donc chaque source est
> restée à sa taille de pixel native — la plupart sont d'ailleurs inconnues.
> Mesuré sur les crops, le diamètre médian d'un noyau va de 13 px sur
> HelaCytoNuc à 42 px sur AitslabBioimaging1. C'est laissé tel quel
> délibérément : l'écart entre les amas du nuage est le fossé de domaine entre
> acquisitions, pas un artefact d'affichage.

`data/` n'est pas versionné. Le dossier doit être peuplé avant de lancer l'app.

### Les dix jeux de crops — un lien vers `cellf-supervised`

Ils sont déjà présents dans le dépôt voisin, qui les synchronise depuis Hugging
Face. Plutôt que d'en dupliquer 10 Go, on pointe dessus par un lien symbolique :

```bash
mkdir -p data && ln -s ../../cellf-supervised/Data/crops data/crops
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

~3,5 min. Produit `data/cloud/features.npz` (111 Mo), 12 colonnes de
descripteurs (63 Mo) et 41 147 vignettes (681 Mo) dans
`src/components/nuclei_cloud/`, le tout gitignoré. Sans eux l'application
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

- **Nuage de noyaux** : les 2 633 390 noyaux des onze sources placés par leurs
  descripteurs morphologiques (une paire d'axes au choix, ou les deux premières
  composantes d'une ACP). Survoler affiche le noyau, cliquer l'ouvre dans
  l'explorateur ci-dessous ; en zoomant, les points deviennent les vignettes.
  Dans la légende, chaque dataset est une « super-classe » qu'un clic masque
  entièrement ; les trois sources étiquetées se déplient en leurs classes.
- **Choix du noyau** : dataset, index, tirage aléatoire.
- **Profil de reconstruction 3D** — sur les dix jeux de crops : aucun (coupe
  centrale seule), gaussien (σ réglable), linéaire (épaisseur réglable), ou
  SAM3D ; sur RESTORE : coupes réelles brutes, ou interpolation linéaire pour
  combler les vides en Z.
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
  cloud.py            # Maths pures : normalisation + ACP (consommé par le build)
  augment.py          # Pont vers cellaug : plan de l'app → ObliqueSection
  visualization.py    # Construction de la figure Plotly
  sam3d_engine.py     # Wrapper subprocess vers l'env conda SAM3D
  run_inference.py    # Script autonome exécuté DANS l'env conda SAM3D
  components/nuclei_cloud/
    index.html        # Composant Streamlit du nuage (canvas, sans build npm)
    atlas/ cols/      # Vignettes et colonnes générées, gitignorées
scripts/
  preprocess_restore.py
  build_cloud.py      # Descripteurs + colonnes + atlas du nuage (11 sources)
  train_sam3d.py
```

Les volumes circulent partout en `(Z, Y, X)`, `float32` normalisé `[0, 1]`, 64³.

Voir [CLAUDE.md](CLAUDE.md) pour les conventions internes et les pièges connus, et
[journal.md](journal.md) pour l'historique du projet.
