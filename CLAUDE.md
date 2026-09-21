# CLAUDE.md — VoCell

Guide de travail pour Claude Code sur ce dépôt. Lire avant toute modification.

## Objectif du projet

Générer des **augmentations 3D de noyaux cellulaires** à partir d'images 2D.
Deux volets :

1. **Explorateur volumétrique** (Streamlit + Plotly) : charge un noyau, reconstruit
   un volume 64³, le coupe selon un plan arbitraire, affiche le rendu voxel
   « Minecraft » + la coupe 2D correspondante.
2. **Pipeline SAM3D** : reconstruction 3D par IA (Segment Anything 3D Objects),
   avec un script de fine-tuning sur le dataset RESTORE.

## Architecture

```
src/                       # Application Streamlit (imports plats, pas de package)
  app.py                   # Point d'entrée : orchestre sidebar → géométrie → rendu
  ui_components.py         # Tous les widgets Streamlit + logique de choix du volume
  data.py                  # Chargement datasets + génération des volumes 3D
  geometry.py              # Maths pures : plan de coupe, clipping, maillage voxel
  augment.py               # Pont vers cellaug : plan de l'app -> ObliqueSection
  visualization.py         # Construction de la figure Plotly
  sam3d_engine.py          # Wrapper subprocess vers l'env conda SAM3D
  run_inference.py         # Script autonome exécuté DANS l'env conda SAM3D
scripts/
  preprocess_restore.py    # .ims (Imaris/HDF5) + masques → data/RESTORE/nuclei.npy
  train_sam3d.py           # Fine-tuning du ss_generator de SAM3D sur RESTORE
```

**Flux de l'app** : `app.py` appelle `ui.*` pour obtenir `volume` + paramètres de
coupe → `geom.get_plane_vectors` → `geom.apply_clipping` → `geom.get_voxel_mesh_data`
→ `vis.create_3d_figure`, et en parallèle `geom.extract_2d_slice` pour la coupe.
Si le panneau ObliqueSection est activé, une troisième colonne applique la même
coupe *sans volume*, via `augment.plane_to_oblique` → `augment.apply_forced`.

### Conventions internes importantes

- **Les volumes sont toujours `(Z, Y, X)` en `float32` normalisé `[0, 1]`, shape 64³.**
  Le stockage sur disque de RESTORE est `(N, 64, 64, 64, 1)` en `uint8`.
- `src/` n'est **pas** un package : les modules s'importent à plat (`import data`).
  Cela ne fonctionne que parce que Streamlit ajoute le dossier du script au
  `sys.path`. Lancer impérativement depuis la racine : `streamlit run src/app.py`.
- **L'UI est en français**, les docstrings et commentaires du code sont en anglais
  (sauf `scripts/preprocess_restore.py`, en français). Conserver cet usage.
- Les chaînes de mode (`"Gaussien"`, `"Linéaire"`, `"Aucune"`, `"Tout afficher"`,
  `"Masquer au-dessus"`, …) sont **du français littéral servant de clés logiques**
  entre `ui_components.py`, `data.py` et `geometry.py`. Les renommer casse tout
  silencieusement — chercher toutes les occurrences avant de toucher.
- Le résultat SAM3D est mis en cache dans `st.session_state[f"ai_vol_{idx}"]`,
  pas dans `@st.cache_data`.

## Datasets

| Dataset | Chemin | Nature | Échelle |
|---|---|---|---|
| CODEX | `data/CODEX/crops.npy` + `crop_metadata.csv` | crops 2D, volume 3D **synthétique** (profil gaussien / linéaire) | ~0.377 µm/px |
| RESTORE | `data/RESTORE/nuclei.npy` | volumes 3D **réels** (confocal, canal DAPI), coupes Z éparses | ~0.15 µm/px |

`data/` est gitignoré, donc l'installation des données est à refaire par machine.

- **CODEX** est un **lien symbolique** vers le dépôt voisin, pour éviter de
  dupliquer 692 Mo :
  `ln -s ../../cellf-supervised/Data/crops/CODEX data/CODEX`
  (relatif : valide tant que `VoCell/` et `cellf-supervised/` sont voisins).
  168 992 crops, `(N, 64, 64)` uint8.
- **RESTORE** n'existe pas dans `cellf-supervised` : il se régénère depuis les
  acquisitions brutes de `~/2021`, également liées symboliquement
  (`ln -s ../../../2021 data/2021`), puis
  `python scripts/preprocess_restore.py` (~1 min, produit 842 Mo).
  3366 noyaux, `(N, 64, 64, 64, 1)` uint8, 4 à 20 coupes Z porteuses de signal
  sur 64 — d'où l'intérêt du mode d'interpolation linéaire dans l'app.
- **Le canal DAPI n'est pas à un index fixe** dans les `.ims` : il est au canal 1
  sur toute la série actuelle, le canal 0 étant CD45 ou SHG. `preprocess_restore.py`
  le repère par son **nom** dans `DataSetInfo/Channel N/Name`. Ne pas revenir à un
  index codé en dur.

`CLASS_MAPPING` dans `data.py` regroupe ~24 classes CODEX brutes en ~14 familles.

## Code partagé avec cellf-supervised

L'augmentation `ObliqueSection` (coupe oblique simulée : atténuation, flou
variable, voile, bruit de photons) **ne vit ni ici ni dans cellf-supervised**,
mais dans un dépôt tiers `cellaug` (https://github.com/Arthur-Chiron/cellaug),
installé par pip des deux côtés. Ne jamais en recopier le code dans `src/` : la
copie diverge dès la première retouche, c'est exactement ce que ce dépôt évite.

Deux entrées : `aug(img)` tire un plan au hasard (mode entraînement SimCLR),
`aug.apply(img, tilt_deg=, phi=, offset=)` impose le plan. **L'app n'utilise que
`apply()`** : un tirage se rejouerait à chaque re-render Streamlit, et surtout
c'est l'imposition du plan qui rend la comparaison avec la coupe géométrique
possible. `src/augment.py` fait la conversion (démonstration dans son
docstring) :

| plan de l'app | ObliqueSection |
|---|---|
| élévation `el` | `tilt_deg = 90 - el` |
| azimut `az` | `phi = az + 180°` |
| `slice_offset` (voxels) | `offset = slice_offset / sin(el) / H_half` (fraction) |

`H_half = aspect × R`, avec `R` le rayon estimé par cellaug. L'augmentation
raisonne en fraction de la demi-hauteur du noyau (~4 px), pas en voxels : le
curseur `slice_offset` (−40…40) sature donc très vite. C'est physique, pas un
bug.

⚠️ `cellf-supervised` est le dépôt de Thomas : **jamais de commit sur `main`**,
uniquement sur la branche `simclr-arthur`.

En développement, un seul clone local installé en éditable dans les deux
environnements — une modification est visible des deux côtés sans réinstaller :

```bash
git clone https://github.com/Arthur-Chiron/cellaug.git ../cellaug
.venv/bin/pip install -e ../cellaug
```

`import cellaug` ne tire pas torch (le wrapper tenseur est à part, dans
`cellaug.torch_wrappers`) : l'app Streamlit reste installable sans GPU.

## Lancer

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

`requirements.txt` = app + prétraitement (installable partout).
`requirements-sam3d.txt` = pipeline GPU, env conda séparé, à ne pas mélanger.

Le fine-tuning et l'inférence SAM3D **ne tournent pas sur cette machine** : ils
dépendent d'une machine GPU distante (chemins en dur `/home/arthur.chiron/…`,
env conda `sam3d-engine`, 2× L40S, dépôt `sam-3d-objects` voisin).

## Pièges connus (état au 2026-09-21)

- Chemins absolus en dur dans `sam3d_engine.py` et `train_sam3d.py`. De plus
  `run_inference.py` résout `sam-3d-objects` en **relatif** (`../../`) alors que
  `sam3d_engine.py` le passe en **absolu** — les deux peuvent diverger.
- `run_inference.py` charge le pipeline SAM3D **de base**, jamais un checkpoint
  fine-tuné : `train_sam3d.py` n'est relié à rien en aval.
- API Streamlit mélangée : `width="stretch"` (récent) cohabite avec
  `use_container_width=True` (déprécié ≥1.49). D'où le `streamlit>=1.49` dans
  `requirements.txt`.
- Aucune version épinglée dans les requirements (pas de lockfile ni d'env de
  référence pour en déduire lesquelles).
- Aucun test, aucune CI, aucune licence (côté VoCell ; `cellaug` a 13 tests).
- `cellaug` : discontinuité de l'atténuation sur le contour du noyau. Dans
  `_render`, `ratio` vaut `inf` dès que `h_local == 0`, **quelle que soit la
  profondeur** — donc même un plan neutre (`tilt=0, offset=0`) assombrit
  l'anneau `rho ∈ [1, 1.25]`, jusqu'à l'effacer complètement en `rho = 1`.
  Invisible à l'entraînement (le fond est nul et `preserve_support` le masque,
  et un plan exactement neutre n'est jamais tiré), visible dans VoCell dès
  qu'on force le plan neutre sur un crop à fond non nul : ~1 % du signal.
  Repéré le 2026-09-21, **non corrigé** : le corriger change le comportement
  de l'entraînement de cellf-supervised, c'est une décision à prendre.

Corrigés lors de la session du 2026-09-21 : bug `bs` dans `train_sam3d.py`,
`requirements.txt` incomplet, `.pyc` versionnés, README obsolète, docstring
erroné de `preprocess_restore.py`.

## Règles de travail

- Ne pas committer de données, checkpoints ou `.npy`.
- Toute nouvelle dépendance doit être ajoutée à `requirements.txt`.
- Garder la logique mathématique dans `geometry.py` (pure, testable, sans
  Streamlit) et n'importer `streamlit` que dans `app.py`, `ui_components.py`
  et `data.py`.
- Tenir `journal.md` à jour à chaque session de travail significative.
