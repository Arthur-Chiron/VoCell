# Journal de bord — VoCell

Chronologie du projet et carnet de sessions. Une entrée par session de travail
significative, la plus récente en haut.

---

## Historique reconstitué (depuis git)

### 2026-03-13 — `276b022` Initial commit: 3D Voxel Explorer App

Socle du projet : `app.py`, `data.py`, `geometry.py`, README, requirements.
Idée fondatrice : prendre un crop 2D de noyau CODEX, l'étaler en profondeur avec
un profil gaussien pour obtenir un pseudo-volume 64³, puis le trancher selon un
plan arbitraire (azimut / élévation / offset) afin de produire des coupes
inédites — donc des augmentations 3D à partir de données 2D.

Rendu « Minecraft » : chaque voxel au-dessus d'un seuil devient un cube, assemblé
en un unique `Mesh3d` Plotly (8 sommets + 12 triangles par voxel, indexation
vectorisée) plutôt qu'en milliers de traces.

### 2026-03-17 — `0aa6215` Class mapping et métadonnées

- `CLASS_MAPPING` : consolidation des ~24 labels CODEX bruts (sous-types T, macro-
  phages, vasculature…) en ~14 familles cohérentes.
- `load_metadata()` lit `crop_metadata.csv` pour afficher la classe du noyau.
- Profils de reconstruction élargis : gaussien (σ réglable), linéaire (épaisseur
  réglable), aucun (coupe centrale seule).

### 2026-05-21 — `4f809ca` Pipeline SAM3D + dataset RESTORE

Le gros tournant du projet : passage du volume **synthétique** au volume **réel**
puis **appris**.

- **Refactor** : `app.py` passe de monolithe à orchestrateur ; extraction de
  `ui_components.py` (widgets + logique de sélection du volume) et
  `visualization.py` (figure Plotly). `geometry.py` gagne `apply_clipping`
  vectorisé par broadcasting (au lieu de `np.mgrid`).
- **Dataset RESTORE** (`scripts/preprocess_restore.py`) : lecture de fichiers
  Imaris `.ims` (HDF5) + masques d'instances, extraction des voxel sizes physiques
  depuis les métadonnées, isolation noyau par noyau, normalisation percentile
  1–99, recadrage XY carré avec zoom arrière ×2, puis replacement des coupes Z
  réelles à leur position **physique** dans une grille 64³ (le pas Z dépend du
  rapport `res_z / res_xy`). Sortie `(N, 64, 64, 64, 1)` uint8.
  → conséquence : les volumes RESTORE sont **creux en Z** (coupes confocales
  éparses), d'où le mode d'interpolation linéaire optionnel côté app.
- **SAM3D en inférence** (`sam3d_engine.py` + `run_inference.py`) : l'app appelle
  un script autonome via `subprocess` dans un env conda séparé (PyTorch3D &
  co. étant incompatibles avec l'env Streamlit). Le script normalise le crop,
  produit un masque binaire, lance le pipeline SAM3D, récupère le mesh, le
  voxelise à 64³, le remplit (`.fill()`), le recentre, puis reporte l'intensité
  des sommets sur les voxels par KD-Tree, et lisse légèrement (σ=0.8).
- **Fine-tuning** (`scripts/train_sam3d.py`) : spécialisation du `ss_generator` de
  SAM3D sur RESTORE. Le décodeur et l'embedder sont gelés, le Stage 0 (depth) est
  supprimé pour libérer de la VRAM, DataParallel sur 2× L40S, autocast bfloat16,
  perte BCE sur l'occupancy 64³. Deux monkeypatches sur les normalizers de
  pointmap pour éviter les crashs sur masque vide, et un pointmap synthétique
  plan à z=5.0 fourni en entrée.

---

## 2026-09-21 — Reprise après 4 mois : état des lieux

Aucune modification de code. Lecture complète du dépôt, création de `CLAUDE.md`
et de ce journal.

**Constat** : le dépôt est cohérent et lisible, mais il est dans un état
« suspendu en plein milieu de la phase SAM3D ». Le code de fine-tuning a été
committé sans trace de résultat : aucun checkpoint, aucun log, aucune métrique,
et `run_inference.py` charge toujours le pipeline **de base** (`checkpoints/hf/
pipeline.yaml`) et non un poids fine-tuné. On ne peut donc pas savoir depuis le
dépôt seul si l'entraînement a tourné, ni ce qu'il a donné.

**Dette identifiée** (détail dans `CLAUDE.md`) :
- bug réel : `bs` utilisé avant affectation dans `train_sam3d.py:173` ;
- `requirements.txt` ne couvre que l'app (ni torch, ni h5py, ni trimesh…), rien
  d'épinglé ;
- chemins absolus `/home/arthur.chiron/…` dans deux fichiers, plus une résolution
  relative divergente dans `run_inference.py` ;
- `src/__pycache__/*.pyc` versionnés par inadvertance ;
- README obsolète : ne mentionne ni RESTORE, ni SAM3D, ni `scripts/` ;
- docstring mensonger en tête de `preprocess_restore.py` ;
- zéro test, zéro CI, pas de licence.

### Lot d'hygiène appliqué dans la foulée

- **Bug corrigé** : `bs` est désormais calculé juste après le collate
  (`train_sam3d.py:170`), avant ses deux usages. L'ancienne définition, placée
  plus bas, est supprimée. Les deux branches `pointmap_scale` / `pointmap_shift`
  ne lèvent plus de `NameError`.
- **Requirements scindés** : `requirements.txt` couvre l'app et le
  prétraitement (ajout de `h5py`, et `streamlit>=1.49` — borne réelle, imposée par
  l'usage de `width="stretch"` sur les boutons) ; `requirements-sam3d.txt`
  regroupe le volet GPU (`torch`, `trimesh`, `Pillow`, `tqdm`, `wandb`) avec la
  mention explicite de ce qui s'installe hors pip (`sam-3d-objects`, `pytorch3d`).
  Séparer les deux fichiers évite de laisser croire qu'un `pip install` unique
  suffirait — PyTorch3D ne cohabite pas avec l'env Streamlit.
- **`.pyc` dégagés** du suivi git, `.gitignore` étendu (`__pycache__/`, `*.py[cod]`,
  `.pytest_cache/`, `.DS_Store`).
- **README réécrit** : les deux volets, les deux datasets et leurs échelles, la
  régénération de RESTORE, la structure des modules, et l'avertissement explicite
  que le volet SAM3D n'est pas exécutable hors machine GPU.
- **Docstring corrigé** en tête de `preprocess_restore.py` (annonçait une liste de
  dicts, produit un array uint8 empilé).

Les fichiers modifiés compilent (`py_compile`). Rien n'a pu être exécuté pour de
vrai : `data/` est absent et Streamlit n'est pas installé localement.

**Non traité, volontairement** : l'épinglage des versions, faute de lockfile ou
d'environnement de référence permettant de savoir lesquelles ont réellement
tourné ; et l'uniformisation de l'API Streamlit (`use_container_width` →
`width`), qui touche au comportement du rendu et sort de l'hygiène.

### Données rebranchées

L'app plantait au démarrage (`FileNotFoundError: data/CODEX/crops.npy`). Les crops
CODEX existent déjà dans le dépôt voisin `cellf-supervised`
(`Data/crops/CODEX/`, synchronisé depuis Hugging Face) : 168 992 crops en
`(N, 64, 64)` uint8, plus `crop_metadata.csv` dont les colonnes `crop_index` et
`classes` correspondent exactement à ce qu'attend `data.py`.

Plutôt que de dupliquer 692 Mo, lien symbolique **relatif** :
`data/CODEX -> ../../cellf-supervised/Data/crops/CODEX`. Il reste valide tant que
les deux dépôts sont voisins. `data/` étant gitignoré, c'est une étape
d'installation à refaire par machine — documentée dans le README.

**Vérifié dans le navigateur** : l'app charge, affiche le noyau #29701 (classe
« B cells »), la reconstruction gaussienne σ=2.5 produit bien un volume arrondi,
et basculer l'élévation de 90° à 0° donne une coupe sagittale où le noyau
n'apparaît plus que comme une bande fine — cohérent avec un profil gaussien
étroit en Z. La chaîne `data → geometry → visualization` est donc saine.

### RESTORE régénéré — et un bug de canal au passage

Les acquisitions brutes sont dans `~/2021` (et non `~/Codes/2021`) : 373 Mo,
22 fichiers `.ims` dont 20 avec masque. Lien symbolique relatif
`data/2021 -> ../../../2021`, comme pour CODEX.

**Avant de lancer le prétraitement, vérification du canal — et bien m'en a pris.**
`preprocess_restore.py` avait `CHANNEL_IDX = 0` codé en dur pour le DAPI. Or dans
les 19 fichiers lisibles, le DAPI est **systématiquement au canal 1** ; le canal 0
est CD45 ou SHG selon la série. Le script aurait produit un dataset de « noyaux »
constitué de signal membranaire ou de second harmonique — silencieusement, sans
lever la moindre erreur, et le résultat aurait *ressemblé* à des noyaux de loin.

Confirmation quantitative sur `@3 CD34AL488…`, intensité moyenne à l'intérieur des
masques de noyaux :

| canal | marqueur | dedans | dehors |
|---|---|---:|---:|
| 0 | SHG | 26.9 | 0.28 |
| **1** | **Dapi** | **72.4** | 0.68 |
| 2 | CD34 | 8.1 | 1.17 |
| 3 | CD45 | 10.2 | 0.83 |
| 4 | PDGFRa | 0.7 | 0.17 |

Le DAPI a 2,7× l'intensité du SHG dans les masques — cohérent avec un marquage
nucléaire, alors que le SHG (collagène) n'a rien à faire dans un noyau. Les masques
ont manifestement été segmentés sur ce canal.

**Correction** : `find_dapi_channel()` repère le canal par son **nom** dans
`DataSetInfo/Channel N/Name`, avec repli sur le canal 1 et avertissement explicite
si aucun nom ne contient « dapi ». `CHANNEL_IDX` devient `CHANNEL_FALLBACK`.
L'index est désormais affiché pour chaque fichier traité.

**Résultat** : 3366 noyaux, `(3366, 64, 64, 64, 1)` uint8, 842 Mo. Deux fichiers
sautés proprement par le script — un `.ims` à signature HDF5 invalide et un
`_mask.npy` illisible comme pickle. Les volumes portent 4 à 20 coupes Z sur 64
(médiane 10), remplissage 1,6 % : c'est bien la structure creuse attendue d'une
pile confocale.

**Vérifié dans le navigateur** : onglet RESTORE, noyau #1591, 17 coupes natives
affichées en bande — on y voit le noyau grossir puis rétrécir à travers Z, la
signature d'une vraie pile confocale (et une confirmation visuelle de plus que le
canal est le bon). En mode « Aucune » le rendu 3D est strié par les coupes
manquantes ; en « Linéaire » le striping disparaît et le volume devient continu.

**Pistes pour la suite** (par ordre de coût croissant) :
1. Externaliser les chemins machine en variables d'environnement / fichier de
   config, pour que le volet SAM3D redevienne exécutable ailleurs.
2. Tests unitaires sur `geometry.py` (module pur : plan, clipping, maillage,
   extraction de coupe) — c'est le cœur mathématique et il est isolable.
3. Reprendre le fil SAM3D : décider si le fine-tuning est relancé, et si oui
   brancher `run_inference.py` sur un checkpoint spécialisé plutôt que sur le
   modèle de base.

---

## 2026-09-21 — `ObliqueSection` sortie en dépôt partagé

**Besoin** : tester dans VoCell l'augmentation `ObliqueSection` écrite dans
`cellf-supervised` (`Lib/data_loading/transforms.py`), en pouvant la modifier
depuis l'un ou l'autre projet sans qu'ils divergent.

**Écarté** : le lien symbolique (comme `data/CODEX`), qui marche localement mais
ne synchronise rien sur GitHub — git ne versionne que le lien. Écarté aussi le
copier-coller, qui garantit deux versions divergentes à la première retouche.

**Décidé** : un troisième dépôt, `cellaug`, installable par pip, consommé par les
deux projets. Une seule implémentation, aucun des deux dépôts n'en détient de
copie. En développement local, un clone unique installé en éditable
(`pip install -e ../cellaug`) dans les deux environnements : une modification est
visible des deux côtés instantanément, et un seul `git push` la publie.

**Fait** : extraction fidèle de `estimate_nucleus`, `estimate_nucleus_cov`,
`variable_blur`, `ObliqueSection` et `TorchObliqueSection`. Deux changements
seulement :

- torch devient **optionnel**. Seul le ré-ensemencement du RNG par worker de
  DataLoader l'utilisait ; sans torch, l'entropie vient de `os.urandom`. Le
  chemin torch est inchangé, donc le comportement côté cellf-supervised l'est
  aussi. VoCell n'a pas torch et n'a pas à l'installer.
- `TorchObliqueSection` est isolé dans `cellaug.torch_wrappers`, pour que
  `import cellaug` ne tire pas torch.

8 tests unitaires (forme, plage, `preserve_support`, `p=0`, multicanal,
anisotropie de l'enveloppe, atténuation nette) passent dans le venv de VoCell,
sans torch. Vérifié sur un vrai crop CODEX : la coupe oblique conserve 90 % de
l'intensité totale et le fond strictement nul.

**Reste à faire** : créer le dépôt sur GitHub, puis remplacer dans
`cellf-supervised` le bloc de `transforms.py` par un ré-export de `cellaug`
(les imports de `train_ssl.py` continuent de fonctionner tels quels). Tant que
ce dépôt n'existe pas, la ligne `cellaug @ git+…` de `requirements.txt` n'est pas
résolvable : seul le clone local éditable fonctionne.

---

## 2026-09-21 — ObliqueSection dans l'app : la 3e colonne

**Idée** : VoCell coupe *géométriquement* (construire un volume 64³, le trancher
selon un plan). `ObliqueSection` approxime la même opération directement en 2D,
sans jamais construire le volume. Les mettre côte à côte sur **le même plan**
répond à une question précise : qu'est-ce que l'augmentation qui entraîne SimCLR
reproduit, et qu'est-ce qu'elle rate ?

**Bloquant levé en amont** : `ObliqueSection.__call__` tire son plan au hasard.
Impossible de comparer, et un re-render Streamlit rejouait un plan différent à
chaque mouvement de curseur. Ajout dans `cellaug` d'un `apply(img, tilt_deg=,
phi=, offset=)` qui impose le plan — `__call__` et `apply` partagent tout le
rendu, seul le tirage diffère. Vérifié : à graine égale, `__call__` redonne bit
pour bit la sortie de l'implémentation d'origine, bruit de photons compris.
C'est le premier aller-retour du dépôt partagé, et il valide le montage : une
modification motivée par VoCell, écrite une fois, disponible des deux côtés.

**Conversion du plan** (`src/augment.py`, démonstration dans le docstring) : le
plan de l'app a pour normale `n = (cos el cos az, cos el sin az, sin el)` ;
résoudre en `z` donne exactement la carte de profondeur affine
d'`ObliqueSection`, avec `tilt = 90° − el`, `phi = az + 180°` et
`offset = slice_offset / sin(el)`. Vérifié dans le navigateur : élévation 65°
affiche bien « inclinaison 25,0° ».

**Calibration** : le bouton « 📐 Mesurer sur le volume » estime l'aplatissement
réel (demi-hauteur / rayon) du volume que la coupe géométrique tranche, au lieu
du 0,4 par défaut des cellules en culture. Sur un CODEX gaussien σ=2.5 il
retourne 0,34 — cohérent. Sur un volume « Aucune » (une seule coupe Z) il
retourne 0, borné à 0,1 : la comparaison n'a alors aucun sens, ce que le
résultat dit de lui-même.

**Vérifié dans le navigateur**, CODEX et RESTORE, les deux modes (plan suivi /
tirage aléatoire), sans erreur console ni serveur. Sur RESTORE l'entrée de
l'augmentation est la projection Z du volume — c'est ce qu'une acquisition 2D
du même noyau donnerait, donc l'entrée honnête pour une augmentation 2D.

**Trouvé au passage, non corrigé** : un plan neutre (`tilt=0, offset=0`) n'est
pas l'identité. Dans `_render`, `ratio` vaut `inf` dès que `h_local == 0`, sans
regarder la profondeur : l'anneau `rho ∈ [1, 1.25]` est assombri, et
complètement effacé en `rho = 1`. À l'entraînement c'est invisible — fond nul
masqué par `preserve_support`, et un plan exactement neutre n'est jamais tiré.
Ici, plan forcé sur un crop à fond non nul, ça coûte ~1 % du signal et ça
dessine un liseré. Le corriger change le comportement de l'entraînement de
cellf-supervised : décision à prendre, pas à prendre en douce.

**Reste à faire** : voir « Reste à faire » de l'entrée précédente (dépôt cellaug
à brancher dans cellf-supervised, **branche `simclr-arthur` uniquement**).
