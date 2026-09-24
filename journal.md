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

---

## 2026-09-21 — Un nuage de 172 358 noyaux comme porte d'entrée

**L'idée.** Remplacer « choisir un dataset, puis tirer un index au hasard » par
une vue d'ensemble : tous les noyaux des deux datasets dans un nuage de points,
survolable, cliquable, qui ouvre l'explorateur 3D existant sur le noyau choisi.

**La difficulté réelle était le survol.** Streamlit expose le clic et le lasso
d'un graphique (`st.plotly_chart(on_select=…)`) mais **pas le survol**, et un
survol qui coûte un aller-retour serveur n'en est pas un. D'où un composant
Streamlit maison — sans chaîne de build : `index.html` parle directement le
protocole `postMessage` (`componentReady`, `render`, `setComponentValue`,
`setFrameHeight`, `apiVersion: 1`). Les 172 358 points sont dessinés et
détectés dans le navigateur ; seul le clic remonte à Python.

**Choix de disposition** : descripteurs morphologiques, pas d'embedding appris.
Dix descripteurs calculés en NumPy pur (aire, élongation, intensité moyenne et
intégrée, contraste, netteté, concentration centrale, remplissage,
hétérogénéité, décentrage), avec au choix une paire d'axes lisible ou les deux
premières composantes d'une ACP. Le checkpoint SimCLR de `cellf-supervised`
aurait donné un nuage plus « sémantique », mais des axes qu'on ne peut pas
nommer — et une dépendance torch dans une app qui s'installe aujourd'hui sans
GPU.

**Échelles volontairement non harmonisées.** CODEX est à ~0,377 µm/px et
RESTORE à ~0,15 µm/px ; tout axe impliquant une taille sépare donc les deux
datasets. C'est visible immédiatement — RESTORE forme un amas compact en haut
à droite de (aire × intensité moyenne) — et c'est le point : ce décalage est le
fossé de domaine qu'un modèle entraîné sur l'un et montré à l'autre affronte.

### Ce que la mesure a corrigé

**Les vignettes étaient illisibles.** Un noyau CODEX occupe ~4 % d'un crop 64²
et culmine vers 130/255 : réduit à 32², c'était une tache noire. Trois
corrections, toutes identiques pour tous les noyaux donc sans casser la
comparaison : recadrage central 44×44 (mesuré : ≥99 % du signal conservé pour
99,5 % des noyaux des deux datasets), étirement p0,5/p99,8 plutôt que min-max,
et gamma 0,65.

**Les atlas étaient trop gros.** Premier jet : 1024 vignettes par planche de
1024². Mesuré en zoomant : 322 points visibles tiraient **144 planches** de
240 Ko, soit ~35 Mo et 600 Mo d'images décodées pour afficher 322 noyaux.
L'accès est aléatoire — deux voisins dans la projection ne sont pas voisins
dans l'index global — donc une zone zoomée touche à peu près autant de planches
qu'elle a de points. Planches ramenées à 8×8 (2694 fichiers de ~16 Ko), seuil
des vignettes à 600 points visibles, cache d'images borné en LRU.

**Deux bugs de transport.** Streamlit sert les fichiers d'un composant en
`Cache-Control: public`, donc une reconstruction d'atlas serait restée
invisible : `meta.json` est désormais lu en `no-store` et porte un `build`
accroché en `?v=` aux autres URL. Et le jeton qui distingue un nouveau clic
était un compteur, remis à 1 à chaque remontage de l'iframe — c'est-à-dire à
chaque retour au nuage : le premier clic suivant passait pour un doublon et
était ignoré. Horodatage à la place.

**Descripteurs dégénérés.** Quelques crops CODEX sont vides ; un masque de deux
pixels a un rapport d'axes de 8200, ce qui écrasait l'axe « élongation » pour
tout le monde. Les noyaux de moins de 4 px sont épinglés à des valeurs neutres.

### Vérifié dans le navigateur

Nuage complet (172 358 points), survol avec aperçu, zoom jusqu'à la mosaïque de
vignettes, clic → explorateur sur le bon noyau pour CODEX **et** RESTORE,
retour au nuage, re-clic immédiat, bascule ACP. Sans erreur console ni serveur.

**Trouvé au passage, non corrigé** : `CLASS_MAPPING` fait pointer
`'CD3+ T cells'` sur `'T Cells'` (C majuscule) là où les six autres entrées T
pointent sur `'T cells'`. 170 crops forment donc une quinzième famille fantôme,
que la légende du nuage rend évidente. Une ligne à changer, mais qui modifie la
classe affichée de ces crops : regroupement à trancher, pas coquille à corriger
en douce.

**Reste à faire** : la disposition par embedding SimCLR, en second mode à côté
des descripteurs — comparer « ce que voit le modèle » à « ce que dit la
morphologie » serait un résultat en soi, et le composant n'aurait pas à bouger.

---

## 2026-09-21 — Les onze sources dans le nuage, 2,6 M de noyaux

Suite directe de l'entrée précédente : le nuage ne montrait que CODEX et
RESTORE. Il couvre maintenant les dix jeux de crops de `cellf-supervised` plus
RESTORE, soit **2 633 390 noyaux** — quinze fois plus.

**Légende à deux niveaux.** Chaque dataset est une « super-classe » : un clic
sur sa ligne masque ou réaffiche toute la source d'un coup, le chevron déplie
ses classes pour les trois sources étiquetées (CODEX 14 familles, HPA 17
lignées, BBBC051 11 types rénaux). Les huit autres n'ont pas d'étiquette et
contribuent une pseudo-classe portant leur nom, pour que la légende ait la même
forme partout. Les couleurs de classes restent dans la famille de teinte de
leur dataset, de sorte que les groupes restent lisibles qu'on colore par
dataset ou par classe — un bouton bascule entre les deux, côté navigateur,
sans rerun.

**Explorateur généralisé.** Afficher onze sources mais n'en rendre que deux
cliquables aurait été incohérent. Les dix jeux de crops sont natifs 2D et
passent tous par la même reconstruction par profondeur synthétique : il n'y a
jamais rien eu de spécifique à CODEX dans le fait d'extruder un crop selon Z.
`get_codex_volume` devient `get_crop_volume(dataset, …)`, le sélecteur de
dataset passe de deux boutons radio à une liste de onze, et SAM3D accepte
n'importe quelle source.

### Un changement d'architecture imposé par la taille

À 172 k points, pousser les positions dans les arguments du composant coûtait
690 ko par changement d'axe. À 2,6 M, c'est 10 Mo sur le websocket à chaque
rerun. Chaque descripteur est donc **normalisé une fois au build et écrit comme
colonne uint16**, que le navigateur va chercher en HTTP et met en cache ;
Python n'envoie plus que deux noms de colonnes. Conséquence : `src/cloud.py` ne
sert plus qu'au build, l'app ne fait plus aucun calcul pour le nuage.

**Rendu.** Peindre 2,6 M de points coûte ~200 ms, et le survol redéclenchait ce
rendu à chaque noyau survolé. La couche de points est maintenant mise en cache
dans son `ImageData` et n'est invalidée que par la vue, les colonnes, les
filtres ou la taille du canvas : le survol se contente de reblitter et est
passé de 200 ms à **17 ms**. Pendant un glisser, le rendu est échantillonné à
~350 k points (44 ms) et la passe complète arrive 140 ms après l'arrêt.

**Atlas.** 41 147 planches au lieu de 2694, réparties en sous-dossiers de 1000
— 41 k fichiers dans un seul répertoire était intenable.

### Ce qu'il a fallu vérifier avant de s'y fier

**L'alignement des étiquettes HPA.** `dataset.py` de `cellf-supervised`
avertit qu'une colonne `slot` existe parce que certaines extractions rangent
les crops dans un ordre différent des métadonnées, « sans quoi chaque étiquette
est appariée au mauvais crop et les labels deviennent effectivement
aléatoires ». Or HPA n'a pas cette colonne ici. Test : avec les vraies
étiquettes, l'aire médiane d'un noyau varie d'un facteur **1,70** entre lignées
cellulaires ; avec les mêmes étiquettes mélangées, **1,08**. C'est aligné, les
lignes sont dans l'ordre des crops.

**Les échelles.** `rescale_images()` est commenté dans le notebook de build de
`cellf-supervised` : rien n'a été ramené à une taille de pixel commune, et
`sources.xlsx` ne donne la valeur que pour CODEX (0,376) et BBBC051 (0,5).
Mesuré sur les crops, le diamètre médian va de 13 px (HelaCytoNuc) à 42 px
(AitslabBioimaging1). Laissé tel quel, comme décidé pour CODEX/RESTORE : c'est
le fossé de domaine, et il saute aux yeux dès l'axe « aire ».

**BBBC051 est en 32²**, seul de son espèce, et n'a jamais été rééchantillonné.
Complété par du noir jusqu'à 64² plutôt que redimensionné : le padding préserve
sa taille de pixel, l'agrandissement doublerait le diamètre apparent de chaque
noyau et inventerait une différence absente des données.

**Seuil de discrétude du jitter.** Il était relatif à N (`uniques > N/20`), ce
qui faisait lire la même colonne `area` comme continue à 30 k noyaux et
discrète à 2,6 M. Remplacé par un plafond absolu : un descripteur compté en
pixels d'un crop 64² ne peut pas prendre plus de 4096 valeurs, quel que soit le
nombre de noyaux.

### Vérifié dans le navigateur

Nuage complet des onze sources, survol avec aperçu, dépliage de HPA en ses 18
classes, masquage d'un dataset entier d'un clic (2 633 390 → 1 959 454,
l'écart exact de HPA), bascule couleur dataset / classe, clic vers
l'explorateur sur TissueNet puis BBBC051, panneau ObliqueSection sur BBBC051.
Sans erreur console ni serveur.

**Reste à faire** : la disposition par embedding SimCLR, toujours en second
mode à côté des descripteurs — l'intérêt grandit avec onze sources, puisque
c'est exactement l'espace où `cellf-supervised` espère que les datasets se
recollent.

---

## 2026-09-21 — Le masque de segmentation au survol

Demandé : afficher le masque de segmentation à côté du noyau dans l'aperçu.

**Il n'y avait rien à segmenter.** `generate_crops` de `cellf-supervised` fait
`img_crop * (mask_crop == row["label"])` : les crops arrivent déjà masqués, et
leur support non nul **est** le masque du noyau. Pas de seuillage à inventer,
pas d'heuristique à régler — `crop > 0`, exactement.

Vérifié plutôt que supposé, sur un échantillon de chaque source : la médiane du
nombre de composantes connexes vaut 1 partout, **sauf HPA**, à 8, avec 40 % des
crops dont le support touche le bord. C'est exactement ce que documente
`clean_hpa_support.py` : HPA est le seul des dix à n'être jamais passé par un
masque de noyau, son fond nul est un seuil d'intensité. L'aperçu écrit donc
« support (seuil) » pour HPA et « masque » pour les dix autres — la différence
est visible à l'œil dans le panneau, le support HPA est franchement déchiqueté
là où les autres donnent un contour net.

**Binarisé avant rééchantillonnage**, pas après : le filtre boîte rend non nul
tout pixel de bord partiellement couvert, donc seuiller la vignette finie
dilaterait le masque d'un pixel. Binariser en pleine résolution puis couper à
0,5 de couverture remet la frontière où elle était.

**Planches séparées plutôt que canal alpha des vignettes.** La mosaïque zoomée
n'a jamais besoin du masque, et l'aperçu en veut exactement un, qui pèse ~3 ko.
Coût total 111 Mo, soit 16 % de l'atlas d'images — les masques binaires se
compressent très bien.

**Attrapé en testant** : le `maskCache` n'avait pas été ajouté à l'objet d'état.
Le remplacement textuel visait une ligne de l'ancienne version du composant et
n'a rien matché, sans erreur — c'est le genre de patch qu'il faut asserter.

**Suite, même session** : le masque est aussi dans l'explorateur, à côté du
crop dans le panneau du haut — et sur RESTORE, une bande de masques sous la
bande de coupes réelles. Les deux vues lisent `data.THRESHOLD_SUPPORT`, le
nuage via le `support` écrit dans `meta.json` au build : elles ne peuvent pas
diverger sur qui a un vrai masque.

Sur HPA l'intérêt saute aux yeux : le support affiché montre les mouchetures
détachées du noyau principal, celles-là mêmes que `clean_hpa_support.py`
cherche à retirer. C'est la première fois qu'on les voit sans les chercher.

**Vérifié au passage, ce n'est pas une régression** : sur RESTORE #26 la coupe
2D est noire en mode « Aucune ». Ses 12 coupes porteuses vont de z=15 à z=48
mais aucune ne tombe sur z=32, là où le plan par défaut tranche. En
« Linéaire » la même coupe vaut 10 905. C'est exactement le comportement des Z
épars, et la raison d'être du mode d'interpolation.

---

## 2026-09-22 — Déploiement Streamlit Cloud : abandonné, et pourquoi

Tentative de mise en ligne sur Streamlit Community Cloud (tier gratuit), pour
rendre le nuage complet accessible publiquement. **Abandonnée**, pas pour un
bug mais pour un coût.

**Le premier mur, le seul qui soit simple** : l'installateur échoue sur
`cellaug @ git+https://github.com/Arthur-Chiron/cellaug.git@main` avec
« could not read Username for 'https://github.com' ». Streamlit clone le dépôt
de l'app avec son propre token, mais pip et uv lancent `git fetch` dans un
sous-processus qui n'hérite d'aucune credential, et `cellaug` est privé. Un
`gh repo edit --visibility public` suffirait. Non fait : ça n'a d'intérêt que
si le reste passe, et le reste ne passe pas.

**Le vrai mur, c'est le volume.** Un clone frais n'a aucune donnée :
`data/` est un lien symbolique gitignoré vers `cellf-supervised`, et les assets
du nuage le sont aussi. Donc `cloud_available()` renvoie `False`, l'app tombe
sur l'explorateur, et `load_crops('CODEX')` lève `FileNotFoundError` au premier
render. Committer de quoi la faire vivre veut dire 732 Mo d'atlas + 202 Mo de
masques + 60 Mo de colonnes pour le seul nuage, 842 Mo pour RESTORE, ~10 Go de
crops — contre ~1 Go de RAM et un clone complet du dépôt à chaque démarrage.

**Ce qui était voulu, c'est le nuage complet en public** : les 2,6 M de points,
pas une démo. Les contournements possibles y renoncent tous — sous-ensemble de
quelques milliers de noyaux, ou assets hébergés ailleurs (S3, HF Datasets) avec
un composant qui irait chercher ses `.bin` hors de l'arbre servi par Streamlit.
Le premier trahit l'objectif, le second est du travail réel pour un résultat
qui reste bridé par la RAM. Tout le reste se paie.

**Décision : on ne déploie pas, et on ne paie pas non plus — pour l'instant.**
Aucune modification retenue, dépôt inchangé. `cellaug` reste privé.

À noter si la question revient : Streamlit Cloud a servi un environnement
Python 3.14.7, assez récent pour que `scipy` et `h5py` n'aient pas forcément de
wheel. Épingler 3.12 dans les *Advanced settings* de l'app.

---

## 2026-09-22 — Un logo dont chaque lettre est un vrai noyau

`scripts/build_logo.py`. Écrire « VoCell » avec six noyaux réels, chacun étant
celui qui, **à rotation près**, ressemble le plus à sa lettre parmi les
2 633 390 des onze sources. Sorties dans `assets/` (quatre PNG + le JSON de
provenance), candidats dans `data/logo/candidates.json`.

### Le cadre canonique, et pourquoi le rayon et pas la boîte

Chaque silhouette — noyau ou glyphe — est recentrée sur son barycentre et mise
à l'échelle pour que son **rayon quadratique moyen** vaille `R0`. Ce choix
n'est pas cosmétique : une boîte englobante grandit d'un facteur allant
jusqu'à √2 quand la forme tourne, le rayon RMS ne bouge pas. C'est la seule
normalisation qui laisse une recherche sur les orientations comparer des
choses comparables. Elle est isotrope, donc l'allongement du `l` est conservé
au lieu d'être écrasé.

**C'est le gabarit qui tourne, pas les noyaux.** 5 glyphes × 72 angles × 2
chiralités = 720 gabarits calculés une fois ; le score d'un lot de noyaux est
alors un produit matriciel `noyaux @ gabarits.T`. Le scan complet des 2,6 M
prend **90 s**, là où une rotation par noyau en prendrait des jours. Le miroir
est admis : un crop n'a pas de chiralité, seule la lettre doit sortir à
l'endroit.

### L'IoU seule classe mal, et ça se voit

Premier scan à l'IoU pure : le meilleur « V » était un **coin plein**
(IoU 0,693) devant un noyau réellement en V (0,669). C'est logique et c'est
quand même faux — un coin plein couvre tous les pixels du V et ne paie que le
creux, qui pèse peu. Pareil pour le `o`, où un disque plein tenait la tête.

D'où le score retenu :

```
score = IoU(silhouette, glyphe) − ½ · (remplissage des contrepoinçons)
```

le **contrepoinçon** étant l'enveloppe convexe du glyphe moins le glyphe : le
creux du V, le trou du o, l'ouverture du C, l'œil du e. Mesurés dans le cadre
32² : 61 px pour le V, 52 pour le o, 102 pour le C, 56 pour le e, **0 pour le
`l`** — qui est convexe, donc classé à l'IoU pure, ce qui est correct. Le
calcul reste linéaire en la silhouette, donc toujours un seul produit
matriciel (deux, avec les contrepoinçons). Après ce changement, le classement
et l'œil sont d'accord sur les cinq lettres.

### Ce que la taille du balayage a changé

Rodage sur 20 000 noyaux par source (154 k au total) : les meilleurs `o` et
`e` étaient des **blobs ronds**, indiscernables l'un de l'autre. Sur les
2,6 M, le `o` est un véritable anneau percé et le `e` a son œil et sa barre.
Autrement dit, le nombre de noyaux n'était pas un luxe ici : ces formes-là
existent à une fréquence de l'ordre de 10⁻⁶.

### Deuxième passe : les vérifications

Les 600 meilleurs par lettre sont repris à 64² et 360 angles, et rejetés si :

- la plus grosse composante connexe fait moins de 90 % du masque — un support
  fragmenté (HPA, dont le fond nul est un seuil d'intensité, pas un masque)
  peut imiter n'importe quelle lettre par accident ;
- le masque touche le bord de son crop — le contour est alors celui de la
  découpe, pas celui de la cellule.

Entre 378 et 572 des 600 passent, selon la lettre.

### Le placement est typographique, la pose ne l'est pas

La pose — angle et miroir — est exactement celle qui a gagné. **L'échelle et
le centrage, non** : ils sont repris sur la boîte englobante du glyphe dans le
mot (moyenne géométrique des deux rapports, centres alignés). Caler six
lettres sur une ligne de base à hauteur commune n'est pas le même problème que
comparer des formes à rotation près, et le rayon RMS donnait un `V` qui
dépassait le `C` d'un bon dixième de cadratin.

### Résultat

| Lettre | Source | Index | score | IoU | Pose |
|---|---|---:|---:|---:|---|
| V | NuInSeg | 13217 | 0,548 | 0,603 | 119° |
| o | BBBC051 | 43702 | 0,656 | 0,787 | 137° |
| C | HPA | 563187 | 0,684 | 0,744 | 313° |
| e | HPA | 630088 | 0,528 | 0,766 | 186°, miroir |
| l | TissueNet | 507283 | 0,976 | 0,976 | 0° |
| l | TissueNet | 540224 | 0,900 | 0,900 | 1°, miroir |

Les deux `l` sont deux noyaux différents — le même crop deux fois serait un
copier-coller, et le deuxième `l` est un noyau à part entière. Quatre sources
sur onze fournissent les six lettres ; `logo_vocell_sources.png` colore chaque
lettre de la teinte de son dataset, ce qui fait du logo sa propre légende.

Un détail de résolution à ne pas prendre pour une incohérence : le score
grossier du `l` vaut 0,817 contre 0,976 après la deuxième passe. Dans le cadre
32², une barre aussi fine ne fait que 3 px de large et l'échantillonnage la
pénalise ; à 64² elle en fait 7.

---

## 2026-09-22 — La similarité cosinus, et où elle veut dire quelque chose

Question de départ : peut-on séparer les noyaux par leur similarité cosinus,
et faut-il pour ça un espace latent « bien formé » ? La question contient une
inversion qu'il fallait défaire d'abord — **le cosinus est une métrique, pas
une méthode**. Il se calcule dans n'importe quel espace vectoriel. Ce qui n'est
pas garanti, c'est que l'angle veuille dire quoi que ce soit. Trois scripts
(`latent_probe.py`, `extract_embeddings.py`, `control_random.py`) mesurent
exactement ça.

**Dans l'espace des descripteurs, l'angle ne dit presque rien.** Les dix
colonnes sont toutes positives, donc tous les vecteurs vivent dans le même
orthant : la similarité cosinus entre deux noyaux au hasard a une médiane de
+0,983 et une étendue p1–p99 de 0,18. Pire, `area` porte 78,6 % de la norme et
`total_intensity` 21,1 % : **99,7 % pour deux colonnes sur dix**, parce que
l'aire se compte en centaines de px² et la netteté vaut 0,029. Un cosinus sur
les descripteurs bruts mesure le rapport `total_intensity / area`, c'est-à-dire
l'intensité moyenne, et ignore les huit autres.

Ce n'est pas un défaut du cosinus, c'est que l'angle exige des axes
commensurables. Vérifié directement : multiplier par 10 l'unité d'**un seul**
descripteur (px² → 0,1 px², un changement d'unité, pas de données) fait perdre
jusqu'à 65 % des dix plus proches voisins — 34,6 % survivent pour `off_center`,
40,5 % pour `elongation`. L'euclidien sur robust-z est invariant par
construction, il divise par l'IQR.

**Les embeddings.** `extract_embeddings.py` passe les 2 633 390 noyaux dans le
ResNet18 SimCLR de `cellf-supervised` (`simclr_resnet18_outHPA_augcell.pt`) et
écrit `h` (512) et `z` (128) en float16, dans l'ordre global du nuage. 797 s
sur le GPU Apple, ~3 300 noyaux/s. Prétraitement calqué sur l'eval du dépôt
voisin (`x/127,5 − 1`), BBBC051 complété plutôt que redimensionné et RESTORE
projeté selon Z, comme partout ailleurs ici.

**Ce que le latent change, et ce qu'il ne change pas.** L'angle y est enfin
bien défini — les 512 dimensions sortent du même ReLU. Et l'information y est
bien dans la direction : ×2,94 le hasard pour la direction seule contre ×1,28
pour la norme seule, cette norme corrélant −0,33 avec l'aire. Le cosinus y
écarte donc une nuisance de taille, exactement ce qu'il ne faisait pas sur les
descripteurs bruts où la norme *était* le signal.

Mais pour le reste, non :

| accord des 10 voisins | CODEX | HPA | BBBC051 |
|---|---:|---:|---:|
| descripteurs eucl./robust-z | ×1,52 | ×2,70 | ×2,25 |
| h cosinus | ×1,44 | ×2,88 | ×2,14 |
| z cosinus (← la loss) | ×1,45 | ×2,72 | ×2,13 |

Égalité. Un ResNet18 fait jeu égal avec dix nombres calculés à la main. Et
surtout **cosinus ≈ euclidien partout**, y compris dans `z` où InfoNCE est
littéralement défini sur le cosinus (0,175 contre 0,174). À k=10, la métrique
ne décide rien ; c'est la représentation qui décide. L'effet source ne
disparaît pas non plus : ×7,06 sur les descripteurs, ×5,97 sur `h`.

**Le contrôle qui manquait.** Un convnet non entraîné est déjà une projection
aléatoire correcte d'une image, donc « ×2,9 le hasard » ne veut rien dire tout
seul. `control_random.py` : non entraîné → entraîné, CODEX ×1,36 → ×1,44,
BBBC051 ×1,90 → ×2,17, **HPA ×1,57 → ×2,99**. L'entraînement a acheté quelque
chose de réel, mais presque uniquement sur HPA — qui était hors du
pré-entraînement SSL (checkpoint `outHPA`), donc c'est du vrai transfert.

Et un chiffre inattendu : **338 des 512 dimensions de `h` sont identiquement
nulles** sur les 2,6 M de noyaux, contre 0 sur le même réseau non entraîné. Il
reste 174 dimensions vivantes, dont 150 portent 95 % de la variance.
Effondrement partiel classique d'un SimCLR sous-entraîné — c'est ce qui borne
ce checkpoint, et pourquoi CP1 latente ne pèse que 12 % de variance.

**Dans l'app.** Troisième disposition « ACP latente », chaque embedding ramené
à une longueur de 1 avant l'ACP (c'est la direction qui porte, cf. ci-dessus),
ACP en deux passes sur la mémoire mappée puisque 2,6 M × 512 en float32 fait
5,4 Go alors que la covariance qu'elle alimente est 512 × 512. **Zéro ligne de
JS** : le composant allait déjà chercher `cols/<clé>.bin` et lisait
`meta.axes[clé]`, il suffisait d'ajouter deux colonnes et deux légendes. Les
axes sont nommés par le descripteur avec lequel ils corrèlent le plus — un axe
latent n'a ni unité ni nom, c'est la seule prise honnête. CP1 latente suit la
netteté du contour (r = +0,69).

Deux bugs attrapés au passage :

- **`load_cloud_meta` était `@st.cache_data` sur le seul chemin du fichier.**
  `build_cloud.py` réécrit `meta.json` sur place : une reconstruction restait
  donc invisible jusqu'au redémarrage du serveur, et le nouveau mode
  n'apparaissait pas. C'est le même trou que côté composant (`Cache-Control:
  public`), déjà bouché là-bas par un fetch `no-store` et un identifiant de
  build. Le cache est maintenant clé sur le mtime.
- **Les indices d'un memmap trié, les étiquettes non triées.** Dans la première
  version de `latent_probe.py`, la fonction qui lit les trois espaces triait
  ses indices pour l'accès mmap pendant que les étiquettes venaient de l'ordre
  d'origine. Features et labels décorrélés en silence, HPA tombait à ×1,00.
  Repéré uniquement parce que la ligne de référence sur les descripteurs ne
  reproduisait plus la mesure de la veille — d'où le `assert` sur des indices
  strictement croissants, et le fait de garder cette ligne de référence dans
  le script.

**Conclusion** : oui, séparer par cosinus est légitime dans ce latent et ne
l'était pas sur les descripteurs. Mais légitime n'est pas meilleur. La
disposition latente est une seconde lecture des mêmes noyaux, pas une lecture
plus juste — et la légende sous le nuage le dit.

---

## 2026-09-22 — Le logo en en-tête, voxelisé et cliquable

Le mot-symbole passe en haut de l'app, au-dessus des deux vues, dans sa
version colorée par dataset. Nouveau composant `src/components/logo/`, sur le
même protocole `postMessage` que le nuage et sans plus de chaîne de build.

### Voxeliser, et sur quelle grille

Les lettres sont rendues en blocs, en écho au rendu « Minecraft » de
l'explorateur. Deux décisions :

- **Un voxel est allumé ou éteint** — seuil à 0,5 sur l'alpha moyennée dans la
  cellule. Garder l'alpha continue frangerait chaque bord de cellules à moitié
  allumées, c'est-à-dire perdrait exactement l'effet recherché. La *couleur*,
  elle, reste la moyenne pondérée par l'alpha : c'est la texture du noyau.
- **La grille est celle du mot entier**, pas d'une lettre. Quantifier chaque
  lettre dans sa propre boîte donnerait six tailles de pixel différentes dans
  le même mot. 171 × 41 voxels pour « VoCell ».

`scripts/build_logo.py --render` écrit `src/components/logo/letters.json` :
six sprites RGBA à un pixel par voxel, **en data URI dans le document**, avec
la provenance de chacun. 15 ko.

C'est versionné, contrairement aux assets du nuage, et c'est délibéré : à
cette taille le logo fonctionne sur un clone frais sans la moindre donnée.
Accessoirement ça contourne le piège du `Cache-Control: public` relevé pour le
nuage — un sprite servi comme fichier séparé survivrait invisiblement à une
reconstruction, alors que le document qui les porte tous est lu `no-store`.

### Survol et clic

Même contrat que le nuage : survoler nomme le noyau, cliquer l'ouvre dans
l'explorateur. La bascule est factorisée dans `ui._open_in_explorer()`, que le
nuage utilise maintenant aussi. Le jeton de clic est un horodatage et pas un
compteur, pour la raison déjà consignée : l'iframe est remontée à chaque
changement de vue.

**Le test de survol lit l'alpha, pas la boîte englobante.** Les boîtes du `V`
et du `o` se chevauchent — le `V` est posé de travers — donc tester le
rectangle attribuerait des pixels du `o` au `V`. Un voxel de marge est ajouté
pour que le `l`, large de 9 voxels, reste facile à attraper. Vérifié dans le
navigateur sur les quatre formes : le centre du `C` **ne** répond pas, ce qui
est correct — c'est son contrepoinçon, il n'y a pas de noyau là.

La légende sous le mot remplace l'infobulle : l'iframe ne fait que 190 px de
haut et une bulle flottante y serait coupée. Elle affiche la vignette du
noyau dans son orientation d'origine, sa source, son score et sa rotation.

### Vérifié dans le navigateur

Survol des six lettres, clic sur le `C` → l'explorateur s'ouvre bien sur
HPA #563187, celui de la planche de provenance. Retour au nuage, dé
d'index, changement de largeur : pas de rebond parasite, le garde par nonce
tient.

Un défaut connu, qui n'est pas dans ce code : redimensionner la fenêtre **sans
déclencher de rerun** laisse Streamlit avec une largeur d'iframe périmée, et
le logo déborde sa colonne jusqu'au rerun suivant. Rien à l'intérieur de
l'iframe ne peut connaître la largeur réelle du conteneur ; la moindre
interaction corrige.

---

## 2026-09-22 — La page d'accueil rendue au nuage

Trois reproches sur la vue d'entrée, tous justes : trop de texte, un nuage qui
n'occupait qu'un tiers de l'écran, et un carré de points qu'on pouvait pousser
hors du cadre à la molette.

### Le texte

Supprimés : le titre « Nuage de noyaux », le paragraphe de mode d'emploi sous
le titre, la note sur les descripteurs en pixels, la note sur la projection
latente, et la phrase permanente du logo (« Chaque lettre est un vrai
noyau… »). Rien de tout cela n'est perdu : c'est consigné ici et dans
`CLAUDE.md`, qui sont les endroits où ça se lit vraiment. La légende du logo
ne s'écrit plus qu'au survol, et le nuage garde ses panneaux — sources, axes,
compteur — dans le canvas, là où ils appartiennent.

### La place

La hauteur d'un composant Streamlit se fixe en pixels, par `setFrameHeight` :
un iframe ne peut pas suivre la fenêtre tout seul. Le contournement est une
règle CSS injectée dans la page hôte —
`iframe[title*="nuclei_cloud"] { height: calc(100vh - Npx) !important }` — le
`!important` l'emportant sur le style en ligne que Streamlit écrit. Côté
composant, la ligne `document.body.style.height = height + "px"` a dû
disparaître : elle se battait avec `html, body { height: 100% }` et
désynchronisait le canvas de son iframe. La hauteur passée depuis Python reste
le repli si jamais le sélecteur cesse de correspondre.

**La barre d'outils de Streamlit (Deploy, menu) est `position: absolute` à
z-index 999990** : elle ne prend aucune place dans la mise en page mais peint
par-dessus ce qui passe dessous. Premier jet raté — le mot-symbole était
tronqué de ses 41 px du haut. Le `padding-top` du bloc est le seul levier, et
il doit valoir au moins la hauteur de la barre (60 px, mesurés) ; cette même
constante entre dans la réserve verticale, pour que dégager la barre ne puisse
pas faire déborder la ligne de contrôles en bas.

Les contrôles (disposition, axes, dispersion) sont maintenant **sous** le
nuage. Ils sont écrits avant lui dans le code — le composant a besoin de leurs
valeurs — mais rendus après, via deux `st.container()` déclarés dans l'ordre
inverse. La légende des sources est descendue en bas à gauche du canvas,
repliée par défaut et à 45 % d'opacité jusqu'au survol ; le compteur est monté
en haut à gauche, lui aussi estompé.

### L'ellipse

Les deux colonnes sont normalisées dans [0,1] chacune : seules, elles
remplissent un carré, et 2,6 M de points dans un carré se lisent comme une
affiche, pas comme un nuage. Le mapping de grille elliptique envoie le carré
unité sur le disque unité :

    u' = u √(1 − v²/2),   v' = v √(1 − u²/2)

puis un étirement à `ASPECT = 1,7` pour remplir un panneau large. C'est une
bijection continue qui déplace beaucoup moins l'intérieur que les coins : les
amas gardent leurs positions relatives, les quatre coins cessent de se faire
passer pour de la donnée. **C'est bel et bien une déformation des axes** —
l'aperçu au survol cite donc les valeurs brutes des colonnes, jamais les
positions elliptiques. Les tableaux transformés sont construits à part ; les
colonnes en cache restent intactes.

### Le cadrage

`fitK()` est le zoom qui inscrit exactement l'ellipse, marge comprise, et il
sert aussi de plancher. `clampView()`, appelé après chaque glisser, chaque
molette et chaque redimensionnement, verrouille le centre sur l'axe dont
l'étendue tient entièrement dans la vue et le borne sur l'autre. Conséquence :
dézoomer ne fait plus flotter le nuage dans le vide, et aucun glisser ne peut
le sortir du panneau.

### Vérifié dans le navigateur

Vue d'ensemble, molette dans les deux sens jusqu'aux butées, glissers vers les
quatre bords, zoom jusqu'aux vignettes, survol (aperçu noyau + masque),
légende dépliée puis repliée, bascule « ACP latente », clic sur un point →
l'explorateur s'ouvre sur TissueNet #1315619. À 800×600 comme à 1440×900, la
ligne de contrôles tombe entière en bas de la fenêtre.

---

## 2026-09-22 — Les logos en PDF vectoriel, et l'artefact de conflation

`--render` sort maintenant six `assets/logo_vocell_voxel*.pdf` en plus des PNG
lisses. **Vectoriels** : un voxel = un rectangle rempli. Un logo finit sur une
affiche autant que dans une slide, et seul le vectoriel garde des arêtes
franches à toute taille — un bitmap agrandi flouterait, ou laisserait chaque
lecteur appliquer sa propre idée du pixel.

Émetteur PDF écrit à la main, une quarantaine de lignes : catalogue, page,
flux comprimé de `re f`. Aucune dépendance ajoutée — `reportlab` dans
`requirements.txt` pour dessiner 2 600 carrés aurait été le plus gros des deux
changements.

### L'artefact, et pourquoi le débord ne suffisait pas

Premier jet : rectangles jointifs. Rendu sur fond sombre, un **quadrillage**
apparaît entre chaque voxel — le fond de page qui transparaît. Réflexe : faire
déborder chaque rectangle d'une fraction de voxel pour qu'ils se recouvrent.
Ça a beaucoup amélioré la chose (comparaison faite à 1400 px, le quadrillage
disparaît), mais à 900 px il revenait.

La raison est que **deux aplats antialiasés jointifs ne composent pas une
couverture pleine**. Chacun dépose un alpha partiel sur le pixel de la
frontière, et il survit `(1−a)(1−b)` de ce qu'il y a dessous : environ un
sixième quand les deux valent un demi. Le recouvrement réduit `a` et `b` mais
ne les amène à 1 que s'il vaut un demi-pixel écran — grandeur qu'un fichier
vectoriel ne connaît pas, par construction.

**Correctif : chaque lettre est peinte deux fois.** D'abord sa silhouette
entière comme un seul chemin — les sous-chemins d'un même `f` partagent un
unique calcul de couverture, donc cette couche-là n'a aucune couture — dans la
couleur moyenne de la lettre. Puis les voxels par-dessus. Ce qui survit aux
coutures est alors la lettre elle-même : une nuance, plus une grille. Vérifié
à 900 px sur les variantes claire et colorée, plus de quadrillage.

### Six fichiers, parce que PDF n'a pas de fond transparent

Une page où rien n'est peint sous les lettres **est** blanche dans tous les
lecteurs. Donc les deux fichiers destinés à être posés dans une maquette le
disent dans leur nom, et les quatre autres portent leur propre fond :

| Fichier | Encre | Fond |
|---|---|---|
| `_dark` | blanc | `#0e1117` |
| `_light` | encre sombre | blanc |
| `_sources` | couleurs dataset | `#0e1117` |
| `_sources_light` | couleurs dataset, rampe inversée | blanc |
| `_white` | blanc | aucun |
| `_ink` | encre sombre | aucun |

Ouvrir `_white` seul montre une page vide. C'est correct : c'est de l'encre
blanche. La rampe de `_sources_light` est inversée et non re-teintée — sur
papier, un voxel dense doit être le plus **sombre** —, pour que chaque lettre
garde la couleur qui identifie sa source.

### Vérifié

Rendu par CoreGraphics (`qlmanage`, à 900 et 1400 px) et par le rasteriseur
pdfium : les six s'ouvrent, la page fait 502 × 138 pt, les blocs sont carrés
et nets. À noter pour la prochaine fois : le volet d'aperçu du navigateur
affichait une page blanche en portrait pour ces mêmes fichiers — c'est son
rendu « instantané statique » des fichiers hors dossier servi, pas le PDF.

---

## 2026-09-22 — Survoler une lettre montre son noyau, et où il vit

Le texte sous le mot disparaît. Survoler une lettre fait maintenant les deux
choses qu'on attend du nuage : montrer le noyau **avec son masque**, et dire
**où le point se trouve** dans le nuage en dessous.

### L'aperçu vient des mêmes fonctions que les atlas

`letters.json` embarque maintenant, par lettre, la vignette *et* le masque,
produits par `build_cloud.to_thumbs` et `to_masks` — celles-là mêmes qui
remplissent les atlas du nuage. Donc mêmes recadrage, même étirement, même
binarisation : un noyau a la même tête partout dans l'app, ce qui est la
seule façon de comparer deux aperçus. Le champ `support` reprend
`data.THRESHOLD_SUPPORT`, si bien que le `C` et le `e`, tous deux HPA,
affichent « support (seuil) » et non « masque ».

Le panneau est celui du nuage, **couché sur le côté** : l'en-tête fait 132 px
et cette hauteur porte la mise en page en dessous, donc le panneau grandit
là où il y a de la place. Il se pose **du côté opposé à la lettre survolée** —
dans l'explorateur le mot tient dans une colonne plus étroite, et un panneau
fixé à un bord masquait le `ll` qu'il était censé décrire.

### Le repère dans le nuage, sans passer par Python

Les deux composants sont deux iframes de même origine : le message va de
l'une à l'autre par `BroadcastChannel`, **sans aller-retour serveur**. C'est
la même raison que pour le nuage lui-même : un survol qui coûte un aller-retour
n'est pas un survol. Le logo envoie `(dataset, index, couleur)`, le nuage
résout l'index global par son propre manifeste — une seule source de vérité
pour l'espace d'index, plutôt que de le recalculer côté logo.

Le repère est **frappé deux fois**, halo noir puis couleur du dataset. Sur
2,6 M de points, un anneau fin d'une seule couleur tombe sur un fond qui la
contient déjà — vérifié en survolant le `l`, dont le point TissueNet tombe en
plein milieu de la zone violette. S'il sort du cadre parce que la vue est
zoomée ailleurs, il est plaqué au bord avec un trait qui indique la direction,
plutôt que d'être dessiné là où le noyau n'est pas.

### Un piège : `img.decode()` ne se résout pas dans un document masqué

Le composant chargeait ses sprites avec `await img.decode()`. Symptôme
observé : `S.letters` vide, `S.grid` renseigné — donc le `fetch` passait et
`load()` restait bloqué après. La cause est que `decode()` ne se règle jamais
tant que le document est masqué. Un en-tête chargé dans un onglet
d'arrière-plan serait donc resté vide indéfiniment, et pas seulement sous un
volet de test caché. Remplacé par `onload`, qui n'a pas ce comportement.

### Vérifié dans le navigateur

Survol du `C` : panneau « HPA #563187 », vignettes noyau + **support (seuil)**,
repère vert au bon endroit. Survol du `l` : « TissueNet #507283 », masque plein,
repère violet lisible en zone dense. Sortie du survol : panneau fermé,
`S.hover` et `S.marked` à −1 des deux côtés. Clic : l'explorateur s'ouvre sur
le bon noyau, et là — sans nuage monté — le survol continue d'afficher le
panneau, le message diffusé n'ayant simplement pas d'auditeur.

---

## 2026-09-22 — L'aperçu de l'en-tête devient celui du nuage

Le panneau que le logo s'était fabriqué disparaît. Survoler une lettre
remplit désormais **le panneau du nuage**, à sa place habituelle (en haut à
droite de la toile) et avec **ses champs** : vignette, masque — ou « support
(seuil) » pour HPA —, `dataset #index`, classe, et les coordonnées brutes des
deux axes. Plus de « lettre l · score 0,900 · rotation 1° », qui était une
information sur la fabrication du logo, pas sur le noyau.

C'est une suppression plus qu'un ajout : le composant du logo n'a plus
d'aperçu du tout. Il envoie `(dataset, index, couleur)` sur le
`BroadcastChannel`, et le nuage appelle son propre `showPreview()`. Un noyau
s'affiche à un seul endroit dans cette app ; l'en-tête l'emprunte au lieu
d'en tenir une copie presque identique qu'il faudrait garder en phase avec
`build_cloud.to_thumbs`. Du coup `letters.json` perd la vignette, le masque
et le drapeau `support` qu'on y avait mis la veille : 16 ko → 11 ko.

Un détail qu'il a fallu corriger dans le nuage : `sheetImage` repeint
l'aperçu quand une planche d'atlas arrive en retard, mais elle ne regardait
que `S.hover`. Un noyau désigné depuis l'en-tête n'est pas survolé, donc son
aperçu serait resté noir jusqu'au prochain évènement. Elle regarde maintenant
`S.hover` puis `S.marked`.

**Conséquence assumée** : dans l'explorateur il n'y a pas de nuage, donc
survoler une lettre n'y montre plus rien — la lettre s'éclaire, le clic
marche. Le panneau « Noyau Original » juste en dessous montre déjà le noyau
courant, et dupliquer l'aperçu pour ce seul cas ramènerait exactement la
copie qu'on vient de supprimer.

### Vérifié dans le navigateur

Survol du `V` : panneau du nuage rempli avec « NuInSeg #13 217 », noyau +
masque, « x 0,478 · y 0,783 », et le repère bleu au bon endroit dans le
nuage. Sortie de la lettre : `S.hover` et `S.marked` à −1, panneau masqué.

Piège de mesure à retenir pour les prochains tests : `getBoundingClientRect()`
appelé **dans** l'iframe donne des coordonnées relatives à l'iframe, pas à la
page. Sans ajouter l'offset de l'iframe, on survole à côté — et comme le
panneau gardait l'état d'un message envoyé à la main depuis la frame du haut,
la vérification semblait passer alors qu'elle ne testait rien.

---

## 2026-09-22 — Le nuage passe au fond de toute la page quand on zoome

Essai, pas encore une décision. Tant qu'on est au cadrage d'ensemble, rien ne
change : le mot-symbole, le nuage dans sa boîte, les contrôles. Dès la
première crantée de molette, l'iframe du nuage s'épingle au viewport
(`position: fixed`, 100 vw × 100 vh) et tout le reste flotte par-dessus — le
mot-symbole, la barre d'outils de Streamlit, la ligne de contrôles sur un
fond flouté. La molette en arrière, ou « Vue d'ensemble », le remet dans sa
boîte.

La bascule est **une classe sur le `<body>` de la page hôte**, posée par le
composant. Les deux documents sont de même origine (c'est déjà ce qui fait
marcher le `BroadcastChannel` avec l'en-tête), donc ça ne coûte rien. Passer
par Python était exclu : un rerun remonte l'iframe et détruirait le zoom qui
vient de demander le mode.

### Le piège, et ce qui le ferme

Le premier jet oscillait. Le seuil était sur `k`, or `fitK()` dépend de la
taille du canvas et la taille du canvas est précisément ce que ce mode
change : passer pleine page augmente `fitK`, `clampView` remonte `k` jusqu'à
lui, le zoom paraît annulé, on rebascule en boîte, `fitK` rebaisse, `k` est
de nouveau au-dessus — et ainsi de suite.

L'état se lit donc sur **`k / fitK()`**, qui vaut 1 au cadrage quelle que
soit la taille du cadre. Et le rapport est mis de côté avant la bascule puis
rejoué au `resize` qui suit (700 ms de validité) : mesuré, il est conservé au
millième près à travers la transition (1,16 avant, 1,16 après), et les deux
états tiennent. C'est aussi le bon comportement en soi — le nuage ne doit pas
sauter quand son cadre grandit. Seuils : 1,15 pour entrer (≈ une crantée),
1,02 pour sortir.

### Trois petites choses qu'il a fallu régler

- L'iframe sort du flux, donc **son conteneur doit garder la place**, sinon
  la ligne de contrôles remonte sous le mot-symbole. Et tout ce qui doit
  passer au-dessus a besoin d'un `z-index` explicite : un bloc statique se
  peint sous une iframe positionnée, quel que soit l'ordre du DOM.
- **La bande du mot-symbole avale le pointeur.** Le logo fait 171 px de large
  dans une iframe qui prend toute la largeur ; au-dessus du nuage, les ~850
  px vides interceptent tout. Le composant du logo publie sa largeur dans
  `--vocell-logo-w` et la bande s'y ajuste. Le padding est renvoyé avec,
  sinon le `resize` que ça déclenche recalcule une taille de voxel plus
  petite et le logo rétrécit à chaque passe.
- La barre d'outils de Streamlit est opaque et pleine largeur : laissée
  telle quelle, c'est un couvercle de 60 px sur une page censée être toute en
  nuage. Elle devient un dégradé. Et les panneaux du composant (compteur,
  aperçu, légende, axes) rentrent de `--inset-t` / `--inset-b`, **mesurés**
  sur la page hôte plutôt que recopiés de `_HEADER_H`.

La classe survit aux reruns, l'iframe non : le composant la retire au
montage, sinon un clic sorti d'une vue zoomée laisserait le nuage suivant mis
en page pour un zoom qu'il n'a plus.

### Vérifié dans le navigateur

Zoom → pleine page ; le survol, l'aperçu, le clic vers l'explorateur, le
retour au nuage et le survol d'une lettre de l'en-tête marchent tous dans le
mode immersif. Dézoom et « Vue d'ensemble » → retour en boîte, rapport à 1,
aucune erreur console.

### Suite : thème forcé, et le mot-symbole va se ranger dans la bande

Deux retouches au mode immersif de tout à l'heure.

**Le thème est fixé en sombre** (`.streamlit/config.toml`, versionné). Ce
n'était pas une préférence : les deux composants maison peignent sur un
`#0e1117` codé en dur dans leur propre CSS — un canvas n'hérite pas du thème
de l'hôte — donc en thème clair l'app encadrait une toile sombre de texte
sombre sur blanc, et pleine page c'était franchement illisible. Mettre la page
d'accord avec la toile coûte trois lignes ; l'inverse voudrait dire deux
palettes à tenir dans chaque composant, pour un mode d'affichage que personne
n'a demandé.

**Le mot-symbole ne reste plus au milieu.** En passant en immersif il glisse
dans la bande d'outils, en haut à gauche, réduit à 44 px. Les deux positions
sont du CSS ; le trajet est un **FLIP** — on mesure où l'iframe est, on
bascule la classe, on mesure où elle a atterri, on la repart de l'ancienne
boîte par un `transform` et une seule transition la ramène. C'est le composant
du **nuage** qui l'exécute, pas celui du logo : c'est le seul à se trouver des
deux côtés de la bascule dans la même tâche, et il agit sur l'élément iframe,
qui vit dans le document hôte — il n'a rien à savoir du script du logo.

Le point non évident : **le cadre doit faire la taille du glyphe dans les deux
états**, pas seulement en immersif. Le mot est centré à taille fixe dans son
cadre, donc cadre et glyphe ne se réduisent du même facteur que si les deux
font la même taille ; avec un cadre pleine largeur au repos et un cadre étroit
à l'arrivée, le FLIP devient une mise à l'échelle non uniforme et le mot
s'étire en route. `--vocell-logo-w` est donc appliqué aussi au repos — ce qui
ne se voit pas, le mot étant centré des deux façons.

Et comme l'iframe du logo passe en `fixed`, sa ligne garde sa hauteur, sinon
la ligne de contrôles remonte de 140 px.

**Piège de vérification**, qui a coûté trois essais : un onglet masqué gèle
les transitions. `getAnimations()` rendait bien une transition sur `transform`
mais son `currentTime` restait à 0 et les rectangles ne bougeaient pas — de
quoi conclure que l'animation ne partait pas. `document.hidden` valait `true`.
Prendre une capture force le rendu : la troisième a attrapé le mot à
mi-parcours entre le centre et le coin.

---

## 2026-09-22 — Le nuage est la page, sans conditions

L'essai est adopté, et du coup simplifié : **il n'y a plus de bascule**. Le
nuage est toujours épinglé au viewport, le mot-symbole toujours rangé en haut
à gauche dans la bande d'outils. Toute la machinerie de la veille — seuils
`k/fitK()`, `lockRatio`, classe `vocell-immersive` sur le `<body>` de l'hôte,
FLIP du mot-symbole — **disparaît**. C'était la bonne façon de faire une
bascule ; la meilleure bascule était de ne pas en avoir.

Ce qu'il en reste, et qui mérite la place : **`--inset-t` / `--inset-b`**,
mesurés sur la page hôte pour que les panneaux du composant ne passent pas
sous la barre d'outils ni sous la ligne de contrôles. Et une cale unique,
`_CLOUD_RESERVE` : les deux iframes étant hors flux, il ne reste dans le flux
que la ligne de contrôles, et c'est ce nombre qui la pousse au pied du
viewport. Un seul nombre à retoucher au lieu de trois constantes chaînées.

### Le mot-symbole restait à cliquer

Le ranger dans la bande d'outils l'avait rendu inerte, et je ne m'en étais pas
aperçu la veille parce que je n'avais testé que le survol *avant* le
déplacement. Deux causes, toutes deux dans le chrome de Streamlit :

- **La barre d'outils avale le pointeur.** C'est un seul élément pleine
  largeur, avec ses deux boutons tassés à l'extrémité droite ; elle prend donc
  tout survol et tout clic sur les 60 px du haut — c'est-à-dire exactement la
  bande où le mot vient de s'installer. Elle passe en `pointer-events: none`,
  ses `button` / `a` / conteneurs d'actions reprennent les leurs. Vérifié :
  « Deploy » et le menu burger répondent toujours, et le milieu de la bande
  atteint maintenant le nuage (on peut y survoler et y glisser).
- **Le mot était sous le dégradé**, pas dessus. La barre est en
  `z-index: 999990` et porte le fondu sombre ; à `z-index: 4` le mot se
  retrouvait *dans* le fondu et perdait la moitié de son contraste — très
  visible au-dessus d'une zone dense et claire du nuage. Il passe au-dessus,
  et le fondu devient ce sur quoi il repose.

Le reste marchait déjà : les coordonnées d'un évènement dans une iframe sont
dans le repère interne de l'iframe, que le parent l'ait mise à l'échelle ou
non, donc le test alpha du logo n'a pas eu une ligne à changer. Vérifié en
cliquant le `V` rangé : NuInSeg #13 217, le même que l'aperçu annonçait.
`_LOGO_PARKED_H` passe de 44 à 48 px, les lettres étant petites à viser.

### Le panneau du haut se vide

`0 visibles / 2 633 390` quittait rarement les deux extrêmes et ne disait rien
qu'on ne voie sur la toile. L'indice « zoomer pour voir les noyaux » part avec
lui : c'était une notice pour un geste que tout le monde essaie de toute façon,
et elle occupait la bande à côté du mot-symbole. Le panneau se réduit à « Vue
d'ensemble ». Les comptes par source restent dans la légende, où ils comparent
quelque chose.

---

## 2026-09-23 — Nuage : cadrage sur la bande libre, palette de classes a priori

Trois retouches de la page principale.

- **Haut et bas du nuage atteignables.** L'ellipse était inscrite dans le
  canvas entier, donc ses bords passaient sous le mot-symbole et sous la ligne
  de contrôles. `fitK()`, `clampView()` et le centre vertical se calent
  désormais sur la bande entre `--inset-t` et `--inset-b`, déjà mesurées sur
  la page hôte. Un `ResizeObserver` sur la ligne de contrôles recadre une vue
  restée au plancher.
- **Bouton « Vue d'ensemble » retiré**, avec son panneau : dézoomer jusqu'au
  plancher ramène à la même vue.
- **Mode « classe » exclusif.** Seuls CODEX, HPA et BBBC051 y sont proposés,
  un à la fois (radio), les autres sources masquées. Les couleurs de classes
  ne suivent plus la teinte du dataset (illisible : 14 à 17 classes dans une
  bande de 20°) mais `cloud.CLASS_KINSHIP` : une parenté morphologique
  **écrite à la main**, familles ordonnées de la plus petite et dense à la
  plus grande et pléomorphe, posées sur un arc OKLCH. Une première version
  dérivait cette parenté des centroïdes de classes (descripteurs + latent) ;
  abandonnée avant d'être livrée : le nuage aurait paru bien trié par
  construction.

---

## 2026-09-23 — README en anglais, et l'app tolère un sous-ensemble de sources

README réécrit en anglais, avec l'objectif qu'un inconnu puisse lancer l'app.
L'écrire a fait apparaître ce qui l'en empêchait :

- **L'app plantait dès qu'une source manquait** : le sélecteur ouvrait sur
  CODEX quoi qu'il arrive, et `build_cloud.py` chargeait les onze. Désormais
  `data.available_datasets()` filtre les deux. Vérifié sur un clone frais avec
  quatre petites sources (build 21 s, nuage, explorateur, toast sur une lettre
  du logo absente), puis sans aucune donnée (message dans la barre latérale).
- **Deux dépendances privées restent bloquantes**, et le README le dit plutôt
  que de le cacher : `cellaug` (GitHub) et le dataset HF `CellfSup/CellfSup`.
- Le README télécharge directement depuis Hugging Face, sans passer par un
  clone de `cellf-supervised`. Piège mesuré : avec `hf` 1.8,
  `--include a b` ne télécharge que `a` (le second est lu comme un nom de
  fichier) — il faut répéter `--include`.

---

## 2026-09-23 — Nuage : UMAP et t-SNE, et les trois modèles SimCLR

Le choix de disposition passe en deux temps : un **espace** (deux descripteurs,
les dix descripteurs, ou le latent d'un modèle) puis une **projection** (ACP,
UMAP, t-SNE). `meta.json["layouts"]` remplace le drapeau `latent`.

- **Trois modèles**, tous les `simclr_resnet18_*.pt` de `cellf-supervised` :
  `outHPA_augcell` (le seul branché jusqu'ici), `outHPA_augcifar`, et
  `cifar_tr` — le run antérieur, dont le commit 413b22b (« wrongly include test
  data in ssl training set ») dit qu'il a vu le test HPA en pré-entraînement ;
  son libellé le dit. `outputs/simclr_classical_losses.pt` n'est qu'une courbe
  de loss. Embeddings déplacés sous `data/cloud/emb/<clé>_{h,z}.npy` ;
  l'extraction écrit en `.part` puis renomme, pour qu'un arrêt en route ne
  laisse pas un tableau à moitié nul que le build prendrait pour fini.
- **UMAP / t-SNE ajustés sur 217 638 noyaux** (200 k uniformes, plancher de
  5 000 par source), le reste placé à la médiane de ses 10 voisins ajustés.
  Temps mesurés (M-series, 10 cœurs) : descripteurs 645 s, chaque latent
  250–330 s ; mis en cache dans `data/cloud/proj/`, un `--meta-only` suivant
  rejoue en secondes. 34 colonnes, 179 Mo.
- Les deux nouveaux latents ont une CP1 plus lourde que celui d'augcell
  (20–21 % contre 12 %), toujours proche de la netteté du contour
  (r = +0,79 à +0,82). Pas mesuré plus loin : `latent_probe.py` prend
  désormais `VOCELL_MODEL=<clé>` pour le refaire sur chacun.
- « Dispersion » est désactivée hors « Deux descripteurs » : les axes
  projetés sont continus, leur pas de dispersion vaut 0.

**Ouvert** : les mesures de CLAUDE.md sur le latent (×2,94, 338 dimensions
mortes…) ne portent que sur `outHPA_augcell`.

---

## 2026-09-24 — Nuage : légende ouverte, dispersion permanente

- Le panneau « Sources » s'ouvre déplié au chargement (il restait replié
  derrière un chevron) ; le chevron le replie toujours.
- L'interrupteur « Dispersion » disparaît : la dispersion est toujours active.
  Elle ne déplace un noyau que de moins d'un pas entre deux valeurs, et les
  axes projetés, continus, n'ont pas de pas — elle n'y fait rien.
- Survoler une source dans la légende fait pâlir tous les autres points (une
  classe, si on survole sa ligne) ; quitter la légende rend le nuage entier.
  Les points pâlis sont peints d'abord, la source survolée par-dessus.
- La légende liste les sources de la plus grosse à la plus petite (ordre
  d'affichage seulement : `S.datasets` reste dans l'ordre de `SOURCES`).
- Maj + clic sur une source n'affiche qu'elle ; le même geste sur une source
  déjà seule réaffiche toutes les autres.
