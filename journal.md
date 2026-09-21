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
