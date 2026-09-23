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

L'entrée de l'app est un **nuage de points** de tous les noyaux des deux
datasets : survol → aperçu du noyau, clic → ouverture dans l'explorateur.

## Architecture

```
src/                       # Application Streamlit (imports plats, pas de package)
  app.py                   # Point d'entrée : orchestre sidebar → géométrie → rendu
  ui_components.py         # Tous les widgets Streamlit + logique de choix du volume
  data.py                  # Chargement datasets + génération des volumes 3D
  geometry.py              # Maths pures : plan de coupe, clipping, maillage voxel
  cloud.py                 # Maths pures : normalisation + ACP des descripteurs,
                           #   palette de classes (parenté écrite à la main)
                           #   (consommé par le build, pas par l'app)
  augment.py               # Pont vers cellaug : plan de l'app -> ObliqueSection
  visualization.py         # Construction de la figure Plotly
  sam3d_engine.py          # Wrapper subprocess vers l'env conda SAM3D
  run_inference.py         # Script autonome exécuté DANS l'env conda SAM3D
  components/nuclei_cloud/
    index.html             # Composant Streamlit du nuage — canvas, sans build npm
    atlas/ masks/ cols/    # Assets générés (gitignorés, 855 Mo)
    meta.json classes.bin  # idem
  components/logo/
    index.html             # Composant du mot-symbole — survol + clic, sans build
    letters.json           # 6 sprites voxel + provenance (15 ko, versionné)
scripts/
  preprocess_restore.py    # .ims (Imaris/HDF5) + masques → data/RESTORE/nuclei.npy
  build_cloud.py           # Descripteurs + colonnes + atlas du nuage (11 sources)
  extract_embeddings.py    # Les 2,6 M de noyaux dans le ResNet18 SimCLR voisin
  latent_probe.py          # La similarité cosinus sépare-t-elle, et dans quel espace
  control_random.py        # Le même test avec un ResNet18 non entraîné (contrôle)
  train_sam3d.py           # Fine-tuning du ss_generator de SAM3D sur RESTORE
  build_logo.py            # Logo « VoCell » : un vrai noyau par lettre
assets/                    # Sorties de build_logo.py (PNG lisses, PDF voxel, JSON)
```

**Flux de l'app** : `app.py` affiche d'abord `ui.render_logo()` (hauteur
`ui._LOGO_H`, dont dépend la réserve verticale du nuage), au-dessus des
deux vues, puis route sur `st.session_state["view"]`
(`"cloud"` par défaut si `data.cloud_available()`). En vue nuage il appelle
`ui.render_cloud_view()` puis `st.stop()`. En vue explorateur il appelle `ui.*` pour obtenir `volume` + paramètres de
coupe → `geom.get_plane_vectors` → `geom.apply_clipping` → `geom.get_voxel_mesh_data`
→ `vis.create_3d_figure`, et en parallèle `geom.extract_2d_slice` pour la coupe.
Si le panneau ObliqueSection est activé, une troisième colonne applique la même
coupe *sans volume*, via `augment.plane_to_oblique` → `augment.apply_forced`.

### Nuage de noyaux

`scripts/build_cloud.py` (~3,5 min) produit, toutes sorties régénérées par
machine et toutes sous des chemins gitignorés :

| Sortie | Contenu |
|---|---|
| `data/cloud/features.npz` | 2 633 390 lignes × 10 descripteurs (111 Mo). Table canonique, pour l'analyse — **l'app ne la lit jamais** |
| `src/components/nuclei_cloud/cols/*.bin` | une colonne uint16 normalisée par descripteur, plus `pca1`/`pca2` (12 × 5,3 Mo) |
| `src/components/nuclei_cloud/atlas/NNN/` | 41 147 PNG de 8×8 vignettes 32² (681 Mo), répartis en sous-dossiers de 1000 |
| `src/components/nuclei_cloud/masks/NNN/` | les masques binaires correspondants, même géométrie et même indexation (111 Mo) |
| `…/meta.json` + `classes.bin` | manifeste des sources, classes, palettes, géométrie des atlas |
| `src/components/nuclei_cloud/cols/latent{1,2}.bin` | ACP de l'espace latent SimCLR — **seulement si** `scripts/extract_embeddings.py` a tourné |

**L'index global est contigu par source**, dans l'ordre de `SOURCES` : le
composant retrouve le dataset d'un point par recherche dichotomique sur les
`start`, sans tableau par point.

- **Les descripteurs sont en pixels, jamais en microns.** Les sources n'ont
  jamais été rééchantillonnées (voir « Datasets ») : tout axe impliquant une
  taille sépare les datasets. Choix explicite, pas un oubli — c'est ce que le
  nuage est là pour montrer.
- **Les vignettes subissent un recadrage central 44×44, un étirement p0,5/p99,8
  et un gamma 0,65** — identiques pour tous, donc comparables. Mesuré : ≥99 %
  du signal conservé pour 99,5 % des noyaux. Sans ça, un noyau CODEX ou
  TissueNet est une tache noire de 10 px dans une vignette de 32.
- **`data.THRESHOLD_SUPPORT` est la source unique** de qui a un vrai masque :
  l'explorateur et le nuage la lisent tous les deux (le second via le
  `support` écrit dans `meta.json` au build), donc ils ne peuvent pas diverger.
- **Le masque de segmentation est `crop > 0`, sans calcul.** `generate_crops`
  de `cellf-supervised` fait `img_crop * (mask_crop == label)` : les crops
  arrivent déjà masqués, et leur support non nul **est** le masque du noyau.
  Mesuré sur des échantillons de chaque source, sa médiane de composantes
  connexes vaut 1 — **sauf HPA**, qui n'est jamais passé par un masque : son
  fond nul est un seuil d'intensité (médiane 8 composantes, 40 % des crops
  touchant le bord). L'aperçu au survol écrit « support (seuil) » pour HPA et
  « masque » ailleurs ; ne pas faire passer le premier pour le second.
  Le masque est binarisé en pleine résolution **puis** rééchantillonné, le
  seuil à 0,5 remettant la frontière où elle était : seuiller la vignette finie
  la dilaterait d'un pixel, le filtre boîte rendant non nul tout pixel de bord
  partiellement couvert.
- **Les masques ont leurs propres planches**, pas le canal alpha des vignettes :
  la mosaïque zoomée n'en a jamais besoin, et l'aperçu en veut exactement une,
  qui pèse ~3 ko.
- **Les atlas sont volontairement petits** (8×8 vignettes). L'accès est
  aléatoire : les voisins dans la projection ne sont pas voisins dans l'index
  global, donc une zone zoomée touche à peu près autant de tuiles qu'elle a de
  points. Avec des planches de 32×32, 322 points tiraient 144 planches de
  240 Ko. Ne pas « optimiser » en regroupant.

**Pourquoi un composant maison et pas Plotly** : Streamlit expose le clic et le
lasso d'un graphique (`on_select`), **pas le survol**. Un survol qui coûte un
aller-retour serveur n'est pas un survol. Le composant dessine et détecte les
2,6 M de points dans le navigateur ; seul le clic remonte à Python.

Le composant n'a **aucune chaîne de build** : `index.html` parle directement le
protocole `postMessage` de Streamlit (`streamlit:componentReady`,
`streamlit:render`, `streamlit:setComponentValue`, `streamlit:setFrameHeight`,
`apiVersion: 1`). Les composants personnalisés exigent `pyarrow`.

**Rien par noyau ne passe par les arguments du composant.** Python envoie deux
noms de colonnes ; le navigateur va chercher les `.bin` en HTTP et les met en
cache. À 2,6 M de points, une paire de positions pèse 10 Mo, qui traverseraient
le websocket à chaque rerun. C'est aussi pourquoi `src/cloud.py` ne sert plus
qu'au build : l'app ne fait plus aucun calcul pour le nuage.

**Le nuage est la page, littéralement.** Il n'est pas cadré : son iframe est
épinglée au viewport (`position: fixed`, 100 vw × 100 vh) et **tout** flotte
par-dessus — le mot-symbole, la barre d'outils de Streamlit, la ligne de
contrôles, et les panneaux du composant (sources, axes, aperçu). Il n'y a
qu'une disposition ; il n'y a jamais de boîte. Points à ne pas défaire :

- **La hauteur ne vient pas de `setFrameHeight`.** Un composant Streamlit se
  dimensionne en pixels ; `_CLOUD_CSS` dans `ui_components.py` prend la main,
  `!important` battant le style en ligne de Streamlit, et le
  `html, body { height: 100% }` du composant suit l'iframe. Le composant ne
  doit **pas** réécrire `document.body.style.height` : il se battrait avec ce
  100 % et désynchroniserait le canvas. La hauteur passée depuis Python n'est
  plus qu'un repli.
- **Les deux iframes étant hors flux, il ne reste que la ligne de contrôles**
  dans le flux, et `_CLOUD_RESERVE` est la cale qui la pousse au pied du
  viewport (bande d'outils + la ligne + les gouttières du bloc). C'est le seul
  nombre à retoucher si la ligne déborde ou flotte trop haut.
- **Tout ce qui doit passer au-dessus déclare son `z-index`** : un bloc
  statique se peint *sous* une iframe positionnée, quel que soit l'ordre du
  DOM.
- **Les contrôles sont écrits avant le nuage et rendus après**, par deux
  `st.container()` déclarés dans l'ordre inverse : le composant a besoin des
  valeurs d'axes, mais il doit occuper le haut de la page. Celui des contrôles
  porte `key="cloud_controls"` — la classe `st-key-…` est ce que le CSS et le
  composant visent tous les deux.
- **Les points sont projetés dans une ellipse, pas dans un carré.** Chaque
  colonne étant normalisée dans [0,1], le nuage brut remplit un carré ; le
  mapping de grille elliptique (`u' = u√(1−v²/2)`, `v' = v√(1−u²/2)`, puis
  étirement à `ASPECT`) l'envoie sur un disque étiré. Bijection continue, qui
  bouge les coins bien plus que l'intérieur. **C'est une déformation des
  axes** : l'aperçu au survol cite `S.rawx`/`S.rawy`, les colonnes brutes, et
  jamais les positions affichées. Ne pas inverser les deux.
- **Le cadrage est borné.** `fitK()` est le zoom qui inscrit l'ellipse et sert
  de plancher ; `clampView()` — après chaque glisser, molette et
  redimensionnement — verrouille le centre sur l'axe entièrement visible et le
  borne sur l'autre. On ne peut ni dézoomer dans le vide ni pousser le nuage
  hors du panneau. Il n'y a **pas de bouton « Vue d'ensemble »** : dézoomer
  jusqu'au plancher y ramène.
- **Le cadrage se fait sur la bande libre, pas sur le canvas.** `fitK()`,
  `clampView()` et le centre vertical (`midY()`) utilisent la bande comprise
  entre `--inset-t` et `--inset-b` : le haut de l'ellipse finit sous le
  mot-symbole et son bas au-dessus de la ligne de contrôles, sinon ses bords
  se voient mais ne s'attrapent pas à la souris. Le canvas reste peint en
  entier, sous le chrome translucide. Un `ResizeObserver` sur la ligne de
  contrôles recadre une vue restée au plancher quand la ligne change de
  hauteur.

#### Le chrome de l'hôte, posé sur le nuage

- **Le mot-symbole vit dans la bande d'outils**, en haut à gauche, à
  `_LOGO_PARKED_H` de haut par un `transform: scale()`. Son cadre est d'abord
  rétréci au mot (`--vocell-logo-w`, publié par le composant du logo, padding
  compris — sinon le `resize` que ça déclenche recalcule un voxel plus petit
  et le mot rétrécit à chaque passe) : le mot fait 171 px css de glyphe dans
  une iframe qui prendrait toute la largeur, et une iframe vide posée sur le
  nuage avale le pointeur partout où elle s'étend.
- **Il est au-dessus de la barre d'outils, pas dessous.** Cette barre est en
  `z-index: 999990` et porte un dégradé sombre ; à un `z-index` ordinaire le
  mot se retrouve *dans* le dégradé et perd la moitié de son contraste.
- **La barre d'outils est rendue transparente au pointeur.** C'est un seul
  élément pleine largeur avec deux boutons à son extrémité droite : laissée
  telle quelle, elle avale tout survol et tout clic sur les 60 px du haut — y
  compris les lettres du mot-symbole, qu'elle recouvre entièrement. Les
  `button`, `a` et les conteneurs d'actions reprennent leurs évènements.
- **Les panneaux du composant rentrent sous ce chrome.** `--inset-t` et
  `--inset-b` sont **mesurés** sur la page hôte (le bas de `stHeader`, le haut
  de `.st-key-cloud_controls`), jamais recopiés d'une constante Python.

Quatre pièges, tous résolus, à ne pas défaire :

- Streamlit sert les fichiers d'un composant en `Cache-Control: public`. Une
  reconstruction resterait donc invisible : `meta.json` est lu en `no-store` et
  porte un `build`, accroché en `?v=` à toutes les autres URL.
- Le jeton de clic est un **horodatage**, pas un compteur : l'iframe est
  remontée à chaque retour au nuage, et un compteur repartant de 1 ferait
  passer le premier clic suivant pour un doublon.
- Streamlit peut livrer un second `render` avant la fin du premier. Le garde
  est une **promesse attendue**, pas un booléen : un booléen laisse le second
  appel continuer avec l'état encore nul.
- La couche de points est **mise en cache dans son `ImageData`**. Peindre 2,6 M
  de points coûte ~200 ms, et un déplacement de souris d'un noyau ne change
  qu'un anneau de 6 px : le survol se contente de reblitter (17 ms). Elle est
  invalidée par la vue, les colonnes, les filtres et la taille du canvas — pas
  par le survol. Pendant un glisser, le rendu est échantillonné (~350 k points,
  44 ms) et la passe complète arrive 140 ms après l'arrêt.

### Disposition latente

Troisième disposition du nuage, offerte **uniquement** quand `meta.json` porte
`"latent": true` — c'est-à-dire quand `scripts/extract_embeddings.py` a écrit
`data/cloud/emb_h.npy`. L'app ne charge jamais de checkpoint pour le savoir :
`build_cloud.py` écrit deux colonnes de plus et pose le drapeau, et le
composant, qui va chercher `cols/<clé>.bin` et lit `meta.axes[clé]`, n'a pas eu
une ligne à changer.

- **Chaque embedding est ramené à une longueur de 1 avant l'ACP.** Mesuré
  (`scripts/latent_probe.py`) : la direction d'un vecteur du backbone s'accorde
  avec la lignée cellulaire ×2,94 au-dessus du hasard, sa norme seule ×1,28, et
  cette norme corrèle −0,33 avec l'aire du noyau. La longueur code surtout la
  taille, qui est déjà la nuisance séparant les onze sources. Le cosinus la
  jette, l'ACP ici aussi.
- **`h` (512, backbone) et pas `z` (128, projecteur)**, bien que la loss
  InfoNCE soit littéralement définie sur le cosinus de `z` : `h` fait mieux ou
  égal partout dans les mesures.
- **Ce n'est pas une meilleure disposition, c'en est une autre.** Cet espace ne
  sépare pas les classes mieux que les dix descripteurs (HPA ×2,88 contre
  ×2,70 ; CODEX ×1,44 contre ×1,52 ; BBBC051 ×2,14 contre ×2,25), le cosinus
  n'y bat pas l'euclidien, et l'écart entre sources n'y tombe que de ×7,06 à
  ×5,97. Ne pas la présenter comme une projection « propre ».
- **338 des 512 dimensions de `h` sont identiquement nulles** sur les 2,6 M de
  noyaux — 0 sur le même réseau non entraîné. C'est ce qui borne le checkpoint,
  et pourquoi CP1 latente ne porte que 12 % de la variance.
- Les axes sont nommés par le descripteur avec lequel ils corrèlent le plus
  (`cloud.closest_descriptor`) : un axe latent n'a ni unité ni nom, et c'est la
  seule prise honnête qu'on puisse donner à l'UI.

`torch` vit dans `requirements-analysis.txt`, **jamais** dans
`requirements.txt` : l'app Streamlit doit rester installable sans GPU.

**Légende hiérarchique** : chaque dataset est une « super-classe ». Un clic sur
sa ligne masque ou réaffiche toute la source d'un coup ; le chevron déplie ses
classes pour les trois sources étiquetées.

**Colorer par classe n'affiche qu'un dataset étiqueté à la fois.** La légende
ne liste alors que CODEX, HPA et BBBC051, en boutons radio ; les autres
sources sont masquées (`S.off`, table de visibilité par classe, distincte des
interrupteurs de l'utilisateur `S.hidden`, qui reviennent intacts en mode
dataset). Les trois vocabulaires n'ont rien en commun et chaque palette
parcourt toute la roue : montrés ensemble, un lymphocyte T et une lignée HPA
porteraient le même bleu.

- **La palette de classes vient de `cloud.CLASS_KINSHIP`, écrite à la main**
  d'après ce qu'on sait de la morphologie des noyaux de chaque type (petits
  ronds denses → lobés → réniformes → allongés stromaux → grands
  pléomorphes, etc.), **jamais mesurée sur les données**. La dériver des
  descripteurs ou de l'espace latent serait tricher : le nuage paraîtrait
  bien trié par construction. Ici la palette est un a priori, que le nuage
  confirme ou non.
- Familles dans l'ordre de parenté, posées sur un arc de 280° de la roue
  OKLCH (perceptuellement uniforme) : un pas entre voisins d'une famille,
  `FAMILY_GAP` pas entre familles ; lumière alternée entre voisins pour qu'ils
  restent distinguables. Les deux extrémités sont les plus dissemblables.
- `Others` (CODEX) et `non étiqueté` (HPA) restent gris : ce ne sont pas des
  types. Une classe inconnue de la table devient une famille à part en fin
  d'arc — mettre la table à jour.
- `meta.json` liste les classes dans l'ordre de la palette : la légende se lit
  comme le dégradé.

### Logo « VoCell »

`scripts/build_logo.py --search` (~3 min) compare les 2,6 M de noyaux aux six
glyphes du mot, **à rotation près**, et `--render` dessine le résultat dans
`assets/` **et** régénère les sprites du composant d'en-tête. Les candidats
sont mis en cache dans `data/logo/candidates.json` (gitignoré) : le rendu se
rejoue sans refaire le balayage.

- **Le gabarit tourne, pas les noyaux.** 5 glyphes × 72 angles × 2 chiralités
  sont calculés une fois ; le scan est alors un produit matriciel par lot.
  90 s pour les 2,6 M. Inverser les deux rôles rendrait le problème infaisable.
- **Les silhouettes sont normalisées par leur rayon quadratique moyen**, pas
  par leur boîte englobante : une boîte grandit jusqu'à √2 en tournant, le
  rayon RMS est invariant. C'est ce qui rend la comparaison sur les
  orientations légitime. Ne pas « simplifier » en revenant à la boîte.
- **Le score n'est pas une IoU nue** :
  `IoU − ½ × remplissage des contrepoinçons`, le contrepoinçon étant
  l'enveloppe convexe du glyphe moins le glyphe. Sans ce terme, un coin plein
  bat un vrai V et un disque plein bat un anneau — mesuré, pas supposé. Le
  `l` étant convexe, son contrepoinçon est vide et il reste classé à l'IoU.
- **La pose vient du match, le placement est typographique.** L'angle et le
  miroir sont ceux qui ont gagné ; l'échelle et le centrage sont repris sur la
  boîte du glyphe dans le mot, sans quoi les six lettres ne tiennent pas une
  ligne de base commune.
- La deuxième passe rejette les masques fragmentés (plus grosse composante
  connexe < 90 %) et ceux qui touchent le bord de leur crop. Sans elle, les
  supports par seuil de HPA remontent en tête par accident.

#### Les PDF voxel

`--render` écrit aussi six `assets/logo_vocell_voxel*.pdf`, **vectoriels** :
un voxel = un rectangle rempli, jamais un bitmap agrandi. Un logo finit sur
une affiche et dans une slide, et seul le vectoriel garde des arêtes franches
à toute taille — un raster flouterait ou laisserait le lecteur inventer sa
propre idée du pixel. Émetteur PDF écrit à la main (~40 lignes) : ajouter
reportlab au `requirements.txt` pour dessiner 2 600 carrés serait le plus
gros changement des deux.

- **Chaque lettre est peinte deux fois.** D'abord sa silhouette entière, en
  un seul `f` dans sa couleur moyenne ; puis les voxels par-dessus. La
  première couche n'est pas décorative : deux aplats antialiasés jointifs ne
  composent pas une couverture pleine — chacun pose un alpha partiel sur le
  pixel de la frontière et il en survit `(1−a)(1−b)`, soit environ un sixième
  à moitié-moitié. Sans le fond d'aplat, ce sixième est la **page**, et un
  quadrillage apparaît dans chaque lettre (constaté, puis corrigé). Avec, ce
  qui survit est la lettre elle-même : une nuance, pas une grille.
- Le débord de `PDF_BLEED` réduit l'artefact sans pouvoir le fermer : il
  faudrait un demi-pixel écran, qu'un fichier vectoriel ne connaît pas.
- **Six fichiers et pas quatre, parce que PDF n'a pas de fond transparent** :
  une page où rien n'est peint sous les lettres *est* blanche dans tous les
  lecteurs. Les deux destinés à être posés dans une maquette le disent dans
  leur nom (`_white`, `_ink`) ; les quatre autres portent leur propre fond et
  s'ouvrent tels quels. Ouvrir `_white` seul montre une page vide : c'est
  correct, c'est de l'encre blanche.

#### L'en-tête de l'app

`ui.render_logo()` monte `src/components/logo/`, qui se comporte comme le
nuage : survoler une lettre montre son noyau, cliquer l'ouvre dans
l'explorateur (`_open_in_explorer`, partagé avec le nuage). Le jeton de clic
est un **horodatage**, pas un compteur, pour la même raison que dans le nuage.

- **Les lettres sont voxelisées**, en écho au rendu « Minecraft » de
  l'explorateur. Un voxel est allumé ou éteint (seuil à 0,5 sur l'alpha) ;
  moyenner l'alpha frangerait chaque bord de cellules à moitié allumées et
  perdrait justement l'effet de bloc. La couleur, elle, est la moyenne
  pondérée par l'alpha de ce qui tombe dans la cellule.
- **La grille est celle du mot entier, pas d'une lettre.** Quantifier chaque
  lettre dans sa propre boîte donnerait six tailles de pixel différentes.
- **`letters.json` embarque ses sprites en data URI** et il est versionné :
  11 ko, donc le logo marche sur un clone frais sans aucune donnée. C'est
  aussi ce qui évite le piège du `Cache-Control: public` de Streamlit — un
  sprite servi à part survivrait invisiblement à une reconstruction, alors
  que le document qui les porte tous est lu en `no-store`.
- **Le survol teste l'alpha, pas la boîte** : les boîtes du `V` et du `o` se
  chevauchent, seuls les voxels allumés répondent (avec un voxel de marge,
  pour que le `l`, large de 9, reste attrapable).

**Survoler une lettre remplit le panneau du nuage**, à sa place habituelle et
avec ses champs habituels — vignette, masque (ou « support (seuil) » pour
HPA), `dataset #index`, classe, coordonnées brutes. Le composant du logo n'a
**pas** d'aperçu à lui : il envoie `(dataset, index, couleur)` et le nuage
appelle son propre `showPreview()`. Un noyau s'affiche à un seul endroit dans
cette app, et l'en-tête l'emprunte plutôt que d'en garder une copie presque
identique. `letters.json` ne transporte donc ni vignette ni masque.

- **Le message passe en `BroadcastChannel` d'iframe à iframe**, jamais par
  Python : les deux composants sont de même origine, et un survol qui coûte
  un aller-retour serveur n'est pas un survol. Le nuage résout
  `(dataset, index)` en index global par son propre manifeste — une seule
  source de vérité pour l'espace d'index.
- Le point est aussi **repéré dans le nuage**. Le repère est frappé deux
  fois, halo noir puis couleur du dataset : sur 2,6 M de points, un anneau
  fin d'une seule couleur tombe sur un fond qui la contient déjà. S'il sort
  du cadre (vue zoomée ailleurs), il est plaqué au bord avec un trait qui
  indique la direction, plutôt que de mentir sur la position.
- `sheetImage` repeint l'aperçu quand une planche arrive en retard ; elle
  regarde `S.hover` **puis** `S.marked`, sinon un noyau désigné depuis
  l'en-tête resterait noir jusqu'au prochain évènement.
- **Conséquence assumée** : dans l'explorateur il n'y a pas de nuage, donc
  survoler une lettre n'y montre plus rien (la lettre s'éclaire, le clic
  marche). Le panneau « Noyau Original » juste en dessous montre déjà le
  noyau courant.
- **`img.decode()` est proscrit ici** : il ne se résout jamais tant que le
  document est masqué, donc un en-tête chargé dans un onglet d'arrière-plan
  restait vide indéfiniment. `onload` n'a pas ce défaut.

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
- Le résultat SAM3D est mis en cache dans
  `st.session_state[f"ai_vol_{dataset}_{idx}"]` (`data.ai_cache_key`), pas dans
  `@st.cache_data`.
- **Les crops sont mmap-és, pas chargés** : `load_crops` est décoré
  `@st.cache_resource`, pas `@st.cache_data`. TissueNet fait 5,5 Go à lui seul,
  et `cache_data` sérialiserait chaque octet pour stocker l'entrée de cache.
- **N'importe quel sous-ensemble des sources suffit.** `data.available_datasets()`
  (existence du fichier, non mise en cache) est la seule liste que lisent le
  sélecteur de l'explorateur et `build_cloud.py` ; `data.DATASETS` reste
  l'ordre canonique. Sans aucune source, la barre latérale le dit au lieu de
  planter ; un clic sur une lettre du logo dont la source est absente affiche
  un toast. Le bouton « classe » du nuage est désactivé sans source étiquetée.
  Un clone frais avec les petits jeux (~300 Mo) doit toujours marcher.
- **L'explorateur accepte les onze sources.** Les dix jeux de crops sont
  natifs 2D et passent tous par la même reconstruction par profondeur
  synthétique — il n'y a jamais rien eu de spécifique à CODEX dans le fait
  d'extruder un crop selon Z. RESTORE est le seul à apporter un vrai volume.

## Datasets

Onze sources, 2 633 390 noyaux. Dix viennent du dépôt voisin `cellf-supervised`
(crops 2D déjà segmentés et découpés), la onzième est propre à VoCell.

| Source | N | Étiquettes | Notes |
|---|---:|---|---|
| CODEX | 168 992 | 14 familles (`CLASS_MAPPING`) | CRC, Hoechst, ~0,376 µm/px |
| HPA | 673 936 | 17 lignées + non étiqueté | Human Protein Atlas |
| BBBC051 | 232 429 | 11 types rénaux | **crops natifs 32²**, ~0,5 µm/px |
| TissueNet | 1 335 905 | — | DAPI, multi-tissus |
| HelaCytoNuc | 146 910 | — | HeLa, DAPI |
| DSB2018 | 32 155 | — | Data Science Bowl 2018 |
| NuInSeg | 29 332 | — | H&E, multi-organes |
| S-BSST265 | 5 380 | — | DAPI |
| NucleusSegData | 3 250 | — | Huh7 / HepG2 |
| AitslabBioimaging1 | 1 735 | — | U2OS, Hoechst |
| RESTORE | 3 366 | — | volumes 3D **réels** (confocal, DAPI), ~0,15 µm/px |

`data/` est gitignoré, donc l'installation des données est à refaire par machine.

- **Les dix jeux de crops** arrivent par **un seul lien symbolique** vers le
  dépôt voisin, pour éviter de dupliquer 10 Go :
  `ln -s ../../cellf-supervised/Data/crops data/crops`
  (relatif : valide tant que `VoCell/` et `cellf-supervised/` sont voisins).
  L'ancien lien `data/CODEX` n'est plus utilisé par le code.
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

### Ce qui n'est pas homogène entre les sources

- **Les échelles ne sont pas harmonisées.** `rescale_images()` est commenté dans
  le notebook de build de `cellf-supervised` : chaque source est restée à sa
  taille de pixel native, et la plupart sont inconnues (`Data/sources.xlsx` ne
  donne la valeur que pour CODEX et BBBC051). Mesuré sur les crops : le diamètre
  médian d'un noyau va de 13 px (HelaCytoNuc) à 42 px (AitslabBioimaging1).
  C'est laissé tel quel, délibérément — voir la section « Nuage de noyaux ».
- **BBBC051 est en 32²** et n'a jamais été rééchantillonné. `data.crop_2d` le
  complète par du noir jusqu'à 64², ce qui **préserve son échelle** ; le
  redimensionner doublerait le diamètre apparent de chaque noyau et
  inventerait une différence absente des données.
- **L'alignement étiquettes / crops diffère.** CODEX porte une colonne
  `crop_index` et se lit par elle ; HPA et BBBC051 n'en ont pas et leurs lignes
  sont dans l'ordre des crops. Vérifié pour HPA, car `dataset.py` de
  `cellf-supervised` avertit d'une colonne `slot` absente ici : avec les vraies
  étiquettes, l'aire médiane d'un noyau varie d'un facteur 1,70 entre lignées,
  contre 1,08 une fois les étiquettes mélangées. C'est aligné.
- **HPA n'a pas de masque de noyau** : son fond nul est un seuil d'intensité, et
  son support est fragmenté. `cellf-supervised` a un `crops_clean.npy`
  expérimental à côté ; VoCell lit `crops.npy` comme pour les dix autres.

`CLASS_MAPPING` dans `data.py` regroupe ~24 classes CODEX brutes en 14 familles.

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

**Le thème est forcé en sombre** (`.streamlit/config.toml`, versionné), et pas
laissé à `prefers-color-scheme`. Le nuage et le mot-symbole peignent sur un
`#0e1117` codé en dur dans leur CSS — un canvas n'hérite pas du thème de
l'hôte — donc en thème clair l'app encadrait une toile sombre de texte sombre
sur fond blanc, et le nuage pleine page était illisible. Mettre la page d'accord
avec la toile coûte trois lignes ; l'inverse voudrait dire deux palettes à
tenir dans chaque composant.

`requirements.txt` = app + prétraitement (installable partout).
`requirements-analysis.txt` = `torch` pour les trois scripts d'analyse du
latent, dans le même env mais jamais requis par l'app.
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
erroné de `preprocess_restore.py`, coquille de casse `'T Cells'` / `'T cells'`
dans `CLASS_MAPPING` (les 170 crops `CD3+ T cells` formaient une famille à part).

## Règles de travail

- Ne pas committer de données, checkpoints ou `.npy`.
- Toute nouvelle dépendance doit être ajoutée à `requirements.txt`.
- Garder la logique mathématique dans `geometry.py` et `cloud.py` (pures,
  testables, sans Streamlit) et n'importer `streamlit` que dans `app.py`,
  `ui_components.py` et `data.py`.
- Tenir `journal.md` à jour à chaque session de travail significative.
