# Projet `vocab-filter` — résumé des recherches et expérimentations

## Objectifs généraux

Le projet cherche à construire une chaîne de traitement capable de travailler sur le texte complet d’un livre et de répondre à trois questions différentes :

1. **Sélectionner, parmi tous les mots du livre, ceux qui sont intéressants à apprendre pour un apprenant avancé d’anglais.**
2. **Pour un mot donné, estimer quels sont les sens réellement les plus fréquents chez les locuteurs natifs.**
3. **Pour une occurrence précise d’un mot dans le livre, identifier le sens réellement utilisé dans cette phrase.**

Ces trois problèmes sont liés, mais ils ne doivent pas être confondus.

---

# 1. Sélectionner les mots intéressants dans un livre

## 1.1 But

Le but n’est pas simplement de sélectionner les mots rares.

On cherche plutôt des mots qui :

- sont connus de la majorité des locuteurs natifs ;
- sont relativement peu fréquents dans les corpus ;
- peuvent donc être inconnus d’un apprenant avancé même s’ils sont parfaitement naturels pour un natif ;
- ne sont pas du vocabulaire élémentaire déjà connu.

Exemples de mots qui ressortaient bien avec cette logique :

- `approachable`
- `monotone`
- `tapering`
- `recliner`
- `effortlessly`
- `subside`
- `roundabout`
- `adhere`
- `amidst`
- `mallet`

---

## 1.2 Première piste : listes de vocabulaire avancé

Un dépôt GitHub de vocabulaire avancé a d’abord été étudié.

Il contenait notamment :

- `stemdict.txt`
- `output/2ormore-justwords-4992.txt`

Cette approche était trop rigide.

Les listes étaient utiles comme référence, mais elles ne permettaient pas de sélectionner correctement les mots difficiles à partir d’un texte arbitraire.

### Conclusion

Les listes de vocabulaire prédéfinies ne sont pas suffisantes comme moteur principal.

---

## 1.3 Age of Acquisition — AoA

Le jeu de données de Kuperman sur l’âge d’acquisition a ensuite été utilisé.

Principe :

- un mot appris très jeune par un natif peut quand même être inconnu d’un apprenant étranger ;
- l’AoA donne donc une information intéressante sur la familiarité native.

Exemples rencontrés :

- `bow` — AoA ~4.95
- `mess` — AoA ~4.28
- `shut` — AoA ~4.89

Ces mots peuvent être parfaitement naturels pour un natif tout en posant problème à un apprenant.

### Problème rencontré

La lemmatisation automatique pouvait produire des erreurs :

- `opera` → `opus`
- certains noms propres étaient également transformés de façon incorrecte.

Une correction a été mise en place :

1. chercher d’abord la forme exacte dans le fichier AoA ;
2. ne lemmatiser que si le mot ressemble réellement à une forme fléchie (`-s`, `-ed`, `-ing`, etc.).

### Conclusion

L’AoA est utile comme signal secondaire, mais ce n’est pas une mesure directe de difficulté pour un apprenant.

---

## 1.4 CEFR

Plusieurs sources CEFR ont été testées.

La source retenue est le fichier direct :

```text
cefr.csv
```

avec des colonnes du type :

```text
headword ; pos ; CEFR
```

Une base SQLite contenant des niveaux estimés a été rejetée car elle produisait des résultats absurdes pour certains mots.

Exemples observés :

- `dollhouse` → C2
- `cupcake` → C2

### Gestion des niveaux multiples

Un même mot peut avoir plusieurs niveaux selon son rôle grammatical.

Exemples :

- `take` → A1/B1
- `set` → A1/A2
- `bow` → B1/B2
- `mess` → B1/B2
- `shut` → A2/B1

La règle retenue est de conserver tous les niveaux connus.

### Filtrage actuel

Les niveaux élémentaires sont exclus seulement si **tous** les niveaux connus sont élémentaires.

```python
EXCLUDED_CEFR = {"A1", "A2"}
```

Exemples :

- `A1` → exclu
- `A1/A2` → exclu
- `A1/A2/B1` → conservé
- `B1` → conservé
- `?` → conservé

### Conclusion

Le CEFR est utile pour éliminer le vocabulaire manifestement basique.

---

## 1.5 Word Prevalence

La piste la plus convaincante pour sélectionner les mots a été le jeu de données de Brysbaert sur la prévalence lexicale.

Fichier utilisé :

```text
word-prevalence.txt
```

Les champs importants sont :

- `word`
- `Pknown`
- `Nobs`
- `Prevalence`
- `FreqZipfUS`

### `Pknown`

`Pknown` représente la proportion de participants natifs qui connaissent le mot.

Dans le livre étudié :

- 2149 mots uniques étaient présents dans le jeu de données ;
- environ 90.6 % avaient `Pknown >= 0.99`.

Cela permet de chercher les mots :

> connus par presque tous les natifs mais relativement rares.

### Filtre qui a donné les meilleurs résultats

```python
MIN_PKNOWN = 0.99
MAX_PKNOWN = 1.01
```

Puis tri par :

```text
FreqZipfUS croissant
```

Cette combinaison donne des mots :

- presque universellement connus des natifs ;
- mais relativement peu fréquents ;
- donc particulièrement intéressants pour un apprenant avancé.

### Conclusion actuelle pour la sélection des mots

Le meilleur pipeline obtenu jusqu’ici est :

```text
texte du livre
    ↓
tokenisation / mots
    ↓
Word Prevalence
    ↓
Pknown très élevé
    ↓
Zipf relativement faible
    ↓
CEFR
    ↓
élimination des mots A1/A2 uniquement
    ↓
liste de vocabulaire avancé intéressant
```

AoA peut rester un signal secondaire.

---

# 2. Trouver les sens d’un mot les plus utilisés par les locuteurs natifs

## 2.1 WordNet

WordNet a servi de base pour définir les différents sens possibles d’un mot.

Exemple :

```text
run.v.01
operate.v.01
function.v.01
campaign.v.01
...
```

Le nom du synset encode :

```text
lemme.POS.numéro_de_sens
```

Par exemple :

```text
run.v.14
```

signifie simplement :

> 14e sens verbal de `run` dans WordNet.

Ce numéro n’est pas un classement de fréquence.

---

## 2.2 WordNet `lemma.count()` / SemCor

La méthode suivante a été testée :

```python
lemma.count()
```

Cette valeur est liée aux annotations de sens de WordNet / SemCor.

Exemple pour `take` :

- environ 732 occurrences verbales ;
- 42 sens possibles ;
- certains sens ont de nombreuses occurrences ;
- plusieurs sens nominaux ont cependant un compte nul.

### Avantage

SemCor est annoté manuellement et constitue donc un signal de bonne qualité quand il contient suffisamment d’exemples.

### Limite

Le corpus est petit et certains sens sont très peu couverts.

### Conclusion

SemCor est utile comme **prior de fréquence des sens**, mais ne couvre pas suffisamment tous les mots.

---

## 2.3 OMSTI

Le corpus OMSTI a ensuite été testé.

Structure locale :

```text
one-million-sense-tagged-instances-wn30/
    noun/
    verb/
    adj/
    adv/
```

avec des fichiers :

```text
take.key
take.xml
run.key
run_away.key
...
```

Les fichiers `.key` permettent de compter directement les occurrences de chaque sens WordNet.

### Exemple : `take`

Pour `take` verbal :

- environ 3690 occurrences ;
- 36 sens représentés.

Les premiers sens couvraient une grande partie des occurrences.

### Problème majeur

La distribution OMSTI peut être très différente de SemCor.

Exemple avec `run` :

SemCor donnait beaucoup d’occurrences de :

```text
run.v.01
= courir
```

alors qu’OMSTI donnait énormément de :

```text
operate.v.01
= diriger / gérer
```

OMSTI est un corpus construit automatiquement pour l’entraînement de systèmes de désambiguïsation.

Il ne faut donc pas l’interpréter comme une distribution fidèle de l’usage naturel.

### Conclusion

OMSTI peut servir :

- de fallback ;
- pour savoir si un sens est attesté ;
- pour augmenter la couverture.

Mais pas comme mesure absolue de fréquence réelle.

---

## 2.4 Exemple `run away`

Pour :

```text
run away
```

SemCor et OMSTI donnaient :

```text
scat.v.01      12 / 13 ≈ 92.3 %
run_away.v.02   1 / 13 ≈ 7.7 %
```

Cela montre que `run away` est beaucoup plus stable sémantiquement que `run`.

Mais l’échantillon est très petit.

---

## 2.5 Spoken BNC2014

Le Spoken BNC2014 a ensuite été ajouté pour observer de vraies conversations de locuteurs natifs.

Corpus local :

```text
spoken-bnc2014/
    spoken/
        tagged/
        metadata/
```

Les fichiers XML contiennent par exemple :

```xml
<w pos="NN1" lemma="water" class="SUBST" usas="O1:2">water</w>
```

Les informations disponibles incluent :

- forme ;
- lemme ;
- POS ;
- classe grammaticale ;
- tag sémantique USAS ;
- locuteur ;
- conversation.

Le corpus analysé contenait environ :

```text
11.26 millions de tokens
1251 conversations
```

Une comparaison d’expressions liées à l’idée de fuite a donné :

```text
get away     359
escape       187
take off     186
run away     169
bolt          50
flee          17
```

Mais les contextes montrent que ces fréquences brutes sont trompeuses.

Exemple :

```text
get away
```

inclut beaucoup de :

```text
get away with it
```

qui signifie :

```text
s’en tirer / ne pas être puni
```

et non :

```text
s’enfuir
```

### Conclusion

Le Spoken BNC2014 est très utile pour mesurer :

- la fréquence réelle d’un mot ou d’une expression dans la conversation ;
- le nombre de conversations où il apparaît ;
- les contextes réels.

Mais ses tags sémantiques ne correspondent pas directement aux synsets WordNet.

### Architecture retenue pour la fréquence des sens

Pour estimer les sens généralement fréquents :

```text
WordNet
    ↓
inventaire des sens

SemCor
    ↓
prior principal quand le sens est suffisamment attesté

OMSTI
    ↓
fallback / couverture supplémentaire

Spoken BNC2014
    ↓
validation sur l’usage conversationnel réel
    + exemples de contexte
```

Il faut éviter de transformer les comptes OMSTI ou Spoken BNC en probabilités absolues de sens sans analyser les contextes.

---

# 3. Trouver le sens réellement utilisé dans une phrase du livre

## 3.1 Nouvelle information disponible

Le texte du livre peut être fourni sous forme bilingue :

```text
phrase anglaise

traduction française produite par un LLM
```

Exemple :

```text
It's an alley full of cigarette butts.

C’est une ruelle pleine de mégots de cigarettes.
```

Cette information change fortement le problème.

On ne cherche plus :

> quel est généralement le sens le plus fréquent de `butt` ?

mais :

> que signifie `butt` dans cette occurrence précise ?

---

## 3.2 Première tentative : embeddings multilingues

Un modèle Sentence Transformers multilingue a été testé :

```text
paraphrase-multilingual-MiniLM-L12-v2
```

Principe :

- récupérer les sens WordNet possibles ;
- comparer la phrase EN + la traduction FR aux définitions WordNet ;
- classer les sens par similarité.

### Bons résultats

Certains mots fonctionnaient bien :

- `duplex`
- `super`
- `alley`
- `courtyard`

### Échecs

#### `view`

Le système préférait :

```text
position.n.03
= opinion / perspective
```

au lieu du sens visuel :

```text
view.n.02
= the visual percept of a region
```

#### `butt`

Dans :

```text
cigarette butts
```

le système préférait le synset :

```text
cigarette.n.01
```

simplement parce que le mot `cigarette` était très proche lexicalement du contexte.

### Conclusion

La similarité d’embeddings est utile pour présélectionner des candidats, mais pas assez fiable comme décision finale.

---

## 3.3 Cross-encoder MMARCO

Une deuxième passe avec un cross-encoder de ranking a été testée.

Le résultat a parfois empiré les choses.

Exemple :

```text
alley
```

a été classé comme :

```text
bowling alley
```

### Conclusion

Un modèle de recherche / ranking n’est pas un modèle de désambiguïsation lexicale.

---

## 3.4 NLI DeBERTa

Un modèle NLI a ensuite été testé.

Principe :

```text
Premise:
phrase anglaise

Hypothesis:
"In this sentence, the word X means Y."
```

### Résultats

Quelques cas ont été améliorés :

- `standard`
- `access`
- `alley`

Mais d’autres ont échoué :

- `view`
- `butt`

Pour `butt`, le modèle a même choisi :

```text
butt.n.08
= a large cask
```

### Conclusion

Le NLI généraliste n’est pas adapté à une tâche de Word Sense Disambiguation WordNet.

---

## 3.5 GlossBERT

GlossBERT, spécifiquement conçu pour la désambiguïsation à partir de définitions WordNet, a ensuite été testé.

Résultats :

### `alley`

Correct :

```text
alley.n.01
= a narrow street with walls on both sides
```

### `standard`

Correct :

```text
standard.a.01
```

### `view`

Incorrect :

```text
position.n.03
= opinion / perspective
```

alors que le contexte parlait clairement de la vue depuis une fenêtre.

### `butt`

Incorrect :

```text
cigarette.n.01
```

au lieu de :

```text
butt.n.09
= cigarette butt / mégot
```

### Conclusion

Même un modèle WSD spécialisé n’est pas suffisamment fiable sur les cas réels du livre.

---

## 3.6 GlossBERT + preuve française pondérée — solution retenue et implémentée

C’est finalement une combinaison de GlossBERT et d’un signal français **pondéré** (et non
plus un simple bonus binaire) qui a permis d’obtenir des résultats fiables. Implémentation :
`sense_in_context.py`.

### Principe général

```text
score_final(synset) = score_GlossBERT(synset) + score_preuve_française(synset)
```

Les deux composantes sont calculées indépendamment puis simplement additionnées ; le synset
au score final le plus élevé est retenu.

### a) Contexte élargi automatique (et non plus la phrase seule)

Une seule phrase isolée ne suffit pas toujours à désambiguïser — y compris pour un humain.
Exemple mesuré :

```text
"I wish you had more of a view."                      → position.n.03 (opinion)   ← faux
  + la réplique suivante seulement                     → view.n.02 (vue physique) ← juste
    ("It's an alley full of cigarette butts.")
```

Le script retrouve donc la phrase dans le texte intégral du livre (recherche par flux de
tokens normalisés, robuste à une mise en page à deux colonnes qui coupe une réplique en deux)
et fournit à GlossBERT une fenêtre de **±2 répliques voisines**, didascalies comprises (« still
staring out the window » porte souvent l’indice décisif). La recherche du mot cible dans ce
contexte élargi reste restreinte à l’empan de la phrase d’origine, pour ne pas viser par
erreur une occurrence du même mot dans une réplique voisine.

### b) Score de preuve française pondéré par pouvoir discriminant

Un bonus plat pour tout lemme français trouvé dans `omw-fr` (WOLF) s’est révélé insuffisant :
« vue » correspond à 7 des 10 sens de `view`, un bonus uniforme sur ces 7 candidats n’apporte
donc aucune information, alors qu’un lemme exclusif à un seul synset (« mégot » pour
`butt.n.09`) doit trancher. Le score est donc pondéré de façon inverse au nombre de synsets
candidats qui partagent le même lemme français (type IDF) :

```text
informativité(lemme) = 1 / (nb de synsets candidats partageant ce lemme)
score_fr(synset)      = base_source × max(informativité(lemme) × facteur_réclamé)
```

`base_source` vaut 1.0 pour `omw-fr`, et seulement 0.15 pour le repli WoNeF (`wonef-precision.xml`,
déjà présent dans le dépôt) — WoNeF est nettement moins fiable et ne doit servir que de
très léger départage, jamais dominer un score GlossBERT net.

### c) Escompte compétitif — y compris pour les emprunts identiques au mot anglais

Un lemme français déjà « expliqué » par un autre mot anglais de la phrase (ex. : « cigarette »
explique le sens `cigarette.n.01`, indépendamment du sens réel de `butt`) est escompté
(facteur ×0.15) plutôt que compté plein pot — cela neutralise le piège où `cigarette.n.01`
battait `butt.n.09` simplement parce que le mot « cigarette » apparaît littéralement dans la
phrase.

Cette même règle a été étendue au **mot cible lui-même** : un lemme français identique au mot
anglais cible (emprunt, ex. : « score », « duplex ») est également escompté. Ce n’est pas un
correctif ad hoc : toute la valeur du signal français vient de ce que le français découpe les
sens autrement que l’anglais ; quand le mot français *est* le mot anglais, cette valeur est
nulle par construction — et c’est aussi exactement là que l’alignement automatique de WOLF
est le moins fiable. Vérifié sur `score` : dans `omw-fr`, le lemme « score » est associé par
erreur à `mark.n.01` (note scolaire) et pas à `score.n.03` (le bon sens sportif, traduit à
tort par « nombre »). Sans l’escompte, ce bruit fait gagner `mark.n.01` (1.011) contre
`score.n.03` (0.670) alors que GlossBERT seul avait déjà raison.

**Aucune ressource française de remplacement à `omw-fr`/WOLF n’existe** dans l’écosystème
`wn` (vérifié via `wn.projects()` — seuls `omw-fr:1.4` et `omw-fr:2.0`, deux versions du même
WOLF, sont proposés). L’escompte des emprunts contourne donc un défaut de qualité de
l’unique ressource disponible sans en changer.

### d) Lemmes français composés

Le découpage en tokens simples empêchait tout match sur les lemmes composés d’`omw-fr`
(« milk-shake », « clef de voûte », « système tonal »...). Ajout d’une comparaison par
sous-chaîne sur la phrase française normalisée (accents retirés, traits d’union et espaces
unifiés), en complément — pas en remplacement — du matching par radical simple.

### Résultats

14 phrases testées (9 initiales + 5 nouvelles tirées du texte réel de la pièce, y compris des
mots jamais vus par le script auparavant : `score`, `plow`, `key`, `gig`, `shake`) →
**14/14 sens corrects**, avec un récapitulatif automatique `OK/ÉCHEC` calculé contre un champ
`expected` déclaré par cas de test.

| mot | piège | sens retenu |
|---|---|---|
| `butt` | `cigarette.n.01` attire par proximité lexicale | `butt.n.09` (mégot) |
| `view` | ambigu sur la phrase seule, tranché par le contexte élargi | `view.n.02` (vue physique) |
| `score` | emprunt mal traduit dans WOLF | `score.n.03` (score sportif) |
| `access` | traduction FR libre (« clé »), aucun lemme ne matche | `access.n.02`, GlossBERT seul décide |
| `shake`, `gig`, `plow` | peu/pas de preuve française | GlossBERT seul décide correctement |

### Conclusion

Contrairement aux tentatives 3.2–3.5, la combinaison GlossBERT + preuve française **pondérée
et escomptée** (plutôt qu’un bonus plat) est fiable sur les cas testés. Le signal français
n’est décisif que lorsqu’il est réellement informatif ; sinon GlossBERT décide seul, et le
script le montre explicitement dans sa sortie (source du match, nombre de sens partageant le
même lemme, marge sur le 2e).

---

# 4. Changement d’approche (exploration parallèle) : alignement bilingue

> Cette piste (SimAlign, §4–6) a été explorée avant d’aboutir à la solution du §3.6, mais
> **ce n’est pas elle qui a été retenue dans l’implémentation finale**. `sense_in_context.py`
> n’utilise pas SimAlign : il interroge directement `omw-fr` synset par synset (traduction
> lexicale ciblée sur le sens candidat, pas un aligneur de mots généraliste). La distinction
> conceptuelle qu’elle a permis de dégager (§5–6 : un alignement EN↔FR n’est pas une
> équivalence dictionnairique) reste en revanche pertinente et a motivé les garde-fous du
> §3.6 (escompte compétitif, escompte des emprunts).

## 4.1 Idée

Puisqu’une traduction française contextualisée existe déjà, il n’est pas nécessaire de demander à un modèle de deviner directement un synset WordNet.

On peut commencer par résoudre un problème plus simple :

> quel élément de la traduction française correspond au mot anglais ciblé ?

C’est un problème d’**alignement bilingue**.

---

## 4.2 SimAlign

SimAlign a été testé avec :

```text
bert-base-multilingual-cased
```

et trois méthodes :

- `mwmf`
- `inter`
- `itermax`

### Résultats

#### `view`

```text
view ↔ vue
```

Accord :

```text
3/3
```

#### `butts`

```text
butts ↔ mégots
```

Accord :

```text
2/3
```

#### `alley`

```text
alley ↔ ruelle
```

Accord :

```text
3/3
```

#### `access`

```text
access ↔ clé
```

Accord :

```text
3/3
```

#### `standard`

```text
standard ↔ normal
```

Accord :

```text
3/3
```

---

# 5. Limite importante de l’alignement

Un alignement ne signifie pas forcément :

```text
mot anglais = traduction lexicale directe
```

Exemple :

```text
He's the only one who has access.

Il est le seul à avoir la clé.
```

SimAlign donne :

```text
access ↔ clé
```

Mais :

```text
access
```

ne signifie pas littéralement :

```text
clé
```

Le sens lexical est plutôt :

```text
accès
droit d’accès
possibilité d’entrer
```

La traduction LLM a reformulé :

```text
has access
```

en :

```text
avoir la clé
```

parce que cela correspond au contexte.

---

## 5.1 La traduction française n’est pas une vérité absolue

La traduction est produite automatiquement par un LLM.

Elle peut donc :

- reformuler ;
- expliciter ;
- condenser ;
- choisir une expression plus naturelle ;
- déplacer une information ;
- traduire un groupe de mots par un seul mot ;
- traduire un mot par plusieurs mots ;
- parfois se tromper.

Il ne faut donc jamais considérer :

```text
alignement EN ↔ FR
```

comme une équivalence dictionnairique certaine.

---

# 6. Distinction importante à conserver

Pour chaque occurrence, il est utile de distinguer trois choses.

## 6.1 Forme anglaise

Exemple :

```text
access
```

## 6.2 Élément français aligné

Exemple :

```text
clé
```

## 6.3 Sens contextuel réel

Exemple :

```text
avoir accès
avoir le droit d’entrer
pouvoir entrer
```

La sortie idéale serait donc :

```text
word                : access
aligned_french      : clé
contextual_meaning  : droit d’accès / possibilité d’entrer
translation_type    : reformulation
```

Alors que pour :

```text
butt
```

on aurait :

```text
word                : butt
aligned_french      : mégot
contextual_meaning  : mégot
translation_type    : équivalence directe
```

---

# 7. Architecture globale retenue à ce stade

## Étape A — sélectionner les mots intéressants

```text
livre
    ↓
tokenisation
    ↓
Word Prevalence
    ↓
Pknown très élevé
    ↓
Zipf relativement faible
    ↓
CEFR
    ↓
suppression des mots uniquement A1/A2
    ↓
liste de mots candidats
```

Signaux secondaires possibles :

```text
AoA
fréquence dans le livre
POS
```

---

## Étape B — connaître les sens généralement utilisés par les natifs

```text
mot / expression
    ↓
WordNet
    ↓
inventaire des sens
    ↓
SemCor
    ↓
fréquence annotée manuellement
    ↓
OMSTI
    ↓
fallback / couverture
    ↓
Spoken BNC2014
    ↓
fréquence dans la conversation réelle
    + contextes
```

Le résultat ne doit pas être présenté comme une probabilité parfaite.

L’objectif est plutôt de savoir :

- quels sens sont bien attestés ;
- quels sens semblent dominants ;
- quels sens sont rares ;
- quels sens apparaissent réellement dans la langue parlée.

---

## Étape C — trouver le sens utilisé dans le livre

Approche retenue et implémentée (`sense_in_context.py`, voir §3.6) :

```text
mot cible + POS
+
phrase EN (retrouvée dans le texte intégral du livre)
    ↓
contexte élargi (±2 répliques voisines, didascalies incluses)
    ↓
GlossBERT sur le contexte élargi   ──┐
                                      ├──→ score_final = somme
omw-fr (WOLF) par synset,            │        (repli WoNeF
pondéré par pouvoir discriminant,  ──┘         si omw-fr muet)
escompté si déjà "réclamé"
(par un autre mot de la phrase,
ou par le mot cible lui-même
s'il s'agit d'un emprunt)
    ↓
synset WordNet le plus probable
    ↓
définition + synonymes + marge sur le 2e sens
```

Validé sur 14 phrases réelles du livre (14/14 sens corrects, récapitulatif automatique
`OK/ÉCHEC`). `omw-fr` fournit une définition anglaise, un synset précis et une référence
lexicale standardisée dès que la traduction française contient un indice exploitable ; sinon
GlossBERT seul décide à partir du contexte élargi.

---

# 8. Ce qui a été rejeté ou rétrogradé

## Comme moteur principal de sélection de vocabulaire

- listes prédéfinies de mots avancés ;
- AoA seul ;
- CEFR estimé automatiquement.

## Comme moteur principal de fréquence des sens

- OMSTI seul.

## Comme moteur principal de sens en contexte

- embeddings multilingues seuls ;
- cross-encoder MMARCO ;
- NLI généraliste (bug additionnel identifié : `CrossEncoder.predict()` renvoie des logits
  bruts, comparés sans softmax entre paires — aggrave un modèle déjà hors-domaine) ;
- GlossBERT seul ;
- SimAlign / alignement bilingue généraliste comme brique centrale (exploré §4–6, remplacé
  par une interrogation ciblée de `omw-fr` synset par synset, plus simple et plus robuste,
  voir §3.6) ;
- un bonus français **plat** (tout lemme trouvé vaut le même score) — insuffisant car un
  lemme partagé par de nombreux sens (« vue » pour 7/10 sens de `view`) n’apporte alors
  aucune information et peut faire gagner un sens faux mais mieux traduit (« mégot » vs
  « cigarette » pour `butt`, avant pondération).

Ces outils peuvent rester utiles comme signaux secondaires, mais aucun n’a été assez fiable
seul. La combinaison retenue (§3.6) est **GlossBERT + preuve française `omw-fr` pondérée par
pouvoir discriminant et escomptée quand elle n’est pas informative** (déjà expliquée par un
autre mot de la phrase, ou emprunt identique au mot cible), sur un contexte élargi extrait
automatiquement du texte source plutôt que sur la phrase isolée.

---

# 9. État actuel et travail restant — désambiguïsation en contexte (§3.6)

## 9.1 Ce qui fonctionne et est implémenté

- `sense_in_context.py` : pipeline complet GlossBERT + preuve française pondérée, contexte
  élargi extrait automatiquement de `The Humans - Stephen Karam.txt`, escompte compétitif et
  escompte des emprunts, repli WoNeF, matching des lemmes français composés.
- Harnais de test : champ `expected` par cas, récapitulatif `OK/ÉCHEC` automatique en fin
  d’exécution, filtre en ligne de commande pour ne relancer qu’un sous-ensemble de mots
  (`python sense_in_context.py view butt`) sans recharger inutilement tous les cas.
- 14 phrases réelles du livre validées manuellement (14/14).

## 9.2 Limites connues, non résolues

- **Le harnais de test reste alimenté à la main.** `TESTS` est une liste écrite manuellement
  (mot, POS, phrase EN, traduction FR inventée) ; il n’y a pas encore de pipeline qui extrait
  automatiquement des triplets (mot, phrase, traduction) depuis le livre entier et les
  Étapes A/B. C’est un harnais de vérification ponctuelle, pas (encore) un traitement
  batch du livre.
- **Traduction française absente du texte source.** Le livre n’existe qu’en anglais dans le
  dépôt ; les traductions françaises de `TESTS` ont été rédigées manuellement pour les
  besoins des tests. Une vraie traduction LLM du livre entier (mentionnée en §3.1 comme
  hypothèse de départ) n’a pas été produite ni branchée.
- **Reformulations non lexicales.** Quand la traduction ne contient aucun lemme français du
  bon synset (`access` → « clé », qui n’est le lemme d’aucun sens de `access` dans WordNet),
  la preuve française vaut 0 et GlossBERT décide seul, sans filet — fonctionne sur les cas
  testés, mais aucune garantie générale.
- **Qualité de `omw-fr` (WOLF) au-delà des emprunts.** Seuls deux défauts précis ont été
  corrigés (emprunts identiques, artefacts WoNeF du type « butte » collée sur tous les sens
  de `butt`) ; WOLF peut comporter d’autres erreurs de traduction non détectées par ces
  heuristiques.
- **Fenêtre de contexte fixe.** `CONTEXT_WINDOW = 2` répliques est une valeur choisie
  empiriquement sur un seul cas (`view`), non ajustée automatiquement par phrase ni validée
  sur un échantillon plus large.
- **Dépendance à la mise en forme du texte source.** L’extraction du contexte élargi repose
  sur un découpage heuristique en répliques (lignes non vides, noms de personnages en
  majuscules écartés) et une recherche de la phrase par flux de tokens avec repli sur préfixe
  — validé sur `The Humans - Stephen Karam.txt`, pas testé sur un texte de mise en forme
  différente. Si la phrase n’est pas retrouvée, le script retombe silencieusement (mais de
  façon signalée dans la sortie) sur la phrase seule, avec le risque de sous-désambiguïsation
  déjà documenté pour `view`.
- **Pas de mesure agrégée.** Le seul indicateur de fiabilité est le récapitulatif 14/14 sur un
  échantillon choisi à la main ; aucune évaluation sur un jeu de phrases plus large ou tiré
  aléatoirement du livre n’a été faite.

## 9.3 Prochaines étapes possibles

- Brancher ce module en aval de l’Étape A (sélection de mots) et de l’Étape B (inventaire des
  sens) pour traiter automatiquement chaque occurrence retenue du livre, plutôt que des cas
  choisis à la main.
- Produire (ou obtenir) une traduction française réelle du livre entier phrase par phrase, et
  vérifier que l’extraction de contexte élargi + la recherche de phrase tiennent à cette
  échelle.
- Élargir l’échantillon de validation (viser un sous-ensemble plus large et si possible tiré
  aléatoirement, pas seulement des cas choisis pour leur intérêt pédagogique).
- Étudier si `CONTEXT_WINDOW` doit varier selon la longueur ou l’ambiguïté de la phrase plutôt
  que rester fixe à 2.

