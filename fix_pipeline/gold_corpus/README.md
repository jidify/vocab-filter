# Gold corpus v0 — *The Humans* (span detection)

Premier prérequis du « Q0-3 — benchmark des architectures de détection » proposé par
l'utilisateur : un corpus de spans annotés à la main, pour mesurer le **rappel** de
plusieurs architectures de détection (spaCy actuel, lexiques/patrons, LLM local,
ensemble) avant de choisir laquelle remplace `pipeline.multi_token`. Ce corpus ne
sert qu'à ça — aucun scorer n'est fourni ici, c'est l'étape suivante.

Livrable : `the_humans_gold_v0.jsonl` — un objet JSON par segment retenu, avec la
liste de ses spans-repères (`gold_spans`).

## Segmentation

Segments extraits via le chemin de production exact : `pipeline.corpus.load_segments()`
→ `PlayAdapter`, filtré sur `kind != "hors_oeuvre"`. **2535 segments** au total pour
`The Humans - Stephen Karam.txt`, identique au compte de la comparaison spaCy
précédente (`pipeline_out/spacy_quick_compare/report.md`) — donc `segment_idx` est
directement comparable aux artefacts déjà produits (`disagreements.jsonl`, etc.).

## Méthode d'échantillonnage

99 segments uniques, combinant quatre lots déterministes (aucun choix ad hoc en
dehors de ceux-ci) :

| Lot | Segments | Méthode |
|---|---:|---|
| `known_difficult` | 12 | Cas déjà identifiés comme difficiles dans la comparaison spaCy précédente (hyphen modifiers, ponctuation de dialogue, faux positifs NER, composé manqué par lg) — relocalisés et ré-annotés en gold plutôt que de faire confiance à la sortie spaCy d'alors. |
| `random_seed_42` | 46 | Tirage aléatoire (`random.Random(42).sample(...)`) sur tous les segments éligibles (≥ 3 tokens) parmi les 2535, pour couvrir la pièce entière et pas seulement la scène d'ouverture déjà vue dans les échanges précédents. |
| `phrasal_verb_pass` | 14 | Recherche ciblée de verbes à particule et d'idiomes (grep sur les particules courantes juste après un verbe, confirmation manuelle des vrais positifs, rejet des faux positifs comme collocations littérales). |
| `phrasal_verb_separable_hardening` | 27 | Passe de renforcement ciblée (voir section dédiée plus bas) sur `phrasal_verb_separable` uniquement, la catégorie jugée quasiment la plus importante par l'utilisateur. |

Aucun chevauchement entre les quatre lots. Script générateur des 72 premiers
segments : `_build_gold.py` ; script de la passe de renforcement :
`_build_hardening_pass.py`. Offsets systématiquement calculés par recherche de
sous-chaîne, jamais tapés à la main.

## Taxonomie des catégories

| Catégorie | Définition |
|---|---|
| `simple_word` | Un seul mot difficile/rare/informel valant la peine d'être signalé à un apprenant. |
| `nominal_compound` | Composé nominal (deux noms ou plus, éventuellement avec modificateur à trait d'union). |
| `multi_token_entity` | Entité nommée multi-tokens (lieu, date, groupe démonymique...). |
| `phrasal_verb_separable` | Verbe à particule séparable (« clean up », « figure out »...). |
| `phrasal_verb_inseparable` | Verbe à particule ou prépositionnel non séparable (« come back », « deal with »...). |
| `idiom` | Expression figée non compositionnelle (« bounce back », « come back to earth »...). |
| `hard_negative` | Span qui RESSEMBLE à un candidat plausible mais ne doit PAS être retenu (`is_gold: false`) — composé compositionnel ordinaire, verbe+préposition littéral, borne erronée réellement produite par un modèle dans le test précédent, etc. |

Chaque span porte aussi `edge_case` (bool) + `edge_type` quand la difficulté est
structurelle plutôt que lexicale : `hyphen_modifier`, `possessive_boundary`,
`dialogue_dash`, `bracket_nonverbal`, `hyphen_tokenization`.

## Comptes

99 segments, **109 spans** au total (**82 positifs / 27 `hard_negative`**),
**22 segments intentionnellement vides** (aucune unité à signaler — utile pour
mesurer les faux positifs, pas seulement le rappel).

| Catégorie (positifs) | Comptes |
|---|---:|
| `phrasal_verb_separable` | 28 |
| `idiom` | 17 |
| `phrasal_verb_inseparable` | 14 |
| `nominal_compound` | 12 |
| `simple_word` | 7 |
| `multi_token_entity` | 4 |
| **`hard_negative` (tous types confondus)** | **27** |

| `edge_type` | Comptes |
|---|---:|
| `hyphen_modifier` | 7 |
| `bracket_nonverbal` | 3 |
| `dialogue_dash` | 2 |
| `possessive_boundary` | 1 |
| `hyphen_tokenization` | 1 |

| Lot | Segments | Spans |
|---|---:|---:|
| `known_difficult` | 12 | 29 |
| `random_seed_42` | 46 | 33 (22 segments vides) |
| `phrasal_verb_pass` | 14 | 18 |
| `phrasal_verb_separable_hardening` | 27 | 29 (23 positifs / 6 `hard_negative`) |

## Exemples représentatifs

- **`nominal_compound` + hyphen_modifier** (idx 281) : `ground-floor apartment` —
  les trois modèles spaCy du test précédent l'avaient tronqué en `floor apartment`
  (agr:06). Le `hard_negative` associé encode explicitement cette mauvaise borne.
- **`idiom` totalement invisible pour spaCy** (idx 156) : `all over the place`
  (désorienté) — ni composé ni entité, donc structurellement hors de portée d'un
  pipeline spaCy seul, quel que soit le modèle. C'est exactement le type d'unité
  que le benchmark Q0-3 doit vérifier.
- **`phrasal_verb_inseparable` vs `hard_negative` dans le même segment** (idx 87) :
  `attention shifts` (sujet+verbe, faux positif que sm ET lg avaient produit comme
  composé — dis:05/dis:26) contre `shifts away` (vraie lecture verbe-particule des
  mêmes mots).
- **`hard_negative` piège d'ambiguïté lexicale** (idx 2107) : `fit in` au sens
  littéral spatial (« ça va rentrer dans la voiture »), pas au sens idiomatique
  social (« s'intégrer ») — teste qu'un détecteur ne généralise pas aveuglément un
  motif de surface.

## Passe de renforcement — `phrasal_verb_separable` (2026-08-28)

L'utilisateur a jugé cette catégorie « quasiment la plus importante » après
lecture des comptes initiaux (seulement 5 exemples, contre 14-17 pour les autres
familles MWE). Passe additive uniquement : les 72 lignes d'origine ne sont ni
modifiées ni réordonnées ; 27 nouvelles lignes sont ajoutées en fin de fichier
avec `sample_reason: "phrasal_verb_separable_hardening"`.

Recherche par motif ciblé sur l'ensemble des 2535 segments (regex sur
verbe + objet-pronom/GN + particule, `_build_hardening_pass.py`), en couvrant
délibérément les trois formes de séparation (objet pronom, GN court, GN long) et
en cherchant des paires même-lemme séparé/non-séparé dans le corpus existant.
Résultat : **23 nouveaux spans positifs** `phrasal_verb_separable` (28 au total
avec les 5 d'origine) et **6 nouveaux `hard_negative`** ciblés sur les pièges
propres à cette catégorie (préposition confondue avec particule, PP littéral
confondu avec particule, ambiguïté lexicale non tranchable).

Exemples marquants de cette passe :

- **Paire même-lemme séparé/non-séparé — `bring up`** : `bringing up marriage`
  (non séparé, idx 1277, déjà dans le corpus d'origine) contre `bring it up`
  (séparé par le pronom objet, idx 1902, ajouté ici). Un détecteur doit
  reconnaître les deux comme la même unité lexicale — c'est le test le plus
  important de cette passe. Même schéma pour `figure out` (idx 624 non séparé /
  idx 2093 séparé).
- **Séparation par long GN coordonné** (idx 2485) : `puts the blanket and pan
  down` — le GN inséré contient un coordinateur (« and »), un piège pour un
  détecteur naïf qui s'arrêterait au premier nom.
- **Paire minimale positif/hard_negative sur la même particule** : `turns it on`
  (idx 2496, activation, phrasal verb réel) contre `places it on the windowsill`
  (idx 2429) et `Puts it on the table` (idx 2480) — même forme de surface
  (verbe + pronom/GN + « on » [+ complément locatif]), seule la première est
  idiomatique.

## Limites (explicites)

- **Un seul texte** : *The Humans* uniquement, en anglais simple (`source: "plain"`).
  Ni la version bilingue, ni un second livre — la proposition Q0-3 complète en
  demande au moins deux. Prochaine étape si ce v0 ne suffit pas à trancher.
- **Annotateur unique et non indépendant** : ces annotations sont faites par
  l'assistant, pas par un relecteur humain aveugle. Comme pour la revue des
  désaccords spaCy précédente, à vérifier par un humain en cas de doute sur une
  conclusion qui s'appuierait dessus.
- **Taille choisie pour la vitesse, pas la puissance statistique** : 109 spans
  suffisent pour distinguer de grosses différences de rappel entre architectures,
  pas pour un intervalle de confiance serré ni pour départager deux architectures
  proches.
- **Pas un substitut au corpus gold complet** demandé dans la proposition Q0-3 —
  seulement de quoi débloquer une première comparaison de rappel entre spaCy,
  lexiques/patrons, LLM local et ensemble. Si les résultats sont serrés entre
  architectures, il faudra revenir à un corpus plus grand, multi-texte, et
  vérifié par un humain indépendant avant de trancher.
- **`e-mails` (idx 1400)** : la tokenisation spéciale `EMAIL_SPECIAL_CASES` de
  production (`pipeline/analyze.py::get_nlp`) n'est pas automatiquement présente
  dans un runner de benchmark isolé — à ajouter explicitement, sinon l'écart
  mesuré sur ce cas précis reflète la méthodologie du test, pas la qualité réelle
  d'une architecture (déjà observé comme artefact dans le test spaCy précédent,
  dis:29).

## Fichiers

- `the_humans_gold_v0.jsonl` — le corpus (livrable principal).
- `verify_offsets.py` — vérifie mécaniquement chaque span contre le texte réel
  chargé via `pipeline.corpus.load_segments()` (le même chemin que la production).
  À relancer après toute modification manuelle du JSONL.
- `_build_gold.py`, `_dump_segments.py`, `_segments_dump.jsonl`,
  `_selected_segments.jsonl`, `_random_sample_idx.json` — scripts et caches
  intermédiaires ayant servi à construire les 72 premiers segments (préfixés `_`,
  non nécessaires pour l'utiliser, gardés pour la traçabilité/reproductibilité).
- `_build_hardening_pass.py`, `_hardening_new_lines.jsonl` — script et sortie
  intermédiaire de la passe de renforcement `phrasal_verb_separable` (mêmes
  raisons de conservation que ci-dessus).
