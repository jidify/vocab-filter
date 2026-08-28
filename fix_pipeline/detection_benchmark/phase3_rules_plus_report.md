# Q0-3 — Phase 3 : `rules_plus`

## Décision (critère d'arrêt n°1) — ATTEINT, on s'arrête ici

**`rules_plus` franchit 3 des 4 seuils du critère d'arrêt n°1 du plan** (un
seul suffisait) : **+23,9 points de rappel MWE exact** (seuil : +10),
**+28,6 points sur les phrasal verbs séparables** (seuil : +15), et
**87,5% des erreurs structurelles spaCy connues corrigées** (seuil : 75%).
Seul le 4ᵉ seuil (rappel global ≥ 95%) n'est pas atteint (70,7%), ce qui
était attendu — ce seuil visait à identifier un remplacement quasi-parfait
d'un coup, pas le critère discriminant ici.

*(Chiffres après une itération sur le rappel — voir "Itération : deux
affinements du rappel" plus bas. La version initiale de `rules_plus`
franchissait déjà les 3 mêmes seuils, avec des marges légèrement plus
faibles : +21,1 / +21,4 / 87,5%.)*

**Conséquence directe (règle du plan, section Phase 4) : la Phase 4 (probe
LLM local) n'est PAS lancée.** Aucun LLM n'a tourné dans cette phase, y
compris "pour voir". `rules_plus` reste la piste à approfondir (Phase 6,
issue 1 : "`rules_plus` suffit").

## Méthodologie

Produit par `fix_pipeline/detection_benchmark/phase3_run_rules_plus.py`
(tracké, ré-exécutable) + `fix_pipeline/detection_benchmark/rules_plus.py`
(les générateurs `rules_plus`, tracké, réutilisable). Mêmes 99 segments
gold que Phase 2, jamais le livre entier. Scoré avec
`fix_pipeline/detection_benchmark/scorer.py` (Phase 1) contre
`fix_pipeline/gold_corpus/the_humans_gold_v0.jsonl` (gelé).

- **Base = Baseline 2 de la Phase 2, reproduite à l'identique** (mêmes
  fonctions de production : `pipeline.analyze.analyze_segments`,
  `pipeline.mwe.find_candidates`/`merge_candidate_sources`/
  `structural_prefilter`, `pipeline.multi_token.detect`) — recalculée dans
  ce script plutôt que réimportée, pour garantir que `rules_plus` s'ajoute
  à un ensemble mesuré dans la même exécution (recall exact 46,3%,
  identique à Phase 2 aux arrondis près — voir `phase3_baseline2_report.json`).
- **`rules_plus` = union(Baseline 2, nouveaux générateurs)** — jamais de
  retrait : spaCy ne reçoit aucun pouvoir de rejet, conformément au plan.
  Un second passage `nlp.pipe` (même tokenizer que la production,
  `pipeline.analyze.get_nlp()`, cas spéciaux e-mail/lexique custom inclus)
  donne accès aux `Doc` spaCy dont `analyze_segments` ne laisse rien
  échapper à l'appelant — nécessaire aux scanners lemme/POS ci-dessous.
  Coût : +5,2s sur 99 segments (négligeable).
- **Nouveaux générateurs** (`rules_plus.py`, aucune réimplémentation d'un
  détecteur de production) :
  1. **PARSEME** (`load_parseme_pairs`) : relit tel quel le magasin gelé
     déjà utilisé en production par `pipeline.vpc`
     (`data/vpc/parseme-en-1.3-train-vpc-lexicon.json`, 86 paires TRAIN) —
     aucun nouveau téléchargement.
  2. **WordNet, verbes multi-mots** (`wordnet_phrasal_verb_lexicon`) :
     lemmes verbaux WordNet (via `nltk.corpus.wordnet`, déjà une
     dépendance de production — `pipeline/analyze.py`) dont tous les mots
     après le verbe appartiennent à une classe fermée de particules.
     Complément nécessaire à PARSEME : vérifié empiriquement que PARSEME
     TRAIN (86 entrées) ne contient PAS `figure_out`, `burn_out`,
     `calm_down`, `clean_up`, `turn_off`, `turn_on`, `steal_away`,
     `stink_up`, `ring_out`, `soak_up`, `keep_down`, `bounce_back`,
     `camp_out` — tous présents comme lemmes WordNet.
  3. **Patron de phrasal verb séparable/inséparable**
     (`scan_phrasal_verb_candidates`) : scanner linéaire sur les
     lemmes/POS spaCy (**jamais sur les dépendances** — contrairement à
     `pipeline.vpc`, spaCy ne reçoit ici aucun pouvoir de rejet), fenêtre
     bornée à 6 tokens interposés (couvre "give something this nice away"
     = 3 mots interposés et "puts the blanket and pan down" = 4, avec
     marge), interrogeant PARSEME ∪ WordNet fusionnés.
  4. **Règles de bornes** :
     - **trait d'union** : chaîne à trait d'union libre
       (`hyphen_chain_candidates`, sans candidat de base requis — couvre
       "turn-of-the-century"/"smart-ass", jamais proposés par
       `multi_token`) + extension à gauche d'un candidat `multi_token`
       existant à travers un trait d'union (`hyphen_extend_existing` —
       couvre "ground-floor apartment"/"triple-A school"/"phys-ed
       classes", tronqués par `multi_token` car spaCy tokenise le trait
       d'union en ponctuation séparée) ;
     - **possessif** : troncature du suffixe possessif d'un candidat
       `multi_token` existant (`possessive_trim_existing` — corrige
       exactement le cas documenté en Phase 0, "New York City's" →
       "New York City") ;
     - **ponctuation de dialogue** (`crosses_hard_boundary`) : tout span
       calculé (scanner phrasal verb ou n-gramme WordNet) est rejeté s'il
       enjambe `?!;:—–[]…` ou une ellipse `. . .` — motivé par les pièges
       "stomps around?—we"/"the Mary statue?—we've" du corpus gold ;
     - **frontières de propositions** : le scanner phrasal verb
       s'interrompt net dès qu'un second verbe/auxiliaire fléchi est
       rencontré avant la particule (jamais sur une simple virgule/"and"
       interne à l'objet interposé — vérifié sur "puts the blanket and
       pan down", où "and" est à l'intérieur de l'objet, pas une frontière).
  5. **Lexiques MWE existants** : déjà dans la Baseline 2 via
     idiomatch/lexique custom (Phase 2) — rien de nouveau à ajouter ici,
     l'union les inclut mécaniquement.

## Les 4 seuils du critère d'arrêt n°1

| Seuil | Exigence | Mesuré | Franchi ? |
|---|---|---:|:---:|
| Rappel MWE exact (`role=lexical_candidate`) | +10 pts vs baseline2 | baseline2 43,7% → rules_plus 67,6% = **+23,9 pts** | **OUI** |
| Rappel phrasal verbs séparables exact | +15 pts vs baseline2 | baseline2 57,1% → rules_plus 85,7% = **+28,6 pts** | **OUI** |
| Erreurs structurelles spaCy connues corrigées | ≥75% | **7/8 = 87,5%** (voir détail ci-dessous) | **OUI** |
| Rappel global exact, sans explosion | ≥95% | 46,3% → 70,7% (n'atteint pas 95%) | non |

*(un seul seuil franchi suffisait pour arrêter — 3 le sont)*

### Détail des 8 erreurs structurelles spaCy connues (`edge_type` `hyphen_modifier`/`possessive_boundary` du corpus gold, Phase 0)

| Segment | Span gold | Baseline 2 | `rules_plus` |
|---:|---|:---:|:---:|
| 75 | `turn-of-the-century` | absent | **corrigé** (chaîne à trait d'union libre) |
| 75 | `ground-floor/basement duplex tenement apartment` | absent | **toujours absent** (slash + composé à 3 têtes ; aucun candidat de base à étendre, cas le plus dur du corpus par construction — voir sa note) |
| 75 | `New York City` | absent (spaCy produit `New York City's`, `York City`, `New York`) | **corrigé** (troncature du possessif) |
| 78 | `mid-century` | déjà correct (NER DATE) | déjà correct |
| 281 | `ground-floor apartment` | absent (spaCy produit `floor apartment`, tronqué) | **corrigé** (extension à gauche à travers le trait d'union) |
| 570 | `triple-A school` | absent (spaCy produit `A school`) | **corrigé** |
| 570 | `phys-ed classes` | absent (spaCy produit `ed classes`) | **corrigé** |
| 1696 | `smart-ass` | déjà correct (lexique custom `pipeline/mwe.py::CUSTOM_IDIOMS`) | déjà correct |

## Tableau complet (comparaison à Baseline 2, mêmes 6 indicateurs que Phase 2)

| Indicateur | Baseline 2 (recalculée dans ce run) | `rules_plus` (union) |
|---|---:|---:|
| Rappel global (exact / chevauchement) | 46,3% / 97,6% | **70,7%** / 97,6% |
| Rappel MWE (`role=lexical_candidate`) | 43,7% / 98,6% | **67,6%** / 98,6% |
| Rappel phrasal verbs séparables | 57,1% / 100% | **85,7%** / 100% |
| Rappel `protective_span` (`multi_token_entity`) | 50% / 100% | **100%** / 100% |
| `boundary_accuracy` globale | 47,5% | **72,5%** |
| Faux positifs sur les 27 `hard_negative` | 10/27 (37,0%) | 13/27 (48,1%, inchangé par l'itération — voir plus bas) |

Par catégorie (exact) : `nominal_compound` 50,0%→**83,3%** ; `multi_token_entity`
50,0%→**100%** ; `phrasal_verb_separable` 57,1%→**85,7%** ; `phrasal_verb_inseparable`
28,6%→**50,0%** ; `idiom` 29,4%→**41,2%** ; `simple_word` 71,4%→85,7% (gain
non attendu — voir note ci-dessous).

Chiffres complets : `pipeline_out/detection_benchmark/phase3_baseline2_report.json`,
`phase3_rules_plus_report.json`, `phase3_new_only_report.json` (régénérés à
chaque run, non trackés).

## Itération : deux affinements du rappel

Après un premier passage de `rules_plus` (chiffres initiaux : rappel MWE
exact 64,8%, phrasal verbs séparables 78,6%), une relecture manuelle des
24 spans gold "vus mais mal bornés" (candidat chevauchant sans borne
exacte — jamais les 2 spans totalement invisibles, distincts) a identifié
deux causes structurelles corrigeables sans toucher au périmètre du plan
(pas de nouveau générateur, pas de règle sémantique) :

1. **Bruit de coordination mal étiqueté par spaCy** (`_is_coordination_noise`
   dans `rules_plus.py`) : sur seg2485 ("Erik puts the blanket and pan
   down"), `en_core_web_sm` étiquette "pan" **VERB, `dep_="conj"` vers
   "puts"** au lieu de NOUN coordonné à "blanket" — une erreur de tagging
   du modèle spaCy sm, pas un vrai second verbe fléchi. Le scanner de
   phrasal verbs avortait donc à tort avant d'atteindre "down" (règle
   "frontière de proposition = second verbe fléchi", trop stricte). Le
   garde-fou ajouté n'avorte plus sur un token VERB/AUX qui est en réalité
   coordonné (`dep_ == "conj"`) au verbe de départ lui-même — ça
   **assouplit** la frontière (laisse passer un candidat que spaCy aurait
   fait manquer), ça ne donne jamais à spaCy un pouvoir de rejet
   supplémentaire.
2. **Rejeu du lexique custom de production à fenêtre large**
   (`scan_custom_idiom_candidates`) : `pipeline/mwe.py::CUSTOM_IDIOMS`
   enregistre "crack open" pour idiomatch, mais le matcher idiomatch
   tourne à `slop=2` (`get_matcher()`, "n=2") — il ne peut jamais relier
   un idiome à travers un objet interposé de 3 mots ou plus. Sur seg197
   ("cracks **the bathroom door** open", exactement l'exemple cité dans le
   plan comme cas canonique), idiomatch échouait donc silencieusement
   malgré l'entrée enregistrée. Rejouer ce même lexique déjà validé
   manuellement en production (`CUSTOM_IDIOMS` + `data/custom_lexicon.jsonl`,
   pas un nouveau lexique) avec le scanner à fenêtre de 6 tokens de
   `rules_plus` contourne la limite de `slop=2` sans toucher à idiomatch
   lui-même.

Effet mesuré (mêmes 99 segments, même scorer) : **+2,8 points de rappel
MWE exact** (64,8%→67,6%), **+7,1 points sur les phrasal verbs séparables**
(78,6%→85,7%), **0 nouveau `hard_negative` capturé** (13/27 inchangé) —
gain propre, sans coût de précision mesurable sur ce corpus. `n_candidates`
561→564 (+3 seulement, l'essentiel du gain vient de bornes corrigées sur
des candidats déjà partiellement trouvés, pas de nouveaux candidats en
masse).

Pistes identifiées mais **délibérément non poursuivies** dans cette
itération (hors périmètre "règles de bornes", relèvent du jugement
sémantique contextuel) : plusieurs spans gold restants (`end up buying`,
`fighting back tears`, `weight's been lifted off his chest`, `cleans up
the mess`, etc.) sont des idiomes/phrasal verbs correctement repérés en
partie, mais dont le gold exige d'inclure un complément qui suit la
particule (objet direct, gérondif, syntagme prépositionnel). Étendre
automatiquement un span de phrasal verb au complément qui suit
recréerait précisément le risque que les `hard_negative`
`starts up the staircase`/`dump her down the spiral staircase` du corpus
sont conçus pour piéger (distinguer un complément idiomatique fixe d'un
complément locatif littéral est une question sémantique, pas structurelle)
— routé vers S3 (`pipeline/mwe_judge.py`), pas vers une règle de bornes
supplémentaire.

## Sur-génération

564 candidats (`rules_plus`) contre 521 (baseline2) sur les mêmes 1131
tokens — **498,7 candidats/1000 tokens contre 460,7**, soit +8,3% relatif.
Pas d'explosion : les nouveaux générateurs à eux seuls n'ajoutent qu'une
poignée de dizaines de candidats, dont l'essentiel recoupe déjà des spans
que baseline2 avait trouvés autrement (dédoublonnage sur
`(segment_idx, start_char, end_char)`).

## Faux positifs `hard_negative` (13/27, +3 par rapport à baseline2)

Les 10 pièges déjà capturés par baseline2 restent capturés (union, jamais
de retrait). 3 nouveaux, tous produits par le scanner de phrasal verbs
(lemme+POS, sans jugement sémantique — attendu, l'arbitrage contextuel est
hors périmètre de cette phase, voir S3) :

- `knocks on` (seg883) — `knock`+`on` est un couple verbe+particule
  attesté (WordNet/PARSEME) ; le piège teste précisément la lecture
  littérale ("knock on the door"), indiscernable d'un patron lemme+POS
  sans contexte sémantique.
- `fit in` (seg2107) — même mécanisme (`fit`+`in`, lecture spatiale
  littérale vs. sociale idiomatique).
- `take her back up` (seg538) — span documenté dans le corpus gold
  lui-même comme "genuinely ambiguous... no single clean phrasal-verb
  reading defensible" (aucune lecture propre n'est défendable même pour
  un relecteur humain) : notre scanner tombe exactement dans
  l'ambiguïté que la note du corpus anticipait.

Conforme à la portée du plan (Phase 1) : le taux de capture des
`hard_negative` est un signal diagnostique documenté, pas un critère de
rejet de `rules_plus` — voir le critère d'arrêt n°1, qui ne le mentionne
pas.

## Note — gain non attendu sur `simple_word`

`hyphen_chain_candidates` (chaîne à trait d'union libre, sans lien avec
les mots simples) matche par construction "e-mails" (seg1400, gold
`simple_word`, `edge_type=hyphen_tokenization`) : une chaîne alphabétique
reliée par un trait d'union est exactement ce que ce span est, même si le
générateur ne visait pas cette catégorie. Effet de bord bénin, pas un
générateur dédié aux mots simples (hors périmètre de `rules_plus`,
Phase 3 du plan).

## Limite assumée

`ground-floor/basement duplex tenement apartment` (seg75, la version
longue à trois têtes nominales et modificateur trait-d'union+slash) reste
non détectée : aucun des générateurs `rules_plus` ne propose de candidat
de base sur ce segment pour "duplex tenement apartment" (spaCy ne produit
STRICTEMENT AUCUN candidat `compound` sur ce segment, vérifié dans
`multi_token_candidates.jsonl`), donc ni l'extension par trait d'union ni
la troncature du possessif n'ont de base à corriger. Une chaîne
nominale générique (regrouper des noms consécutifs sans dépendance)
recouvrerait ce cas mais sort du périmètre du plan pour cette phase
(patrons de phrasal verbs + règles de bornes nommées, pas un
réimplémentation générique du compound-chunking de `multi_token`) — à
documenter pour une Phase 6/implémentation ultérieure si `rules_plus` est
repris en production.
