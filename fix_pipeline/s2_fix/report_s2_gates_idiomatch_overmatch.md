# Rapport avant/après — Portes de validation S2 (idiomatch)

## Statut

**Corrigé.** Complète `bug_idiomatch_slot_overmatch.md`, qui documentait le
symptôme (`know someone` = 111 occurrences) sans encore proposer de
correctif. Ce rapport couvre l'implémentation, la mesure avant/après et les
limites restantes.

## Ce qui a été corrigé

Trois mécanismes distincts, tous propres à la dépendance `idiomatch`
(voir `pipeline/mwe_gates.py` pour l'analyse complète en commentaire) :

- **Porte A — slots ouverts saturés sans lien syntaxique** (`know someone`
  ← `I know`/`you know`/`it is not known`). Cause : `idiomatch/builders.py
  ::openslot_passive` réordonne les tokens *et* rappelle `slop()` sur un
  motif déjà slopé, produisant un motif passif qui accepte n'importe quel
  pronom à distance. Corrigé par trois règles cumulatives (alternative
  réordonnée admise seulement sur un vrai passif à ≥ 2 ancres lexicales,
  ancrage syntaxique direct du slot, interdiction d'être sujet d'une ancre
  précédente) — voir `pipeline/mwe_gates.py::_slot_gate`.
- **Porte C — idiomes tout-grammaticaux réalisés en emploi auxiliaire**
  (`I do` ← `I've done`/`Did you see it`). Corrigé en exigeant une surface
  strictement contiguë et en rejetant tout basculement VERB (citation) →
  AUX (occurrence) — voir `_grammatical_gate`. Les entrées de forme
  `[verbe léger][particule]` (`do in`, `do up`…) sont explicitement
  exclues de cette porte : structurellement indiscernables d'un vrai
  phrasal verb séparable, elles restent hors périmètre plutôt que d'être
  rejetées à tort sur un futur livre.
- **Porte D — ancre lexicale corrompue** (`wing it`, compilé sur le lemme
  erroné `we`). `nlp("wing it")` étiquette `wing` en VBG et applique la
  règle `-ing → -e`. Réparé au chargement du matcher (`get_matcher()`) en
  reconstruisant le motif à partir de `nlp("winging it")`, correctement
  lemmatisé — l'idiome redevient détectable au lieu d'être simplement
  neutralisé.

Hors périmètre, documenté dans le plan : la sur-fusion par remplissage
lexical (`go to` ← `Go home to`, Correction S2-1) — risque réel de perdre
de vrais phrasal verbs, nécessite une règle de connexité syntaxique plus
fine que celles ci-dessus, validée sur davantage d'occurrences.

## Mesure avant/après (*The Humans*, run complet)

| | Avant | Après |
|---|---:|---:|
| occurrences idiomatch brutes | 1044 | 1042 (−2 : interaction du greedy-matching avec la porte D) |
| occurrences écartées avant S3 | 0 | 259 (130 slot, 91 grammatical_gap, 38 grammatical_auxiliary) |
| `know someone` | 111 | **3** (`knew her`, `knew me`, `knows all this`) |
| `I do` | 62 | **12** |
| `wing it` | 0 candidat réel, 14 faux `we ... it` | **0** faux match ; idiome réel désormais détectable |
| `shut one's mouth` | 2 | **2** (préservé — passif réel à 2 ancres) |
| `let someone go` | 7 | **7** (préservé) |
| `or something` | 16 | **0** (déjà tout rejeté à raison — aucun cas réel dans ce livre) |
| types MWE candidats | 542 | 522 |
| occurrences MWE candidates | 1580 | 1313 |

Gates nommés du plan d'action, contrôlés après coup — tous intacts :
`burn out`=4, `let someone go`=7, `figure out`=4, `grow up`=8,
`talk about`=11, `clean up`=5, `crack open`=2, `smart ass`=4.

## Écart avec le plan initial (transparence)

Le plan de conception estimait 152 occurrences rejetées par la porte C
(85 gardées). Le chiffre réel mesuré après implémentation et tests est
**129 rejetées / 109 gardées** (91 gap + 38 aux, sur 238 occurrences
appartenant à la classe tout-grammaticale). L'écart vient d'un bug dans le
script d'exploration utilisé pendant la planification : sa condition de
rejet testait `citation.pos in ("VERB","AUX")` au lieu de `citation.pos ==
"VERB"` strictement. Cette version large aurait rejeté à tort `how are
you`/`being that` (dont le verbe léger est *déjà* un auxiliaire dans sa
forme de citation — ce n'est pas un basculement parasite). Le test
`test_contiguous_light_verb_expressions_are_kept` (`test_mwe_gates.py`) a
détecté l'écart avant merge, en échouant sur `How are you?` avec la
condition large ; `pipeline/mwe_gates.py::_grammatical_gate` implémente la
version stricte, correcte, et le test passe.

Autre correction en cours d'implémentation : le commentaire de
`classify()` affirmait initialement que les portes A et C sont
« mutuellement exclusives par construction ». Faux — `someone`/`something`
sont à la fois des slots pour idiomatch (`POS: PRON`) et des `PRON`
ordinaires pour la classe tout-grammaticale (`or something`, `up to
something` satisfont les deux définitions). L'ordre d'évaluation de
`classify()` (slot d'abord) tranche correctement sans double comptage —
vérifié : `or something` sort à 0/16 rejetés, jamais compté deux fois — le
commentaire a été corrigé pour refléter ça plutôt que la fausse hypothèse.

## Fichiers modifiés

- `pipeline/mwe_gates.py` *(nouveau)* — les trois portes.
- `pipeline/mwe_alignment.py` — `_iter_assignments` (nouveau, affectations
  complètes spec↔token) ; `_all_alignments` réécrit comme réduction de
  celui-ci, comportement externe inchangé (`test_mwe_alignment.py` vert
  sans modification).
- `pipeline/mwe.py` — `get_matcher()` répare la porte D après
  `from_pretrained` ; `find_candidates()` pose `rejected_by` sur chaque
  candidat idiomatch ; `run()` partitionne et écrit
  `mwe_rejected_candidates.jsonl` en plus de `mwe_candidates.jsonl`.
- `pipeline/config.py` — `MWE_REJECTED_CANDIDATES_PATH`.
- `test_mwe_gates.py` *(nouveau)* — 15 tests contre le vrai matcher de
  production, y compris le test négatif demandé à l'étape 4 du rapport de
  bug (`know someone` ne capture plus `you know`/`I know` intransitifs).
- `fix_pipeline/q0_2_stratified_corpus.json` /
  `test_q0_2_regression.py` — nouvelle strate
  `sur_appariement_idiomatch`, enregistrée `passes` (pas
  `known_failure` : c'est une correction, pas un défaut caractérisé).

## Vérifications exécutées

- `test_mwe_gates.py` : 15/15.
- `test_mwe_alignment.py`, `test_mwe_fusion.py`, `test_s4_reservation.py`,
  `test_rules_plus.py`, `test_q0_2_regression.py` : verts (une seule
  exception, pré-existante et confirmée identique sur la base non modifiée
  via `git stash` : `mwe_occurrences_heterogenes` dépend d'un
  `mwe_decisions.jsonl` figé d'un run S3 antérieur, non affecté par ce
  correctif).
- Suite complète du dépôt (`python -m unittest discover`) : 316 tests,
  même unique échec pré-existant.
- `python -m pipeline.mwe` exécuté deux fois de suite : 259 rejets stables
  (déterminisme confirmé).

## Ce qui reste ouvert

- La famille B (sur-fusion par remplissage lexical, `come to`/`go to`) —
  Correction S2-1, plus risquée, non traitée ici (voir le plan).
- `mwe_decisions.jsonl`/`mwe_confirmed_spans.jsonl` restent dans l'état
  intermédiaire signalé par `bug_idiomatch_slot_overmatch.md` (run S3
  arrêté à ~3-6 lots sur 32) — un nouveau run S3 complet bénéficiera
  directement de la baisse de ~25 % des candidats idiomatch, mais n'a pas
  été relancé dans le cadre de ce correctif.
