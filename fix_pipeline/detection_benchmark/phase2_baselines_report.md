# Q0-3 — Phase 2 : deux baselines spaCy

Rapport produit par `fix_pipeline/detection_benchmark/phase2_run_baselines.py`
(tracké, ré-exécutable) + `fix_pipeline/detection_benchmark/normalize_adapter.py`
(l'adaptateur de normalisation). Artefacts bruts (candidats produits, occurrences)
dans `pipeline_out/detection_benchmark/` (gitignored, régénérable — relancer le
script pour les reproduire). Scoré avec `fix_pipeline/detection_benchmark/scorer.py`
(Phase 1) contre `fix_pipeline/gold_corpus/the_humans_gold_v0.jsonl` (gelé,
99 segments / 109 spans).

## Méthodologie

- **Segments** : uniquement les 99 segments couverts par le corpus gold (retrouvés
  dans `pipeline.corpus.load_segments()` par `segment_idx` — assertion dans le
  script que les 99 sont bien tous présents). Jamais le livre entier : le corpus
  gold n'est utilisé que pour sélectionner CES segments, jamais copié ni utilisé
  pour biaiser un détecteur.
- **Aucune réimplémentation** : chaque candidat vient d'un appel direct au code de
  production — `pipeline.analyze.analyze_segments()` (boucle spaCy unique, comme
  `analyze.py::run()`, y compris `EMAIL_SPECIAL_CASES`/`custom_lexicon` via
  `get_nlp()`), `pipeline.multi_token.detect()`, `pipeline.mwe.find_candidates()` /
  `load_vpc_candidates()` / `merge_candidate_sources()` / `structural_prefilter()`
  (mêmes fonctions que `mwe.py::run()`), et le filtre à 4 conditions de
  `pipeline/select.py::iter_content_occurrences` pour les mots simples (sans son
  `is_covered` — cette réservation appartient à S3/S4, hors détection).
- **Schémas de sortie confirmés par lecture du code, pas supposés** (voir
  `normalize_adapter.py` pour le détail) :
  - `pipeline.multi_token.detect()` → `start_char`/`end_char` absolus,
    `candidate_types` (ex. `"named_entity:GPE"`, `"nominal_compound"`).
  - `pipeline.vpc` (`PhrasalVerbDetection`) → schéma `*_char_span`/`*_char_spans`
    différent, jamais consommé directement : normalisé via `mwe.load_vpc_candidates()`
    (la fonction de production réelle qui projette VPC vers le même schéma
    qu'idiomatch), pas réimplémenté ici.
  - `pipeline.mwe.find_candidates()` (idiomatch) + VPC fusionnés → `start_char`/
    `end_char` = enveloppe complète du match (inclut l'objet interposé d'un
    phrasal verb séparable), `member_char_spans` = membres verbe+particule(s).
  - `pipeline/custom_lexicon.py` : confirmé qu'il ne produit **aucun span propre**
    — c'est un magasin de données consommé par `mwe.get_matcher()`
    (`add_idioms`) et `analyze.get_nlp()` (cas spéciaux du tokenizer). Son effet
    est donc déjà entièrement inclus dans les candidats idiomatch/occurrences
    ci-dessus ; pas d'adaptateur séparé nécessaire pour lui.
  - Le scorer n'utilise la `category`/`source` d'un candidat que pour l'audit —
    l'appariement avec le gold se fait uniquement sur les offsets caractères
    (voir `scorer.py`), donc une catégorie de candidat approximative
    n'affecte aucune métrique.
- **Baseline 1** : `pipeline.multi_token` seul → 35 candidats.
- **Baseline 2 (l'ensemble réel du pipeline actuel)** : multi_token (35) ∪
  mwe fusionné+filtré idiomatch/VPC (77, sur 64 idiomatch bruts + 31 VPC non
  rejetés) ∪ mots simples (409, filtre `select.py` sans `is_covered`) = 521
  candidats.
- `total_tokens=1131` (compté sur le texte des 99 segments réellement analysés,
  passé explicitement au scorer plutôt que de compter sur son repli implicite).

## Les 6 indicateurs demandés par le plan (Phase 2)

| Indicateur | Baseline 1 — `multi_token` seul | Baseline 2 — pipeline complet |
|---|---:|---:|
| **Rappel global** (exact / chevauchement) | **8.5%** / 20.7% (7/82 exact, 17/82 chevauché) | **46.3%** / 97.6% (38/82 exact, 80/82 chevauché) |
| **Rappel MWE** (`role=lexical_candidate` : nominal_compound + idiom + phrasal verbs) | **7.0%** / 18.3% (5/71 exact, 13/71 chevauché) | **43.7%** / 98.6% (31/71 exact, 70/71 chevauché) |
| **Rappel phrasal verbs séparables** | **0%** / 7.1% (0/28 exact, 2/28 chevauché) | **57.1%** / 100% (16/28 exact, 28/28 chevauché) |
| **Rappel `protective_span`** (`multi_token_entity`) | **50%** / 75% (2/4) | **50%** / 100% (2/4) — inchangé, seul multi_token les détecte dans les deux baselines |
| **Erreurs de bornes** (chevauché mais pas exact) | 10/17 spans chevauchés mal bornés (`boundary_accuracy` = 41.2%) | **42/80 spans chevauchés mal bornés** (`boundary_accuracy` = 47.5%) |
| **Faux positifs sur les 27 pièges `hard_negative`** | **8/27 (29.6%)** | **10/27 (37.0%)** |

Chiffres complets (par catégorie, par rôle, temps d'exécution, taux de génération) :
`pipeline_out/detection_benchmark/baseline1_report.json` et `baseline2_report.json`
(régénérés à chaque run, non trackés).

## Lecture

- **Baseline 2 bat largement baseline 1** sur tout ce qui compte pour la priorité
  projet (MWE > NER) : +36.7 points de rappel MWE exact, et surtout les phrasal
  verbs séparables passent de 0% à 57.1% — `multi_token` seul ne les voit
  structurellement jamais (il ne détecte que composés nominaux et entités,
  jamais les constructions verbe-particule).
- **Le rappel par chevauchement de baseline 2 est déjà très haut (97.6%)** :
  presque tous les spans gold sont "vus" par au moins un détecteur. L'écart
  avec le rappel exact (46.3%) est donc presque entièrement un problème de
  **bornes**, pas de couverture — cohérent avec le constat du benchmark spaCy
  précédent (`pipeline_out/spacy_quick_compare/report.md`) sur les troncatures.
  Cible privilégiée pour `rules_plus` (Phase 3) : des règles de bornes
  (traits d'union, ponctuation de dialogue, objets interposés dans les phrasal
  verbs séparables), pas de nouveaux générateurs de candidats.
- **`protective_span` reste identique entre les deux baselines** (2/4 exact) :
  seul `multi_token` (NER + `compound`) produit ce type de span dans le
  pipeline actuel ; idiomatch/VPC/mots simples n'y contribuent jamais. Les
  2 entités manquées restent donc un point aveugle commun aux deux baselines,
  pas un problème résolu par l'union des détecteurs.
- **Faux positifs `hard_negative` en hausse avec l'union (10/27 vs 8/27)** :
  les 8 pièges de baseline 1 (`York City`, `attention shifts`, `floor
  apartment`, `fluorescent light`...) restent capturés par `multi_token` dans
  les deux baselines ; l'union en ajoute 2 de plus, `opens the door` (seg528 et
  seg629) — un candidat VPC/idiomatch qui matche un piège de phrasal verb à
  interprétation littérale. Attendu : `structural_prefilter` ne fait aucun
  jugement sémantique (voir sa docstring), donc l'union ne peut qu'ajouter des
  pièges, jamais en retirer.
- **Mots simples (409 candidats, filtre `select.py` sans `is_covered`)** :
  rappel exact 71.4% (5/7), chevauchement 85.7% (6/7) — déjà proche du plancher
  pratique attendu (le filtre à 4 conditions n'exclut presque rien), cohérent
  avec la note du plan : ce n'est pas une fonction de "détection" à améliorer,
  c'est un filtre déjà quasi-permissif par construction. Le seul span
  `simple_word` manqué en chevauchement est un cas hors du périmètre normal du
  filtre (à vérifier en Phase 3 si le budget le permet, sans réouvrir le corpus
  gold).

## Sur-génération (taux de candidats / 1000 tokens)

- Baseline 1 : 30.9 candidats / 1000 tokens (35 candidats sur 1131 tokens).
- Baseline 2 : 460.7 candidats / 1000 tokens (521 candidats sur 1131 tokens) —
  dominé par les 409 candidats "mots simples" (par construction : ce filtre
  n'est pas censé être sélectif, la sélectivité pédagogique vient plus tard en
  S4). En excluant les mots simples, le sous-total MWE+entités (112 candidats)
  donne ~99 / 1000 tokens, à comparer à la Phase 3 (`rules_plus`).

## Ne pas perdre de vue pour la Phase 3

- Le goulot n'est pas la couverture (97.6% de chevauchement) mais les
  **bornes exactes** — `rules_plus` doit prioriser les règles de bornes
  (traits d'union, ponctuation de dialogue/didascalie, objets interposés des
  phrasal verbs séparables) plutôt que de nouveaux générateurs de candidats.
- `protective_span` (entités) n'a qu'une seule source dans le pipeline actuel
  (`multi_token`/NER spaCy) — un point à surveiller si `rules_plus` ne touche
  pas à cette famille.
- Les 2 nouveaux `hard_negative` capturés par l'union (`opens the door` ×2)
  sont un signal que la fusion sans arbitrage sémantique (attendu, S3 pas
  encore lancé) augmente mécaniquement le risque de faux positifs — pas un bug
  du merge, juste son comportement documenté (`merge_candidate_sources` ne
  retire jamais un piège que l'une des deux sources aurait accepté).
