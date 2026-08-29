# Baseline batch — inventaire des tâches LLM (Lot M0)

Lot **M0** du plan `./fix_pipeline/multi_models/plan_multi_models.md` (§0, §1, §2, §6). Photo de
l'état actuel du code, **lecture seule** : aucun code de routage n'a été modifié pour produire ce
rapport. Toutes les ancres `fichier.py:ligne` ont été vérifiées en lisant le code, pas déduites.

Note : `pipeline/config.py` apparaît modifié (`M`) dans `git status`, mais `git diff pipeline/config.py`
est vide — seule une différence de fin de ligne (LF/CRLF) est en cause. Il n'y a rien à préserver ni à
annuler pour ce lot.

---

## 1. Trois notions à ne pas confondre (§0, §2.2)

Le plan interdit explicitement de fusionner ces trois mécanismes. Les trois coexistent déjà dans le
code, à des endroits différents :

| Notion | Définition | Où, aujourd'hui |
|---|---|---|
| **Batch prompt** (`mode_batch`) | Jusqu'à `batch_size` cas métier dans **un seul** prompt ; la réponse JSON liste une décision par clé/id. C'est l'objet de ce plan. | `S6-translate-frontier`, `S6-backtranslate`, `S6-judge-dossier`, `S6-reassign` (voir §3) |
| **Parallélisme `litellm.batch_completion`** | Plusieurs **prompts distincts** (unitaires ou lots) envoyés en parallèle via des workers HTTP. Indépendant de `mode_batch`. | `sense_fr_frontier.py:310` et `sense_fr_reassign.py:310`, tous deux avec `max_workers=config.SENSE_FR_FRONTIER_MAX_WORKERS` (`config.py:280`, valeur `10`). `run_stage_b`/`run_stage_c` (`sense_fr_adjudicate.py`) utilisent `litellm.completion` (pas `batch_completion`) : leurs lots sont envoyés **séquentiellement**, aucun parallélisme HTTP. |
| **`nlp.pipe(batch_size=64)`** | Micro-batching spaCy, aucun LLM impliqué. Hors sujet de ce plan. | `pipeline/analyze.py:227` ; `fix_pipeline/detection_benchmark/phase3_run_rules_plus.py:112`, `phase_tokenizer_fix_probe.py:119`, `tokenizer_boundary_fix.py:73` |

---

## 2. Inventaire — tâches production (`run_pipeline.py`)

| `task_id` | Étape | Module / point d'entrée | Client actuel | Batch **prompt** aujourd'hui | Constante & valeur | Prompt **unitaire** présent | Prompt **lot** présent | `batch_allowed` cible |
|---|---|---|---|---|---|---|---|---|
| `S3-judge-occurrence` | S3 | `mwe_judge.judge_occurrence` (`mwe_judge.py:437`), appelée en boucle par `run()` (`:656`) | `llm.call_json` (stdlib, backend ollama/catgpt) | **non** — 1 occurrence / appel | `S3_JUDGE_BATCH_SIZE = 50` (`config.py:141`) — **non branché sur ce chemin** (voir §5) | oui — `OCC_SYSTEM_PROMPT` (`:165`) + `OCC_PROMPT_TEMPLATE` (`:176`) | **non** en production (un prototype existe hors production, voir §4) | **oui** (prototype eval à 50) |
| `S3-definition-cluster` | S3 | `mwe_judge.choose_cluster_definition` (`:353`), boucle `assign_cluster_definitions` (`:395`) | `llm.call_json` (aucun `model=` passé → `config.llm_model()` global) | **non** — 1 cluster / appel | aucune | oui — `DEFINITION_SYSTEM_PROMPT` (`:306`) + `DEFINITION_PROMPT_TEMPLATE` (`:314`) | **non** | **oui** (à introduire) |
| `S5-arbitrate` | S5 | `senses.arbitrate` (`senses.py:703`), appelée à `:893` quand la politique calibrée le demande | `llm.call_json` (aucun `model=` passé → global) | **non** | aucune | oui — `ARBITRATION_SYSTEM` (`:685`) + `ARBITRATION_TEMPLATE` (`:692`) | **non** | **oui** (à introduire) |
| `S6-translate-frontier` | S6b-1 | `sense_fr_frontier.run` → `_translate_batches` (`sense_fr_frontier.py:282`) | LiteLLM — `litellm.batch_completion` (`:310`) | **oui** | `SENSE_FR_FRONTIER_BATCH_SIZE = 40` (`config.py:279`), lue à `sense_fr_frontier.py:531` | **non** — aucun chemin N=1 dédié ni testé | oui — `build_user_prompt` (`:173`) + schéma `BatchTranslations` (`:97`) | **oui** (déjà) |
| `S6-backtranslate` | S6b-2 B | `sense_fr_adjudicate.run_stage_b` (`:422`) → `_backtranslate_batch` (`:372`) | LiteLLM — `litellm.completion`, **un appel par lot** (`:409`) | **oui**, découpage réel `targets[i:i+batch_size]` (`:427`) | `batch_size: int = 40` — **littéral de signature**, pas de constante `config` ; l'appelant `run()` ne le passe pas (`:620`) | **non** | oui — system inline (`:389`) + schéma `_BatchGuesses` (`:386`) | **oui** (déjà) |
| `S6-judge-dossier` | S6b-2 C | `sense_fr_adjudicate.run_stage_c` (`:445`) | LiteLLM — `litellm.completion` (`:525`) | **oui, mais jamais découpé** — voir écart §5 | `batch_size: int = 20` — **paramètre mort** | **non** | oui — system inline (`:478`) + schéma `_BatchVerdicts` (`:475`) | **oui** (déjà) |
| `S6-reassign` | S6c | `sense_fr_reassign.run` (`:510`) → `_translate_batches` (`:286`) | LiteLLM — `litellm.batch_completion` (`:310`) | **oui** | `SENSE_FR_REASSIGN_BATCH_SIZE = 10` (`config.py:295`), lue à `sense_fr_reassign.py:532` | **non** | oui — `build_user_prompt` (`:269`) + schéma `ReassignBatch` (`:187`) | **oui** (déjà) |

---

## 3. Hors run par défaut (conservés, même registre)

| `task_id` | Module | Rôle | Batch prompt aujourd'hui | Mécanisme réel | `batch_allowed` v1 |
|---|---|---|---|---|---|
| `S6-translate-local` | `sense_fr.llm_translate_votes` (`sense_fr.py:230`) | Traduction « dictionnaire », sans contexte livre (`--retry-pending`) | **non** | `SENSE_FR_LLM_DRAWS = 3` appels séquentiels, 3 **formulations** de prompt différentes (`TRANSLATE_INSTRUCTIONS`, `:182`) — un mécanisme de **votes/consensus** (`SENSE_FR_LLM_MIN_AGREE = 2`), pas un lot au sens du plan | **non** en v1 (votes = autre mécanisme) |
| `S6-backtranslate-local` | `sense_fr.llm_backtranslate` (`:263`) | Rétro-traduction locale | **non** | 1 appel par candidat | **non** en v1 (lié au chemin local) |

---

## 4. Hors production (eval) — alignement optionnel

| `task_id` | Module | Note |
|---|---|---|
| `EVAL-S3-judge` | `fix_pipeline/evaluate_s3_judges.py` | `_run_local_batch` (`:131`) est un **prototype**, opt-in via `--batch` (`:223`) ; il garde `len(cases) > config.S3_JUDGE_BATCH_SIZE` (`:133`) — **c'est le seul consommateur de cette constante dans tout le dépôt**. Schéma compact propre à l'eval, distinct du schéma S3 de production : `{"decisions":[{"case_id","label","canonical_form","pos","confidence"}]}` (`:157`), `cache_metadata.protocol = "s3-judge-eval-1-compact-batch-prompt-2"` (`:163`). Il **prouve la faisabilité** d'un prompt lot S3 et fournit un patron réutilisable (détection manquants/doublons `:169-186`, repli `incertain` sur `batch_error`), mais n'est pas branché sur `mwe_judge.run()` et n'est pas directement réutilisable en production sans découplage. Modèles en dur : `FRONTIER_MODEL`/`LOCAL_MODEL` (`:22-23`). |
| `EVAL-S6-joint` | `pipeline/eval_frontier_ablation.py` | `run_joint(cases, model, batch_size=10)` (`:340`) — batch 10 déjà aligné sur `S6-reassign`. `DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-sol"` (`:37`) est **délibérément** hors `config.ALLOWED_FRONTIER_MODELS` : le juge de l'ablation doit rester indépendant du modèle candidat (voir commentaire `config.py:305-307`). |

---

## 5. Étapes sans LLM (hors registre, rappel §1 du plan)

`corpus`, `analyze` (S1), `mwe` (S2), `select` (S4), `sense_fr_adjudicate` Stage A (offline, aucun
appel réseau), `export` (S7). spaCy / GlossBERT / LaBSE ne sont pas des tâches de ce plan.

---

## 6. Constats figés (§2.3 du plan)

- **`S3_JUDGE_BATCH_SIZE = 50` (`config.py:141`) n'est PAS branché sur `mwe_judge.run()`.** Son
  unique lecteur dans tout le dépôt est `evaluate_s3_judges.py:133`. La boucle de production S3
  (`mwe_judge.py:640-663`) juge strictement une occurrence par appel LLM, sans jamais consulter cette
  constante.
- **`evaluate_s3_judges._run_local_batch` est un prototype**, pas un chemin de production : il prouve
  qu'un prompt lot S3 est faisable (schéma `decisions[]`, détection manquants/doublons, repli
  `incertain`), mais reste couplé à l'eval (opt-in `--batch`, `cache_metadata` versionné séparément du
  protocole S3 de production `S3_PROMPT_VERSION`/`S3_DECISION_SCHEMA_VERSION`).
- **S6c (`SENSE_FR_REASSIGN_BATCH_SIZE = 10`)** : le commentaire `config.py:284-294` documente qu'un
  run réel avec batch 40 (24 entrées dans un seul appel) a produit des réassignations fausses (ex.
  `beat.n.08` → `beat.n.06` pour `fr="petite pause"`), corrigées en rejouant les mêmes entrées seules
  dans un lot de 6. Le batch 40 est **dégradant, mesuré** ; le défaut 10 — celui réellement validé par
  `eval_frontier_ablation.run_joint(batch_size=10)` — est à **préserver comme défaut de configuration**,
  pas un simple choix arbitraire à écraser au premier réglage global.

---

## 7. Autres écarts et pièges relevés (au-delà du strict §2.3)

- **`run_stage_c` (`S6-judge-dossier`) : `batch_size=20` est un paramètre mort.** La fonction construit
  **un seul** prompt à partir de `targets` en entier (`sense_fr_adjudicate.py:493-516`, un seul
  `litellm.completion` à `:525`) — aucun découpage `targets[i:i+batch_size]` comme dans `run_stage_b`
  ou `_translate_batches`. L'appelant `run(...)` ne passe même pas ce paramètre (`:646`). Le lot
  réellement envoyé au modèle = `len(residual)`, sans plafond. Pour M2 : « préserver le défaut 20 »
  n'est **pas** le comportement actuel (qui n'a aucun plafond) — c'est un choix à acter explicitement,
  pas une simple reprise de constante.
- **`run_stage_b` (`S6-backtranslate`) : `batch_size=40` est un littéral de signature**, pas une
  constante dans `config.py`. Le découpage est réel (`sense_fr_adjudicate.py:427`), mais rien n'est
  configurable et l'appelant ne le passe pas explicitement (`:620`, valeur par défaut silencieuse).
- **Aucune des 4 tâches S6 déjà batchées n'a de prompt/schéma unitaire.** Elles n'ont qu'une seule
  famille de prompts (lot). La « règle des deux prompts » (§0, §6) est donc **non satisfaite** pour
  `S6-translate-frontier`, `S6-backtranslate`, `S6-judge-dossier`, `S6-reassign` — le plan le
  reconnaît déjà en listant ces quatre tâches comme « à compléter / expliciter le chemin N=1 » (§6).
- **Aucun test ne couvre le basculement unitaire/lot.** `test_sense_fr_frontier.py`,
  `test_sense_fr_adjudicate.py`, `test_sense_fr_reassign.py`, `test_llm_backends.py` ne contiennent
  aucune occurrence du mot « batch » (vérifié par recherche texte).
- **Deux familles de cache disque, aucune n'inclut explicitement `task_id` ni `mode_batch` :**
  - `llm._cache_path` (`llm.py:26-53`) : clé = `{backend, model, system, prompt, temp, cache_metadata}` ;
  - côté S6 : `_cache_path(model, system, user)` propre à chaque module
    (`sense_fr_frontier.py:275`, `sense_fr_reassign.py:279`), plus des digests inline dédiés
    `backtranslate_*` / `judge_*` (`sense_fr_adjudicate.py:402-405`, `:518-521`).
  Le mode (unitaire vs lot) n'est distingué qu'**indirectement**, via le texte du prompt qui diffère.
  Le §4.3 du plan exige que `task_id`, `mode_batch` et `batch_size` effectif entrent explicitement
  dans la clé de cache — ce n'est le cas nulle part aujourd'hui.
- **Magasin S3** (`mwe_stores`) : `occurrence_store_key` (`mwe_judge.py:222-234`) inclut déjà
  `backend`, `model`, `prompt_version` et `schema_version` — un changement de backend/modèle invalide
  correctement les entrées. Il **n'inclut pas** le mode batch : à ajouter en M3 si `mode_batch` devient
  configurable pour `S3-judge-occurrence`.
- **Modèles hérités à migrer (§3.4 du plan)** : `SENSE_FR_FRONTIER_MODEL = "openai/gpt-5-mini"`
  (`config.py:278`) et la liste blanche mono-élément `ALLOWED_FRONTIER_MODELS` (`:308`), gardée par
  `require_frontier_model` (`:311-317`), appliquée dans `sense_fr_frontier.run:471` et
  `sense_fr_reassign.run:510`. `run_pipeline.py --llm-model` (`:115`, appliqué via
  `config.configure_llm` à `:120`) ne pilote **que** le backend local (`OLLAMA_MODEL`/`CATGPT_MODEL`)
  — il ne touche **aucune** des quatre tâches S6.
- **`PROVIDER=chatgpt`** : aucune occurrence dans le code du dépôt (recherche exhaustive) — bien
  décoratif aujourd'hui, conforme à la description du §0 du plan.

---

## 8. Rappel — la règle des deux prompts (§0, §6 du plan)

Pour toute tâche `batch_allowed: true`, le livrable final (hors périmètre de ce lot M0) devra fournir
**deux familles de prompts** distinctes :

| Artefact | Unitaire | Batch |
|---|---|---|
| System prompt | dédié ou partagé | souvent partagé + consignes de lot |
| User template | 1 item | liste d'items + règle « une décision par clé » |
| Schéma / response_format | objet unique | `{decisions: [...]}` (ou équivalent) |
| Parser | 1 décision | map id → décision ; détection manquants/doublons |
| Test | 1 fixture | lot de 2–3 fixtures |

Le code sélectionnera la famille selon la config effective (`mode_batch` **et** `batch_size >= 2`).
Si `batch_allowed: false`, seule la famille unitaire existe ; activer `mode_batch` doit lever une
erreur de configuration explicite, jamais un repli silencieux vers un faux lot.

**État actuel par tâche `batch_allowed: true` :**

| `task_id` | Prompt unitaire | Prompt lot | Travail restant (hors M0) |
|---|---|---|---|
| `S6-translate-frontier` | absent | présent (`build_user_prompt` + `BatchTranslations`) | créer/expliciter le chemin N=1 |
| `S6-backtranslate` | absent | présent (`_backtranslate_batch`) | créer/expliciter le chemin N=1 |
| `S6-judge-dossier` | absent | présent, mais **sans plafond réel** (`run_stage_c`) | créer le chemin N=1 **et** rebrancher le découpage en lots |
| `S6-reassign` | absent | présent (`build_user_prompt` + `ReassignBatch`) | créer/expliciter le chemin N=1 |
| `S3-judge-occurrence` | présent (production) | absent en production (prototype eval seulement) | créer un prompt lot distinct, inspiré de `_run_local_batch` sans coupler à l'eval |
| `S3-definition-cluster` | présent | absent | créer un prompt lot |
| `S5-arbitrate` | présent | absent | créer un prompt lot |

---

## 9. Défauts de registre — proposition pour M1 (non implémentée ici)

Figés sur le comportement **actuel**, sans changer de valeur :

- `S3-judge-occurrence`, `S3-definition-cluster`, `S5-arbitrate` : backend global
  (`VOCAB_LLM_BACKEND=ollama`, modèle `OLLAMA_MODEL=mistral-small:24b` par défaut), `mode_batch=false`.
- `S6-translate-frontier` : `openai/gpt-5-mini`, `mode_batch=true`, `batch_size=40`.
- `S6-backtranslate` : `openai/gpt-5-mini`, `mode_batch=true`, `batch_size=40` (à faire remonter en
  constante `config` — aujourd'hui un littéral de signature, voir §7).
- `S6-judge-dossier` : `openai/gpt-5-mini`, `mode_batch=true`, `batch_size=20` — **à acter** comme
  nouveau comportement (le code actuel n'a aucun plafond), pas comme reprise d'un défaut existant.
- `S6-reassign` : `openai/gpt-5-mini`, `mode_batch=true`, `batch_size=10` (préserver — dégradation
  mesurée au-delà, voir §6).

---

## 10. Gate M0

- [x] Chaque `task_id` du §1 du plan apparaît dans ce rapport (7 production, 2 hors-run, 2 eval).
- [x] Batch prompt vs non-batch correctement distingué de `batch_completion` et de `nlp.pipe`.
- [x] `S3_JUDGE_BATCH_SIZE` noté comme non branché sur `mwe_judge.run()`.
- [x] `evaluate_s3_judges._run_local_batch` noté comme prototype.
- [x] Règle des deux prompts rappelée, avec état actuel par tâche.
- [x] Aucun code de routage modifié : `git status`/`git diff` ne montrent que ce fichier en plus
      (la modification préexistante de `pipeline/config.py` est une différence de fin de ligne, sans
      contenu, non liée à ce lot).

## Prochain gate

Lot **M1** — registre `pipeline/llm_tasks.py` (ou section dédiée de `config.py`) : descripteurs de
tâche, parseur d'overrides env (`VOCAB_LLM_<TASK_ID>=model;batch=...;batch_size=...`), repli sur
`VOCAB_LLM_BACKEND`/modèles globaux pour S3–S5, alias `PROVIDER=chatgpt` → `catgpt`, validation stricte
(`mode_batch=true` interdit si `batch_allowed=false`, `batch_size` requis ≥ 1 si batch). Tests
unitaires hors réseau. Ne pas brancher S3–S6 sur ce registre dans M1 (sauf lecture de défauts si
nécessaire aux tests). **M1 n'est pas commencé.**
