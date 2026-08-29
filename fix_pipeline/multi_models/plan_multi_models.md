# Plan — configuration multi-modèles et batch par tâche LLM

## 0. Objectif et contraintes

L'objectif est de pouvoir configurer **indépendamment pour chaque tâche LLM** du pipeline :

1. le **modèle** (provider + nom), ex. `ollama/mistral-small:24b`, `openai/gpt-5-mini`, `catgpt/catgpt-browser` ;
2. le **mode batch** (`mode_batch: true|false`) : plusieurs cas dans **un seul prompt / un seul appel** ;
3. la **taille de batch** (`batch_size`), obligatoire et respectée uniquement si `mode_batch` est autorisé et activé.

Ce plan ne change pas la logique métier S3–S6 (jugement, désambiguïsation, traduction). Il unifie le routage modèle/batch et impose un contrat de prompts double (unitaire vs lot).

### Contraintes confirmées

- Travailler dans `C:\DOCS\_perso\vocab-filter`.
- Ne pas casser les caches disque : la clé de cache doit inclure le modèle **et** le mode (unitaire / batch) **et** le contenu du prompt.
- Ne pas fusionner `litellm.batch_completion` (parallélisme HTTP de plusieurs prompts) avec `mode_batch` (n items dans **le même** prompt). Les deux peuvent coexister : lots multi-items + plusieurs lots en parallèle.
- `PROVIDER=chatgpt` dans `.env` est aujourd'hui décoratif ; le plan le mappe vers le provider `catgpt` (gateway OpenAI-compatible).
- Les batch sizes S6 déjà mesurés ne sont **pas** un n global : S6c reste plus petit que S6b-1 (dégradation mesurée à 40 pour la réassignation).
- Aucune exception livre-spécifique (*The Humans*). Les evals hors production (`evaluate_s3_judges`, `eval_frontier_ablation`) adoptent le même registre de tâches ou un sous-ensemble explicite.

### Définition opérationnelle de `mode_batch`

| Terme | Signification |
|---|---|
| `mode_batch: false` | Un cas métier = un prompt = un appel. |
| `mode_batch: true` | Jusqu'à `batch_size` cas métier dans **un seul** prompt ; la réponse JSON liste une décision par clé/id. |
| Parallélisme (`MAX_WORKERS`) | Indépendant : plusieurs prompts (unitaires ou lots) envoyés en parallèle. Hors périmètre de la config `mode_batch`. |

### Règle des deux prompts (obligatoire)

Pour **chaque** tâche dont `mode_batch` est autorisé (`batch_allowed: true` dans le registre) :

- il faut **deux familles de prompts** (system + user template + schéma JSON) :
  - **mode unitaire** : un item, réponse scalaire ;
  - **mode batch** : N items numérotés / clés explicites, réponse `{ "decisions": [ ... ] }` (ou équivalent) avec **exactement une décision par clé**, ordre stable ;
- le code sélectionne la famille selon la config effective (`mode_batch` et `batch_size >= 2`) ;
- si `batch_allowed: false`, une seule famille unitaire existe ; activer `mode_batch` lève une erreur de configuration explicite, jamais un repli silencieux vers un faux lot.

Les tâches déjà batchées (S6b-1, S6b-2 B/C, S6c) ont déjà le prompt lot ; le mode unitaire peut être le même prompt avec `N=1`, ou un prompt dédié plus léger — le plan exige que les deux chemins soient **testés** et que le basculement config soit sans ambiguïté.

---

## 1. Inventaire des tâches LLM (Sx-role)

Identifiants stables à utiliser dans la config, le CLI et les rapports. Format : `S{n}-{role}`.

### Production (`run_pipeline.py`)

| ID tâche | Étape | Module | Rôle | Client actuel | Batch prompt aujourd'hui | `batch_allowed` cible |
|---|---|---|---|---|---|---|
| `S3-judge-occurrence` | S3 | `pipeline/mwe_judge.py` | Jugement lexical **par occurrence** (label, canon, POS, paraphrase) | `llm.call_json` | **non** (1 occ / appel) | **oui** (prototype eval à 50) |
| `S3-definition-cluster` | S3 | `pipeline/mwe_judge.py` | Choix de glose par cluster de sens | `llm.call_json` | **non** | **oui** (à introduire) |
| `S5-arbitrate` | S5 | `pipeline/senses.py` | Arbitrage WordNet quand la politique calibrée le demande | `llm.call_json` | **non** | **oui** (à introduire) |
| `S6-translate-frontier` | S6b-1 | `pipeline/sense_fr_frontier.py` | Traduction primaire contextuelle + sense_fit | LiteLLM | **oui**, `SENSE_FR_FRONTIER_BATCH_SIZE=40` | **oui** (déjà) |
| `S6-backtranslate` | S6b-2 B | `pipeline/sense_fr_adjudicate.py` | Rétro-traduction FR→EN (opt-in `--with-backtranslation`) | LiteLLM | **oui**, défaut 40 | **oui** (déjà) |
| `S6-judge-dossier` | S6b-2 C | `pipeline/sense_fr_adjudicate.py` | Juge sur dossier, peut réécrire `fr` (opt-in `--with-judge`) | LiteLLM | **oui**, défaut 20 | **oui** (déjà) |
| `S6-reassign` | S6c | `pipeline/sense_fr_reassign.py` | Réassignation POS/sense_id sur pending structurels | LiteLLM | **oui**, `SENSE_FR_REASSIGN_BATCH_SIZE=10` | **oui** (déjà) |

### Hors run par défaut (conservés, même registre)

| ID tâche | Module | Rôle | Batch prompt aujourd'hui | `batch_allowed` |
|---|---|---|---|---|
| `S6-translate-local` | `pipeline/sense_fr.py` | Traduction « dictionnaire » sans contexte livre (`--retry-pending`) | non (×3 formulations / sens) | **non** en v1 (votes = autre mécanisme) |
| `S6-backtranslate-local` | `pipeline/sense_fr.py` | Rétro-traduction locale | non | **non** en v1 (lié au chemin local) |

### Hors production (eval) — alignement optionnel

| ID tâche | Module | Note |
|---|---|---|
| `EVAL-S3-judge` | `fix_pipeline/evaluate_s3_judges.py` | A déjà un `_run_local_batch` (50) ; doit lire le même slot `S3-judge-occurrence` ou un override eval |
| `EVAL-S6-joint` | `pipeline/eval_frontier_ablation.py` | Batch 10 déjà ; aligner sur `S6-reassign` ou slot eval dédié |

### Étapes **sans** LLM (hors registre)

`corpus`, `analyze` (S1), `mwe` (S2), `select` (S4), `sense_fr_adjudicate` Stage A (offline), `export` (S7). spaCy / GlossBERT / LaBSE ne sont pas des tâches de ce plan.

---

## 2. État des lieux batch (baseline à figer avant code)

Recalculer et documenter dans un mini-rapport `fix_pipeline/multi_models/baseline_batch_inventory.md` :

1. Pour chaque ID du §1 : unitaire / batch, constante, valeur, deux prompts déjà présents ou non.
2. Distinguer explicitement :
   - **batch prompt** (n cas / prompt) — objet de ce plan ;
   - **batch_completion parallèle** (`SENSE_FR_FRONTIER_MAX_WORKERS`) — hors config `mode_batch` ;
   - **nlp.pipe batch_size=64** (spaCy) — hors sujet.
3. Constats figés :
   - `S3_JUDGE_BATCH_SIZE=50` existe mais **n'est pas branché** sur `mwe_judge.run()` ;
   - `evaluate_s3_judges._run_local_batch` prouve qu'un prompt lot S3 est faisable ;
   - S6c : batch 40 mesuré comme dégradant ; défaut 10 à préserver comme défaut de config.

**Gate** : le rapport baseline existe ; aucun code de routage n'a encore changé.

---

## 3. Schéma de configuration

### 3.1 Enregistrement d'une tâche

Chaque tâche expose un descripteur stable :

```text
task_id: "S6-translate-frontier"
batch_allowed: true          # si false, mode_batch=true est une erreur config
default_model: "openai/gpt-5-mini"
default_mode_batch: true
default_batch_size: 40       # ignoré si mode_batch false ; min 1
```

### 3.2 Résolution runtime

Entrée utilisateur (env / YAML / CLI) pour une tâche :

```text
model: "ollama/mistral-small:24b"   # provider/name
mode_batch: false | true
batch_size: <int>                   # requis si mode_batch true ; >= 2 recommandé
```

Résolution :

1. override CLI si présent ;
2. sinon env `VOCAB_LLM_TASK_<ID>` (voir §3.3) ;
3. sinon fichier optionnel `data/llm_tasks.toml` ou équivalent versionné ;
4. sinon défauts du descripteur §3.1 (comportement actuel du pipeline).

`model` se parse en `(provider, bare_name)` avec providers autorisés : `ollama`, `openai`, `catgpt`. Alias `.env` : `PROVIDER=chatgpt` → provider `catgpt`.

Helpers uniques (à placer dans `pipeline/config.py` ou `pipeline/llm_tasks.py`) :

- `task_config(task_id) -> TaskLlmConfig`
- `litellm_model(task_id)` / `llm_call_params(task_id)` (`api_base`, `api_key` selon provider)
- `effective_batch_size(task_id) -> int` : `1` si `mode_batch` false, sinon `batch_size` validé

### 3.3 Variables d'environnement (proposition)

Une ligne par tâche, parseable, sans ambiguïté :

```text
VOCAB_LLM_S3_JUDGE_OCCURRENCE=ollama/mistral-small:24b;batch=false
VOCAB_LLM_S5_ARBITRATE=ollama/mistral-small:24b;batch=false
VOCAB_LLM_S6_TRANSLATE_FRONTIER=openai/gpt-5-mini;batch=true;batch_size=40
VOCAB_LLM_S6_REASSIGN=openai/gpt-5-mini;batch=true;batch_size=10
VOCAB_LLM_S6_JUDGE_DOSSIER=openai/gpt-5-mini;batch=true;batch_size=20
VOCAB_LLM_S6_BACKTRANSLATE=openai/gpt-5-mini;batch=true;batch_size=40
VOCAB_LLM_S3_DEFINITION_CLUSTER=ollama/mistral-small:24b;batch=false
```

Endpoints partagés (inchangés dans l'esprit) :

```text
OLLAMA_URL=...
CATGPT_BASE_URL=...
CATGPT_API_TOKEN=...
OPENAI_API_KEY=...          # déjà chargé via LiteLLM / .env
PROVIDER=chatgpt            # alias optionnel → catgpt pour les tâches catgpt/*
```

### 3.4 Compatibilité descendante

- Sans aucune nouvelle variable : comportement **identique** à aujourd'hui (S3/S5 → backend ollama/catgpt global actuel ; S6 → `openai/gpt-5-mini` + batch sizes actuels).
- `VOCAB_LLM_BACKEND` / `OLLAMA_MODEL` / `CATGPT_MODEL` restent lus comme **repli global** pour les tâches S3/S5 tant que leur slot dédié est vide.
- `SENSE_FR_FRONTIER_MODEL` et `ALLOWED_FRONTIER_MODELS` : migrer vers « modèle résolu de la tâche » + validation « le modèle demandé est celui configuré pour la tâche », plus une liste blanche figée mono-modèle.

**Gate** : tests unitaires du parseur ; erreur claire si `batch=true` et `batch_allowed=false` ; erreur si `batch=true` sans `batch_size` ou `batch_size < 1`.

---

## 4. Couche d'appel unifiée

### 4.1 Deux chemins clients, un registre

Aujourd'hui : `pipeline/llm.py` (stdlib) pour S3/S5 ; LiteLLM pour S6. Le plan **conserve** les deux clients si besoin, mais **toute** tâche passe par :

```text
pipeline.llm_tasks.resolve(task_id) → model, provider, params, mode_batch, batch_size
pipeline.llm_tasks.completion(task_id, messages, response_schema, ...)
```

qui délègue à `llm.call_json` ou `litellm.completion` / `batch_completion` selon le provider et les capacités (`reasoning_effort` seulement si supporté, typiquement OpenAI frontier).

### 4.2 Capacités provider

| Provider | JSON structuré | `reasoning_effort` | Endpoint |
|---|---|---|---|
| `ollama` | `format: json` / json_object selon client | non | `OLLAMA_URL` |
| `openai` | response_format Pydantic ou json_object | oui | API OpenAI |
| `catgpt` | json_object (gateway) | non (sauf si le modèle sous-jacent le permet — traiter comme non en v1) | `CATGPT_BASE_URL` |

Un appel avec paramètre non supporté doit **omettre** le paramètre, pas échouer silencieusement après retry opaque.

### 4.3 Cache

Clé de cache = hash(`task_id`, `model`, `mode_batch`, `batch_size` effectif, system, user, schéma). Changer de modèle ou de mode invalide l'ancien cache (comportement attendu, à documenter dans le README).

**Gate** : `test_llm_backends.py` étendu ; un test par provider en mock ; aucun appel réseau dans les tests ordinaires.

---

## 5. Lots de livraison (ordre impératif)

Chaque lot : baseline du lot → tests qui échouent → code minimal → tests verts → mini-rapport → lot suivant (§8 du plan qualité global : même discipline).

### Lot M0 — baseline et contrat

- **Pré-requis** : aucun.
- **Description** : écrire §2 (`baseline_batch_inventory.md`) ; figer le tableau des `task_id` ; documenter la règle des deux prompts.
- **Résultat** : inventaire signé ; aucun changement runtime.
- **Vérifications** : chaque ID du §1 apparaît ; batch vs non-batch correct ; `S3_JUDGE_BATCH_SIZE` noté comme non branché.

### Lot M1 — registre + parseur de config

- **Pré-requis** : M0.
- **Description** : module `pipeline/llm_tasks.py` (ou section dédiée dans `config.py`) : descripteurs, parse env, `task_config()`, validation `batch_allowed`.
- **Résultat** : défauts = comportement actuel ; overrides env testables.
- **Vérifications** : tests parse ; repli `VOCAB_LLM_BACKEND` ; alias `PROVIDER=chatgpt` ; refus `mode_batch` sur tâche non autorisée.

### Lot M2 — branchement S6 existants (déjà batch)

- **Pré-requis** : M1.
- **Description** : `sense_fr_frontier`, `sense_fr_adjudicate` (B/C), `sense_fr_reassign` lisent `task_config` pour modèle + `batch_size`. Remplacer constantes hardcodées par défauts du registre. Garantir chemin **unitaire** (`mode_batch=false` ou `batch_size=1`) avec **prompt / schéma unitaire** testé (deux prompts).
- **Résultat** : bascule config sans changer les défauts numériques (40 / 40 / 20 / 10).
- **Vérifications** : dry-run / mocks ; `require_frontier_model` adapté ; caches toujours indexés par modèle.

### Lot M3 — S3-judge-occurrence : modèle + batch optionnel

- **Pré-requis** : M1 ; prompt lot inspiré de `evaluate_s3_judges._run_local_batch`.
- **Description** :
  - brancher le modèle via `S3-judge-occurrence` ;
  - si `mode_batch=false` : conserver le prompt unitaire actuel ;
  - si `mode_batch=true` : regrouper jusqu'à `batch_size` occurrences, **prompt batch distinct**, parser `decisions[]`, écrire le magasin occurrence comme aujourd'hui ;
  - défaut production : `mode_batch=false` jusqu'à validation qualité (ne pas activer 50 en prod sans mesure).
- **Résultat** : double chemin testé ; défaut unitaire = non-régression.
- **Vérifications** : tests fixtures offline pour unitaire et lot (2–3 items) ; clé de cache distincte ; panne LLM → `incertain` comme aujourd'hui.

### Lot M4 — S3-definition-cluster

- **Pré-requis** : M1.
- **Description** : même patron (modèle dédié ; batch optionnel avec **deux prompts** si `batch_allowed`).
- **Défaut** : unitaire.
- **Vérifications** : un test unitaire + un test lot minimal.

### Lot M5 — S5-arbitrate

- **Pré-requis** : M1.
- **Description** : `arbitrate()` utilise `task_config("S5-arbitrate")`. Batch optionnel : plusieurs (mot, contexte, candidats) dans un prompt ; **deux prompts**. Défaut unitaire.
- **Vérifications** : mock ; politique S5-2 inchangée (le batch ne change pas *quand* on arbitre, seulement *comment* on appelle).

### Lot M6 — chemins locaux S6 (`sense_fr.py`) et CLI

- **Pré-requis** : M1.
- **Description** : enregistrer `S6-translate-local` / `S6-backtranslate-local` ; `run_pipeline.py` et modules CLI exposent `--llm-task ID=model;batch=...` ou documentation env claire. README : tableau des tâches + exemples ollama / openai / catgpt.
- **Vérifications** : `--help` / doc ; pas de régression si env vide.

### Lot M7 — alignement evals (optionnel mais recommandé)

- **Pré-requis** : M3 pour S3 ; M2 pour S6.
- **Description** : `evaluate_s3_judges` et `eval_frontier_ablation` consomment le registre (plus de `FRONTIER_MODEL` / `LOCAL_MODEL` en dur).
- **Vérifications** : les scripts tournent avec overrides ; juge d'eval reste indépendant du candidat si le plan d'ablation l'exige.

### Lot M8 — gate final multi-modèles

- **Pré-requis** : M0–M6 (M7 si fait).
- **Description** : matrice manuelle minimale (au moins un run mock par provider) ; checklist :
  - trois providers joignables en config (même sans appel réel) ;
  - chaque tâche production a un modèle résolu ;
  - chaque tâche `batch_allowed` a **deux** chemins de prompt couverts par des tests ;
  - défauts = baseline comportementale ;
  - documenter invalidation de cache lors d'un switch modèle.
- **Résultat** : `fix_pipeline/multi_models/report_multi_models.md`.
- **Gate** : aucun défaut critique ; bascule `.env` documentée pour S6-2 ensuite.

---

## 6. Deux prompts par tâche batchable — détail

Pour chaque tâche avec `batch_allowed: true`, le livrable code + tests doit nommer explicitement :

| Artefact | Unitaire | Batch |
|---|---|---|
| System prompt | `*_SYSTEM` ou partagé | souvent partagé + consignes de lot |
| User template | 1 item | liste d'items + règle « une décision par clé » |
| Schéma / response_format | objet unique | `{decisions: [...]}` |
| Parser | 1 décision | map id → décision ; détecter manquants / doublons |
| Test | 1 fixture | lot de 2–3 fixtures |

Tâches déjà pourvues d'un prompt lot (à compléter / expliciter le chemin N=1) :

- `S6-translate-frontier` — `build_user_prompt(batch)` + `BatchTranslations`
- `S6-backtranslate` — `_backtranslate_batch`
- `S6-judge-dossier` — `run_stage_c`
- `S6-reassign` — `build_user_prompt` + `ReassignBatch`

Tâches à créer pour le mode lot :

- `S3-judge-occurrence` — s'inspirer de `_run_local_batch` sans le coupler à l'eval
- `S3-definition-cluster` — nouveau
- `S5-arbitrate` — nouveau

---

## 7. Interactions avec S6-2 et le plan qualité

- Ce chantier **précède ou accompagne** S6-2 (`plan_action_fix_pipeline.md` §6 Correction S6-2) : le juge Stage C et la frontier doivent pouvoir viser des modèles différents (traduction vs adjudication).
- Ne pas lire `vocab_corrige.csv` dans les modules de production.
- Après M8, S6-2 peut préciser dans `.env` p.ex. frontier OpenAI + Stage C OpenAI (ou CatGPT) sans retoucher le code.

---

## 8. Politique de livraison (rappel)

Chaque lot M0–M8 :

1. lire ce plan + la section du lot ;
2. ajouter les tests qui échouent avant le correctif ;
3. modifier le minimum nécessaire ;
4. régénérer uniquement les artefacts requis ;
5. exécuter tests ciblés + non-régression LLM mock ;
6. publier un mini-rapport avant/après ;
7. ne passer au lot suivant que si résultat attendu + vérifications sont verts.

Une hausse de « flexibilité config » ne suffit jamais si elle casse le défaut actuel (S3 unitaire, S6 batch 40/10/20) ou si `mode_batch=true` sans second prompt testé.
