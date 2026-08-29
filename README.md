# vocab-filter

## Backend LLM : Ollama ou CatGPT-Gateway

Les appels JSON de S3/S5 utilisent Ollama par défaut. Pour les envoyer à une
instance [CatGPT-Gateway](https://github.com/GautamVhavle/CatGPT-Gateway) déjà
lancée et authentifiée :

```powershell
uv run python run_pipeline.py --llm-backend catgpt `
  --llm-base-url http://localhost:8000/v1 `
  --catgpt-api-token dummy123 `
  --llm-model catgpt-browser
```

Pour un module lancé directement, utiliser `VOCAB_LLM_BACKEND=catgpt` avec
`CATGPT_BASE_URL`, `CATGPT_API_TOKEN`, `CATGPT_MODEL` et `CATGPT_TIMEOUT`.
Le gateway est un service distinct : ce projet ne le lance pas et ne gère pas
sa connexion à ChatGPT. Sans configuration, Ollama reste utilisé (`OLLAMA_URL`
et `OLLAMA_MODEL` permettent d'en changer l'adresse et le modèle).

**Important** : `--llm-backend`/`--llm-model` (et `VOCAB_LLM_BACKEND`/
`OLLAMA_MODEL`/`CATGPT_MODEL`) ne configurent qu'un backend **global de
repli**, utilisé par 5 des 9 tâches LLM du pipeline tant qu'aucun slot dédié
n'est posé (voir tableau ci-dessous). Ils ne changent **jamais** le modèle des
4 tâches S6 routées par LiteLLM (`S6-translate-frontier`, `S6-backtranslate`,
`S6-judge-dossier`, `S6-reassign`), configurées uniquement via `VOCAB_LLM_S6_*`.

## Configuration multi-modèles par tâche (S3–S6)

Chaque tâche LLM du pipeline (`pipeline/llm_tasks.py::TASK_REGISTRY`) se
configure **indépendamment** — modèle, et pour les tâches qui l'autorisent,
mode lot (`mode_batch`) et taille de lot (`batch_size`) — via une variable
d'environnement dédiée `VOCAB_LLM_<TASK_ID>` :

```text
VOCAB_LLM_<TASK_ID>=<provider>/<modèle>[;batch=true|false][;batch_size=<n>]
```

`batch_size` est obligatoire si `batch=true`. `batch=true` sur une tâche
`batch_allowed: false` (les deux tâches "-local", ci-dessous) lève une erreur
de configuration explicite au démarrage, jamais un repli silencieux. Sans
aucune de ces variables : comportement **identique à aujourd'hui** (baseline
figée dans `fix_pipeline/multi_models/baseline_batch_inventory.md`).

| `task_id` | Étape | Module | Client | `batch_allowed` | Défaut modèle | Défaut mode/taille | Variable d'env |
|---|---|---|---|---|---|---|---|
| `S3-judge-occurrence` | S3 | `pipeline/mwe_judge.py` | `pipeline/llm.py` | oui | backend global (`ollama/mistral-small:24b`) | unitaire | `VOCAB_LLM_S3_JUDGE_OCCURRENCE` |
| `S3-definition-cluster` | S3 | `pipeline/mwe_judge.py` | `pipeline/llm.py` | oui | backend global | unitaire | `VOCAB_LLM_S3_DEFINITION_CLUSTER` |
| `S5-arbitrate` | S5 | `pipeline/senses.py` | `pipeline/llm.py` | oui | backend global | unitaire | `VOCAB_LLM_S5_ARBITRATE` |
| `S6-translate-frontier` | S6b-1 | `pipeline/sense_fr_frontier.py` | LiteLLM | oui | `openai/gpt-5-mini` | lot 40 | `VOCAB_LLM_S6_TRANSLATE_FRONTIER` |
| `S6-backtranslate` | S6b-2 B | `pipeline/sense_fr_adjudicate.py` | LiteLLM | oui | `openai/gpt-5-mini` | lot 40 | `VOCAB_LLM_S6_BACKTRANSLATE` |
| `S6-judge-dossier` | S6b-2 C | `pipeline/sense_fr_adjudicate.py` | LiteLLM | oui | `openai/gpt-5-mini` | lot 20 | `VOCAB_LLM_S6_JUDGE_DOSSIER` |
| `S6-reassign` | S6c | `pipeline/sense_fr_reassign.py` | LiteLLM | oui | `openai/gpt-5-mini` | lot 10 | `VOCAB_LLM_S6_REASSIGN` |
| `S6-translate-local` | hors run par défaut | `pipeline/sense_fr.py` (`--retry-pending`) | `pipeline/llm.py` | **non** | backend global | unitaire (seul chemin) | `VOCAB_LLM_S6_TRANSLATE_LOCAL` |
| `S6-backtranslate-local` | hors run par défaut | `pipeline/sense_fr.py` | `pipeline/llm.py` | **non** | backend global | unitaire (seul chemin) | `VOCAB_LLM_S6_BACKTRANSLATE_LOCAL` |

« backend global » = `VOCAB_LLM_BACKEND` (ou l'alias `PROVIDER=chatgpt` →
`catgpt`) + `OLLAMA_MODEL`/`CATGPT_MODEL` (ou `--llm-backend`/`--llm-model` en
CLI), lus tant qu'aucune variable `VOCAB_LLM_<TASK_ID>` dédiée n'est posée
pour cette tâche précise.

### Exemples

```bash
# S3/S5 : Ollama local, modèle différent du défaut
VOCAB_LLM_S3_JUDGE_OCCURRENCE="ollama/gemma3:27b;batch=false"

# S3-definition-cluster : lot de 5, même backend
VOCAB_LLM_S3_DEFINITION_CLUSTER="ollama/mistral-small:24b;batch=true;batch_size=5"

# S6-translate-frontier : OpenAI, lot réduit à 20 (mesure de dégradation en cours)
VOCAB_LLM_S6_TRANSLATE_FRONTIER="openai/gpt-5-mini;batch=true;batch_size=20"

# S6-judge-dossier : Ollama plutôt qu'OpenAI pour Stage C seul — LiteLLM lit
# sa propre variable OLLAMA_API_BASE (pas OLLAMA_URL), voir note providers ci-dessous
VOCAB_LLM_S6_JUDGE_DOSSIER="ollama/mistral-small:24b;batch=true;batch_size=20"

# S6-backtranslate : CatGPT-Gateway — adaptateur pipeline/llm_litellm_catgpt.py,
# voir note providers ci-dessous pour ses limites (pas de json_schema natif)
VOCAB_LLM_S6_BACKTRANSLATE="catgpt/catgpt-browser;batch=true;batch_size=40"

# S6-translate-local / S6-backtranslate-local : batch_allowed=false, jamais de ;batch=true
VOCAB_LLM_S6_TRANSLATE_LOCAL="ollama/mistral-small:24b"
VOCAB_LLM_S6_BACKTRANSLATE_LOCAL="ollama/mistral-small:24b"
```

### Providers réellement joignables, par client

- `pipeline/llm.py` (S3, S5, `S6-*-local`) : `ollama`, `catgpt` et `openai`
  fonctionnent tous les trois — client stdlib maison, un branchement HTTP par
  backend (`OLLAMA_URL`, `CATGPT_BASE_URL`/`CATGPT_API_TOKEN`,
  `OPENAI_API_KEY`/`OPENAI_BASE_URL`).
- LiteLLM (les 4 tâches S6 batchées) : `openai/*` fonctionne nativement.
  `ollama/*` fonctionne aussi, mais LiteLLM lit sa **propre** variable
  `OLLAMA_API_BASE` (pas `OLLAMA_URL` de ce projet) — à poser séparément si on
  pointe une de ces 4 tâches vers Ollama. `catgpt/*` **n'est pas** un provider
  natif de LiteLLM, mais un adaptateur minimal le câble
  (`pipeline/llm_litellm_catgpt.py`, `litellm.CustomLLM`) : envoi HTTP direct
  au gateway (`CATGPT_BASE_URL`/`CATGPT_API_TOKEN`/`CATGPT_TIMEOUT`, comme
  `pipeline/llm.py`), et — le gateway ne sachant pas produire de JSON
  contraint par schéma — le schéma attendu (`response_format=<modèle
  Pydantic>`) est rajouté en texte dans le message système avant l'envoi, le
  gateway ne recevant que `response_format: {"type":"json_object"}`. Rien
  d'autre n'est câblé (pas de streaming, pas d'appel asynchrone, pas de
  comptabilité de coût réelle — `litellm.completion_cost` reste non mappé
  pour ce provider et échoue silencieusement, comme pour tout modèle inconnu
  de LiteLLM). Conséquence : un modèle du gateway qui répond hors schéma lève
  une `ValidationError` non rattrapée par l'appelant (seule l'erreur réseau
  l'est, via `isinstance(response, Exception)`) — à valider sur un petit lot
  avant de basculer une tâche coûteuse.

### Invalidation du cache disque au changement de modèle/mode

La clé de cache de chaque appel LLM (`pipeline/llm.py::_cache_path`,
`sense_fr_frontier.py::_cache_path`, et équivalents dans
`sense_fr_adjudicate.py`/`sense_fr_reassign.py`) inclut désormais `task_id`,
`model`, `mode_batch` et la taille de lot effective, en plus du prompt exact.
**Changer le modèle, le backend ou le mode (unitaire ↔ lot) d'une tâche
invalide donc son cache disque existant** (`pipeline_out/cache/`) : le run
suivant repaie les appels LLM déjà cachés pour cette tâche, même si le modèle
choisi in fine est identique à l'ancien mais que la clé de cache, elle, a
changé de forme. C'est un effet de bord attendu (comportement voulu par
`fix_pipeline/multi_models/plan_multi_models.md` §4.3), pas une régression —
mais il a un coût réel (temps + coût API) au run suivant et doit être anticipé
avant de relancer un run de production après un changement de config.
