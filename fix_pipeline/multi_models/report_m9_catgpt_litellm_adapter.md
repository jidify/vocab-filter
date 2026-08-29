# Rapport — adaptateur LiteLLM → CatGPT-Gateway (Lot M9, hors plan initial)

Ferme l'écart documenté dans `report_multi_models.md` §2.1 et §6 : `catgpt/*`
passait la validation de configuration (`task_config`/`require_frontier_model`)
mais échouait à l'appel réel pour les 4 tâches S6 routées par LiteLLM
(`S6-translate-frontier`, `S6-backtranslate`, `S6-judge-dossier`,
`S6-reassign`), `catgpt` n'étant pas un provider natif de LiteLLM.

Périmètre volontairement minimal, sur demande explicite : envoyer un prompt,
récupérer le texte de la réponse — pas de streaming, pas d'async, pas
d'embeddings, pas de comptabilité de coût réelle.

## Implémentation

- **Nouveau** `pipeline/llm_litellm_catgpt.py` : `_CatGptLLM(litellm.CustomLLM)`
  implémente uniquement `completion()` (seule méthode requise pour
  `litellm.completion`/`batch_completion` synchrones non streamés — le seul
  chemin emprunté par ce dépôt) ; réutilise la mécanique HTTP de
  `pipeline/llm.py` (POST OpenAI-compatible sur `CATGPT_BASE_URL`, Bearer
  `CATGPT_API_TOKEN`, `response_format: {"type":"json_object"}`). Deux
  fonctions publiques : `register()` (idempotent, alimente
  `litellm.custom_provider_map`) et `call_kwargs(model)` (retourne
  `{"allowed_openai_params": ["reasoning_effort"]}` + enregistre le provider
  pour tout modèle `catgpt/...`, `{}` sans effet de bord sinon).
- **Injection du schéma dans le prompt** : le gateway ne sait produire que du
  JSON libre, pas du JSON contraint par schéma. Le JSON Schema que LiteLLM
  produit à partir de `response_format=<PydanticModel>` (présent dans
  `optional_params`) est donc extrait et rajouté en suffixe du message
  `system` avant l'envoi — même motif que celui déjà utilisé à la main par
  les 4 appelants CatGPT existants via `pipeline/llm.py`
  (`mwe_judge.PROMPT_SCHEMA`, `senses.ARBITRATION_TEMPLATE`,
  `sense_fr.BACKTRANSLATE_TEMPLATE`, `OCC_BATCH_PROMPT_TEMPLATE`). **Invariant
  respecté** : cette dégradation est strictement locale au provider
  `catgpt` — un modèle `openai/*` (ou tout autre provider `json_schema`
  natif) continue de recevoir exactement le même `response_format` Pydantic
  qu'avant ce lot, `call_kwargs()` étant un no-op pour lui (vérifié par
  `test_multi_models_catgpt_litellm.py::NoRegressionOnJsonSchemaModelsTests`).
- **4 sites d'appel modifiés, une ligne chacun** (`**catgpt_call_kwargs(model)`
  ajouté aux kwargs existants, rien d'autre) :
  `pipeline/sense_fr_frontier.py` (`batch_completion`),
  `pipeline/sense_fr_reassign.py` (`batch_completion`),
  `pipeline/sense_fr_adjudicate.py` (`completion`, Stage B et Stage C).
- Aucune modification de `TASK_REGISTRY`, `ALLOWED_PROVIDERS`,
  `task_config()`, `require_frontier_model`, des clés de cache disque, des
  prompts existants ou des CLI.

## Tests

`test_multi_models_catgpt_litellm.py` (7 tests, nouveaux) :
- `call_kwargs()` : effet uniquement sur `catgpt/...`, idempotent, no-op sur
  `openai/*`/`ollama/*`.
- Non-régression `json_schema` : `frontier`/`adjudicate` avec un modèle
  `openai/*` reçoivent toujours `response_format=<PydanticModel>` (identité
  d'objet vérifiée) et aucun `allowed_openai_params`.
- Bout en bout, HTTP mocké (`urllib.request.urlopen`), `litellm.batch_completion`
  réel (pas mocké) : POST vers `{CATGPT_BASE_URL}/chat/completions`, en-tête
  `Bearer`, `response_format={"type":"json_object"}`, schéma de
  `BatchTranslations` bien présent dans le message système, réponse
  correctement reliée par `sense_id`.
- Erreur réseau (`urllib.error.URLError`) : lot sauté, pas de crash — même
  chemin `isinstance(response, Exception)` qu'ollama/openai.

Isolation `config.CACHE_DIR` par `tempfile.TemporaryDirectory()` reprise pour
tous les tests qui empruntent un chemin réel de `frontier`/`adjudicate` — le
piège de cache disque déjà documenté au §3 de `report_multi_models.md`
(un cache déjà écrit par un autre test avec le même modèle/prompt aurait
court-circuité les mocks HTTP en `discover`, jamais en isolé). Repéré et
corrigé pendant ce lot (`NoRegressionOnJsonSchemaModelsTests` échouait en
`discover` mais pas en isolé, avant l'ajout de l'isolation).

```text
uv run python -m unittest test_multi_models_catgpt_litellm -v   # 7/7 OK
uv run python -m unittest discover -p "test_*.py"                # 273 tests, 0 échec, x2 consécutifs
```

(266 tests de la baseline M8 + 7 nouveaux = 273 — aucun test préexistant
modifié.)

## Documentation mise à jour

- `README.md` (section « Providers réellement joignables, par client ») :
  l'avertissement « `catgpt/*` échouera » est remplacé par la description de
  l'adaptateur et de sa limite (schéma *instruit*, pas *contraint* —
  `ValidationError` possible, non rattrapée par l'appelant contrairement à
  l'erreur réseau).
- `README.md` (§ Exemples) : ajout d'un exemple `VOCAB_LLM_S6_BACKTRANSLATE`
  sur `catgpt/catgpt-browser`.

## Limite honnête, assumée et documentée

Le gateway ne *contraint* pas la sortie comme le fait le `json_schema` natif
d'OpenAI : il l'*instruit*, au même niveau de garantie que S3/S5/S6-\*-local
sur CatGPT via `pipeline/llm.py`. Si le modèle derrière le gateway répond du
JSON hors schéma, c'est une `ValidationError` chez l'appelant. Les 4 tâches
S6 ne la rattrapent pas explicitement (seule l'`Exception` d'un lot
`batch_completion` l'est) — un modèle gateway trop bavard fait donc échouer
l'étape plutôt que dégrader en `pending`. Recommandé : mesurer sur un petit
lot avant de basculer une tâche coûteuse en production.

Non câblé (hors périmètre demandé) : `acompletion`/streaming (jamais
empruntés par ce dépôt), comptabilité de coût réelle (`completion_cost` reste
non mappé pour `catgpt`, déjà absorbé par le `try/except Exception: pass`
existant aux 2 sites qui l'appellent).
