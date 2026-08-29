# Rapport — unification du client LLM + `custom_prompt` (Lots U1–U6)

Ferme `report_multi_models.md` §4bis : deux clients LLM indépendants
(`pipeline/llm.py`, stdlib `urllib`, pour S3/S5/S6-\*-local ; LiteLLM appelé
directement dans les 3 modules S6 batchés) fusionnés en un seul —
`pipeline/llm_client.py`. Ajoute aussi `custom_prompt`, demandé par
l'utilisateur pour adapter un prompt à un modèle sans toucher au code.
Ferme au passage le Lot M7 resté ouvert depuis `report_multi_models.md` §5
(evals hors registre).

## 1. Vue d'ensemble avant / après

| Sujet | Avant (fin M9) | Après (U6) |
|---|---|---|
| Clients LLM | 2 indépendants — `pipeline/llm.py` (S3/S5/S6-\*-local) et LiteLLM appelé en dur dans 3 modules S6 | 1 — `pipeline/llm_client.py`, seul point d'appel LiteLLM du dépôt |
| Tâches sans slot | `mwe_judge.judge_type` (aucun descripteur, défauts globaux implicites) | enregistrée `S3-judge-type`, résolue comme les 9 autres |
| `evaluate_s3_judges.py`/`eval_frontier_ablation.py` | `FRONTIER_MODEL`/`LOCAL_MODEL`/`DEFAULT_JUDGE_MODEL` en dur, appels LLM ad hoc (M7 non fait) | modèle local/frontière résolus via `task_config()` ; mécanique d'appel routée par `llm_client` ; indépendance juge/candidat de l'ablation intacte |
| Prompt par tâche | figé dans le module (`OCC_PROMPT_TEMPLATE`, etc.), aucun levier de config | `custom_prompt` (registre) + `;prompt=<nom>` (env) + catalogue `pipeline/prompt_variants.py` |
| Cache des 4 tâches S6 LiteLLM | clé propre à chaque module (`_cache_path`) | même formule, même préfixe, **octet pour octet** — vérifié (`test_llm_client_cache_parity.py`) |
| Cache S3/S5/S6-local | interne à `pipeline/llm.py` | invalidé par la migration (accepté — modèles locaux, re-run gratuit) |
| `S3-judge-occurrence` calibration | `complete` = 4 champs dont `reason`, jamais relu ailleurs | `complete` = 3 champs tous relus en aval (canon/pos/paraphrase) ; même formule pour toute variante de prompt |
| ollama via LiteLLM | `api_base` non posé — LiteLLM lisait sa propre `OLLAMA_API_BASE`, jamais `OLLAMA_URL` | `pipeline/llm_client.py` pose `api_base=config.OLLAMA_URL` sur chaque appel ollama — écart documenté en M8 fermé |
| Tests | 266 (rapport M8) → 306 (avant suppression `pipeline/llm.py`) | 303, 0 échec, stable sur 2 exécutions consécutives |

## 2. Architecture livrée

- **`pipeline/llm_client.py`** — `call()` (un appel, dict JSON libre ou
  modèle Pydantic), `call_batch_completion()` (plusieurs appels indépendants
  en parallèle HTTP via `litellm.batch_completion` — jamais confondu avec
  `mode_batch`), `is_available()` (ping, reste en `urllib`, ne passe pas par
  LiteLLM). `build_cache_key()` pour les tâches migrées depuis l'ancien
  client A ; `cache_path_for()`/`BatchItem.cache_key_fields` pour les tâches
  qui préservaient déjà leur propre clé (les 4 tâches S6).
- **`pipeline/prompt_variants.py`** — catalogue `PROMPT_VARIANTS` (nom →
  `PromptOverride` : system/template unitaire, system/template lot, chacun
  optionnel — `None` conserve le texte standard du module). `render()`
  applique `format_map` avec un placeholder manquant transformé en
  `PromptVariantError` explicite (jamais un `KeyError` nu).
- **`pipeline/llm_tasks.py`** — `TaskDescriptor`/`TaskLlmConfig` portent
  `custom_prompt` (nom de variante, résolu en objet `PromptOverride` par
  `task_config()`) ; `_parse_override` accepte `;prompt=<nom>` en plus de
  `batch`/`batch_size`. Nom absent du catalogue → `TaskConfigError` au
  démarrage.

## 3. Lots

- **U1** — client unifié écrit et testé hors ligne (`test_llm_client.py`,
  13 → 14 tests), aucun appelant branché.
- **U2** — 5 tâches de l'ancien client A + `judge_type` (nouveau slot
  `S3-judge-type`) migrées vers `llm_client.call`. 25 patches de tests
  (`test_llm_backends.py`, `test_multi_models_m3_m4.py`,
  `test_multi_models_m5_s5.py`, `test_mwe_fusion.py`,
  `test_q0_2_regression.py`, `test_sense_fr.py`) réécrits sur le nouveau
  contrat d'appel (`model=` qualifié provider/modèle, `cache_key_fields=`
  au lieu de `cache_metadata=`).
- **U3** — 4 tâches S6 (LiteLLM en dur) migrées vers `llm_client.call`/
  `call_batch_completion`. Cache préservé octet pour octet — digest recalculé
  avec la formule exacte d'avant migration, comparé en dur dans
  `test_llm_client_cache_parity.py` (4 tests). Un vrai écart trouvé pendant
  ce lot et corrigé avant de le clore : aucun (contrairement à M8, ce lot
  n'a pas eu besoin de correctif après-coup — la préservation de clé a été
  vérifiée avant tout commit).
- **U4** — `custom_prompt` + variante `s3-occurrence-tags` (récupérée de
  `fix_pipeline/evaluate_s3_judges.py`). `evidence` en 1-2 étiquettes
  fermées (`EVIDENCE_TAGS`) plutôt qu'un indice en texte libre — c'est le
  texte libre qui rendait ce champ coûteux chez catgpt. `reason` retiré du
  prompt de cette variante et des champs de `complete` **pour toute
  variante** (12 tests, `test_prompt_variants_u4.py`).
- **U5** — gate de parité sur verdicts réels, voir §4. `pipeline/llm.py`
  et `test_llm_backends.py` supprimés après décision utilisateur.
- **U6** — `evaluate_s3_judges.py`/`eval_frontier_ablation.py` consomment le
  registre et le client unifié (ferme M7). Indépendance juge/candidat de
  l'ablation non touchée : `DEFAULT_CANDIDATE_MODEL`/`DEFAULT_JUDGE_MODEL`
  restent des constantes libres, jamais résolues via `task_config`/
  `ALLOWED_FRONTIER_MODELS` (config.py:333-336) — seule la mécanique
  d'appel (`_completion`/`_completions`) route désormais par `llm_client`.

## 4. Gate de parité (U5) — résultat et décision

`fix_pipeline/parity_llm_client.py` (supprimé après usage, résultat ci-dessous
conservé ici) a rejoué les 50 cas de `fix_pipeline/s3_judge_eval_cases.json`
via l'ancien client (`pipeline/llm.py`) et le nouveau (`pipeline/llm_client.py`),
même modèle (`ollama/mistral-small:24b`, forcé via
`VOCAB_LLM_S3_JUDGE_OCCURRENCE` — la résolution par défaut de ce dépôt tombe
sur `catgpt/catgpt-browser`, voir §5), température 0.

**Résultat réel (exécuté, pas simulé)** : 1 écart sur 50, sur la frontière
lexicalisé/non-lexicalisé — critère de passage du gate, donc gate rouge
littéralement.

**Cas concerné** : `literal-look-up-sky` (« She looked up at the sky and
watched the clouds. », gold = `littéral`). Ancien client → `phrasal_verb`
(faux). Nouveau client → `littéral` (**correct**, conforme au gold).

**Cause identifiée** (pas laissée en hypothèse) : LiteLLM template lui-même
`system` + `prompt` en un seul texte avant de les envoyer à Ollama
(`ollama_pt()`, dans
`litellm/llms/ollama/completion/transformation.py::transform_request`), et
ne pose **pas** le champ `system` séparé du payload `/api/generate`. L'ancien
client, lui, envoyait `system` et `prompt` comme deux champs distincts,
laissant Ollama appliquer le template de son propre Modelfile. Les deux
clients envoient donc un texte structurellement différent au même modèle —
écart réel de mécanique, pas du bruit de génération (température 0 des deux
côtés).

**Décision (utilisateur, explicite)** : accepter l'écart, supprimer
`pipeline/llm.py`. Justification retenue : 1/50, et dans le sens d'une
amélioration (verdict correct) sur le seul cas mesuré — pas une preuve que
l'écart est toujours favorable, mais aucune preuve du contraire non plus, et
le chantier de fusion des deux clients est le but explicite du §4bis.
**Point de vigilance retenu, pas fermé** : si un désaccord modèle/gold
apparaît après un changement de tâche vers `ollama/*`, revenir sur ce
gate — documenté dans le README (« Limite connue, mesurée »).

## 5. Écart trouvé pendant ce chantier, distinct de celui du gate

En diagnostiquant la résolution de modèle pour le gate, `task_config()`
résolvait `catgpt/catgpt-browser` alors que `config.LLM_BACKEND` valait
`ollama` et qu'aucune variable d'env `PROVIDER`/`VOCAB_LLM_BACKEND` n'était
posée dans le process. Cause : **`import litellm` charge `.env` via
`python-dotenv`** (`litellm/__init__.py:27`,
`_dotenv.load_dotenv(override=...)`), qui active la ligne `PROVIDER=chatgpt`
de ce dépôt — jusqu'ici documentée comme inerte
(`report_m6_local_cli_readme.md`, `report_multi_models.md` §4 : « aucun
python-dotenv... ne configure rien par défaut »), et qui l'était réellement
**avant ce chantier**, tant que S3/S5/S6-local passaient par
`pipeline/llm.py` (aucun import `litellm`). Depuis U2, `pipeline/llm_client.py`
est importé par la quasi-totalité du pipeline et importe `litellm` en tête
de module — la ligne `.env` s'active donc désormais **pour de vrai**, comme
effet de bord de l'unification elle-même, pas d'un changement volontaire.

**Pas corrigé dans ce lot** (hors périmètre de la demande initiale) :
neutraliser ou documenter ce comportement reste à faire séparément. À
retenir pour la suite : ce dépôt bascule maintenant réellement sur CatGPT
par défaut dès que `.env` est présent dans le répertoire de lancement,
contrairement à ce que les rapports précédents affirmaient.

## 6. Vérification

```text
uv run python -m unittest discover -p "test_*.py"      # 303 tests, 0 échec, x2 consécutifs
uv run python -m unittest test_llm_client -v            # 14 tests, client seul
uv run python -m unittest test_llm_client_cache_parity  # 4 tests, digests pré-migration
uv run python -m unittest test_prompt_variants_u4 -v     # 12 tests, custom_prompt
uv run python fix_pipeline/evaluate_s3_judges.py --models local --limit 3   # run réel, catgpt/catgpt-browser
# Gate de parité (script depuis supprimé, résultat conservé ci-dessus) :
#   VOCAB_LLM_S3_JUDGE_OCCURRENCE="ollama/mistral-small:24b" \
#     uv run python fix_pipeline/parity_llm_client.py     # 50 cas réels, 1 écart, décision §4
```

## 7. Écarts encore ouverts, explicitement documentés

- **Chargement de `.env` par `litellm`** (§5) — comportement nouveau, non
  neutralisé, à trancher séparément.
- **Limite du gate de parité** (§4) — 1 écart mesuré sur 50, accepté par
  décision explicite, pas un 0 garanti sur un corpus plus large.
- **`custom_prompt` n'est câblé que sur `S3-judge-occurrence`** — le champ
  existe au niveau du registre pour toute tâche, mais seul ce site d'appel
  consulte `task.custom_prompt` aujourd'hui (seul besoin concret exprimé).
  Étendre à `S3-definition-cluster`/`S5-arbitrate`/les tâches S6 est possible
  sans changement de schéma, pas fait faute de besoin identifié.
- **Adaptateur catgpt** (`pipeline/llm_litellm_catgpt.py`) — limites déjà
  actées en M9 (`report_m9_catgpt_litellm_adapter.md`) inchangées : schéma
  *instruit* dans le prompt, pas *contraint*.

Chantier d'unification (U1–U6) **clos**. `pipeline/llm.py` supprimé,
`§4bis` fermé, M7 fermé.
