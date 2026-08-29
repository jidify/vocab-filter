# Rapport M6 — chemins locaux S6 (`sense_fr.py`), CLI, README

Lot **M6** du plan `./fix_pipeline/multi_models/plan_multi_models.md` (§1 hors run,
§3.3–3.4, Lot M6). Pré-requis M1 (déjà commité) : `S6-translate-local` et
`S6-backtranslate-local` existaient déjà dans `pipeline/llm_tasks.py::TASK_REGISTRY`
(`batch_allowed=False`, `default_mode_batch=False`, `default_batch_size=1`,
`global_model_fallback=True`), mais **non branchés** dans `pipeline/sense_fr.py` —
c'est le travail de ce lot.

## Écart avant / après

| Sujet | Avant | Après |
|---|---|---|
| `llm_translate_votes` (S6-translate-local) | `llm.call_json(prompt, system=TRANSLATE_SYSTEM, timeout=120)` — modèle/backend implicites (`config.LLM_BACKEND`/`config.llm_model()`), `cache_metadata` absent | `task = task_config("S6-translate-local")` ; `model=task.bare_model`, `backend=task.provider`, `cache_metadata={"task_id", "model", "mode_batch": False, "batch_size": 1}` |
| `llm_backtranslate` (S6-backtranslate-local) | Idem, implicite | `task = task_config("S6-backtranslate-local")` ; mêmes kwargs explicites |
| `llm_is_available()` (ping mémoïsé, un seul par process) | `llm.is_available()` — pingait `config.LLM_BACKEND` brut | `llm.is_available(backend=task_config("S6-translate-local").provider)` — ping le backend réellement résolu, alias `.env` `PROVIDER=chatgpt` inclus |
| `llm.is_available()` | Pas de paramètre, seulement `catgpt`/`ollama` | `backend: str \| None = None` optionnel (repli `config.LLM_BACKEND`, inchangé sans argument) + branche `openai` ajoutée (miroir de `llm.call_json`) |
| Message d'indisponibilité dans `run()` | `f"LLM {config.LLM_BACKEND} injoignable à {llm_url}"` | Même backend que celui pingé par `llm_is_available()` (`task_config("S6-translate-local").provider`), jamais désynchronisé du ping réel |
| `run_pipeline.py --llm-backend`/`--llm-model` | Aide argparse minimale, aucune indication de périmètre | Aide + commentaire explicites : ne couvrent QUE le backend global de repli (S3/S5/S6-\*-local), jamais les 4 tâches S6 LiteLLM |
| `pipeline.config.configure_llm` | Docstring d'une ligne | Docstring détaillant le périmètre exact (mêmes termes que ci-dessus) |
| README | Section « Backend LLM » (S3/S5 seulement) | + tableau des 9 `task_id`, syntaxe `VOCAB_LLM_<TASK_ID>`, exemples ollama/openai/catgpt, note providers réellement joignables par client, section invalidation de cache |

`S6-translate-local`/`S6-backtranslate-local` restent `batch_allowed: false` :
aucun second prompt lot n'est requis (règle des deux prompts, plan §0/§6, ne
s'applique qu'aux tâches `batch_allowed: true`). Le mécanisme
`SENSE_FR_LLM_DRAWS` (3 formulations de prompt pour un consensus interne) reste
intact et n'est PAS un lot au sens du plan — rappelé dans le nouveau
commentaire de `llm_translate_votes`.

## Un vrai écart trouvé et corrigé en cours de lot (pas seulement les deux fonctions ciblées)

Ce dépôt contient un `.env` réel avec `PROVIDER=chatgpt` (pas une variable de
test). `pipeline.llm_tasks._global_model()` honore cet alias depuis M1
(`PROVIDER=chatgpt` → backend `catgpt`), mais `config.LLM_BACKEND` — lu une
seule fois à l'import (`os.getenv("VOCAB_LLM_BACKEND", "ollama")`) — l'ignore
totalement. En branchant `llm_translate_votes`/`llm_backtranslate` sur
`task_config()`, le modèle réellement appelé s'est mis à suivre l'alias
`PROVIDER`, alors que `llm_is_available()` continuait de pinger
`config.LLM_BACKEND` (donc Ollama) : le ping de disponibilité et l'appel réel
pouvaient diverger silencieusement dès qu'un `.env` pose `PROVIDER=chatgpt`
sans poser aussi `VOCAB_LLM_BACKEND`. Détecté par la suite complète
(`test_sense_fr.py`, tests de non-régression « env vide » échouant dans le
vrai environnement du dépôt, pas dans un environnement de test isolé) avant
tout commit. Corrigé : `llm.is_available()` accepte désormais un `backend`
explicite (branche `openai` ajoutée au passage, miroir de `llm.call_json`) ;
`sense_fr.llm_is_available()` et le message d'indisponibilité de `run()`
utilisent tous deux `task_config("S6-translate-local").provider`. Limite
assumée en v1 : un seul ping mémoïsé par process pour les deux tâches locales
— correct tant qu'elles ne reçoivent pas de slots dédiés à des providers
différents (cas non couvert, non demandé par le plan pour ce lot).

## Vérifications

- Tests offline ajoutés à `test_sense_fr.py` (10 nouveaux, tous mockés, aucun
  appel réseau) :
  - `llm_translate_votes`/`llm_backtranslate` utilisent le modèle/provider du
    slot dédié quand `task_config` est mocké avec un modèle différent du
    défaut (`catgpt/dict-translator`, `catgpt/dict-backtranslator`) ;
  - `cache_metadata` de chaque appel porte `task_id`, `mode_batch: false`,
    `batch_size: 1` ;
  - `llm_is_available()` pingue le backend résolu par `task_config`, pas
    `config.LLM_BACKEND` brut ;
  - env réellement vide (`VOCAB_LLM_S6_TRANSLATE_LOCAL`,
    `VOCAB_LLM_S6_BACKTRANSLATE_LOCAL`, `VOCAB_LLM_BACKEND`, `PROVIDER`,
    `OLLAMA_MODEL`, `CATGPT_MODEL` tous absents) → défaut de registre
    `ollama/mistral-small:24b`, `mode_batch=false`, `batch_size=1` —
    comportement identique à avant ce lot.
- `test_run_pipeline_cli.py` (2 nouveaux tests) :
  - `run_pipeline.py --help` documente explicitement, texte non wrappé
    compris, que `--llm-backend`/`--llm-model` ne couvrent pas
    `S6-translate-frontier` (et les 3 autres tâches LiteLLM) — les variables
    `VOCAB_LLM_S6_*` restent le seul levier ;
  - `config.configure_llm(backend=None, base_url=None, api_token=None,
    model=None, timeout=None)` reste un no-op strict (tous les globals LLM
    inchangés) — non-régression explicite du cas « aucune option CLI ».
- Suite complète (`uv run python -m unittest discover -p "test_*.py"`) :
  **261 tests, 0 échec, 4 skip, 11 échecs attendus** — même profil que les
  lots précédents (M2 : 238 tests ; M5 : 252 tests), pas de régression sur
  `test_llm_tasks.py`, `test_llm_backends.py`, `test_multi_models_m2_s6.py`,
  `test_multi_models_m3_m4.py`, `test_multi_models_m5_s5.py`,
  `test_sense_fr_frontier.py`, `test_sense_fr_adjudicate.py`,
  `test_sense_fr_reassign.py`.
- Aucune exception livre-spécifique ajoutée ; aucun import de
  `vocab_corrige.csv`.

## Commandes exécutées

```text
uv run python -m unittest test_sense_fr -v   # rouge (4 erreurs, task_config absent) -> branchement -> vert
uv run python -m unittest discover -p "test_*.py"   # 260 tests, 2 échecs (alias PROVIDER non honoré par le ping)
# -> correctif llm.is_available(backend=...) + sense_fr.llm_is_available()
uv run python -m unittest test_sense_fr -v test_run_pipeline_cli -v   # 12 tests, vert
uv run python -m unittest discover -p "test_*.py"   # 261 tests, 0 échec, 4 skip, 11 échecs attendus
```

## Gate M6

- [x] `S6-translate-local`/`S6-backtranslate-local` déjà au registre M1
      (`batch_allowed: false`) — confirmé, inchangé par ce lot.
- [x] Branchement réel dans `pipeline/sense_fr.py` (`llm_translate_votes`,
      `llm_backtranslate`) sur `task_config`, modèle/provider/cache_metadata
      explicites — plus de résolution implicite via `config.llm_model()`.
- [x] README : tableau des 9 `task_id`, variables d'env, exemples
      ollama/openai/catgpt (avec avertissement honnête : `catgpt` non câblé
      côté LiteLLM pour les 4 tâches S6 batchées — pas de faux exemple qui
      échouerait à l'usage), invalidation de cache documentée.
- [x] `run_pipeline.py`/`configure_llm` ne laissent plus croire qu'un seul
      `--llm-model` couvre S6 (aide CLI + docstring alignées).
- [x] Tests de non-régression si env vide — et un écart réel (alias
      `PROVIDER=chatgpt` non honoré par le ping de disponibilité) trouvé et
      corrigé avant de clore le lot, pas seulement contourné dans les tests.
- [x] Suite complète verte (261/261, profil skip/échecs attendus inchangé).

## Prochain gate

Lot **M7** (optionnel mais recommandé) — aligner `fix_pipeline/evaluate_s3_judges.py`
et `pipeline/eval_frontier_ablation.py` sur le registre (`FRONTIER_MODEL`/
`LOCAL_MODEL` actuellement en dur), en conservant l'indépendance juge/candidat
exigée par l'ablation (`DEFAULT_JUDGE_MODEL` hors `ALLOWED_FRONTIER_MODELS`,
volontairement). **M8 non lancé**, conformément à la consigne de ce lot.
