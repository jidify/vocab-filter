# Rapport M5 — S5-arbitrate sur slot modèle et prompt lot

Lot **M5** du plan `./fix_pipeline/multi_models/plan_multi_models.md` (§5 Lot M5, §6).
Pré-requis M1 (déjà commité, Lot M1+M2) : le descripteur `S5-arbitrate` existait déjà dans
`pipeline/llm_tasks.py` (`batch_allowed=True`, `default_mode_batch=False`,
`global_model_fallback=True`), inchangé par ce lot.

## Écart avant / après

| Sujet | Avant | Après |
|---|---|---|
| `arbitrate()` | Backend/modèle globaux (`config.LLM_BACKEND`/`config.llm_model()` implicites via `llm.call_json`) | Slot `S5-arbitrate` ; modèle/provider résolus via `task_config()`, mêmes défauts (repli global tant qu'aucune variable dédiée n'est posée) |
| Prompt arbitrage | Seul prompt scalaire (`ARBITRATION_TEMPLATE`) | Prompt scalaire conservé à l'identique + enveloppe lot `decisions[]` indexée par `request_id` (`ARBITRATION_BATCH_TEMPLATE`) |
| Nouvelle fonction | — | `arbitrate_batch(requests)` : jusqu'à `batch_size` cas `(request_id, word, pos, context_text, synsets)` dans un seul prompt |
| Panne / réponse lot incomplète | (n/a, pas de lot avant) | `request_id` manquant ou dupliqué → même forme d'échec que la panne LLM unitaire (`selected_sense: None`, `confidence: 0.0`, `error`) |
| Cache | `cache_metadata` absent de l'appel `arbitrate()` | `cache_metadata` inclut `task_id`, `model`, `mode_batch`, `batch_size` (unitaire ou lot) |
| Point d'appel (`analyze_occurrence:968-969`) | `if needs_arbitration and allow_arbitration: arb = arbitrate(word, pos, context_text, synsets)` | **Inchangé, ligne pour ligne** — voir §3 |

`mode_batch` reste strictement le regroupement de plusieurs cas dans un même
prompt. Aucun usage de `litellm.batch_completion` n'a été ajouté.

## 3. Politique S5-2 : preuve d'absence de changement

- `calibrated_resolution_policy()` (`senses.py`) : **aucune ligne modifiée**. Les tests
  existants `test_s5_calibrated_policy.py` (5 tests, non touchés) continuent de couvrir
  cette fonction et passent à l'identique.
- Le point de déclenchement de l'arbitrage dans `analyze_occurrence` —
  `if needs_arbitration and allow_arbitration:` puis l'appel à `arbitrate(...)` — n'a
  pas été touché : seul le corps de `arbitrate()` change (modèle/provider/cache), pas
  *quand* elle est appelée ni ce qui décide `needs_arbitration`.
- Constat, sans lien avec ce lot mais vérifié en creusant l'appelant : dans le chemin de
  production actuel (`run()` → `resolve_joint_occurrence()`), `analyze_occurrence` est
  toujours invoquée avec `allow_arbitration=False` (idem dans
  `verify_senses_regression.py`) — l'arbitrage LLM n'est donc concrètement déclenché par
  aucun appelant de production aujourd'hui. Ce lot ne change rien à ce fait : il rend la
  fonction `arbitrate()`/`arbitrate_batch()` correctement configurable pour le jour où un
  appelant repassera `allow_arbitration=True` (ou pour un usage eval), sans toucher au
  déclenchement lui-même.
- Pas de branchement automatique de `arbitrate_batch()` dans `analyze_occurrence`/`run()` :
  aucun appelant de production ne collecte aujourd'hui plusieurs arbitrages en attente
  (contrairement à S3 où `run()` boucle déjà sur des occurrences indépendantes). Ajouter un
  tel regroupement aurait exigé de restructurer `analyze_occurrence`/`resolve_joint_occurrence`
  au-delà de ce que demande le plan pour M5 (« batch optionnel : plusieurs cas dans un
  prompt ; deux prompts » — pas d'exigence de rebranchement du point d'appel unique
  existant). `arbitrate_batch()` est livrée testée et utilisable par un futur appelant
  (eval ou pipeline) sans modification supplémentaire.

## Vérifications

- Tests offline (`test_multi_models_m5_s5.py`, 6 tests, tous mockés, aucun appel réseau) :
  - chemin unitaire utilise le modèle/provider du slot, schéma scalaire, `cache_metadata`
    non-lot ;
  - panne LLM unitaire → `selected_sense: None` (comportement identique à avant) ;
  - chemin lot renvoie une décision normalisée par `request_id`, `cache_metadata` en mode
    lot avec la bonne taille ;
  - `request_id` manquant/dupliqué → résultat en erreur par entrée concernée ;
  - dépassement de `batch_size` configuré → `ValueError` explicite (pas de repli silencieux) ;
  - panne LLM en mode lot → toutes les entrées du lot marquées en erreur.
- Suite complète : 252 tests verts (246 avant + 6 nouveaux), 4 skip, 11 échecs attendus —
  identique à avant ce lot. Aucune régression sur `test_s5_calibrated_policy.py` ni sur
  les lots précédents (`test_llm_tasks.py`, `test_multi_models_m2_s6.py`,
  `test_multi_models_m3_m4.py`).
- Défaut du registre M1 reste `mode_batch=false`, `batch_size=1` pour `S5-arbitrate` :
  sans variable dédiée, le comportement est donc unitaire, avec le même repli modèle
  global qu'avant (`VOCAB_LLM_BACKEND`/`OLLAMA_MODEL`/`CATGPT_MODEL`).

## Gate M5

- [x] `arbitrate()` utilise `task_config("S5-arbitrate")`.
- [x] Batch optionnel avec deux prompts distincts (`ARBITRATION_TEMPLATE` /
      `ARBITRATION_BATCH_TEMPLATE`), parser `decisions[]`.
- [x] Défaut unitaire.
- [x] Mock ; politique S5-2 inchangée (diff nul sur `calibrated_resolution_policy` et sur
      le point d'appel `analyze_occurrence:968-969`).

## Gate suivant

Lot **M6** — chemins locaux S6 (`sense_fr.py`) et CLI : enregistrer
`S6-translate-local`/`S6-backtranslate-local` (déjà au registre M1, non branchés dans
`sense_fr.py`), exposer `--llm-task ID=model;batch=...` ou documentation env dans
`run_pipeline.py`, tableau des tâches + exemples ollama/openai/catgpt dans le README.
