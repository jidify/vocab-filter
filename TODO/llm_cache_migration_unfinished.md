# TODO — migration `pipeline_out/cache/` → `data/llm_results.sqlite3` inachevée

**Statut : constaté, pas traité.** Observé le 2026-08-30, en vérifiant si
`pipeline_out/cache/` pouvait être supprimé après le commit `7e4f39e`
("Decouple LLM batch calls from unit-level result storage").

## Le constat

Ce commit a migré les 7 tâches `batch_allowed=True` du registre
(`pipeline/llm_tasks.py`) vers `llm_client.run_units()` /
`pipeline/llm_store.py`, dont la clé de stockage porte sur des valeurs métier
(`task_id`, `model`, `protocol`, `unit_id` + hash de la charge sémantique) —
jamais sur `batch_size`, `mode_batch`, ni le texte du prompt rendu.

Mais 3 sites d'appel n'ont pas été migrés et passent toujours par
`llm_client.call()`, dont le cache reste sur disque
(`pipeline_out/cache/`, via `llm_client.cache_path_for`, `llm_client.py:99-104`)
et clé sur le texte du prompt rendu :

- `pipeline/mwe_judge.py:138` — S3-judge-type
- `pipeline/sense_fr.py:270` — S6-translate-local
- `pipeline/sense_fr.py:295` — S6-backtranslate-local

Aujourd'hui ce n'est pas un bug : les 3 tâches sont déclarées
`batch_allowed=False` dans le registre (`pipeline/llm_tasks.py:71,79,80`), et
`task_config()` fait respecter cette contrainte en levant `TaskConfigError` si
on tente `batch=true` dessus (`llm_tasks.py:206-208`).

**Mais ce registre est du code de développement écrit en dur, destiné à
devenir variable** (confirmé par l'utilisateur, 2026-08-30) — pas une
décision d'architecture figée. Le jour où `batch_allowed` s'ouvre pour l'une
de ces 3 tâches, le cache disque réintroduit exactement le couplage
lot/stockage que le commit `7e4f39e` venait de supprimer pour les 7 autres :
changer `batch_size`, ou avoir un seul item invalide dans un lot, redevient
susceptible de rejouer tout le lot.

Autre écart déjà constaté : les valeurs inscrites en dur dans le registre
(`model="ollama/mistral-small:24b"`, `mode_batch=False`, `batch_size=1`) ne
décrivent plus les runs réels. Le cache LLM actuellement présent dans
`pipeline_out/cache/` a été produit avec `catgpt` et des lots de taille 50, via
les overrides d'environnement `VOCAB_LLM_*` que `task_config()` sait déjà
résoudre (`llm_tasks.py:125,177-188`) — le registre en dur n'est donc déjà
qu'un défaut de repli, pas la réalité des appels effectués.

## Pourquoi c'est pertinent

- Sans cette migration, le bénéfice du commit `7e4f39e` (repayer un lot en cas
  de changement de `batch_size` ou d'échec partiel) ne couvre que 7 tâches sur
  10 — et parmi les 3 restantes, S6-translate-local/S6-backtranslate-local
  utilisent déjà un mécanisme de plusieurs tirages par item
  (`config.SENSE_FR_LLM_DRAWS`, voir `sense_fr.py:240-284`), donc plusieurs
  appels réseau par entrée métier, avec le même risque de cache non réutilisable
  si le prompt bouge.
- `run_units()` couvre déjà le cas unitaire sans changement de mécanisme :
  `batch_size=1`/`mode_batch=False` produit une tranche d'un seul item par
  appel (point 2 du docstring, `llm_client.py:427-432`) — la migration des 3
  tâches n'a donc pas besoin d'un nouveau chemin de code, seulement de
  remplacer leurs 3 sites d'appel.
- Effet de bord déjà visible : `llm_client.call_batch_completion()`
  (`llm_client.py:236`) n'a plus aucun appelant en production — seulement
  `test_llm_client.py:172,191,206`. Code mort qui traîne l'ancien chemin de
  cache par lot, à retirer avec `cache_path_for` une fois les 3 dernières
  tâches migrées.
- Dérive documentaire à corriger en même temps que la migration :
  `pipeline/sense_fr_frontier.py:55,284` et `pipeline/sense_fr_reassign.py:287`
  décrivent encore `pipeline_out/cache/frontier_*.json` /
  `pipeline_out/cache/reassign_*.json` comme le mécanisme vivant, alors que ces
  deux tâches sont déjà sur `run_units()`.

## Pas encore fait

1. Migrer `mwe_judge.py:138` (S3-judge-type), `sense_fr.py:270`
   (S6-translate-local) et `sense_fr.py:295` (S6-backtranslate-local) vers
   `llm_client.run_units()`.
2. Écrire un récupérateur pour le cache disque existant de ces 3 tâches, sur le
   modèle de `tools/migrate_llm_cache.py` — sinon la migration perd sèchement
   tout ce cache-là (contrairement aux 7 tâches déjà migrées, où
   `migrate_llm_cache.py` avait récupéré 2965 lignes sans écart).
3. Supprimer `llm_client.call_batch_completion()` et `cache_path_for()` une
   fois qu'ils n'ont plus d'appelant réel — nettoyer leurs tests associés
   (`test_llm_client.py:172,191,206`).
4. Corriger les commentaires périmés de `sense_fr_frontier.py:55,284` et
   `sense_fr_reassign.py:287`.
5. Voir aussi [[cache_dir_mixes_llm_and_downloads]] — une fois cette migration
   faite, `pipeline_out/cache/` ne contiendrait plus que les téléchargements
   dbnary/apertium, ce qui simplifie (ou rend inutile) la séparation proposée
   là-bas.
