# TODO — `pipeline_out/cache/` mélange réponses LLM et téléchargements dbnary/apertium

**Statut : constaté, pas traité.** Observé le 2026-08-30, en vérifiant si
`pipeline_out/cache/` pouvait être supprimé après le passage à
`data/llm_results.sqlite3` (commit `7e4f39e`).

## Le constat

`config.CACHE_DIR = ROOT / "pipeline_out" / "cache"` (`pipeline/config.py:66`)
sert à deux usages sans rapport l'un avec l'autre :

- **Réponses LLM** — `llm_client.cache_path_for()` (`llm_client.py:99-104`)
  écrit dedans pour les tâches pas encore migrées vers `run_units()`, voir
  [[llm_cache_migration_unfinished]].
- **Dictionnaires téléchargés** — `pipeline/lex_bilingual.py:72-73` :
  `DBNARY_CACHE_PATH` (`en_dbnary_ontolex.ttl.bz2`) et `APERTIUM_CACHE_PATH`
  (`apertium-fra-eng.dix`), sans aucun rapport avec un appel LLM.

Conséquence pratique : impossible de raisonner sur le dossier comme un tout.
`lex_bilingual.py:419` documente l'option `--reuse-cache` comme "réutilise les
fichiers déjà présents dans `pipeline_out/cache/` sans retélécharger" — une
description qui ne vaut que pour dbnary/apertium, alors que le même dossier
contient aussi des réponses LLM déjà payées.

## Pourquoi c'est pertinent

- Vider ou déplacer `pipeline_out/cache/` pour repartir sur un cache LLM propre
  force aussi le retéléchargement de dbnary/apertium (fichiers volumineux), et
  inversement.
- Le dossier entier est gitignored comme un seul bloc (voir `.gitignore`,
  ajouté par `7e4f39e`) — impossible de committer/versionner l'un sans l'autre
  si un jour ça devient utile pour l'un des deux usages.
- Cette question est directement ce qui a déclenché la remarque de
  l'utilisateur (2026-08-30) : "je ne peux pas supprimer
  `pipeline_out/cache/`" — la réponse correcte aujourd'hui est bien "non", pour
  deux raisons indépendantes (tâches LLM pas migrées + fichiers dbnary/apertium
  dedans).

## Pas encore fait

1. Décider où déplacer les téléchargements dbnary/apertium — piste évoquée :
   `pipeline_out/downloads/`, en laissant `pipeline_out/cache/` aux seules
   réponses LLM restantes.
2. Migrer les fichiers déjà présents sur disque (ou accepter un
   retéléchargement au prochain run de `lex_bilingual.py`).
3. Mettre à jour `.gitignore`, `lex_bilingual.py:419` (aide de
   `--reuse-cache`) et toute doc qui nomme `pipeline_out/cache/` comme cache de
   téléchargement.
4. À rapprocher de [[llm_cache_migration_unfinished]] avant de trancher : une
   fois les 3 dernières tâches LLM migrées vers `run_units()`,
   `pipeline_out/cache/` ne contiendrait plus QUE dbnary/apertium — la
   séparation proposée ici pourrait alors devenir un simple renommage du
   dossier plutôt qu'un vrai découpage, ce qui peut changer l'ordre dans lequel
   traiter les deux TODO.
