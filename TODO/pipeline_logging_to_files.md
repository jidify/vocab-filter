# TODO — mécanisme de logs fichier pour suivre l'avancement et les erreurs du pipeline

**Statut : pas encore traité.** Constaté à répétition le 2026-08-30 en
suivant en direct un run S1→S5 sur *The Humans* (voir aussi
`TODO/s3_occurrence_batch_prompt_redundancy.md`, trouvé pendant le même
suivi).

## Le problème

Chaque étape du pipeline (`pipeline/*.py::run()`) n'écrit sa progression
et ses erreurs que via `print()` sur stdout. Tant que le process tourne,
il n'y a **aucun moyen d'observer l'avancement en direct** dès que stdout
est redirigé vers un fichier plutôt qu'un terminal interactif : Python
bufferise stdout en mode bloc quand la sortie n'est pas un tty, donc rien
n'apparaît dans le fichier de log avant que le process se termine (succès,
échec, ou kill).

Concrètement pendant ce run :
- `senses.py` (S5) a tourné plus de 50 minutes sans qu'aucune ligne de
  progression ne soit visible dans le fichier de log — seul un fichier
  `pipeline_out/senses_run.log` PÉRIMÉ (5 jours plus vieux, issu d'un run
  antérieur) montrait un format `x/2922 (taux, ETA)`, ce qui a d'abord
  semblé être un log courant.
- La seule façon de distinguer "ça travaille" de "c'est bloqué" a été de
  sonder l'état du process OS (CPU cumulé, présence d'une connexion
  TCP établie vers le gateway LLM) — un proxy indirect, pas une source
  fiable de vérité sur ce que fait réellement le pipeline.
- Pareil pour `mwe_judge.py` (S3) : les pannes LLM individuelles
  (`panne(s) LLM d'occurrence`) ne sont visibles qu'à la toute fin du run,
  agrégées en un seul compteur — impossible de savoir EN COURS DE ROUTE
  combien d'occurrences ont déjà échoué, ni pourquoi (timeout ? HTTP 500 ?
  JSON invalide ?) sans attendre la fin ou tuer le process pour lire
  `mwe_decisions.jsonl`.

## Ce qui serait utile

1. Un vrai fichier de log par run (horodaté ou à chemin fixe type
   `pipeline_out/run.log`), écrit avec `flush=True` (ou via le module
   `logging` configuré avec un `FileHandler` non bufferisé/`flush`
   immédiat) — pas juste une redirection de stdout côté appelant.
2. Progression incrémentale visible en direct : au minimum un compteur
   "x/N" par étape, écrit à intervalle raisonnable (comme le fait déjà
   `mwe_judge.py` avec ses lignes `i/522`, mais dans un fichier qui se met
   à jour réellement pendant l'exécution, pas seulement à la fin).
3. Les erreurs (pannes LLM, réponses invalides, exceptions) journalisées
   individuellement AU MOMENT où elles surviennent, avec assez de contexte
   (occurrence_id, tâche, raison) pour diagnostiquer sans attendre la fin
   du run ni fouiller dans les caches.
4. Cohérent avec `run_pipeline.py::main()` qui imprime déjà des séparateurs
   `=== {name} ({module_name}) ===` par étape — le même mécanisme devrait
   couvrir l'intérieur de chaque étape, pas seulement leurs frontières.

## Pas encore fait

- Décider de l'outil : `logging` stdlib (configurable, niveaux, rotation)
  vs. simple `print(..., flush=True)` généralisé — probablement `logging`
  pour avoir aussi les niveaux (INFO progression, WARNING panne
  individuelle récupérable, ERROR échec bloquant).
- Décider où : un seul fichier par run dans `pipeline_out/`, ou un fichier
  par étape (cohérent avec la structure actuelle `mwe_decisions.jsonl`,
  `senses.jsonl`, etc.) ?
- Vérifier l'impact sur les tests existants qui capturent stdout
  (`capsys`/`subprocess.run(capture_output=True)`) avant de migrer les
  `print()` de progression vers `logging`.
