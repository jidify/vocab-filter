# Rapport M2 — branchement S6 déjà batchées

## Avant

- `S6-translate-frontier` et `S6-reassign` utilisaient les constantes globales et
  un seul schéma lot.
- Stage B utilisait une taille 40 littérale ; Stage C acceptait `batch_size=20`
  mais envoyait tous les résidus dans un seul appel.
- Les caches S6 n'indexaient pas explicitement tâche, mode et taille effective.

## Après (branchement initial)

- Les quatre tâches S6 lisent `task_config` pour le modèle et la taille effective,
  avec défauts conservés : 40 / 40 / 20 / 10.
- Chaque tâche possède un contrat unitaire scalaire et un contrat lot enveloppé,
  avec sélection unitaire quand `mode_batch=false` ou taille effective 1.
- Stage C découpe réellement les cibles selon sa taille de lot.
- `require_frontier_model` valide le modèle contre le slot de tâche résolu ; les
  CLI S6 n'imposent plus la whitelist mono-modèle historique.
- Les caches frontier, reassign, backtranslation et judge incluent `task_id`,
  modèle, mode, taille effective et prompt.
- Aucun changement de routage S3/S5 production.

## Revue et correctifs (même lot M2, avant clôture)

Une relecture du branchement initial (avant tout commit) a identifié 6 anomalies
empêchant de déclarer M2 terminé. Tests d'abord (échec constaté), puis correctif :

1. **Chemin unitaire du juge Stage C dégradé** — `build_judge_unit_prompt`
   utilisait un prompt appauvri (sans phrases réelles du livre, sans POS, sans
   lemmes, sans mélange des candidats), alors que le dossier complet est
   documenté comme nécessaire (« sans elles, le juge travaillerait à l'aveugle »).
   Corrigé : `_format_judge_item` factorise désormais la ligne de dossier,
   réutilisée à l'identique par `build_judge_batch_prompt` et
   `build_judge_unit_prompt` — le chemin unitaire est un vrai N=1, pas un
   prompt distinct appauvri. `_judge_batch` utilise ces deux fonctions au lieu
   de dupliquer la construction inline.
2. **Provenance modèle fausse sous override** — `evidence.frontier_model`
   restait figé sur `config.SENSE_FR_FRONTIER_MODEL` même quand
   `VOCAB_LLM_S6_TRANSLATE_FRONTIER`/`VOCAB_LLM_S6_REASSIGN` pointait vers un
   autre modèle. `build_entry` (frontier) et `_build_reassigned_entry`/
   `apply_decision` (reassign) acceptent désormais `model` et l'écrivent tel
   que résolu par `run()`.
3. **`batch=true;batch_size=1` n'activait pas le chemin unitaire** pour
   frontier/reassign (Stage B/C géraient déjà ce cas). Nouvelle fonction
   `pipeline.llm_tasks.use_batch_prompt(task, batch_size=None)` — vrai
   seulement si `mode_batch` **et** taille effective >= 2 — centralisée et
   utilisée par les quatre tâches.
4. **Dérivation `mode_batch` illisible dans Stage B/C**, déduite de la présence
   d'un paramètre `model` plutôt que de la config. Remplacée par la même
   fonction `use_batch_prompt`, indépendante de `model`.
5. **Nettoyage** : `_Verdict`/`_BatchVerdicts` redéfinies localement dans
   `_judge_batch` (doublons morts des classes module-niveau) supprimées ;
   `require_frontier_model` distingue désormais son message selon qu'il valide
   un slot de tâche ou l'ancienne liste blanche globale ; aides `--model`
   (frontier/reassign) réécrites pour ne plus prétendre restreindre à
   `ALLOWED_FRONTIER_MODELS` ; `SENSE_FR_BACKTRANSLATE_BATCH_SIZE` (40) et
   `SENSE_FR_JUDGE_BATCH_SIZE` (20) ajoutées à `config.py` (avant : littéral
   de signature pour Stage B, paramètre mort pour Stage C) et lues par le
   registre `llm_tasks.py` à la place des littéraux `40`/`20` en dur ;
   garde-fous ajoutés (`_translate_batches` frontier/reassign,
   `_backtranslate_batch`, `_judge_batch`) : un appel unitaire avec plus d'un
   item lève désormais une erreur explicite au lieu de jeter silencieusement
   les items surnuméraires.
6. **Rapport incomplet** — voir « Écart avant/après » ci-dessous.

Une régression annexe a été détectée par la suite complète (hors des 6
anomalies mais bloquante pour la non-régression) : `test_tranches.py` mockait
`sense_fr_frontier._translate_batches` avec une signature `(batches, model)`
sans les nouveaux paramètres `mode_batch`/`batch_size` introduits par le
branchement M2 initial — corrigé en alignant le mock sur la signature réelle.

## Écart avant/après — invalidation de cache

L'ajout de `task_id` (et, pour Stage B/C, de `mode_batch`/`batch_size`) à la
clé de cache **invalide tous les caches disque S6 existants, même à
configuration inchangée** (modèle et taille identiques à avant M2) : les
anciennes clés ne contenaient ni `task_id` ni le mode. Conséquence concrète :
le prochain run réel de `sense_fr_frontier`, `sense_fr_reassign`, ou
`sense_fr_adjudicate --with-backtranslation/--with-judge` repaiera
l'intégralité des appels LLM déjà cachés avant ce lot, plutôt que de les
servir depuis `pipeline_out/cache/`. C'est un effet de bord attendu et accepté
(le plan §4.3 l'exige explicitement — « changer de modèle ou de mode invalide
l'ancien cache »), mais il a un coût réel au prochain run et doit être
communiqué avant de lancer un run de production, pas découvert après coup.

## Vérifications

- Tests mock M2 : `test_multi_models_m2_s6.py` — 32 tests verts (8 initiaux +
  24 ajoutés lors de la revue : invariant d'information unitaire/lot du juge,
  provenance modèle, sélection `batch_size=1`, garde-fous unitaires, variance
  de clé de cache par taille de lot).
- `test_llm_tasks.py` — 12 tests verts (9 initiaux + 3 sur `use_batch_prompt`).
- Tests ciblés historiques (`test_sense_fr_frontier`, `test_sense_fr_adjudicate`,
  `test_sense_fr_reassign`, `test_llm_backends`, `test_s5_joint_reassignment`,
  `test_sense_fr`) : 37 tests verts, aucune régression.
- Suite complète du dépôt (`python -m unittest discover -p "test_*.py"`) :
  238 tests, 0 échec (4 skip, 11 échecs attendus préexistants) — confirme
  l'absence de régression ailleurs, y compris sur `test_tranches.py` après
  correction du mock.
- Défauts numériques préservés dans les 4 tâches : 40 / 40 / 20 / 10.
- Aucun changement de routage S3/S5 production.

## Prochain gate

M2 clos. Prochain lot : **M3** — `S3-judge-occurrence` : brancher le modèle sur
son slot de tâche, défaut production `mode_batch=false` (non-régression), et,
si `mode_batch=true`, prompt batch distinct inspiré de
`fix_pipeline/evaluate_s3_judges.py::_run_local_batch` sans le coupler à
l'eval (voir plan §5 Lot M3). Ne pas toucher à `S3-definition-cluster` (M4) ni
à S5 dans ce lot.
