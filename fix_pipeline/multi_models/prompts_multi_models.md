# Prompts autosuffisants — multi-modèles / batch par tâche

## Mode d'emploi

Exécuter **un seul prompt à la fois**, dans l'ordre M0 → M8. Après validation (et commit éventuel), faire `/clear`, puis coller le prompt suivant.

Chaque prompt impose de lire d'abord le plan :

`./fix_pipeline/multi_models/plan_multi_models.md`

Compléments utiles selon le lot (ne remplacent pas le plan multi-modèles) :

- `./fix_pipeline/plan_action_fix_pipeline.md` — contexte S3–S6 et discipline §8 ;
- `./pipeline/config.py`, `./pipeline/llm.py`, `./run_pipeline.py` — état actuel des backends.

### Règles communes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M0 — baseline batch + contrat des tâches

Lis intégralement `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§0, §1, §2 et §6**. Sans modifier le runtime, produis `./fix_pipeline/multi_models/baseline_batch_inventory.md` qui, pour chaque `task_id` du §1, indique : module, client (`llm` / LiteLLM), batch prompt oui/non, constante et valeur actuelles, présence déjà d'un prompt unitaire et/ou lot, et la valeur cible `batch_allowed`. Distingue clairement batch prompt, parallélisme `batch_completion`, et `nlp.pipe`. Signale que `S3_JUDGE_BATCH_SIZE` n'est pas branché sur `mwe_judge.run()` et que `evaluate_s3_judges._run_local_batch` est un prototype. Rappelle la règle des deux prompts. Ne passe pas à M1.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M1 — registre et parseur de config

Lis le plan `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§3, Lot M1, §8**. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0) pour les défauts/constantes actuels de chaque tâche. Implémente le registre des tâches (`task_id`, `batch_allowed`, défauts modèle / `mode_batch` / `batch_size`) et le parseur d'overrides (env du type `VOCAB_LLM_S6_TRANSLATE_FRONTIER=openai/gpt-5-mini;batch=true;batch_size=40`, repli sur `VOCAB_LLM_BACKEND` / modèles globaux pour S3–S5, alias `PROVIDER=chatgpt` → `catgpt`). Expose `task_config(task_id)` (ou équivalent) avec validation stricte : `mode_batch=true` interdit si `batch_allowed=false` ; `batch_size` requis et ≥ 1 si batch. Sans brancher encore les modules S3–S6 sur ce registre (sauf lectures de défauts si nécessaire aux tests). Ajoute des tests unitaires hors réseau. Ne passe pas à M2.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M2 — brancher les tâches S6 déjà batchées

Lis le plan `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§1 (S6-*), §4, §6, Lot M2**. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0) pour les défauts numériques actuels et les écarts déjà relevés (ex. `run_stage_c` sans découpage réel). Branche `sense_fr_frontier` (`S6-translate-frontier`), `sense_fr_adjudicate` Stage B/C (`S6-backtranslate`, `S6-judge-dossier`) et `sense_fr_reassign` (`S6-reassign`) sur `task_config` pour modèle et `batch_size`. Conserve les défauts numériques actuels (40 / 40 / 20 / 10). Pour chaque tâche, garantis un chemin **unitaire** (`mode_batch=false` ou taille 1) avec prompt/schéma unitaire **distinct ou explicitement N=1**, et un chemin **lot** — deux prompts testés (plan §6). Adapte `require_frontier_model` / liste blanche au modèle résolu par tâche. Cache indexé par modèle + mode. Tests mock uniquement. Ne touche pas encore à S3/S5 production.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M3 — S3-judge-occurrence : modèle + batch optionnel

Lis le plan `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§1 `S3-judge-occurrence`, §5 Lot M3, §6**. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0) pour l'état actuel de cette tâche et du prototype `_run_local_batch`. Branche `mwe_judge.judge_occurrence` / `run()` sur le slot `S3-judge-occurrence`. Défaut production : `mode_batch=false` (non-régression). Si `mode_batch=true`, implémente un **prompt batch distinct** (s'inspirer de `fix_pipeline/evaluate_s3_judges.py::_run_local_batch` sans coupler l'eval), taille `batch_size`, parsing `decisions[]` avec détection manquants/doublons, mêmes écritures magasin / panne → `incertain` qu'aujourd'hui. Tests fixtures offline pour unitaire et lot (2–3 items), clés de cache distinctes. Ne change pas la définition de cluster (M4) ni S5.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M4 — S3-definition-cluster

Lis le plan `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§1 `S3-definition-cluster`, Lot M4, §6**. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0) pour l'état actuel de cette tâche. Branche `choose_cluster_definition` sur son slot modèle. Si `batch_allowed` et `mode_batch`, ajoute un **second prompt** (lot de clusters) + parser ; sinon chemin unitaire seul. Défaut : unitaire. Tests unitaire + lot minimal hors réseau. Ne touche pas à S5.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M5 — S5-arbitrate

Lis le plan `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§1 `S5-arbitrate`, Lot M5, §6**, et le contrat actuel de `pipeline/senses.py::arbitrate` / politique S5-2. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0) pour l'état actuel de cette tâche. Branche l'arbitrage sur `task_config("S5-arbitrate")`. Le batch optionnel doit utiliser un **prompt lot distinct** ; `mode_batch` ne doit pas changer *quand* l'arbitrage est déclenché, seulement le regroupement d'appels. Défaut : unitaire. Tests mock. Ne réouvre pas S6.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M6 — chemins locaux S6 + CLI + README

Lis le plan `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§1 hors run, §3.3–3.4, Lot M6**. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0), §3 (hors run par défaut) et §7 (modèles hérités à migrer). Enregistre `S6-translate-local` et `S6-backtranslate-local` (`pipeline/sense_fr.py`) dans le registre (`batch_allowed: false` en v1). Documente dans le README le tableau des `task_id`, les variables d'env, des exemples ollama / openai / catgpt, et l'invalidation de cache au switch modèle. Aligne `run_pipeline.py` / `configure_llm` pour ne plus laisser croire qu'un seul `--llm-model` couvre S6. Tests de non-régression si env vide. Ne lance pas M8.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M7 — alignement des evals (optionnel)

Lis le plan `./fix_pipeline/multi_models/plan_multi_models.md`, **Lot M7**. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0), §4 (evals hors production). Fais consommer le registre à `fix_pipeline/evaluate_s3_judges.py` et `pipeline/eval_frontier_ablation.py` (plus de modèle frontier/local en dur). Conserve l'indépendance juge/candidat exigée par l'ablation. Vérifie que les overrides de tâche fonctionnent. Saute ce prompt si M7 est reporté ; documente alors le report dans le rapport M8.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

---

## Prompt M8 — gate final multi-modèles

Lis intégralement `./fix_pipeline/multi_models/plan_multi_models.md`, surtout **§5 Lot M8, §6, §7, §8**. Relis aussi `./fix_pipeline/multi_models/baseline_batch_inventory.md` (baseline M0) pour comparer avant/après dans le rapport final. Exécute la checklist : chaque tâche production a un modèle résolu ; chaque tâche `batch_allowed` a deux chemins de prompt couverts par des tests ; défauts = baseline ; providers ollama/openai/catgpt configurables ; pas de régression mock. Produis `./fix_pipeline/multi_models/report_multi_models.md` (avant/après, écarts, caches, comment configurer S6-2 ensuite). Ne commence pas S6-2 dans ce lot sauf demande explicite.

### Consignes

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents ; ne pas annuler le travail utilisateur ;
- **ne pas** confondre `mode_batch` (n cas dans un prompt) avec `litellm.batch_completion` (parallélisme HTTP) ;
- pour toute tâche `batch_allowed: true`, livrer et tester **deux** chemins de prompt (unitaire + lot) — voir plan §0 et §6 ;
- défauts runtime = comportement actuel du pipeline tant qu'aucune env dédiée n'est posée ;
- pas d'exception livre-spécifique ; pas d'import de `vocab_corrige.csv` dans la production ;
- commencer par des tests qui échouent, puis implémenter ; ne pas annoncer le lot terminé si les vérifications du plan ne sont pas toutes satisfaites ;
- terminer par commandes exécutées, écarts avant/après, et prochain gate.

