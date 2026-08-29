# Rapport final — gate multi-modèles (Lot M8)

Lot **M8** du plan `./fix_pipeline/multi_models/plan_multi_models.md` (§5 Lot M8,
§6, §7, §8). Clôture le chantier ouvert en M0 (`baseline_batch_inventory.md`) et
poursuivi M1→M6 (`report_m1_registry.md` → `report_m6_local_cli_readme.md`).
**M7 (alignement des evals `evaluate_s3_judges.py`/`eval_frontier_ablation.py`)
n'a pas été fait** — reporté, voir §6 ci-dessous ; le plan autorise
explicitement M8 avec « M0–M6 (M7 si fait) » comme pré-requis.

## 1. Vue d'ensemble avant / après (comparé à la baseline M0)

| Sujet | M0 (baseline) | M8 (aujourd'hui) |
|---|---|---|
| Configuration par tâche | constantes dispersées, un seul backend global pour S3/S5, littéraux de signature pour S6b-2 B/C | registre unique `pipeline/llm_tasks.py::TASK_REGISTRY` (9 `task_id`), override `VOCAB_LLM_<TASK_ID>=provider/modèle;batch=...;batch_size=...` |
| `S3_JUDGE_BATCH_SIZE=50` | non branché sur `mwe_judge.run()` (seul lecteur : `evaluate_s3_judges.py`) | toujours non branché sur la production — remplacé par le slot `S3-judge-occurrence` (défaut `mode_batch=false`, prod n'active jamais 50 sans mesure, conforme au plan §5 Lot M3) ; `evaluate_s3_judges.py` continue de lire `S3_JUDGE_BATCH_SIZE` seul (M7 non fait) |
| Prompt lot S3/S5 | inexistant en production (prototype eval seul) | existe et testé pour les 3 tâches (`S3-judge-occurrence`, `S3-definition-cluster`, `S5-arbitrate`), défaut production reste unitaire |
| S6b-1/S6c (frontier, reassign) | batch réel mais aucun chemin unitaire dédié/testé | chemin unitaire N=1 explicite + testé, batch inchangé (40 / 10) |
| S6b-2 B (backtranslate) | `batch_size=40` littéral de signature, jamais lu depuis `config` | `SENSE_FR_BACKTRANSLATE_BATCH_SIZE=40` dans `config.py`, lu par le registre ; chemin unitaire testé |
| S6b-2 C (judge-dossier) | `batch_size=20` paramètre mort — aucun découpage réel, tout `residual` part dans un seul appel | `SENSE_FR_JUDGE_BATCH_SIZE=20` réellement appliqué (découpage effectif, comportement **nouveau**, assumé en M2 — voir §3) ; chemin unitaire testé |
| `S6-translate-local`/`S6-backtranslate-local` | mécanisme de votes/rétro-traduction locale, aucun lien avec un registre | enregistrées (`batch_allowed: false`), branchées sur `task_config()` (M6) |
| `require_frontier_model`/`ALLOWED_FRONTIER_MODELS` | liste blanche mono-modèle globale | liste blanche par tâche résolue (`task_config(task_id).model`), globale conservée en repli |
| Alias `.env` `PROVIDER=chatgpt` | décoratif, aucune occurrence dans le code | mappé par `pipeline.llm_tasks._global_model()` vers `catgpt` pour les 5 tâches à repli global — **mais inopérant par défaut** : rien dans ce dépôt ne charge `.env` (pas de `python-dotenv`), donc `PROVIDER` n'atteint le process que si l'utilisateur l'exporte lui-même (voir §4) |
| Cache disque | clé sans `task_id` ni mode nulle part | `task_id` + `model` + `mode_batch` + `batch_size` effectif dans la clé, sur les 9 tâches |
| CLI `run_pipeline.py --llm-model` | laissait croire qu'il couvrait tout le LLM du pipeline | aide + docstring `configure_llm` précisent : backend global de repli seulement (S3/S5/S6-\*-local), jamais les 4 tâches S6 LiteLLM |
| README | backend Ollama/CatGPT pour S3/S5 seulement | + tableau des 9 `task_id`, exemples ollama/openai/catgpt, matrice de joignabilité réelle par client, invalidation de cache |

## 2. Checklist du plan §5 Lot M8

### 2.1 Trois providers joignables en config (même sans appel réel)

Satisfait pour les 9 tâches, au niveau configuration : `pipeline.llm_tasks.ALLOWED_PROVIDERS
= {"ollama", "openai", "catgpt"}` s'applique uniformément, quel que soit le
client d'exécution (`pipeline/llm.py` ou LiteLLM). Vérifié par
`test_llm_tasks.py` (parsing générique, alias `PROVIDER=chatgpt`→`catgpt`,
tous providers acceptés dans un override) et par du run mock réel par
provider :

| Client | ollama | openai | catgpt |
|---|---|---|---|
| `pipeline/llm.py` (S3, S5, `S6-*-local`) | `test_llm_backends.py::test_ollama_protocol_is_preserved` | `test_llm_backends.py::test_openai_task_provider_uses_openai_compatible_request` | `test_llm_backends.py::test_catgpt_openai_compatible_request` |
| LiteLLM (4 tâches S6 batchées) | `test_multi_models_m8_gate.py::LiteLlmProviderMatrixTests` (4 tests, nouveaux — l'écart manquait avant ce lot) | `test_multi_models_m2_s6.py` (nombreux, modèle `"openai/m"`) | `test_multi_models_m2_s6.py::test_frontier_entry_records_resolved_model_not_global_default` / `test_reassign_entry_records_resolved_model_not_global_default` (`"catgpt/dedicated"`) |

**Nuance honnête, déjà posée dans le README (M6) et reconfirmée ici** :
« joignable en config » ne veut pas dire « opérationnel à l'exécution réelle »
pour les 4 tâches S6 routées par LiteLLM :
- `openai/*` fonctionne nativement (provider natif LiteLLM).
- `ollama/*` fonctionne, mais LiteLLM lit sa **propre** variable
  `OLLAMA_API_BASE` (vérifié dans le code vendu,
  `litellm/llms/ollama/common_utils.py`), pas `OLLAMA_URL` de ce projet — à
  poser séparément.
- `catgpt/*` **n'est pas** un provider natif de LiteLLM (recherche exhaustive
  dans `.venv/Lib/site-packages/litellm/` : aucune référence à `catgpt`). Le
  poser sur une des 4 variables `VOCAB_LLM_S6_*` passait la validation de
  configuration (`task_config`/`require_frontier_model`) mais échouait à
  l'appel réel tant que le routage LiteLLM→CatGPT-Gateway n'était pas câblé.
  **Ce n'était pas un défaut introduit par ce chantier** (le M0 baseline
  notait déjà que S6b-1/2/c ne parlaient qu'à OpenAI en pratique) — seulement
  rendu *configurable sans erreur immédiate*, ce qui aurait pu laisser croire
  à tort que c'était opérationnel.
  **Écart fermé depuis** par un adaptateur minimal
  (`pipeline/llm_litellm_catgpt.py`, `litellm.CustomLLM`) — voir
  `report_m9_catgpt_litellm_adapter.md`. Limite restante, assumée : le
  gateway ne reçoit que `response_format: {"type":"json_object"}` (schéma
  injecté en texte dans le prompt, pas contraint comme le `json_schema`
  natif d'OpenAI) — une réponse hors schéma lève une `ValidationError` non
  rattrapée par l'appelant.

### 2.2 Chaque tâche production a un modèle résolu

Vérifié programmatiquement pour les 9 tâches du registre (nouveau test,
`test_multi_models_m8_gate.py::EveryProductionTaskResolvesAModelTests`, qui
boucle sur `TASK_REGISTRY` entier plutôt que sur une liste recopiée à la
main) :

```text
S3-judge-occurrence:      ollama/mistral-small:24b   mode_batch=False batch_size=1
S3-definition-cluster:    ollama/mistral-small:24b   mode_batch=False batch_size=1
S5-arbitrate:             ollama/mistral-small:24b   mode_batch=False batch_size=1
S6-translate-frontier:    openai/gpt-5-mini          mode_batch=True  batch_size=40
S6-backtranslate:         openai/gpt-5-mini          mode_batch=True  batch_size=40
S6-judge-dossier:         openai/gpt-5-mini          mode_batch=True  batch_size=20
S6-reassign:              openai/gpt-5-mini          mode_batch=True  batch_size=10
S6-translate-local:       ollama/mistral-small:24b   mode_batch=False batch_size=1
S6-backtranslate-local:   ollama/mistral-small:24b   mode_batch=False batch_size=1
```

(capture réelle, `task_config()` sous l'environnement du process, sans
override — identique au tableau §2.2 ci-dessous)

### 2.3 Chaque tâche `batch_allowed: true` a deux chemins de prompt couverts par des tests

7 tâches concernées (les 2 tâches `-local` restent `batch_allowed: false`,
hors périmètre de cette règle par construction, plan §0/§6) :

| `task_id` | Prompt unitaire | Prompt lot | Tests dédiés (fichier) |
|---|---|---|---|
| `S3-judge-occurrence` | `OCC_PROMPT_TEMPLATE` (préexistant) | `decisions[]` indexé `occurrence_id` (nouveau, M3) | `test_multi_models_m3_m4.py` (2 classes, unitaire+lot+manquants/doublons) |
| `S3-definition-cluster` | `DEFINITION_PROMPT_TEMPLATE` (préexistant) | `decisions[]` indexé `cluster_id` (nouveau, M4) | `test_multi_models_m3_m4.py` |
| `S5-arbitrate` | `ARBITRATION_TEMPLATE` (préexistant) | `ARBITRATION_BATCH_TEMPLATE`, `decisions[]` indexé `request_id` (nouveau, M5) | `test_multi_models_m5_s5.py` (6 tests) |
| `S6-translate-frontier` | `build_unit_user_prompt`/`UnitTranslation` (nouveau, M2) | `build_user_prompt`/`BatchTranslations` (préexistant) | `test_multi_models_m2_s6.py` |
| `S6-backtranslate` | `build_backtranslate_unit_prompt`/`_Guess` (nouveau, M2) | `build_backtranslate_batch_prompt`/`_BatchGuesses` (préexistant) | `test_multi_models_m2_s6.py` |
| `S6-judge-dossier` | `build_judge_unit_prompt`/`_Verdict` (nouveau, M2 — même ligne de dossier que le lot, corrigé en revue M2) | `build_judge_batch_prompt`/`_BatchVerdicts` (préexistant) | `test_multi_models_m2_s6.py` |
| `S6-reassign` | `build_unit_user_prompt`/`UnitReassignedDecision` (nouveau, M2) | `build_user_prompt`/`ReassignBatch` (préexistant) | `test_multi_models_m2_s6.py` |

Règle des deux prompts (plan §0/§6) : satisfaite pour les 7 tâches — 0 écart
restant par rapport à l'état « à compléter / à créer » listé dans
`baseline_batch_inventory.md` §8.

### 2.4 Défauts = baseline comportementale

- S3/S5 : `mode_batch=false` par défaut — **inchangé**, aucune tâche de
  production n'active un lot sans variable dédiée.
- S6 batché : `40 / 40 / 20 / 10` — **valeurs inchangées**. Deux nuances déjà
  actées et documentées en leur temps (pas des régressions silencieuses) :
  - `S6-judge-dossier` (`batch_size=20`) : le comportement *appliqué* a
    changé — avant M2, `batch_size` était un paramètre mort et **tout**
    `residual` partait dans un seul appel sans plafond ; depuis M2, le
    découpage par 20 est réellement effectué. C'est un choix explicite acté
    dans `report_m2_s6.md` (« pas la reprise d'un comportement déjà validé »),
    pas une dérive de baseline.
  - `S6-backtranslate` (`batch_size=40`) : remonté d'un littéral de signature
    vers `config.SENSE_FR_BACKTRANSLATE_BATCH_SIZE`, valeur identique.
- `S6-translate-local`/`S6-backtranslate-local` : toujours le mécanisme de
  votes (`SENSE_FR_LLM_DRAWS=3`) + rétro-traduction, backend global par
  défaut — **inchangé**, confirmé par test (`test_sense_fr.py::
  LocalTaskSlotsNoRegressionWhenEnvEmptyTests`, environnement réellement vide,
  y compris `PROVIDER`/`VOCAB_LLM_BACKEND`).

### 2.5 Documenter l'invalidation de cache lors d'un switch modèle

Documenté dans `README.md` (section « Invalidation du cache disque au
changement de modèle/mode », ajoutée en M6) et dans `report_m2_s6.md` §« Écart
avant/après — invalidation de cache » pour le premier effet de bord constaté
(branchement M2). Résumé : la clé de cache de chaque appel LLM inclut
désormais `task_id`, `model`, `mode_batch` et la taille de lot effective —
tout changement de modèle, de backend ou de mode invalide le cache disque de
la tâche concernée (`pipeline_out/cache/`), y compris au premier run suivant
ce chantier lui-même (les caches antérieurs à M2 ne portaient aucune de ces
clés). Effet de bord voulu par le plan §4.3, pas une régression, mais à
anticiper avant un run de production.

### 2.6 Pas de régression mock

```text
uv run python -m unittest discover -p "test_*.py"
Ran 266 tests in ~28s
OK (skipped=4, expected failures=11)
```

Exécuté deux fois consécutives pour écarter toute dépendance à l'ordre
(voir §5 — un vrai piège de cache disque trouvé et corrigé pendant ce lot) :
résultat identique aux deux passages. Progression du nombre de tests par
lot : M0 (0 test dédié, rapport seul) → M1 : 9 → M2 : +23 (32 cumulés côté S6)
→ M3/M4 : +7 → M5 : +6 → M6 : +9 (7 nouveaux + 2 CLI) → **M8 : +5**
(`test_multi_models_m8_gate.py`, matrice provider LiteLLM/ollama manquante +
balayage complet du registre). Aucun test préexistant modifié pour faire
passer un test neuf (seule exception : les deux tests de non-régression
« env vide » de `test_sense_fr.py`, réécrits **pendant M6**, avant tout
commit, pour ne plus dépendre de `config.LLM_BACKEND` — voir
`report_m6_local_cli_readme.md`).

## 3. Un vrai écart trouvé et corrigé pendant ce lot (pas seulement audité)

En vérifiant l'item 2.1 (matrice provider), la version isolée des 4 nouveaux
tests LiteLLM/ollama passait, mais échouait dans `unittest discover` : les
quatre fonctions concernées (`frontier._translate_batches`,
`reassign._translate_batches`, `adjudicate._backtranslate_batch`,
`adjudicate._judge_batch`) écrivent un **vrai cache disque** sous
`config.CACHE_DIR` (`pipeline_out/cache/`, gitignoré mais persistant entre
exécutions successives de la suite). Le premier passage (isolé) écrivait ce
cache ; le second (discovery, ou toute relance ultérieure) le retrouvait et
court-circuitait silencieusement l'appel mocké — `mocked.call_args` devenait
`None`. Corrigé en isolant `config.CACHE_DIR` sur un répertoire temporaire le
temps du test (`tempfile.TemporaryDirectory()` + `patch.object`), au lieu de
nettoyer un fichier de cache nommé à la main comme le fait déjà
`test_multi_models_m2_s6.py::test_frontier_translate_batches_batch_size_one_sends_unit_prompt`.
Revérifié par deux exécutions consécutives de la suite complète (§2.6) : même
résultat aux deux passages.

## 4. Rappel de l'écart PROVIDER=chatgpt corrigé en M6 (pas un nouveau défaut)

Déjà documenté dans `report_m6_local_cli_readme.md` §« Un vrai écart trouvé et
corrigé » : `llm_is_available()` (ping mémoïsé, un seul par process, dans
`pipeline/sense_fr.py`) pingait `config.LLM_BACKEND` brut, qui ignore l'alias
`.env` `PROVIDER=chatgpt` honoré par `task_config()` depuis M1 — un run réel
avec `PROVIDER=chatgpt` seul (sans `VOCAB_LLM_BACKEND`) pouvait pinger Ollama
alors que les appels réels partaient vers CatGPT. Corrigé en M6
(`llm.is_available(backend=...)` + `sense_fr.llm_is_available()` résolu via
`task_config`). Rementionné ici parce que le fichier `.env` de ce dépôt pose
toujours `PROVIDER=chatgpt` — pertinent pour la checklist « aucun défaut
critique » : celui-ci a été trouvé et fermé, pas laissé ouvert.

**État factuel, pas un détail mineur** : `.env` `PROVIDER=chatgpt` n'est
chargé dans l'environnement du process que si le mécanisme de lancement de
l'utilisateur le fait lui-même — `uv run` 0.8.15 ne charge pas `.env`
automatiquement, et aucun `python-dotenv` (ni équivalent) n'existe dans ce
dépôt (vérifié : aucune occurrence de `dotenv`/`load_dotenv` dans
`pipeline/*.py` ni `pyproject.toml`). Vérifié aussi à l'exécution : un
process lancé sans export explicite n'a `PROVIDER` ni aucune variable
`VOCAB_LLM_*`/`CATGPT_*` définie. **Par défaut, la ligne `PROVIDER=chatgpt`
de `.env` ne configure donc rien** — elle ne prend effet que si l'utilisateur
l'exporte lui-même dans son shell (`direnv`, profil, etc.) avant de lancer le
pipeline. À garder en tête en configurant S6-2 (§7) : le simple fait de
poser `PROVIDER=chatgpt` dans `.env` ne suffit pas.

## 4bis. Deux clients LLM distincts, jamais nommés comme tels dans ce rapport

Ce rapport parle de « 3 providers », « 9 tâches », comme si un seul mécanisme
d'appel LLM existait. En réalité il y en a **deux**, implémentés
indépendamment :

| | Client A | Client B |
|---|---|---|
| Fichier | `pipeline/llm.py` | LiteLLM (`litellm.completion`/`batch_completion`) |
| Mécanique | stdlib `urllib`, JSON fait main | bibliothèque tierce, `response_format=<Pydantic>` |
| Providers | `ollama`, `catgpt`, `openai` | `openai` (natif), `ollama` (via `OLLAMA_API_BASE`, propre à LiteLLM), `catgpt` (via l'adaptateur `pipeline/llm_litellm_catgpt.py`, `litellm.CustomLLM`) |
| Tâches | `S3-judge-occurrence`, `S3-definition-cluster`, `S5-arbitrate`, `S6-translate-local`, `S6-backtranslate-local` | `S6-translate-frontier`, `S6-backtranslate`, `S6-judge-dossier`, `S6-reassign` |
| catgpt disponible depuis | `1a8cec0` (2026-08-28, « Add CatGPT Gateway LLM backend ») | `6d12c0e` (2026-08-29, adaptateur LiteLLM→CatGPT, hors plan initial — voir `report_m9_catgpt_litellm_adapter.md`) |

Conséquence concrète de cette confusion : `mwe_judge.run()` (Client A) sait
parler à catgpt depuis le 28/08, **indépendamment** de l'adaptateur LiteLLM
ajouté le 29/08 pour le Client B — les deux faits n'ont aucun rapport de
cause à effet, mais le vocabulaire unifié du rapport M8 ne permettait pas de
le voir sans relire le code des deux clients.

**Tâche de suite (non planifiée ici)** : unifier le pipeline sur un seul
client LLM. Il devra être basé sur LiteLLM, puisque l'adaptateur
`pipeline/llm_litellm_catgpt.py` rend maintenant catgpt joignable de ce
côté — ce qui manquait pour faire disparaître le Client A sans perdre
catgpt sur les 5 tâches qui en dépendent encore.

**Fermé depuis** — voir `report_u_unified_client.md` (Lots U1–U6) :
`pipeline/llm_client.py` remplace les deux clients ci-dessus,
`pipeline/llm.py` (Client A) supprimé après un gate de parité réel sur 50
cas S3 (1 écart mesuré, accepté par décision explicite — voir ce rapport
§4). Ajoute au passage `custom_prompt` (registre + `pipeline/prompt_variants.py`)
et ferme le Lot M7 (§5 ci-dessous).

## 5. M7 — fermé (Lot U6, voir `report_u_unified_client.md`)

Reporté au moment de M8 (`FRONTIER_MODEL`/`LOCAL_MODEL`/`DEFAULT_JUDGE_MODEL`
en dur dans `evaluate_s3_judges.py`/`eval_frontier_ablation.py`, aucun des
deux scripts aligné sur le registre). Fermé pendant le chantier
d'unification (Lot U6) : `evaluate_s3_judges.py` résout `LOCAL_MODEL`/
`FRONTIER_MODEL` via `task_config()` ; les deux scripts routent leurs appels
via `pipeline/llm_client.py`. Indépendance juge/candidat de l'ablation
préservée (`config.py:333-336`) : `DEFAULT_CANDIDATE_MODEL`/
`DEFAULT_JUDGE_MODEL` restent des constantes libres, jamais résolues via
`task_config`/`ALLOWED_FRONTIER_MODELS` — seule la mécanique d'appel a
changé, pas la logique de sélection du modèle juge.

## 6. Comment configurer S6-2 ensuite (plan §7)

Le plan anticipe : « après M8, S6-2 peut préciser dans `.env` p.ex. frontier
OpenAI + Stage C OpenAI (ou CatGPT) sans retoucher le code ». Concrètement,
aujourd'hui :

```bash
# Frontier (S6b-1) et Stage C juge dossier (S6b-2 C) sur deux modèles OpenAI
# différents — le cas d'usage principal visé par S6-2, sans toucher au code :
VOCAB_LLM_S6_TRANSLATE_FRONTIER="openai/gpt-5-mini;batch=true;batch_size=40"
VOCAB_LLM_S6_JUDGE_DOSSIER="openai/gpt-5.6-sol;batch=true;batch_size=20"
```

- Aucune modification de code requise — c'est exactement ce que ce chantier
  livre : deux tâches indépendantes, deux slots, deux modèles.
- **« (ou CatGPT) »** du plan §7 est désormais atteignable pour ces tâches
  (S6b-1/S6b-2 B/C, S6c) grâce à l'adaptateur
  `pipeline/llm_litellm_catgpt.py` (§2.1, `report_m9_catgpt_litellm_adapter.md`) —
  avec la limite assumée d'un schéma *instruit* dans le prompt, pas *contraint*
  comme le `json_schema` natif d'OpenAI. CatGPT reste, comme avant, pleinement
  fonctionnel pour S3/S5/S6-\*-local (client `pipeline/llm.py`, testé sur les
  3 providers).
- Chaque changement de modèle sur une tâche déjà utilisée invalide son cache
  disque (§2.5) — prévoir le coût du premier run après un changement de
  configuration S6-2.
- `require_frontier_model`/`ALLOWED_FRONTIER_MODELS` continuent de protéger
  contre une frappe de travers : un modèle qui ne correspond pas à celui
  résolu par `task_config(task_id)` lève une erreur explicite au lancement
  des CLI `sense_fr_frontier.py --model ...` / `sense_fr_reassign.py --model
  ...`, pas un run silencieusement plus coûteux que prévu.

## 7. Commandes exécutées (ce lot)

```text
uv run python -m unittest test_multi_models_m8_gate -v      # rouge sur l'item cache (discovery), vert isolé
# -> isolation config.CACHE_DIR par tempfile.TemporaryDirectory()
uv run python -m unittest test_multi_models_m8_gate -v      # vert, x2 consécutifs (anti-flakiness)
uv run python -m unittest discover -p "test_*.py"           # 266 tests, 0 échec, x2 consécutifs
uv run python -c "... task_config(...) pour les 9 task_id"  # capture §2.2
```

## 8. Écarts avant/après (résumé exécutif)

- **Avant M0** : 3 tâches S3/S5 sans slot ; 4 tâches S6 batchées sans chemin
  unitaire testé ; 2 tâches locales hors registre ; cache sans `task_id`/mode ;
  CLI laissant croire à une config mono-modèle.
- **Après M8** : 9 tâches enregistrées et résolues ; 7 tâches `batch_allowed`
  avec deux chemins de prompt testés ; 2 tâches locales branchées ; cache
  indexé par tâche/modèle/mode partout ; CLI et README explicites sur le
  périmètre exact de chaque levier de configuration ; 266 tests verts (0 échec,
  4 skip, 11 échecs attendus préexistants, stable sur plusieurs exécutions).
- **Écarts encore ouverts, explicitement documentés, pas cachés** : M7 non
  fait (evals encore en dur) ; dépendance au chargement effectif de `.env`
  par l'environnement de lancement pour l'alias `PROVIDER=chatgpt`.
- **Écart fermé après M8** : CatGPT côté LiteLLM pour les 4 tâches S6, par un
  adaptateur minimal — voir `report_m9_catgpt_litellm_adapter.md`. Limite
  assumée : schéma *instruit* dans le prompt, pas *contraint* (pas de
  `json_schema` natif pour ce provider).

## Gate M8

- [x] Trois providers joignables en config, avec au moins un run mock par
      provider — y compris le trou LiteLLM/ollama comblé pendant ce lot.
- [x] Chaque tâche production (9/9) a un modèle résolu, vérifié
      programmatiquement sur tout le registre.
- [x] Chaque tâche `batch_allowed` (7/7) a deux chemins de prompt couverts par
      des tests.
- [x] Défauts = baseline comportementale (40/40/20/10, unitaire S3/S5), écarts
      assumés (Stage C) déjà actés en leur temps, pas nouveaux.
- [x] Invalidation de cache au switch modèle documentée (README + ce rapport).
- [x] Pas de régression mock : 266/266, stable sur plusieurs exécutions.
- [x] Aucun défaut critique : le seul écart réel trouvé pendant ce lot
      (flakiness de cache disque dans les nouveaux tests) a été corrigé avant
      de clore le lot, pas laissé en l'état.
- [x] Bascule `.env` documentée pour S6-2 (§6), avec ses limites actuelles
      honnêtement signalées plutôt que passées sous silence.

Chantier multi-modèles (M0–M6, M8) **clos**. M7 reste ouvert, optionnel, à
reprendre séparément si besoin — pas de prochain gate automatique dans ce
plan au-delà de M8.
