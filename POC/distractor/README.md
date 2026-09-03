# Générateur de distracteurs

Pour chaque mot/expression (MWE) d'un CSV fusionné + localisé (sortie de
`POC/pipeline/stages/localize_words_and_mwe.py`), demande à catgpt 2 ou 3
distracteurs français pour un QCM de traduction — jamais une traduction
possible de l'expression, quel que soit le sens.

## Commande

```bash
uv run python POC/distractor/generate_distractors.py --in <votre_fichier.csv>
```

Sans `--in`, utilise le fichier d'exemple (`inputs/vocabulary_input_example.csv`).

Le résultat est écrit dans `out/<slug-du-fichier-d'entrée>/<slug>-distractors.csv`
(jamais un chemin libre, pour ne jamais écraser le résultat d'un autre
fichier). Un cache persistant de distracteurs (`cache/distractors_cache.csv`,
partagé entre TOUS les fichiers traités) évite de rappeler le LLM pour une
expression déjà connue d'un traitement précédent.

## Options

| Option | Défaut | Rôle |
|---|---|---|
| `--in` | `inputs/vocabulary_input_example.csv` | CSV d'entrée à traiter |
| `--out-dir` | `out/` | Répertoire racine des résultats (le résultat va dans `<out-dir>/<slug>/`) |
| `--limit N` | `0` (= toutes) | Ne traite que les N premières expressions uniques |
| `--batch-size N` | `50` | Expressions par appel LLM (`0` = un appel par expression) |
| `--restart` | off | Repart de zéro pour ce fichier (supprime `out/<slug>/`) |
| `--ignore-cache` | off | Ignore le cache de distracteurs en lecture (le réécrit quand même) |
| `--cache-path` | `cache/distractors_cache.csv` | Chemin du cache persistant de distracteurs |
| `--rejected-out` | `out/<slug>/audit/distractors_rejected.csv` | Journal (`expression,cause`) des distracteurs rejetés par le garde-fou de ce run |
| `--no-cache` | off | Désactive le cache disque de DSPy (~/.dspy_cache) — sans rapport avec `--cache-path` |
| `--dry-run` | off | Dédoublonne et affiche le plan de lots sans appeler le LLM ni rien écrire |
| `--replay-rejected` | off | Mode rejeu : relit `--rejected-out` et redemande au LLM les expressions concernées, en lui interdisant de reproposer les causes déjà rejetées. Remplace le flux normal pour ce run ; incompatible avec `--restart` |

## Exemples

```bash
# Traitement complet d'un fichier
uv run python POC/distractor/generate_distractors.py --in mon_livre/word_and_mwe_analysis.csv

# Test rapide sur 10 expressions, sans appeler le LLM (juste vérifier le plan)
uv run python POC/distractor/generate_distractors.py --in mon_livre.csv --limit 10 --dry-run

# Mode séquentiel (1 appel LLM par expression, au lieu de lots de 50)
uv run python POC/distractor/generate_distractors.py --in mon_livre.csv --batch-size 0

# Repartir de zéro pour ce fichier (le cache de distracteurs, lui, est conservé)
uv run python POC/distractor/generate_distractors.py --in mon_livre.csv --restart

# Forcer un recalcul LLM même pour des expressions déjà en cache
uv run python POC/distractor/generate_distractors.py --in mon_livre.csv --ignore-cache

# Résultats dans un répertoire différent
uv run python POC/distractor/generate_distractors.py --in mon_livre.csv --out-dir /tmp/essai

# Rejouer les expressions dont des distracteurs ont été rejetés (voir
# out/<slug>/audit/distractors_rejected.csv) — le LLM est explicitement
# invité à ne pas reproposer les causes déjà rejetées
uv run python POC/distractor/generate_distractors.py --in mon_livre.csv --replay-rejected
```

## Format d'entrée

Le CSV attendu est celui produit par `localize_words_and_mwe.py` (colonnes
`type`, `lemme`, `lexicalized_form`, `translations`, ...) — voir
`inputs/vocabulary_input_example.csv` pour un exemple. Avec ou sans ligne
d'en-tête, les deux sont détectés automatiquement.

Détails de conception (dédoublonnage, garde-fou anti-traduction, cache,
prompts DSPy) : voir le docstring en tête de `generate_distractors.py`.
