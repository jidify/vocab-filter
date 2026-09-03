# Pipeline POC de sélection de vocabulaire

Orchestrateur qui enchaîne les six étapes du pipeline (`stages/`) pour
produire, à partir d'un livre `.txt`, le vocabulaire à apprendre : extraction
des mots/MWE, traduction+analyse via catgpt, fusion, localisation dans le
texte.

## Commande

```bash
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py --file "<livre.txt>"
```

Le résultat est écrit dans `out/<slug-du-livre>/<slug>-vocabulary.csv`
(jamais un chemin libre, pour ne jamais écraser le résultat d'un autre livre),
avec les fichiers intermédiaires et l'audit de chaque filtre dans
`out/<slug>/transient/`.

## Étapes

`word_extract` → `word_translate` (LLM) → `mwe_extract` → `mwe_translate`
(LLM) → `merge` → `localize`.

## Options

| Option | Défaut | Rôle |
|---|---|---|
| `--file` | *(requis)* | Livre `.txt` en entrée |
| `--skip-lines N` | `0` | Lignes de tête à ignorer (hors-œuvre) en plus de la détection automatique |
| `--out-dir` | `out/<slug-du-livre>` | Répertoire racine de sortie pour ce livre |
| `--from ÉTAPE` | — | Reprend à partir de cette étape (voir les 6 noms ci-dessus) |
| `--only ÉTAPE` | — | Ne lance que cette étape |
| `--force` | off | Rejoue les étapes déterministes même si leur sortie existe déjà |
| `--restart` | off | Étapes LLM : ignore et réécrit leurs CSV de sortie/reprise |
| `--no-cache` | off | Étapes LLM : désactive le cache disque DSPy (~/.dspy_cache) |
| `--batch-max-phrases N` | `50` | Étapes LLM : phrases visées par lot (`0` = un appel par lemme/candidat) |
| `--limit N` | `0` (= tous) | Étapes LLM : plafond de lemmes/candidats traités |
| `--max-phrases N` | `0` (= toutes) | Plafond de phrases affichées par entrée dans les CSV de contextes |
| `--zone-percent P` | `5.0` | Taille des tranches de localisation, en % |

## Exemples

```bash
# Traitement complet d'un extrait
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py \
    --file "books_excerpts/The Humans - Stephen Karam - excerpt.txt"

# Livre complet, avec le hors-œuvre (copyright/sommaire) à sauter
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py \
    --file "books/The Humans - Stephen Karam.txt" --skip-lines 182

# Reprendre à partir d'une étape (les précédentes doivent déjà avoir tourné)
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py --file "<livre.txt>" --from mwe_extract

# Ne lancer qu'une seule étape (ex. relancer juste la traduction des mots)
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py --file "<livre.txt>" --only word_translate

# Rejouer les étapes déterministes (extraction/fusion/localisation) même si déjà faites
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py --file "<livre.txt>" --force

# Repartir de zéro sur les étapes LLM (ignore leur reprise habituelle)
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py --file "<livre.txt>" --restart

# Test rapide : 10 lemmes/candidats max par étape LLM
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py --file "<livre.txt>" --limit 10

# Résultats dans un répertoire différent
uv run python POC/pipeline/build_vocabulary_to_learn_pipeline.py --file "<livre.txt>" --out-dir /tmp/essai
```

## Prérequis

Les étapes `word_translate`/`mwe_translate` appellent catgpt : la passerelle
(`CATGPT_BASE_URL`, défaut `http://localhost:8000/v1`) doit être joignable,
sauf si `--only`/`--from` ne sélectionne que des étapes déterministes
(`word_extract`, `mwe_extract`, `merge`, `localize`).

Détails de conception (structure de `out/`, reprise par étape, garde-fous) :
voir le docstring en tête de `build_vocabulary_to_learn_pipeline.py`.
