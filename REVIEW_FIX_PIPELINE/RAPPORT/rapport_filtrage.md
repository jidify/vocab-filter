# Rapport de filtrage — `vocab_filtered.csv`

Ce filtrage est un test hors-pipeline (scripts jetables, conservés dans
`REVIEW_FIX_PIPELINE/filter_tests/`) : il n'a modifié aucun code de
`pipeline/`. Il part de `pipeline_out/vocab.csv` (2974 lignes, run S1→S6-2
avec catgpt) et ne concerne **que les lignes `unit_type == "word"`**
(2476/2974) — les 498 lignes `unit_type == "mwe"` traversent le filtrage
sans y être soumises (les référentiels utilisés sont lexicaux, mot à mot)
et sont réintégrées telles quelles à la fin.

Deux ressources absentes du dépôt manquaient pour ce filtrage
(`word-prevalence.txt`, `kuperman-aoa.csv` — voir `pipeline/config.py`) ;
elles ont été fournies dans `DATASETS/`, en plus de `DATASETS/cefrj.csv`
utilisé comme référentiel CEFR pour ce test (différent de `cefr.csv` à la
racine, utilisé par `pipeline/select.py`).

## Chaîne de filtres appliquée, dans l'ordre

| # | Filtre | Source | Valeur/seuil | Règle |
|---|---|---|---|---|
| 1 | Word Prevalence | `DATASETS/word-prevalence.txt`, colonne `Pknown` | **`Pknown > 0.90`** (> 90 %) | mot connu par plus de 90 % des natifs interrogés ; mot absent du fichier → **exclu** |
| 2 | CEFR | `DATASETS/cefrj.csv`, colonne `CEFR` | **exclure si tous les niveaux connus ⊆ {A1, A2}** | jointure par POS (mapping `noun/verb/adjective/adverb` → `n/v/a/r`), repli sur l'union de tous les POS si le POS du mot ne matche aucune entrée ; niveau **inconnu → conservé** (non exclu) |
| 3 | Repêchage Zipf | `wordfreq.zipf_frequency(lemme, "en")` | **`zipf_wordfreq_en < 4.5`** | s'applique **seulement** aux mots exclus par le filtre 2 (A1/A2 exclusif) : un mot A1/A2 assez rare dans `wordfreq` (< 4.5) est réintégré malgré son niveau CEFR basique |

Colonnes ajoutées au passage, à titre informatif (non filtrantes) :
`aoa_test` (âge d'acquisition, `DATASETS/kuperman-aoa.csv`, colonne
`Rating.Mean`) et `zipf_freqzipfus` (colonne `FreqZipfUS` de
`word-prevalence.txt`, source de fréquence différente de `wordfreq`,
non interchangeable — écart absolu moyen mesuré de 0,31 point sur ce
périmètre, voir échanges précédents).

## Entonnoir (mots simples uniquement, 2476 lignes de départ)

| Étape | Sortie | Effet |
|---|---|---|
| Types `word` dans `vocab.csv` | 2476 | — |
| Filtre 1 — `Pknown > 90 %` | 2128 | -348 (317 absents de `word-prevalence.txt`, 31 avec `Pknown ≤ 90 %`) |
| Filtre 2 — exclusion A1/A2 exclusif | 936 | -1192 |
| Filtre 3 — repêchage `zipf_wordfreq_en < 4.5` | 1113 | +177 |
| **+ 498 MWE réintégrées telles quelles** | **1611** | contenu actuel de `vocab_filtered.csv` (régénéré avec ce seuil) |

## Pour reproduire / ajuster

Les trois valeurs à modifier pour resserrer ou élargir le périmètre sont :

- `MIN_PKNOWN = 0.90` (filtre 1, strict — `Pknown > 0.90`)
- `EXCLUDED_CEFR = {"A1", "A2"}` (filtre 2)
- `ZIPF_RESCUE_THRESHOLD = 4.5` (filtre 3)

Aucune n'est aujourd'hui câblée dans `pipeline/` (S4 calcule ces signaux
mais ne filtre pas dessus — voir le premier diagnostic de cette session) ;
ce rapport documente uniquement les valeurs utilisées pour produire
`vocab_filtered.csv`, pas une configuration du pipeline lui-même.
