# Rapport de dédoublonnage — `vocab_deduped.csv`

Ce dédoublonnage est un test hors pipeline (scripts jetables, conservés dans
`REVIEW_FIX_PIPELINE/dedup_tests/`) : il n'a modifié aucun code de `pipeline/`.
Il part de `REVIEW_FIX_PIPELINE/vocab_filtered.csv` (1611 lignes, sortie de
`filter_book_vocab.py` — voir `rapport_filtrage.md`).

**Contrainte : le dédoublonnage intervient AVANT S6 (traduction).** Aucune règle
de fusion ci-dessous ne lit `meaning_fr`, `meaning_fr_official`, `meaning_fr_alt`
ni `fr_status` — vérifié par introspection du code source en tête de
`dedup_vocab.py::main()` (`assert_fr_fields_unused`), pas seulement déclaré ici.
Ces colonnes sont recopiées telles quelles dans `vocab_deduped.csv`, prises sur
la ligne représentante de chaque fusion, à titre informatif seulement.

## Où le dédoublonnage est censé être fait

Une seule fois, en amont — et une seule fois seulement :

| Étape | Fonction | Rôle |
|---|---|---|
| **S3** | `pipeline/mwe_judge.py:307` `assign_sense_ids()` | **LE point de dédoublonnage MWE** — plan `fix_pipeline/plan_action_fix_pipeline.md` §3, *Correction S3-2*. Regroupe sur (canon, POS, paraphrase compatible), Jaccard ≥ 0,6 sur gloses normalisées. |
| **S5** | `pipeline/senses.py` (résolution conjointe lemme/POS/sens) | équivalent côté mots : l'identité est le `best_sense` WordNet. |
| **S4** | `pipeline/select.py:295-302` `build_mwe_units()` | agrégateur pur : `defaultdict` sur `(canonical_form, pos, sense_id)`. Hérite du découpage S3, ne décide rien. |
| **S6/S7** | `pipeline/score.py:331`/`:427` | groupement mécanique pour les mots ; aucun groupement pour les MWE. |

S3/S5 décident de l'identité sémantique ; S4/S6/S7 n'ont pas le droit de
fusionner deux clés. **Ce point de dédoublonnage ne fonctionne pas** pour les
MWE (voir cause racine ci-dessous).

### Cause racine côté MWE — deux défauts distincts

1. **Ordre inversé.** `assign_sense_ids` (`mwe_judge.py:307`) regroupe sur
   `contextual_paraphrase`, puis `assign_cluster_definitions`
   (`mwe_judge.py:546`, appelée juste après en ligne 1041) choisit une
   définition *par cluster déjà figé*. La définition est le vrai normalisateur
   de sens mais arrive après que l'identité a été frappée.
2. **Le mauvais champ.** Dans `pipeline_out/mwe_decisions.jsonl`,
   `contextual_paraphrase` n'est souvent pas une glose mais une paraphrase de
   la *phrase* : pour `be going to`, les 32 valeurs observées sont
   « I am not going to do that. », « We are going to have Thanksgiving at your
   granddaughter's new place. »… Aucune méthode de similarité ne peut
   regrouper ça. Les `definition_en` produites ensuite, elles, sont de vraies
   gloses (« Used to express a planned or expected future action »). Le
   signal exploitable est `definition_en`, pas `contextual_paraphrase`.

## Choix de la méthode de similarité — mesures

Comparaison sur les 497 unités de `pipeline_out/selected_mwe.jsonl` (champ
`definition_en`, groupées par `(canonical_form, pos)`), clustering agglomératif
cosinus contre le Jaccard actuel de S3 —
voir `dedup_tests/compare_similarity.py` :

| seuil | Jaccard (méthode S3) | embeddings LaBSE | embeddings all-MiniLM-L6-v2 |
|---|---|---|---|
| 0,85 | 450 | 428 | 423 |
| 0,80 | 448 | 413 | 413 |
| 0,70 | 439 | 397 | 396 |
| **0,60** | 435 | **384** | 387 |
| 0,50 | 427 | 375 | 380 |

*(baseline sans fusion : 497)* — **les embeddings dominent le Jaccard à tous les
seuils.** Le Jaccard sature parce que deux gloses synonymes ne partagent presque
aucun mot (« To be required or obliged to do something. » vs « To need or be
strongly obliged or advised to do something. »).

Deux modèles comparés : `sentence-transformers/LaBSE` (bitext mining
multilingue, déjà en cache local, historiquement utilisé ailleurs dans ce
dépôt pour l'alignement FR/EN) et `sentence-transformers/all-MiniLM-L6-v2`
(modèle STS anglais dédié, léger). Résultats proches ; **LaBSE retenu** car il
sépare `get it`/VERB exactement selon ses deux sens réels à 0,60
(« comprendre » vs « prendre en charge / aller chercher »), MiniLM le scinde
en 3 clusters (sur-séparation légère). Aucun des deux n'a été jugé
définitivement supérieur — à revoir si le seuil est ajusté.

### La similarité ne marche pas pour les sens WordNet voisins (mots)

Même protocole sur les 164 lignes mot dupliquées de `vocab_filtered.csv` :

| méthode | lignes retirées |
|---|---|
| embeddings LaBSE sur `definition_en`, seuil 0,70 | −7 |
| embeddings LaBSE, seuil 0,60 | −17 |
| WordNet `wup_similarity` ≥ 0,90 | −0 |
| WordNet `wup_similarity` ≥ 0,80 | −2 |
| hyperonyme direct commun | −4 |

Les gloses WordNet sont trop courtes et lexicalement disjointes, les
hiérarchies verbales trop plates (`proceed.v.02`/`proceed.v.04`/`go.v.02`
restent séparés à tous les seuils raisonnables). **Aucune fusion automatique
fiable ici** — traité par une politique de sélection, pas une fusion (famille
C ci-dessous).

## Constantes utilisées (`dedup_tests/dedup_vocab.py`)

| Constante | Valeur | Rôle |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/LaBSE` | modèle de similarité, famille A |
| `MWE_SIM_THRESHOLD` | **0,60** | seuil cosinus de fusion MWE — point mesuré sur 8 cas témoins, pas validé sur l'ensemble |
| `MWE_LINKAGE` | `average` | linkage du clustering agglomératif |
| `MAX_SENSES_PER_LEMMA_POS` | **2** | famille C, sens max gardés par (lemme, POS) |
| `MIN_OCCURRENCES_TO_KEEP_EXTRA_SENSE` | **2** | un sens au-delà du 1er n'est gardé que s'il a ≥ 2 occurrences |
| `WORD_SIM_THRESHOLD` | `None` (désactivé) | fusion optionnelle par embeddings pour la famille C, marginale (−7 à 0,70) |

## Invariant de sécurité (famille A)

Deux lignes portant chacune un `sense_id` WordNet ou DBnary *différent*
(préfixe autre que `mwe-custom-v1:`) ne sont **jamais** fusionnées entre elles,
même dans le même cluster d'embedding : ces inventaires ont déjà tranché. Un
cluster contenant ≥ 2 identifiants non-custom distincts est explosé — chaque
ligne autoritaire redevient singleton, les lignes custom du cluster restent
groupées entre elles séparément (`enforce_authority_invariant`). Une fusion
implique donc toujours au moins un `mwe-custom-v1:*`.

Quand un cluster mélange un identifiant autoritaire et des lignes custom du
même sens (ex. `go on`/VERB, `put in`/VERB, `end up`/VERB, `get rid of`/VERB
ci-dessous), la ligne représentante retenue est **toujours** l'identifiant
WordNet/DBnary, jamais le custom, même à `book_count` égal.

## Entonnoir

| Étape | Lignes | Effet |
|---|---|---|
| `vocab_filtered.csv` | 1611 (1113 word, 498 mwe) | — |
| Famille A — fusion MWE par embedding (48 fusions) | 498 → 387 | −111 |
| Famille B — identités non résolues S5 (1 fusion) | 16 → 15 | −1 |
| Famille C — sélection top-N sens WordNet (75 groupes réduits) | 1097 → 1042 | −55 (61 occurrences écartées, jamais redistribuées) |
| **`vocab_deduped.csv`** | **1444** | **−167** |

## Contrôles exécutés par le script

1. **Conservation des occurrences** (familles A+B, qui fusionnent) :
   2834 occurrences avant fusion, 61 explicitement écartées par la famille C,
   2773 conservées après — assertion `total_occurrences_after == before - dropped`,
   passe.
2. **Aucun champ FR lu** : `assert_fr_fields_unused()` scanne le code source des
   règles de fusion et échoue si un champ FR y est indexé.
3. **Invariant WordNet/DBnary** : implémenté dans `enforce_authority_invariant`,
   appliqué avant toute fusion de la famille A.
4. **Cas témoins** (sortie du script) :

   | groupe | résultat |
   |---|---|
   | `care package`/NOUN/mwe | 1 ligne |
   | `piece of work`/NOUN/mwe | 1 ligne |
   | `be going to`/VERB/mwe | 2 lignes |
   | `have got to`/VERB/mwe | 2 lignes |
   | `get it`/VERB/mwe | 2 lignes |
   | `good`/a/word | 0 ligne (déjà absent de `vocab_filtered.csv` — filtré en amont, A1 courant) |
   | `good`/r/word | 2 lignes (`better.r.02`, `well.r.01` — sous le seuil `MAX_SENSES_PER_LEMMA_POS`, aucune réduction nécessaire) |

## Échantillon de fusions MWE (famille A), pour relecture manuelle

| groupe | lignes fusionnées | sense_id retenu | book_count |
|---|---|---|---|
| `there we go`/OTHER | 2 | `mwe-custom-v1:6d9d8c6b…` | 6 |
| `there you go`/OTHER | 3 | `mwe-custom-v1:9dcd4bf2…` | 4 |
| `smart-ass`/NOUN | 3 | `mwe-custom-v1:b08216e8…` | 4 |
| `go out of one's way`/VERB | 2 | `mwe-custom-v1:e740ce8e…` | 2 |
| `no way`/OTHER | 5 | `mwe-custom-v1:3c7cf399…` | 5 |
| `let someone go`/VERB | 2 | `mwe-custom-v1:05489353…` | 2 |
| `piece of work`/NOUN | 4 | `mwe-custom-v1:7c968147…` | 4 |
| `messed up`/ADJ | 2 | `mwe-custom-v1:b100ab58…` | 2 |
| `roll one's eyes`/VERB | 3 | `mwe-custom-v1:982fa35b…` | 3 |
| `Jesus Christ`/OTHER | 2 | `mwe-custom-v1:1c13838e…` | 2 |
| `can't wait`/VERB | 2 | `mwe-custom-v1:0321d78d…` | 2 |
| `make fun of`/VERB | 2 | `mwe-custom-v1:9a8623c6…` | 2 |
| `go on`/VERB | 2 | **`happen.v.01`** (custom absorbé) | 6 |
| `grow up`/VERB | 2 | `mwe-custom-v1:cd824f0a…` | 2 |
| `you guys`/OTHER | 3 | `mwe-custom-v1:c64c5b6d…` | 30 |
| `put in`/VERB | 2 | **`install.v.01`** (custom absorbé) | 4 |
| `end up`/VERB | 2 | **`finish_up.v.02`** (custom absorbé) | 3 |
| `join in`/VERB | 4 | `mwe-custom-v1:96f60dd7…` | 5 |
| `all right`/OTHER | 3 | `mwe-custom-v1:6bef36be…` | 3 |
| `get rid of`/VERB | 2 | **`obviate.v.01`** (custom absorbé) | 2 |

Les 4 lignes en gras illustrent l'invariant : un `sense_id` déjà tranché par
WordNet absorbe une variante `mwe-custom-v1:*` de définition identique ou
quasi-identique, sans jamais l'inverse.

## Échantillon de lignes écartées (famille C), pour relecture manuelle

Sens minoritaire d'un `(lemme, POS)` déjà représenté par ≥ 2 sens plus
prioritaires (`score_default`), et sous le seuil `MIN_OCCURRENCES_TO_KEEP_EXTRA_SENSE` :

| lemme/POS | sense_id écarté | occurrences | glose |
|---|---|---|---|
| `half`/a | `half.s.02` | 3 | partial |
| `fix`/v | `pay_back.v.02` | 1 | take vengeance on or get even |
| `apologize`/v | `apologize.v.01` | 1 | acknowledge faults or shortcomings or failing |
| `cell`/n | `cell.n.01` | 1 | any small compartment |
| `settle`/v | `settle.v.01` | 2 | settle into a position, usually on a surface or ground |
| `straw`/n | `straw.n.01` | 1 | plant fiber used e.g. for making baskets and hats |
| `proceed`/v | `proceed.v.02` | 1 | move ahead; travel onward in time or space |
| `proceed`/v | `go.v.02` | 1 | follow a procedure or take a course |
| `lead`/v | `lead.v.01` | 1 | take somebody somewhere |
| `beat`/n | `beat.n.06` | 2 | the sound of stroke or blow |
| `beat`/n | `pulse.n.02` | 3 | the rhythmic contraction and expansion of the arteries |

`beat`/n illustre la limite de la politique : `pulse.n.02` (3 occurrences) est
écarté alors qu'il a plus d'occurrences que certains sens gardés, parce que
`score_default` (pas `occurrences`) ordonne le classement — à surveiller si un
tri par occurrences s'avère préférable.

## Le multi-POS n'est pas touché

103 lemmes / 235 lignes ont plusieurs POS distincts (`good`/a + `good`/r,
`right`/a + `right`/n + `right`/v + `right`/r…) — c'est une distinction réelle
pour l'apprenant, **conservée telle quelle**, jamais fusionnée.

## Pour reproduire / ajuster

```powershell
uv run python REVIEW_FIX_PIPELINE/dedup_tests/compare_similarity.py   # comparaison Jaccard/embeddings
uv run python REVIEW_FIX_PIPELINE/dedup_tests/dedup_vocab.py          # dédoublonnage effectif
```

Les constantes à ajuster sont en tête de `dedup_vocab.py` (voir le tableau
ci-dessus). Aucune n'est câblée dans `pipeline/` — ce rapport documente
uniquement les valeurs utilisées pour produire `vocab_deduped.csv`, pas une
configuration du pipeline lui-même.

## Hors périmètre (à noter, pas à faire ici)

La correction durable est dans `pipeline/mwe_judge.py` : re-frapper le
`sense_id` depuis la `definition_en` choisie, **après**
`assign_cluster_definitions`, et non depuis `contextual_paraphrase` avant —
plus le constat que `contextual_paraphrase` n'est pas une glose et ne peut pas
servir de clé de sens. Voir `TODO/` pour le dépôt d'une note (convention du
dépôt : un `.md` par régression documentée), sans toucher au code.

Un second passage de dédoublonnage, cette fois informé par la traduction
(`meaning_fr_official`), reste une question ouverte pour après S6 — non traité
ici par construction.
