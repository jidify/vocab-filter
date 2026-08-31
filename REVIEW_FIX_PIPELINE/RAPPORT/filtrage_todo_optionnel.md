# TODO optionnel — brancher le BNC2014 spoken comme 4e source de fréquence

**But de ce fichier** : donner à une future session (après un `/clear`,
sans mémoire de cette conversation) tout ce qu'il faut pour câbler ce
filtre dans `REVIEW_FIX_PIPELINE/filter_tests/filter_book_vocab.py` sans
avoir à réexplorer le dépôt. Coller ce fichier entier dans le prompt
suffit comme point de départ.

## Contexte (pourquoi)

`filter_book_vocab.py` filtre le vocabulaire d'un livre par AoA
(`kuperman-aoa.csv`), Word Prevalence (`Pknown`, `word-prevalence.txt`) et
CEFR (`cefrj.csv`), avec un repêchage CEFR basé sur la fréquence Zipf
(`ZIPF_RESCUE_THRESHOLD`). Deux sources de fréquence Zipf existent déjà et
ont été comparées : `wordfreq.zipf_frequency(lemme, "en")` (mélange
web/actualités/livres) et `FreqZipfUS` (colonne de `word-prevalence.txt`,
= SUBTLEX-US, sous-titres de films). Les deux sont biaisées vers l'écrit
ou le semi-scripté — aucune ne mesure la fréquence dans de la
**conversation spontanée réelle**, ce qui serait pourtant la mesure la
plus pertinente pour filtrer le vocabulaire d'un dialogue de théâtre
(voir `pipeline-vocabulaire-proposition_1.md`, section 7 : SUBTLEX/Zipf
pour les films, **Spoken BNC2014 pour la conversation réelle**).

Le corpus est déjà présent en local : `DATASETS/spoken-bnc2014/` (British
National Corpus 2014, volet oral — transcriptions de conversations
spontanées entre locuteurs natifs britanniques). Il n'est câblé nulle
part dans `pipeline/` aujourd'hui (`pipeline/score.py` mentionne l'idée
en commentaire — "BNC2014 en confirmation, non branché par défaut" —
mais aucun code réel ne le lit). Un seul précédent existe :
`bnc_escape_compare.py` (racine du dépôt), qui compte des expressions
ciblées dans le corpus — pas un loader générique de fréquences.

## Ce qui a déjà été mesuré (pas besoin de reconfirmer)

- **1251 fichiers XML** dans `DATASETS/spoken-bnc2014/spoken/tagged/`
  (motif `*.xml`), 845 Mo sur disque pour tout `DATASETS/spoken-bnc2014/`.
- Format confirmé (extrait réel, `S23A-tgd.xml`) :
  ```xml
  <text id="S23A">
  <u n="1" who="S0094" trans="nonoverlap" whoConfidence="high">
  <w pos="NN2" lemma="word" class="SUBST" usas="Q3">words</w>
  </u>
  <u n="2" ...>
  <w pos="PPH1" lemma="it" class="PRON" usas="Z8">it</w>
  <w pos="VBZ" lemma="be" class="VERB" usas="A3">'s</w>
  ...
  </u>
  ```
  Le lemme est **déjà fourni** dans l'attribut `lemma=` (pas besoin de
  relemmatiser). L'attribut `class=` porte une classification POS
  **simplifiée** (11 valeurs mesurées sur un échantillon de 20 fichiers :
  `VERB, PRON, ADV, SUBST, ADJ, PREP, INTERJ, CONJ, ART, STOP, UNC`),
  plus proche du vocabulaire de `cefrj.csv` que le tagset CLAWS7 complet
  de l'attribut `pos=` (ex. `NN2`, `PPH1`, `VBZ`).
- **Parsing complet du corpus mesuré** : 1251/1251 fichiers, **0 erreur
  de parse XML**, **11 422 602 tokens**, **49 987 lemmes distincts**,
  **28,5 secondes** avec `xml.etree.ElementTree` (parcours simple,
  `tree.getroot().iter('w')`, aucun traitement NLP). Un cache sur disque
  n'est donc même pas strictement indispensable en termes de temps, mais
  reste recommandé pour ne pas re-parser à chaque lancement du script de
  test.
- Table `class=` → catégories `cefrj.csv` observée dans le sample (à
  affiner si besoin, mapping direct plausible) :
  `SUBST→noun, VERB→verb, ADJ→adjective, ADV→adverb, PRON→pronoun,
  PREP→preposition, ART→determiner, CONJ→conjunction, INTERJ→interjection`.
  `STOP` (ponctuation) et `UNC` (incertain/illisible) sont à exclure du
  comptage.

## Formule Zipf-like

Même échelle que `wordfreq`/SUBTLEX (`FreqZipfUS`) pour rester comparable
aux deux colonnes déjà produites par le script :

```python
import math
zipf_bnc = math.log10(count_par_million + 1) + 3
# count_par_million = occurrences_lemme / total_tokens_corpus * 1_000_000
```

(`total_tokens_corpus` mesuré ci-dessus : 11 422 602 — recalculer si le
corpus change, ne pas figer cette constante en dur sans le recalculer.)

## Plan d'implémentation (dans `filter_book_vocab.py`)

1. **Nouveau bloc de constantes**, à côté de `CEFR_PATH` :
   ```python
   BNC_TAGGED_DIR = ROOT / "DATASETS" / "spoken-bnc2014" / "spoken" / "tagged"
   BNC_CACHE_PATH = ROOT / "REVIEW_FIX_PIPELINE" / "filter_tests" / "bnc_spoken_zipf_cache.csv"
   ```

2. **Fonction `load_bnc_spoken_zipf()`** : si `BNC_CACHE_PATH` existe,
   charger directement (dict `lemme -> zipf_bnc`, format CSV simple
   `lemma,zipf_bnc`). Sinon : parcourir `BNC_TAGGED_DIR.glob("*.xml")`
   avec `xml.etree.ElementTree`, compter les lemmes (`w.get("lemma")`,
   casefold, exclure `class in {"STOP", "UNC"}` ou `class is None`),
   calculer le Zipf-like avec la formule ci-dessus, écrire le cache CSV,
   retourner le dict. Suivre le style déjà utilisé dans le script pour
   `load_prevalence`/`load_aoa`/`load_cefr` (mêmes conventions
   d'encodage, mêmes conventions de nommage).

3. **Nouvelle colonne** `zipf_bnc_spoken` dans le tuple `vocabulary` et
   dans l'écriture CSV (`writer.writerow([...])`, ligne ~421-428) — mot
   absent du corpus BNC → chaîne vide, pas 0.0 (qui serait faussement
   interprété comme "extrêmement rare" plutôt que "non mesuré").

4. **Deux options d'usage, à trancher selon ce qu'on veut observer** (pas
   décidé dans cette session, à choisir en fonction du besoin) :
   - **Option A (informative seulement)** : juste une colonne de plus
     dans le CSV de sortie, comme `zipf_freqzipfus` l'est déjà pour
     `build_vocab_filtered.py` — sert à comparer visuellement les 3
     sources de fréquence sans changer le filtrage.
   - **Option B (second critère de repêchage CEFR)** : élargir la
     condition ligne ~379 (`if is_basic_only and not (zipf < ZIPF_RESCUE_THRESHOLD): continue`)
     pour repêcher aussi un mot A1/A2 dont `zipf_bnc_spoken` dépasse un
     seuil à définir (mot réellement fréquent en conversation orale,
     même si basique) — nécessite de choisir une valeur de seuil
     (pas de valeur "canonique" connue, à calibrer empiriquement,
     contrairement à `4.5` sur `wordfreq` qui vient de `hello-aop-zipf.py`).

5. **Limites à documenter dans le script** (ne pas les découvrir plus
   tard en pensant que c'est un bug) :
   - Corpus **britannique**, alors que `word-prevalence.txt` (Brysbaert)
     et `kuperman-aoa.csv` (Kuperman, Mechanical Turk) sont très
     majoritairement calibrés sur de l'anglais **américain** — même
     divergence régionale que celle déjà mesurée entre `wordfreq` et
     `FreqZipfUS` (écart absolu moyen 0,31 point sur le périmètre filtré
     de cette session — voir les échanges qui ont précédé ce fichier,
     non reproduits ici).
   - Corpus plus petit (11,4M tokens) que SUBTLEX-US → données creuses
     pour les lemmes rares, même caveat que `MIN_NOBS` sur `Pknown`
     (`pipeline/config.py`) : envisager un seuil minimal d'occurrences
     avant de faire confiance à un `zipf_bnc_spoken` mesuré sur très peu
     d'exemples.

## Estimation d'effort (mesurée au moment de l'écriture de ce fichier)

Loader + cache : ~30-45 min. Branchement colonne + option A ou B dans le
script : ~15-30 min. Premier run de constitution du cache : ~30s (mesuré,
voir ci-dessus). Vérification : ~15-30 min. Total : une session de
travail, pas un chantier pluri-session.
