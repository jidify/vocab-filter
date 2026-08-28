"""Probe hors plan (question posée après Phase 3/6, pas une phase numérotée
du plan) : un fix de TOKENIZER spaCy, pas un nettoyage du texte source.

Constat (voir la conversation qui a mené ici) : `en_core_web_sm` échoue à
segmenter un tiret cadratin/demi-cadratin quand il est collé SANS ESPACE à
une ponctuation fermante précédente (``?``, ``!``, ``.``, ``,``, ``;``,
``:``, guillemets, parenthèse/crochet fermants) — la règle d'infixe par
défaut de spaCy n'exige un tiret qu'ENTRE deux caractères alphabétiques.
Exemple mesuré : ``"around—we"`` se scinde correctement (``around``, ``—``,
``we``), ``"around?—we"`` reste un seul token ``'around?—we'`` — c'est
exactement la famille de piège ``edge_type=dialogue_dash`` du corpus gold
(``fix_pipeline/gold_corpus/the_humans_gold_v0.jsonl``, idx102/idx1277).

Scan sur les 2535 segments du livre (hors_oeuvre exclu) : 30 tokens
suspects sur 30327 (~0,1%), tous cette même famille — voir
``scan_suspect_tokens`` ci-dessous, un diagnostic générique (pas calibré
sur *The Humans* : il détecte la SIGNATURE du bug — ponctuation
hétérogène ≥2 caractères collée à de l'alphanumérique des deux côtés —
pas une liste de caractères figée).

``patch_dash_after_punctuation`` AJOUTE deux motifs d'infixe (jamais de
retrait) à un `Tokenizer` spaCy déjà chargé — donc n'importe quel texte,
offsets caractère intacts (contrairement à une réécriture du texte source,
qui casserait l'alignement avec le corpus gold). Vérifié : 29/30 corrigés
sur le livre entier, zéro régression sur les traits d'union légitimes
(``ground-floor``, ``e-mail``, ``and/or``, ``smart-ass``...). Le seul
résidu (``"that—'member"``, une élision par apostrophe) n'est pas chassé
— trop ambigu à généraliser sans risquer de casser des contractions
légitimes ailleurs.

**Ceci ne modifie AUCUN fichier de production** (`pipeline/analyze.py`
n'est jamais importé en écriture ici) : le patch s'applique au `Tokenizer`
d'un `nlp` déjà obtenu via `pipeline.analyze.get_nlp()`, en mémoire, pour
la durée du script appelant seulement.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from spacy.util import compile_infix_regex

DASH_AFTER_CLOSING_PUNCT = r"(?<=[?!.,;:'\")\]’”])(?:-|–|—|--|---)(?=\w)"
DASH_BEFORE_OPENING_PUNCT = r"(?<=\w)(?:-|–|—|--|---)(?=[\[\(‘“])"

# Signature générique du bug (pas calibrée sur les caractères de ce livre) :
# un token dont le texte mélange une suite de 2+ signes de ponctuation
# hétérogènes ET des caractères alphanumériques des deux côtés.
SUSPECT_TOKEN_RE = re.compile(r"\w[^\w\s]{2,}\w", re.UNICODE)


def patch_dash_after_punctuation(nlp) -> None:
    """Mute en place le `Tokenizer` de `nlp` : ajoute les deux motifs
    d'infixe ci-dessus aux motifs par défaut du modèle (jamais de retrait).
    Idempotent (peut être appelé plusieurs fois sans effet cumulatif) : les
    motifs sont réappliqués sur `nlp.Defaults.infixes`, pas sur l'état déjà
    patché du tokenizer."""
    new_infixes = list(nlp.Defaults.infixes) + [
        DASH_AFTER_CLOSING_PUNCT,
        DASH_BEFORE_OPENING_PUNCT,
    ]
    nlp.tokenizer.infix_finditer = compile_infix_regex(new_infixes).finditer


def scan_suspect_tokens(nlp, texts: Iterable[str]) -> Counter:
    """Diagnostic générique et réutilisable sur n'importe quel texte :
    fait tourner `nlp.pipe` et compte les tokens dont le texte matche
    `SUSPECT_TOKEN_RE` — le signal qui a permis de détecter le bug sur
    *The Humans* sans connaître sa forme exacte à l'avance."""
    hits: Counter = Counter()
    for doc in nlp.pipe(texts, batch_size=64):
        for tok in doc:
            if SUSPECT_TOKEN_RE.search(tok.text):
                hits[tok.text] += 1
    return hits
