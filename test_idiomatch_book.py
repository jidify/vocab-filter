import unittest

if __name__ != "__main__":
    raise unittest.SkipTest("manual whole-book exploration; run this file directly")

from pathlib import Path
from collections import Counter

from pipeline.mwe import get_matcher


# ============================================================
# CONFIG
# ============================================================

BOOK_PATH = Path(
    "The Humans - Stephen Karam.txt"
)

MAX_CONTEXT_CHARS = 180


# ============================================================
# CHARGEMENT DU LIVRE
# ============================================================

text = BOOK_PATH.read_text(
    encoding="utf-8"
)

print(
    f"Texte chargé : {len(text):,} caractères"
)

print()


# ============================================================
# CHARGEMENT IDIOMATCH
#
# Peut prendre un certain temps au premier lancement,
# car les patterns sont construits.
# ============================================================

print("Chargement de Idiomatcher...")

idiomatcher = get_matcher()

print("Idiomatcher chargé.")
print()


# ============================================================
# TRAITEMENT SPACY
#
# IMPORTANT :
# on utilise bien le modèle NLP fourni par idiomatch.
# ============================================================

doc = idiomatcher.nlp(
    text
)


# ============================================================
# DÉTECTION
# ============================================================

matches = idiomatcher(
    doc
)

print(
    f"Occurrences détectées : "
    f"{len(matches)}"
)

print()


# ============================================================
# COMPTER LES IDIOMES
# ============================================================

idiom_counts = Counter(
    match["idiom"]
    for match in matches
)

print("=" * 100)
print("IDIOMES DÉTECTÉS")
print("=" * 100)
print()

for idiom, count in (
    idiom_counts.most_common()
):

    print(
        f"{count:>4}  {idiom}"
    )


# ============================================================
# AFFICHER LES OCCURRENCES AVEC CONTEXTE
# ============================================================

print()
print()
print("=" * 100)
print("OCCURRENCES AVEC CONTEXTE")
print("=" * 100)


for i, match in enumerate(
    matches,
    start=1
):

    idiom = match["idiom"]
    span_text = match["span"]

    meta = match.get(
        "meta"
    )

    print()
    print(
        f"[{i}] {idiom}"
    )

    print(
        f"Match : {span_text}"
    )


    # --------------------------------------------------------
    # meta est documenté sous forme :
    # (match_id, start_token, end_token)
    # --------------------------------------------------------

    if (
        meta
        and len(meta) >= 3
    ):

        start_token = meta[1]
        end_token = meta[2]

        matched_span = doc[
            start_token:end_token
        ]

        start_char = (
            matched_span.start_char
        )

        end_char = (
            matched_span.end_char
        )

        context_start = max(
            0,
            start_char - MAX_CONTEXT_CHARS
        )

        context_end = min(
            len(text),
            end_char + MAX_CONTEXT_CHARS
        )

        context = text[
            context_start:
            context_end
        ]

        context = (
            context
            .replace("\n", " ")
            .replace("\r", " ")
        )

        context = " ".join(
            context.split()
        )

        print(
            f"Contexte : ...{context}..."
        )


# ============================================================
# RÉSUMÉ
# ============================================================

print()
print()
print("=" * 100)
print("RÉSUMÉ")
print("=" * 100)

print(
    f"Idiomes différents : "
    f"{len(idiom_counts)}"
)

print(
    f"Occurrences totales : "
    f"{len(matches)}"
)
