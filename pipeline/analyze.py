"""S1 — Analyse linguistique et indexation des occurrences.

Traite le texte segment par segment (pas le livre entier en un seul
`nlp()` — voir la remarque sur `test_idiomatch_book.py` dans le plan :
pousser 140 000 caractères dans un seul appel casse l'attribution à la
phrase/segment). Produit `occurrences.jsonl` : un token de contenu par
ligne, avec surface, lemme, POS, dépendance, tête, position dans le
segment.
"""

from __future__ import annotations

import json

import spacy

from pipeline import config
from pipeline.corpus import Segment, load_segments

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def analyze_segments(segments: list[Segment]):
    nlp = get_nlp()

    # nlp.pipe traite chaque segment séparément (un Doc par segment) —
    # c'est le point qui diffère de test_idiomatch_book.py.
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]
    texts = (s.en for s in play_segments)

    for seg, doc in zip(play_segments, nlp.pipe(texts, batch_size=64)):
        for token in doc:
            if token.is_space or token.is_punct:
                continue

            wn_pos = config.UPOS_TO_WN.get(token.pos_)

            yield {
                "segment_idx": seg.idx,
                "kind": seg.kind,
                "speaker": seg.speaker,
                "token_i": token.i,
                "surface": token.text,
                "lemma": token.lemma_.lower(),
                "upos": token.pos_,
                "wn_pos": wn_pos,
                "tag": token.tag_,
                "dep": token.dep_,
                "head_i": token.head.i,
                "head_lemma": token.head.lemma_.lower(),
                "is_alpha": token.is_alpha,
                "is_stop": token.is_stop,
                "start_char": token.idx,
                "end_char": token.idx + len(token.text),
            }


def run() -> int:
    config.ensure_out_dir()
    segments = load_segments()

    n = 0
    with config.OCCURRENCES_PATH.open("w", encoding="utf-8") as f:
        for occ in analyze_segments(segments):
            f.write(json.dumps(occ, ensure_ascii=False) + "\n")
            n += 1

    print(f"{n} occurrences écrites dans {config.OCCURRENCES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
