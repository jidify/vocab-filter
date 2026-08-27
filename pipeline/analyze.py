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
from spacy.symbols import ORTH

from pipeline import config, custom_lexicon
from pipeline.corpus import Segment, load_segments

_NLP = None

# Sans ça, spaCy coupe "e-mail" en 3 tokens ("e" / "-" ponctuation / "mail")
# : le "e" (trop court) disparaît et "mail" seul se fait désambiguïser vers
# mail.v.01 "envoyer par la poste" au lieu du sens email — mesuré sur The
# Humans (4 occurrences verbales mal rattachées, voir le plan du 2026-08-27
# "Correction manuelle smart-ass / e-mail sans re-run complet", qui corrige
# ce livre sans rejouer S1-S5 via data/manual_corrections.jsonl ; ces cas
# spéciaux ne servent qu'aux PROCHAINS livres). Vérifié empiriquement :
# gardé en un seul token, spaCy lui assigne tout seul le lemme "e-mail",
# que WordNet reconnaît nativement (lemme alternatif du synset
# electronic_mail.n.01 / e-mail.v.01) — pas besoin de forcer le lemme ici.
EMAIL_SPECIAL_CASES = ["e-mail", "e-mails", "e-mailing", "e-mailed"]


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
        # EMAIL_SPECIAL_CASES (socle en dur) + data/custom_lexicon.jsonl
        # (ajouté sans édition de code depuis pipeline/review_ui.py).
        for surface in EMAIL_SPECIAL_CASES + custom_lexicon.load_tokenizer_surfaces():
            _NLP.tokenizer.add_special_case(surface, [{ORTH: surface}])
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
