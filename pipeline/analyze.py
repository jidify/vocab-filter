"""S1 — Analyse linguistique et indexation des occurrences.

Traite le texte segment par segment (pas le livre entier en un seul
`nlp()` — voir la remarque sur `test_idiomatch_book.py` dans le plan :
pousser 140 000 caractères dans un seul appel casse l'attribution à la
phrase/segment). Produit `occurrences.jsonl` : un token de contenu par
ligne, avec surface, lemme, POS, dépendance, tête, position dans le
segment.
"""

from __future__ import annotations

import spacy
from spacy.symbols import ORTH

from pipeline import atomic, config, custom_lexicon
from pipeline.corpus import Segment, load_segments
from pipeline.vpc import adapter as vpc_adapter
from pipeline.vpc import service as vpc_service

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


_CHAR_SPAN_PAIR_FIELDS = ("verb_char_span",)
_CHAR_SPAN_LIST_FIELDS = (
    "particle_char_spans", "token_char_spans",
    "normalized_char_spans", "original_char_spans",
)


def _offset_char_spans(record: dict, offset: int) -> None:
    """Convertit en place les offsets de `record` (un `PhrasalVerbDetection`
    sérialisé) de "relatif à la phrase" à "absolu dans le segment" — voir le
    commentaire au point d'appel."""

    for field in _CHAR_SPAN_PAIR_FIELDS:
        a, b = record[field]
        record[field] = [a + offset, b + offset]
    for field in _CHAR_SPAN_LIST_FIELDS:
        spans = record.get(field)
        if spans is None:
            continue
        record[field] = [[a + offset, b + offset] for a, b in spans]


def analyze_segments(segments: list[Segment], vpc_sink: list[dict]):
    nlp = get_nlp()
    # Lot 2 — le détecteur VPC tourne DANS cette même boucle nlp.pipe, sur le
    # Doc déjà annoté (pas de second appel spaCy séparé — voir le plan,
    # point 10). `vpc_sink` accumule les décisions (rejets compris) au fil du
    # parcours ; `run()` les écrit une fois ce générateur épuisé.
    detector = vpc_service.build_detector()
    nlp_model = f"en_core_web_sm:{nlp.meta.get('version', 'unknown')}"

    # nlp.pipe traite chaque segment séparément (un Doc par segment) —
    # c'est le point qui diffère de test_idiomatch_book.py.
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]
    texts = (s.en for s in play_segments)

    for seg, doc in zip(play_segments, nlp.pipe(texts, batch_size=64)):
        # `pipeline.vpc.adapter.sentences_from_doc` rend les offsets de
        # chaque SyntaxToken relatifs au DÉBUT DE LA PHRASE (`token.idx -
        # sentence_start`, voir sa docstring) — un choix du contrat
        # SyntaxSentence/SyntaxToken hérité du projet source, où une phrase
        # peut être analysée hors de tout document. `mwe.py` (Lot 3) doit
        # fusionner les candidats VPC avec idiomatch SUR LES SPANS DE
        # CARACTÈRES, qui eux sont absolus dans le SEGMENT (occurrences.jsonl,
        # mwe_candidates.jsonl) — donc on convertit ici, une fois, avant
        # d'écrire vpc_candidates.jsonl : record["*_char_span(s)"] passent de
        # "relatif à la phrase" à "absolu dans le segment", pour que toute
        # comparaison de spans en aval soit dans le même référentiel.
        sentence_starts = {
            f"seg{seg.idx}:s{i}": s.start_char for i, s in enumerate(doc.sents)
        }
        for sentence in vpc_adapter.sentences_from_doc(
            doc,
            sentence_id_prefix=f"seg{seg.idx}",
            nlp_engine_version=spacy.__version__,
            nlp_model=nlp_model,
        ):
            sentence_start = sentence_starts[sentence.sentence_id]
            for detection in detector.analyze(sentence):
                record = detection.model_dump(mode="json")
                record["segment_idx"] = seg.idx
                _offset_char_spans(record, sentence_start)
                vpc_sink.append(record)

        for token in doc:
            if token.is_space or token.is_punct:
                continue

            wn_pos = config.UPOS_TO_WN.get(token.pos_)

            yield {
                # Lot 0 — identité stable d'occurrence (plan Partie 2,
                # point F) : remplace la clé fragile (lemma, pos,
                # segment_idx, surface) utilisée jusqu'ici pour dédupliquer
                # senses.jsonl. "w:" = word, distingue des occurrence_id
                # "m:" des expressions multi-mots (voir pipeline/mwe.py).
                "occurrence_id": f"w:{seg.idx}:{token.i}",
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

    vpc_candidates: list[dict] = []
    n = atomic.atomic_write_jsonl(
        config.OCCURRENCES_PATH, analyze_segments(segments, vpc_candidates)
    )
    n_vpc = atomic.atomic_write_jsonl(config.VPC_CANDIDATES_PATH, vpc_candidates)

    print(f"{n} occurrences écrites dans {config.OCCURRENCES_PATH}")
    print(f"{n_vpc} candidats VPC (rejets compris) écrits dans {config.VPC_CANDIDATES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
