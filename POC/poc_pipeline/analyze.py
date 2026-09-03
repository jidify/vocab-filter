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
from functools import lru_cache

from poc_pipeline import atomic, config, custom_lexicon, multi_token, rules_plus, zones
from poc_pipeline.corpus import Segment, load_segments
from poc_pipeline.vpc import adapter as vpc_adapter
from poc_pipeline.vpc import service as vpc_service
from nltk.corpus import wordnet as nwn

_NLP = None

# Version du contrat d'analyse morphosyntaxique stocke dans chaque occurrence.
# Elle est volontairement independante de la version du schema JSONL : toute
# modification des hypotheses produites doit invalider les digests aval.
ANALYSIS_VERSION = "s1-morphosyntax-v1"


@lru_cache(maxsize=8192)
def _wordnet_sense_ids(lemma: str, wn_pos: str | None) -> tuple[str, ...]:
    return tuple(sorted({s.name() for s in nwn.synsets(lemma, pos=wn_pos)})) if wn_pos else ()


def _candidate(lemma: str, upos: str, wn_pos: str | None, source: str) -> dict:
    normalized = lemma.casefold()
    sense_ids = list(_wordnet_sense_ids(normalized, wn_pos))
    return {"lemma": normalized, "upos": upos, "wn_pos": wn_pos, "source": source,
            "wordnet_sense_ids": sense_ids}


@lru_cache(maxsize=1)
def _deprefix_verbs_by_radical() -> dict[str, tuple[str, ...]]:
    found: dict[str, set[str]] = {}
    for synset in nwn.all_synsets(pos=nwn.VERB):
        for item in synset.lemmas():
            name = item.name().casefold()
            if name.startswith("de_") and len(name) > 3:
                found.setdefault(name[3:], set()).add(name)
            elif name.startswith("de") and len(name) > 2:
                found.setdefault(name[2:], set()).add(name)
    return {radical: tuple(sorted(names)) for radical, names in found.items()}


def morphosyntactic_analysis(token) -> dict:
    """Construit une analyse principale et des hypotheses morphologiques.

    Les hypotheses sont des ouvertures d'inventaire, pas des decisions de
    sens. Elles reposent sur des alternances productives et, quand possible,
    sur l'inventaire WordNet local ; aucun contexte ni resultat du benchmark
    n'est encode ici. S5 reste donc strictement inchange.
    """

    surface = token.text.casefold()
    lemma = (token.lemma_ or token.text).casefold()
    upos = token.pos_ or "X"
    wn_pos = config.UPOS_TO_WN.get(upos)
    tag = token.tag_ or ""
    alternatives: list[dict] = []

    def add(candidate_lemma: str, candidate_upos: str, candidate_pos: str | None, source: str):
        candidate = _candidate(candidate_lemma, candidate_upos, candidate_pos, source)
        if (candidate["lemma"], candidate["wn_pos"]) == (lemma, wn_pos):
            return
        if not any((a["lemma"], a["wn_pos"]) == (candidate["lemma"], candidate["wn_pos"])
                   for a in alternatives):
            alternatives.append(candidate)

    # Un VBG peut etre verbal, adjectival ("creeping horror") ou un nom
    # lexicalise ("frosting"). Garder les trois lectures ne les departage pas.
    if tag == "VBG" or surface.endswith("ing"):
        add(surface, "ADJ", "a", "present_participle_adjective")
        if nwn.synsets(surface, pos=nwn.NOUN):
            add(surface, "NOUN", "n", "lexicalized_gerund_noun")

    # Certains pluriels ont une entree lexicale propre (p. ex. un pluriel
    # tantum). Le lemme spaCy singulier reste bien l'analyse principale.
    if tag in {"NNS", "NNPS"} or (surface.endswith("s") and surface != lemma):
        add(surface, "NOUN", "n", "surface_plural_lexeme")

    # Ouvre les variantes derivationnelles WordNet dont le radical est le
    # lemme principal (de-stress/de_stress, etc.). C'est une regle generale
    # d'inventaire, bornee aux verbes effectivement attestes dans WordNet.
    if wn_pos == "v":
        for name in _deprefix_verbs_by_radical().get(lemma, ()):
            add(name, "VERB", "v", "wordnet_deprefix_derivation")

    # Une forme non flechie peut etre homographe entre categories (bitch,
    # stress...). On conserve seulement les POS attestes dans WordNet.
    for candidate_upos, candidate_pos in (("NOUN", "n"), ("VERB", "v"),
                                           ("ADJ", "a"), ("ADV", "r")):
        if nwn.synsets(surface, pos=candidate_pos):
            add(surface, candidate_upos, candidate_pos, "wordnet_surface_homograph")

    return {
        "version": ANALYSIS_VERSION,
        "primary": _candidate(lemma, upos, wn_pos, "spacy"),
        "alternatives": alternatives,
        "morphology": {
            "is_inflected": surface != lemma,
            "is_participle": tag in {"VBG", "VBN"},
            "is_plural": tag in {"NNS", "NNPS"},
            "is_nominalization_candidate": any(a["wn_pos"] == "n" for a in alternatives),
        },
    }


def validate_occurrence_schema(occurrences: list[dict], segments: list[Segment]) -> None:
    """Bloque l'ecriture si un span ou le nouveau contrat S1 est invalide."""
    source_by_idx = {segment.idx: segment.en for segment in segments}
    for occ in occurrences:
        source = source_by_idx[occ["segment_idx"]]
        if source[occ["start_char"]:occ["end_char"]] != occ["surface"]:
            raise ValueError(f"offsets invalides pour {occ['occurrence_id']}")
        analysis = occ.get("analysis")
        if occ.get("analysis_version") != ANALYSIS_VERSION or not isinstance(analysis, dict):
            raise ValueError(f"schema d'analyse invalide pour {occ['occurrence_id']}")
        if analysis.get("version") != ANALYSIS_VERSION:
            raise ValueError(f"version d'analyse invalide pour {occ['occurrence_id']}")
        primary = analysis.get("primary", {})
        if (primary.get("lemma"), primary.get("upos"), primary.get("wn_pos")) != (
                occ["lemma"], occ["upos"], occ["wn_pos"]):
            raise ValueError(f"analyse principale incoherente pour {occ['occurrence_id']}")
        if not isinstance(analysis.get("alternatives"), list):
            raise ValueError(f"alternatives invalides pour {occ['occurrence_id']}")

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


@lru_cache(maxsize=1)
def _rules_plus_custom_idiom_sequences():
    """Q0-3 Phase 6 — import différé de ``pipeline.mwe.CUSTOM_IDIOMS`` (pas
    au niveau module : ``pipeline.rules_plus`` ne doit jamais dépendre de
    ``pipeline.mwe`` pour éviter un import circulaire, voir sa docstring —
    c'est ``analyze.py`` qui fait le pont, dans le sens S2 -> S1 dont
    dépend déjà indirectement ce module ailleurs)."""
    from poc_pipeline.mwe import CUSTOM_IDIOMS

    return rules_plus.custom_idiom_sequences(CUSTOM_IDIOMS + custom_lexicon.load_idioms())


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


def analyze_segments(play_segments: list[Segment], vpc_sink: list[dict],
                     zone_sink: list[int], multi_token_sink: list[dict] | None = None,
                     rules_plus_sink: list[dict] | None = None):
    multi_token_sink = multi_token_sink if multi_token_sink is not None else []
    rules_plus_sink = rules_plus_sink if rules_plus_sink is not None else []
    nlp = get_nlp()
    # Lot 2 — le détecteur VPC tourne DANS cette même boucle nlp.pipe, sur le
    # Doc déjà annoté (pas de second appel spaCy séparé — voir le plan,
    # point 10). `vpc_sink` accumule les décisions (rejets compris) au fil du
    # parcours ; `run()` les écrit une fois ce générateur épuisé. Lot 5 :
    # `zone_sink` accumule, dans le même esprit et pour la même raison
    # (jamais un second passage spaCy), le segment_idx de CHAQUE token
    # non-espace rencontré — l'unité du layout de zones (point I).
    detector = vpc_service.build_detector()
    nlp_model = f"en_core_web_sm:{nlp.meta.get('version', 'unknown')}"

    # Q0-3 Phase 6 — mêmes principes que VPC ci-dessus : les scanners
    # "rules_plus" Groupe B (phrasal verb PARSEME+WordNet, rejeu du
    # lexique custom, composés nominaux WordNet — voir
    # pipeline/rules_plus.py) tournent DANS cette même boucle, sur le
    # même `doc` déjà annoté, jamais un second passage spaCy séparé.
    # Groupe A (trait d'union/possessif) est géré à l'intérieur de
    # `multi_token.detect()` lui-même, pas ici.
    rules_plus_pv_lexicon = rules_plus.merged_phrasal_verb_lexicon()
    rules_plus_nominal_lexicon = rules_plus.wordnet_nominal_lexicon()
    rules_plus_custom_sequences = _rules_plus_custom_idiom_sequences()

    # nlp.pipe traite chaque segment séparément (un Doc par segment) —
    # c'est le point qui diffère de test_idiomatch_book.py.
    texts = (s.en for s in play_segments)

    for seg, doc in zip(play_segments, nlp.pipe(texts, batch_size=64)):
        segment_multi_tokens = multi_token.detect(doc, seg.idx)
        multi_token_sink.extend(segment_multi_tokens)

        rules_plus_sink.extend(
            rules_plus.scan_phrasal_verb_candidates(doc, seg.idx, seg.kind, rules_plus_pv_lexicon)
        )
        rules_plus_sink.extend(
            rules_plus.scan_custom_idiom_candidates(doc, seg.idx, seg.kind, rules_plus_custom_sequences)
        )
        rules_plus_sink.extend(
            rules_plus.scan_wordnet_nominal_candidates(doc, seg.idx, seg.kind, rules_plus_nominal_lexicon)
        )
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
            if token.is_space:
                continue

            # Lot 5 (point I) : TOUS les tokens non-espace, ponctuation
            # comprise — l'unité du layout de zones. Les occurrences
            # lexicales ci-dessous continuent d'exclure la ponctuation
            # après ce point, comme avant ce lot.
            zone_sink.append(seg.idx)
            if token.is_punct:
                continue

            wn_pos = config.UPOS_TO_WN.get(token.pos_)
            analysis = morphosyntactic_analysis(token)

            occurrence = {
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
                "analysis_version": ANALYSIS_VERSION,
                "analysis": analysis,
            }
            # Projection informative pour les consommateurs aval. Ce champ
            # n'est consulté ni par is_covered ni par une porte de suppression.
            occurrence["multi_token_candidates"] = [
                {"candidate_id": row["candidate_id"], "surface": row["surface"],
                 "start_char": row["start_char"], "end_char": row["end_char"],
                 "candidate_types": row["candidate_types"], "score": row["score"],
                 "provenance": row["provenance"]}
                for row in multi_token.covering(occurrence, segment_multi_tokens)
            ]
            yield occurrence


def run() -> int:
    config.ensure_out_dir()
    segments = load_segments()
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]

    vpc_candidates: list[dict] = []
    multi_token_candidates: list[dict] = []
    zone_token_segments: list[int] = []
    rules_plus_candidates: list[dict] = []
    # Matérialisé (pas streamé directement dans atomic_write_jsonl comme
    # avant ce lot) : le layout de zones (ci-dessous) a besoin d'avoir vu
    # TOUS les tokens du livre avant de pouvoir assigner un zone_id à quoi
    # que ce soit — donc le générateur doit être épuisé d'abord.
    occurrences = list(analyze_segments(
        play_segments, vpc_candidates, zone_token_segments, multi_token_candidates,
        rules_plus_candidates,
    ))
    validate_occurrence_schema(occurrences, play_segments)
    multi_token.validate(multi_token_candidates, {s.idx: s.en for s in play_segments})

    # Lot 5 — layout de zones (plan Partie 2 point H/I, Partie 4 Lot 5) :
    # toujours sur le livre entier, jamais par tranche (Partie 3). Recalculé
    # à chaque run, pas mis en cache — l'algorithme est rapide (un simple
    # découpage d'une liste de segment_idx déjà en mémoire).
    source_text = "\n".join(s.en for s in play_segments)
    layout = zones.build_layout(zone_token_segments, source_text, config.ZONE_PERCENT)
    zones.write(layout)
    seg_zone = zones.segment_zone_map(layout)
    for occ in occurrences:
        occ["zone_id"] = seg_zone.get(occ["segment_idx"])

    n = atomic.atomic_write_jsonl(config.OCCURRENCES_PATH, occurrences)
    n_multi = atomic.atomic_write_jsonl(
        config.MULTI_TOKEN_CANDIDATES_PATH, multi_token_candidates
    )
    n_vpc = atomic.atomic_write_jsonl(config.VPC_CANDIDATES_PATH, vpc_candidates)
    n_rules_plus = atomic.atomic_write_jsonl(
        config.RULES_PLUS_CANDIDATES_PATH, rules_plus_candidates
    )

    print(f"{n} occurrences écrites dans {config.OCCURRENCES_PATH}")
    print(f"{n_multi} hypothèses multi-tokens écrites dans "
          f"{config.MULTI_TOKEN_CANDIDATES_PATH}")
    print(f"{n_vpc} candidats VPC (rejets compris) écrits dans {config.VPC_CANDIDATES_PATH}")
    print(f"{n_rules_plus} candidats rules_plus écrits dans {config.RULES_PLUS_CANDIDATES_PATH}")
    print(f"{layout['zone_count']} zones ({config.ZONE_PERCENT}%) -> {config.ZONE_LAYOUT_PATH} "
          f"(layout_id={layout['layout_id'][:19]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
