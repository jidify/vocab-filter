"""S2 — Détecter les expressions avant les mots simples.

`idiomatch` reste le générateur de candidats à haut rappel (§10 du
résumé), réglé à `n=2` (repris de la conclusion pratique §10.9). Mais
son inventaire n'a AUCUN marqueur "idiomatique" vs "littéral/composable"
exploitable automatiquement (vérifié : "know someone", "go to",
"talk about" ont des définitions dans idioms.yml au même titre que
"figure out" ou "wing it" — la différence n'est pas dans les données,
elle doit être jugée). Le pré-filtre ici ne fait donc QUE ce qui est
structurellement certain ; la décision idiome/littéral/compositionnel
revient entièrement à mwe_judge.py (S3), comme le veut §3.3 de
proposition_1.
"""

from __future__ import annotations

import json
import builtins
from collections import Counter, defaultdict

from idiomatch import Idiomatcher

from poc_pipeline import atomic, config, custom_lexicon, mwe_alignment, mwe_gates
from poc_pipeline.corpus import load_segments
from poc_pipeline.tokenizer_setup import configure_tokenizer

_MATCHER = None
_IDIOMS_YML: dict[str, dict] | None = None

# Idiomes absents de la base idiomatch/Wiktionary mais identifiés dans
# vocab-filter-resume.md §10.4-10.6 comme manquants et ajoutés
# dynamiquement via add_idioms() (schéma de test_crack_open.py:10-30).
# "crack open" est confirmé présent dans le texte : "cracks the
# bathroom door open" et "cracks open another beer".
#
# "smart ass" : absent d'idiomatch (vérifié), et le tiret de "smart-ass"
# est tokenisé comme ponctuation séparée par spaCy — sans cette entrée,
# S2 ne détecte jamais le composé et S5 désambiguïse "ass" tout seul
# (a fini sur ass.n.02 "un crétin" — voir le plan du 2026-08-27
# "Correction manuelle smart-ass / e-mail sans re-run complet", qui
# corrige *The Humans* sans rejouer S1-S5 via data/manual_corrections.jsonl ;
# cette entrée-ci ne sert qu'aux PROCHAINS livres). Vérifié empiriquement :
# add_idioms() avec ce lemme matche bien "smart-ass" malgré le tiret
# interposé comme token de ponctuation (span "smart - ass", slop=2).
CUSTOM_IDIOMS = [
    {
        "etymology": None,
        "lemma": "crack open",
        "senses": [{"content": "To cause something to open.", "examples": []}],
        "source": "custom",
    },
    {
        "etymology": None,
        "lemma": "smart ass",
        "senses": [{
            "content": "A person who makes clever, sarcastic, or impertinent remarks.",
            "examples": [],
        }],
        "source": "custom",
    },
]


def _all_custom_idioms() -> list[dict]:
    """CUSTOM_IDIOMS (socle en dur ci-dessus) + data/custom_lexicon.jsonl
    (ajouté sans édition de code depuis pipeline/review_ui.py) — lu une
    fois par processus, comme CUSTOM_IDIOMS lui-même. Une entrée du
    lexique avec le même `lemma` qu'une entrée en dur la remplace (ordre
    d'insertion dans le dict de get_idiom_definition/get_matcher)."""
    return CUSTOM_IDIOMS + custom_lexicon.load_idioms()


def get_idiom_senses(idiom: str) -> list[dict]:
    """Toutes les définitions idiomatch/custom, sans privilégier la première.

    L'ordre de la ressource est conservé uniquement pour rendre l'inventaire
    reproductible ; il ne constitue jamais un signal de sélection de sens.
    """

    global _IDIOMS_YML
    if _IDIOMS_YML is None:
        import yaml
        from idiomatch.idiomatcher import RESOURCES_DIR

        data = yaml.safe_load((RESOURCES_DIR / "idioms.yml").read_text(encoding="utf-8"))
        _IDIOMS_YML = {entry["lemma"]: entry for entry in data}
        for entry in _all_custom_idioms():
            _IDIOMS_YML[entry["lemma"]] = entry

    entry = _IDIOMS_YML.get(idiom)
    return [dict(sense) for sense in (entry or {}).get("senses", []) if sense.get("content")]


def get_idiom_definition(idiom: str) -> str | None:
    """Glose anglaise de l'idiome telle qu'écrite dans idioms.yml
    (idiomatch/resources/idioms.yml) — sert directement de "sens" pour
    les unités multi-mots en S5/S6 : ces expressions n'ont
    généralement pas d'entrée WordNet propre, donc GlossBERT/omw-fr
    n'ont rien à désambiguïser pour elles."""

    senses = get_idiom_senses(idiom)
    if len(senses) != 1:
        return None
    # Une glose n'est non ambiguë que lorsque l'inventaire n'en contient
    # réellement qu'une. Les inventaires polysémiques passent par S3-3.
    return next(iter(senses))["content"]


def get_matcher():
    global _MATCHER
    if _MATCHER is None:
        # idiomatch 0.x opens its UTF-8 YAML/JSON resources without an
        # explicit encoding.  On Windows that means cp1252 and crashes on
        # perfectly valid Unicode.  Scope the compatibility shim to the
        # dependency module and to this load only; application files keep
        # using the normal built-in open.
        import idiomatch.idiomatcher as idiomatcher_module

        def open_utf8(path, mode="r", *args, **kwargs):
            if "b" not in mode:
                kwargs.setdefault("encoding", "utf-8")
            return builtins.open(path, mode, *args, **kwargs)

        previous_open = getattr(idiomatcher_module, "open", None)
        idiomatcher_module.open = open_utf8
        try:
            _MATCHER = Idiomatcher.from_pretrained(n=2)
        finally:
            if previous_open is None:
                del idiomatcher_module.open
            else:
                idiomatcher_module.open = previous_open
        # Stage 0 : patch de tiret SEUL sur ce tokenizer (hyphen_whitelist=
        # special_cases=False) — voir tokenizer_setup.py. La liste blanche à
        # tirets figerait des composés (ex. "able-bodied") en un seul token,
        # ce qui casserait les motifs idiomatch construits token par token
        # dessus (voir mwe_alignment.py::patterns_for_idiom). Les cas
        # spéciaux (email/custom_lexicon) sont exclus eux aussi : mesuré sur
        # *The Humans* complet, les admettre ici change le compte de slop
        # d'un match et fait apparaître 1 candidat "to the letter" sur "to
        # e-mail the rec letter" — pas un vrai idiome. Appliqué AVANT tout
        # matching, donc avant repair_corrupt_anchor_lemmas/add_idioms
        # ci-dessous.
        configure_tokenizer(_MATCHER.nlp, hyphen_whitelist=False, special_cases=False)
        # Porte D (fix_pipeline/s2_fix/) : répare les entrées idiomatch dont
        # l'ancre a été compilée sur un lemme spaCy erroné (voir
        # mwe_gates.CORRUPT_ANCHOR_REPAIRS) AVANT d'ajouter les idiomes
        # custom du projet ci-dessous — jamais corrompus par construction,
        # rien à réparer pour eux.
        mwe_gates.repair_corrupt_anchor_lemmas(_MATCHER)
        _MATCHER.add_idioms(_all_custom_idioms())
    return _MATCHER


def find_candidates(segments):
    """Fait tourner idiomatch SEGMENT PAR SEGMENT (pas le livre entier
    en un seul nlp() comme test_idiomatch_book.py:57 — un match ne doit
    jamais pouvoir chevaucher deux segments)."""

    matcher = get_matcher()
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]

    for seg in play_segments:
        doc = matcher.nlp(seg.en)
        matches = matcher(doc)

        for m in matches:
            match_id, start, end = m["meta"]
            span = doc[start:end]

            # Lot 1 — défaut A (plan Partie 2, point D) : dérive les
            # membres RÉELS du span (pipeline/mwe_alignment.py), pas son
            # enveloppe entière. Un candidat "ambigu" (motif introuvable,
            # ou plusieurs alignements distincts possibles) ne reçoit
            # aucun member_char_spans : il reste visible pour audit dans
            # cet artefact, mais mwe_judge.py ne le réservera jamais
            # (abstention, jamais de suppression sur une hypothèse
            # incertaine).
            alignment = mwe_alignment.align_members(m["idiom"], list(span), matcher.nlp, matcher.n)
            if alignment.ambiguous:
                member_char_spans = []
            else:
                member_char_spans = [
                    [span[i].idx, span[i].idx + len(span[i].text)]
                    for i in sorted(alignment.member_indices)
                ]

            # Portes S2 (fix_pipeline/s2_fix/) : `None` si le candidat passe,
            # sinon la famille de rejet — jamais omis, pour que run() puisse
            # partitionner sans perdre trace de ce qui a été écarté et
            # pourquoi (voir MWE_REJECTED_CANDIDATES_PATH).
            rejected_by = mwe_gates.classify(m["idiom"], list(span), matcher.nlp, matcher.n)

            yield {
                # Lot 0 — identité stable (plan Partie 2, point F) : "m:"
                # distingue des occurrence_id "w:" des mots simples
                # (pipeline/analyze.py). Basé sur les offsets caractères,
                # pas les indices de tokens : le Doc d'idiomatch (matcher.nlp
                # ci-dessus) n'est pas le même pipeline spaCy que celui
                # d'analyze.py (pas les mêmes cas spéciaux de tokenizer),
                # donc seuls les offsets caractères sont comparables entre
                # les deux sources — voir la règle de fusion en S3.
                "occurrence_id": f"m:{seg.idx}:{span.start_char}:{span.end_char}",
                "segment_idx": seg.idx,
                "kind": seg.kind,
                "idiom": m["idiom"],
                "surface": span.text,
                "start_token": start,
                "end_token": end,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "n_tokens_span": end - start,
                "n_tokens_lemma": len(m["idiom"].split()),
                "member_char_spans": member_char_spans,
                "ambiguous_alignment": alignment.ambiguous,
                "rejected_by": rejected_by,
                "source": "idiomatch",
                # Lot 3 (point C) : seuls les candidats VPC signalés par le
                # garde-fou directionnel du détecteur portent ce flag à True
                # (voir load_vpc_candidates) — idiomatch n'a pas ce concept,
                # jamais escaladé à l'occurrence pour cette source.
                "directional_context_dependent": False,
            }


def load_vpc_candidates(segments) -> list[dict]:
    """Lit `vpc_candidates.jsonl` (Lot 2, écrit par analyze.py) et projette
    chaque détection VPC non rejetée vers le MÊME schéma que
    `find_candidates` ci-dessus, pour que `structural_prefilter`/
    `group_by_type`/`mwe_judge.py`/`select.py` puissent traiter les deux
    sources uniformément à partir d'ici.

    `member_char_spans` = les spans (verbe + particule(s)) tels
    qu'`analyze.py` les a déjà convertis en absolu-dans-le-segment (voir son
    commentaire sur `_offset_char_spans`) — jamais les indices de tokens du
    Doc idiomatch, incompatibles (pipeline spaCy différent, voir le
    commentaire sur `occurrence_id` dans `find_candidates`).

    Les rejets (`decision == "rejected_syntax"`) restent dans
    `vpc_candidates.jsonl` pour audit (point 5 du plan) mais ne deviennent
    jamais candidats ici — jamais réservables."""

    if not config.VPC_CANDIDATES_PATH.exists():
        return []

    segments_by_idx = {s.idx: s for s in segments}
    candidates = []
    with config.VPC_CANDIDATES_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["decision"] == "rejected_syntax":
                continue

            member_char_spans = sorted(tuple(span) for span in rec["token_char_spans"])
            start_char = min(a for a, _ in member_char_spans)
            end_char = max(b for _, b in member_char_spans)
            seg_idx = rec["segment_idx"]
            seg = segments_by_idx.get(seg_idx)

            candidates.append({
                # Même convention d'identité que find_candidates (plan
                # Partie 2, point F) : c'est PAR CETTE CLÉ, calculée sur les
                # spans de caractères des deux sources indépendamment, que la
                # fusion détecte qu'idiomatch et VPC ont vu la même occurrence
                # (merge_candidate_sources ci-dessous).
                "occurrence_id": f"m:{seg_idx}:{start_char}:{end_char}",
                "segment_idx": seg_idx,
                "kind": seg.kind if seg else None,
                "idiom": rec["normalized_expression"],
                "surface": rec["observed_expression"],
                "start_token": None,
                "end_token": None,
                "start_char": start_char,
                "end_char": end_char,
                "n_tokens_span": len(member_char_spans),
                "n_tokens_lemma": len(rec["normalized_expression"].split()),
                "member_char_spans": [list(s) for s in member_char_spans],
                "ambiguous_alignment": False,
                "source": "vpc",
                # Point 14 du plan : le garde-fou directionnel
                # (_MOTION_VERBS x _DIRECTIONAL_PARTICLES,
                # pipeline/vpc/detectors/phrasal_verbs.py) a signalé cette
                # occurrence comme potentiellement littérale (verbe de
                # mouvement + particule directionnelle) mais l'a acceptée
                # quand même à cause d'un frame PARSEME attesté — ce sont
                # précisément les cas "walk up the stairs" vs "walk up to
                # someone" dépendants du contexte que mwe_judge.py doit
                # trancher occurrence par occurrence, pas au niveau du type.
                "directional_context_dependent": rec["rule_id"].startswith(
                    "dependency_prt_train_frame_override"
                ),
                "vpc_decision": rec["decision"],
                "vpc_decision_reason": rec["decision_reason"],
            })
    return candidates


def load_rules_plus_candidates(segments) -> list[dict]:
    """Lit `rules_plus_candidates.jsonl` (Q0-3 Phase 6, écrit par
    analyze.py dans la même boucle nlp.pipe que VPC — voir
    pipeline/rules_plus.py) : déjà au schéma commun (`find_candidates`/
    `load_vpc_candidates`), aucune projection nécessaire ici.

    Contrairement à `load_vpc_candidates`, aucun filtre de rejet : les
    scanners `rules_plus` ne rejettent jamais rien par construction (union
    avec spaCy sans pouvoir de rejet, voir
    fix_pipeline/detection_benchmark/phase3_rules_plus_report.md)."""

    if not config.RULES_PLUS_CANDIDATES_PATH.exists():
        return []
    with config.RULES_PLUS_CANDIDATES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def merge_candidate_sources(
    idiomatch_candidates: list[dict],
    vpc_candidates: list[dict],
    rules_plus_candidates: list[dict] = (),
) -> list[dict]:
    """Fusionne les sources SUR occurrence_id (donc sur les spans de
    caractères de l'enveloppe candidate — jamais les indices de token, les
    pipelines spaCy internes n'étant pas les mêmes, voir les docstrings
    ci-dessus). Ordre de priorité sur collision (même segment, même
    enveloppe détectée par plusieurs sources), du plus fort au plus
    faible : idiomatch > VPC > rules_plus.

    - idiomatch gagne sur VPC : son alignement de membres a été validé au
      Lot 1 (rejeu déterministe des motifs slop_N), alors que VPC ne
      connaît que verbe+particule.
    - `rules_plus` (Q0-3 Phase 6) ne gagne JAMAIS une collision — voir sa
      propre conclusion ("union avec spaCy sans pouvoir de rejet") : il ne
      comble que les occurrences qu'AUCUNE des deux autres sources n'a
      trouvées.
    - Le flag `directional_context_dependent` de VPC est conservé même
      quand idiomatch gagne — c'est un signal sur le CONTEXTE de
      l'occurrence, indépendant de la source qui a fourni les spans
      exacts. `rules_plus` ne porte jamais ce flag (toujours False), donc
      rien à préserver de sa part sur ce point."""

    by_occurrence: dict[str, dict] = {c["occurrence_id"]: c for c in rules_plus_candidates}
    for c in vpc_candidates:
        by_occurrence[c["occurrence_id"]] = c
    for c in idiomatch_candidates:
        existing = by_occurrence.get(c["occurrence_id"])
        if existing is not None and existing["directional_context_dependent"]:
            c = {**c, "directional_context_dependent": True}
        by_occurrence[c["occurrence_id"]] = c
    return list(by_occurrence.values())


def structural_prefilter(candidates: list[dict]) -> list[dict]:
    """Ne rejette QUE ce qui est certain sans jugement sémantique :
    - déjà garanti par construction (un match idiomatch ne franchit
      jamais un segment, on le garde comme filet de sécurité) ;
    - candidats à un seul token (bruit de tokenisation, pas une MWE) ;
    - même (idiome, segment, span) en double.
    """

    seen = set()
    kept = []
    for c in candidates:
        if c["n_tokens_span"] < 2:
            continue
        key = (c["idiom"], c["segment_idx"], c["start_char"], c["end_char"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept


def group_by_type(candidates: list[dict]) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_type[c["idiom"]].append(c)
    return dict(by_type)


def run() -> int:
    config.ensure_out_dir()
    segments = load_segments()

    print("Chargement d'idiomatch (n=2)...")
    idiomatch_raw = list(find_candidates(segments))
    print(f"{len(idiomatch_raw)} occurrences idiomatch brutes.")

    idiomatch_accepted = [c for c in idiomatch_raw if c["rejected_by"] is None]
    idiomatch_rejected = [c for c in idiomatch_raw if c["rejected_by"] is not None]
    if idiomatch_rejected:
        print(f"{len(idiomatch_rejected)} écartée(s) par les portes S2 avant fusion "
              f"({config.MWE_REJECTED_CANDIDATES_PATH.name}).")

    vpc_raw = load_vpc_candidates(segments)
    print(f"{len(vpc_raw)} occurrences VPC (Lot 2, {config.VPC_CANDIDATES_PATH.name}).")

    rules_plus_raw = load_rules_plus_candidates(segments)
    print(f"{len(rules_plus_raw)} occurrences rules_plus (Q0-3 Phase 6, "
          f"{config.RULES_PLUS_CANDIDATES_PATH.name}).")

    raw = merge_candidate_sources(idiomatch_accepted, vpc_raw, rules_plus_raw)
    n_merged = len(idiomatch_accepted) + len(vpc_raw) + len(rules_plus_raw) - len(raw)
    if n_merged:
        print(f"{n_merged} occurrence(s) détectée(s) par les deux sources (fusionnées).")

    filtered = structural_prefilter(raw)
    print(f"{len(filtered)} après pré-filtre structurel.")

    by_type = group_by_type(filtered)
    print(f"{len(by_type)} types distincts.")

    atomic.atomic_write_jsonl(
        config.MWE_CANDIDATES_PATH,
        (
            {"idiom": idiom, "count": len(occs), "occurrences": occs}
            for idiom, occs in sorted(by_type.items(), key=lambda kv: -len(kv[1]))
        ),
    )

    print(f"-> {config.MWE_CANDIDATES_PATH}")

    atomic.atomic_write_jsonl(config.MWE_REJECTED_CANDIDATES_PATH, idiomatch_rejected)
    print(f"-> {config.MWE_REJECTED_CANDIDATES_PATH} ({len(idiomatch_rejected)} lignes, audit)")

    top = Counter({idiom: len(occs) for idiom, occs in by_type.items()}).most_common(15)
    print("\nTop 15 par fréquence (à juger en S3, pas à faire confiance ici) :")
    for idiom, count in top:
        print(f"  {count:4d}  {idiom}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
