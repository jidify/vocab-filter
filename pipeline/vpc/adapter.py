"""Adaptateur `Doc` spaCy -> `SyntaxSentence`/`SyntaxToken` (plan, Lot 2, point 10).

Ce n'est PAS un portage : `book-lexical-analyzer` construit ses
`SyntaxSentence` dans `booklex/nlp/spacy_engine.py::SpacyNlpEngine.analyze_syntax`,
qui appelle lui-même `nlp(text)` sur une chaîne. Ici, le `Doc` est déjà produit
par la boucle `nlp.pipe()` existante de `pipeline/analyze.py` (voir le plan,
"pas de nouvelle passe spaCy séparée") — cette fonction projette un `Doc` déjà
vivant, elle n'en produit pas un nouveau. La logique de projection (quels
champs spaCy vont où) suit le même principe que `analyze_syntax` : seuls les
tokens `is_space` sont exclus (la ponctuation est conservée — le détecteur VPC
en a besoin pour la structure de dépendance, contrairement à la boucle
lexicale de `pipeline/analyze.py` qui l'exclut après coup)."""

from __future__ import annotations

from spacy.tokens import Doc

from pipeline.vpc.domain.phrasal_verbs import SyntaxSentence, SyntaxToken


def sentences_from_doc(
    doc: Doc,
    *,
    sentence_id_prefix: str,
    nlp_engine_version: str,
    nlp_model: str,
) -> tuple[SyntaxSentence, ...]:
    """Projette les phrases d'un `Doc` spaCy déjà annoté (tagger + parser)
    vers le contrat `SyntaxSentence`/`SyntaxToken` consommé par
    `pipeline.vpc.detectors.PhrasalVerbDetector`.

    `sentence_id_prefix` doit être unique pour le `Doc` appelant (typiquement
    `f"seg{segment_idx}"`) : les identifiants de phrase et de token en
    dérivent, ce qui rend `PhrasalVerbDetection.detection_id` déterministe
    d'un run à l'autre (même segment, même index de phrase, même index de
    token -> même hash)."""

    results: list[SyntaxSentence] = []
    for ordinal, sent in enumerate(doc.sents):
        sentence_id = f"{sentence_id_prefix}:s{ordinal}"
        sentence_start = sent.start_char
        sentence_tokens = tuple(token for token in sent if not token.is_space)
        sentence_indices = {token.i: index for index, token in enumerate(sentence_tokens)}
        tokens = tuple(
            SyntaxToken(
                token_id=f"{sentence_id}:tok-{sentence_indices[token.i]}",
                sentence_id=sentence_id,
                index=sentence_indices[token.i],
                surface_form=token.text,
                lemma=(token.lemma_ or token.text).lower(),
                pos=token.pos_ or "X",
                fine_pos=token.tag_ or None,
                dependency_relation=token.dep_ or "dep",
                dependency_head_index=sentence_indices.get(
                    token.head.i, sentence_indices[token.i]
                ),
                start_char=token.idx - sentence_start,
                end_char=token.idx - sentence_start + len(token.text),
            )
            for token in sentence_tokens
        )
        if not tokens:
            continue
        results.append(
            SyntaxSentence(
                sentence_id=sentence_id,
                text=sent.text,
                tokens=tokens,
                nlp_engine="spacy",
                nlp_engine_version=nlp_engine_version,
                nlp_model=nlp_model,
            )
        )
    return tuple(results)
