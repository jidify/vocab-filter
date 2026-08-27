"""Vendoré depuis `book-lexical-analyzer` (booklex/detectors/phrasal_verbs.py,
commit 5bd3f5fce03d4b467f0c8431d0aa91c154cc2841, module booklex 0.1.0) — voir
`pipeline/vpc/__init__.py` pour l'attribution complète.

Changements par rapport à l'original (documentés au point 10/12 du plan) :

- `from booklex.domain.models import Document` supprimé, ainsi que
  `syntax_sentences_from_document()` et `PhrasalVerbDetector.detect_document()`
  qui en dépendaient. `domain/models.py` (l'ingestion + le type `Document` du
  projet source) n'est volontairement pas porté ici : ce dépôt construit ses
  `SyntaxSentence`/`SyntaxToken` directement depuis un `Doc` spaCy déjà vivant
  dans `pipeline/analyze.py`, via `pipeline/vpc/adapter.py`. `analyze()` et
  `detect()` (qui opèrent sur un `SyntaxSentence` déjà construit) sont
  inchangés.
- `from booklex import __version__ as BOOKLEX_VERSION` remplacé par
  `pipeline.vpc.VENDORED_SOURCE_VERSION` (même valeur : "0.1.0" — c'est la
  version du module `booklex` source au moment du portage, pas celle de
  vocab-filter).
- Chemins d'import ajustés vers `pipeline.vpc.domain.*`.

Le reste (logique de détection, règles syntaxiques, gardes directionnelles,
sérialisation) est inchangé caractère pour caractère."""

from __future__ import annotations

from collections import defaultdict
import hashlib

from pipeline.vpc import VENDORED_SOURCE_VERSION as BOOKLEX_VERSION
from pipeline.vpc.domain.phrasal_verbs import (
    PhrasalVerbCategory,
    PhrasalVerbDecision,
    PhrasalVerbDetection,
    PhrasalVerbEvidence,
    PhrasalVerbEvidenceType,
    PhrasalVerbProvenance,
    SyntaxSentence,
    SyntaxToken,
)
from pipeline.vpc.domain.ports import VpcLexiconProvider, VpcReferenceProvider
from pipeline.vpc.domain.resources import ResourceMetadata
from pipeline.vpc.domain.vpc_frames import (
    ProductionVpcFrameSelection,
    VpcFrameSelection,
    syntactic_frame,
)


_MOTION_VERBS = frozenset(
    {"come", "drop", "get", "go", "head", "move", "run", "sit", "stand", "step", "stream", "walk"}
)
_DIRECTIONAL_PARTICLES = frozenset(
    {"around", "back", "down", "forward", "in", "off", "on", "out", "over", "round", "up"}
)
_LOCATIVE_CONTINUATIONS = frozenset(
    {"at", "for", "from", "in", "into", "on", "onto", "through", "to", "toward", "towards"}
)
_LEXICALIZED_CONTINUATION_EXCEPTIONS = frozenset(
    {("get", "around", "to"), ("get", "over", "with"), ("go", "on", "to")}
)
_DIRECT_OBJECT_PREP_PAIRS = frozenset({("get", "at"), ("look", "after")})
_EXTERNAL_ADVMOD_PAIRS = frozenset({("come", "forward"), ("kick", "back")})
_EXTERNAL_PREP_PAIRS = frozenset({("throw", "out")})


class PhrasalVerbDetector:
    """Deterministic English VPC detector with contextual, inspectable evidence."""

    def __init__(
        self,
        references: VpcReferenceProvider,
        external_lexicon: VpcLexiconProvider | None = None,
        *,
        policy_version: str = "phase2.6",
        frame_selection: VpcFrameSelection | ProductionVpcFrameSelection | None = None,
        frozen_selection_sha256: str | None = None,
    ) -> None:
        if policy_version not in {"phase2.4", "phase2.6", "phase2.7", "phase2.8"}:
            raise ValueError(
                "policy_version must be 'phase2.4', 'phase2.6', 'phase2.7', or 'phase2.8'"
            )
        if policy_version == "phase2.4" and external_lexicon is not None:
            raise ValueError("Phase 2.4 baseline does not use an external lexicon")
        if policy_version in {"phase2.7", "phase2.8"} and frame_selection is None:
            raise ValueError("a frame policy requires a frozen TRAIN/DEV frame selection")
        if policy_version not in {"phase2.7", "phase2.8"} and frame_selection is not None:
            raise ValueError("frame selection is only valid for Phase 2.7 or Phase 2.8")
        if frozen_selection_sha256 is not None and (
            len(frozen_selection_sha256) != 64
            or any(character not in "0123456789abcdef" for character in frozen_selection_sha256)
        ):
            raise ValueError("frozen selection SHA-256 must be lowercase hexadecimal")
        if policy_version == "phase2.8":
            assert frame_selection is not None
            if (
                frame_selection.minimum_occurrences != 1
                or frame_selection.minimum_documents != 2
            ):
                raise ValueError("Phase 2.8 requires support >= 1 and document_count >= 2")
            if frame_selection.generic_frames:
                raise ValueError("Phase 2.8 does not license generic frames")
            if any(item.train_documents < 2 for item in frame_selection.lexical_frames):
                raise ValueError("Phase 2.8 excludes single-document lexical frames")
            if any(
                item.verb_lemma == "pick" and item.particle_lemmas == ("up",)
                for item in frame_selection.rejected_lexical_frames
            ):
                raise ValueError("Phase 2.8 excludes the single-document pick up veto")
        self._policy_version = policy_version
        self._frozen_selection_sha256 = frozen_selection_sha256
        self._resource = references.resource
        index: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
        for reference in references.vpc_references():
            key = (
                reference.verb_lemma.casefold(),
                tuple(particle.casefold() for particle in reference.particle_lemmas),
            )
            index[key].add(reference.category)
        self._reference_index = {
            key: tuple(sorted(categories)) for key, categories in index.items()
        }
        self._external_resource = (
            external_lexicon.resource if external_lexicon is not None else None
        )
        self._external_index = {
            (candidate.verb_lemma, candidate.particle_lemmas): candidate
            for candidate in (
                external_lexicon.candidates() if external_lexicon is not None else ()
            )
        }
        self._frame_selection = frame_selection
        self._lexical_frames = {
            ((item.verb_lemma, item.particle_lemmas), item.frame): item
            for item in (frame_selection.lexical_frames if frame_selection else ())
        }
        self._generic_frames = {
            item.frame: item
            for item in (frame_selection.generic_frames if frame_selection else ())
        }
        self._rejected_lexical_frames = {
            ((item.verb_lemma, item.particle_lemmas), item.frame): item
            for item in (
                frame_selection.rejected_lexical_frames if frame_selection else ()
            )
        }

    @property
    def detector_version(self) -> str:
        if self._policy_version == "phase2.4":
            return "dependency-parseme-1.0"
        if self._policy_version == "phase2.7":
            return "dependency-train-frames-3.0"
        if self._policy_version == "phase2.8":
            return "dependency-document-stable-frames-4.0"
        return "dependency-evidence-2.0"

    @property
    def reference_resource(self) -> ResourceMetadata:
        return self._resource

    def analyze(
        self,
        sentence: SyntaxSentence,
        *,
        document_id: str | None = None,
    ) -> tuple[PhrasalVerbDetection, ...]:
        children: dict[int, list[SyntaxToken]] = defaultdict(list)
        for token in sentence.tokens:
            if token.dependency_head_index != token.index:
                children[token.dependency_head_index].append(token)

        decisions: list[PhrasalVerbDetection] = []
        for verb in sentence.tokens:
            pos_recovery = (
                self._policy_version in {"phase2.6", "phase2.7", "phase2.8"}
                and self._is_pos_recovery_verb(verb, children)
            )
            if verb.pos != "VERB" and not pos_recovery:
                continue
            dependents = children.get(verb.index, ())
            particles = sorted(
                (
                    token
                    for token in dependents
                    if token.dependency_relation.lower() == "prt"
                    and token.pos in {"ADP", "PART"}
                ),
                key=lambda token: token.index,
            )
            for particle in particles:
                directional_reason = None
                frame_detail = self._selected_frame_detail(sentence, verb, (particle,))
                frame_rejection = self._rejected_frame_detail(
                    sentence, verb, (particle,)
                )
                if self._policy_version in {"phase2.6", "phase2.7", "phase2.8"}:
                    directional_reason = self._directional_rejection_reason(
                        sentence, verb, particle
                    )
                if frame_rejection is not None:
                    decisions.append(
                        self._reject_candidate(
                            sentence,
                            verb,
                            (particle,),
                            document_id=document_id,
                            rule_id="dev_validated_train_frame_rejection",
                            reason=frame_rejection,
                            context_detail=frame_rejection,
                        )
                    )
                elif (
                    directional_reason is not None
                    and self._policy_version in {"phase2.7", "phase2.8"}
                    and frame_detail is not None
                ):
                    decisions.append(
                        self._classify_particle_candidate(
                            sentence,
                            verb,
                            (particle,),
                            document_id=document_id,
                            syntax_rule="dependency_prt_train_frame_override",
                            context_detail=(
                                f"{frame_detail}; attested lexical frame overrides the "
                                "generic directional guard"
                            ),
                            train_frame_detail=frame_detail,
                        )
                    )
                elif directional_reason is not None:
                    decisions.append(
                        self._reject_candidate(
                            sentence,
                            verb,
                            (particle,),
                            document_id=document_id,
                            rule_id="directional_context_rejection",
                            reason=directional_reason,
                            context_detail="locative continuation supports a literal direction",
                        )
                    )
                else:
                    decisions.append(
                        self._classify_particle_candidate(
                            sentence,
                            verb,
                            (particle,),
                            document_id=document_id,
                            syntax_rule=(
                                "dependency_prt_pos_recovery"
                                if pos_recovery
                                else "dependency_prt"
                            ),
                            context_detail=(
                                "coarse POS recovered because fine POS is verbal and "
                                "external lexical evidence supports the exact prt pair"
                                if pos_recovery
                                else None
                            ),
                            train_frame_detail=frame_detail,
                        )
                    )

            prepositions = sorted(
                (
                    token
                    for token in dependents
                    if token.dependency_relation.lower() in {"prep", "case"}
                    and token.pos == "ADP"
                ),
                key=lambda token: token.index,
            )
            for preposition in prepositions:
                frame_detail = self._selected_frame_detail(
                    sentence, verb, (preposition,)
                )
                if self._policy_version == "phase2.4":
                    prep_frame = (
                        "Phase 2.4 lexical override"
                        if self._reference_categories(verb, (preposition,))
                        else None
                    )
                else:
                    prep_frame = self._accepted_preposition_frame(
                        sentence, verb, preposition, children
                    )
                    if prep_frame is None and self._policy_version in {"phase2.7", "phase2.8"}:
                        if frame_detail is not None:
                            prep_frame = frame_detail
                if prep_frame is not None:
                    frame_rejection = self._rejected_frame_detail(
                        sentence, verb, (preposition,)
                    )
                    if frame_rejection is not None:
                        decisions.append(
                            self._reject_candidate(
                                sentence,
                                verb,
                                (preposition,),
                                document_id=document_id,
                                rule_id="dev_validated_train_frame_rejection",
                                reason=frame_rejection,
                                context_detail=frame_rejection,
                            )
                        )
                    else:
                        decisions.append(
                            self._classify_particle_candidate(
                                sentence,
                                verb,
                                (preposition,),
                                document_id=document_id,
                                syntax_rule="dependency_prep_override",
                                context_detail=prep_frame,
                                train_frame_detail=frame_detail,
                            )
                        )
                else:
                    decisions.append(
                        self._reject_preposition(
                            sentence,
                            verb,
                            preposition,
                            document_id=document_id,
                        )
                    )

            lexical_adverbs = sorted(
                (
                    token
                    for token in dependents
                    if token.dependency_relation.lower() == "advmod"
                    and token.pos in {"ADV", "ADP", "PART"}
                    and (
                        self._reference_categories(verb, (token,))
                        or (
                            self._policy_version in {"phase2.6", "phase2.7", "phase2.8"}
                            and self._accepted_external_advmod(
                                sentence, verb, token, children
                            )
                        )
                    )
                ),
                key=lambda token: token.index,
            )
            for adverb in lexical_adverbs:
                frame_rejection = self._rejected_frame_detail(
                    sentence, verb, (adverb,)
                )
                if frame_rejection is not None:
                    decisions.append(
                        self._reject_candidate(
                            sentence,
                            verb,
                            (adverb,),
                            document_id=document_id,
                            rule_id="dev_validated_train_frame_rejection",
                            reason=frame_rejection,
                            context_detail=frame_rejection,
                        )
                    )
                else:
                    decisions.append(
                        self._classify_particle_candidate(
                            sentence,
                            verb,
                            (adverb,),
                            document_id=document_id,
                            syntax_rule="dependency_advmod",
                            context_detail="advmod pair is explicitly licensed by lexical and contextual evidence",
                            train_frame_detail=self._selected_frame_detail(
                                sentence, verb, (adverb,)
                            ),
                        )
                    )
        return tuple(
            sorted(
                decisions,
                key=lambda item: (item.verb_token_index, item.particle_token_indices),
            )
        )

    def detect(
        self,
        sentence: SyntaxSentence,
        *,
        document_id: str | None = None,
    ) -> tuple[PhrasalVerbDetection, ...]:
        return tuple(
            decision
            for decision in self.analyze(sentence, document_id=document_id)
            if decision.is_detection
        )

    def _classify_particle_candidate(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        particles: tuple[SyntaxToken, ...],
        *,
        document_id: str | None,
        syntax_rule: str,
        context_detail: str | None = None,
        train_frame_detail: str | None = None,
    ) -> PhrasalVerbDetection:
        particle_lemmas = tuple(token.lemma.casefold() for token in particles)
        key = (verb.lemma.casefold(), particle_lemmas)
        reference_categories = self._reference_index.get(key, ())
        if reference_categories:
            decision = PhrasalVerbDecision.MATCHED_REFERENCE
            category = _reference_category(reference_categories)
            rule_id = f"{syntax_rule}+parseme_lemma_match"
            reason = (
                f"spaCy {verb.dependency_relation}/{particles[0].dependency_relation} syntax "
                "and lemma pair attested as a PARSEME VPC"
            )
        else:
            decision = PhrasalVerbDecision.NOT_MATCHED_REFERENCE
            category = PhrasalVerbCategory.VPC_UNSPECIFIED
            rule_id = f"{syntax_rule}_without_parseme_match"
            reason = (
                "contextual syntax and candidate evidence support a VPC; "
                "PARSEME absence is not rejection"
            )
        return self._build_decision(
            sentence,
            verb,
            particles,
            document_id=document_id,
            decision=decision,
            category=category,
            reference_categories=reference_categories,
            rule_id=rule_id,
            reason=reason,
            syntax_rule=syntax_rule,
            context_detail=context_detail,
            train_frame_detail=train_frame_detail,
        )

    def _selected_frame_detail(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        particles: tuple[SyntaxToken, ...],
    ) -> str | None:
        if self._policy_version not in {"phase2.7", "phase2.8"}:
            return None
        frame = syntactic_frame(sentence, verb, particles)
        lexical_key = (
            verb.lemma.casefold(),
            tuple(item.lemma.casefold() for item in particles),
        )
        selected = self._lexical_frames.get((lexical_key, frame))
        if selected is not None:
            if self._policy_version == "phase2.8":
                return f"PARSEME TRAIN lexical frame (n={selected.train_occurrences})"
            return (
                "PARSEME TRAIN lexical frame "
                f"(n={selected.train_occurrences}, DEV FP=0)"
            )
        selected = self._generic_frames.get(frame)
        if selected is not None:
            return (
                "PARSEME TRAIN generic frame "
                f"(n={selected.train_occurrences}, DEV TP={selected.dev_true_positives}, DEV FP=0)"
            )
        return None

    def _rejected_frame_detail(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        particles: tuple[SyntaxToken, ...],
    ) -> str | None:
        if self._policy_version not in {"phase2.7", "phase2.8"}:
            return None
        frame = syntactic_frame(sentence, verb, particles)
        key = (
            verb.lemma.casefold(),
            tuple(item.lemma.casefold() for item in particles),
        )
        rejected = self._rejected_lexical_frames.get((key, frame))
        if rejected is None:
            return None
        return (
            "PARSEME TRAIN lexical frame was rejected on DEV "
            f"({rejected.dev_false_positives} FP, 0 TP)"
        )

    def _external_candidate(
        self,
        verb: SyntaxToken,
        particles: tuple[SyntaxToken, ...],
    ):
        return self._external_index.get(
            (
                verb.lemma.casefold(),
                tuple(token.lemma.casefold() for token in particles),
            )
        )

    def _is_pos_recovery_verb(
        self,
        token: SyntaxToken,
        children: dict[int, list[SyntaxToken]],
    ) -> bool:
        if token.pos == "VERB" or not (token.fine_pos or "").upper().startswith("VB"):
            return False
        if token.dependency_relation.lower() not in {
            "root",
            "conj",
            "xcomp",
            "ccomp",
            "advcl",
        }:
            return False
        return any(
            child.dependency_relation.lower() == "prt"
            and child.pos in {"ADP", "PART"}
            and self._external_candidate(token, (child,)) is not None
            for child in children.get(token.index, ())
        )

    def _accepted_external_advmod(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        adverb: SyntaxToken,
        children: dict[int, list[SyntaxToken]],
    ) -> bool:
        key = (verb.lemma.casefold(), adverb.lemma.casefold())
        if key not in _EXTERNAL_ADVMOD_PAIRS or self._external_candidate(verb, (adverb,)) is None:
            return False
        if key == ("come", "forward"):
            return self._directional_rejection_reason(sentence, verb, adverb) is None
        return any(
            child.index < adverb.index and child.dependency_relation.lower() in {"dobj", "obj"}
            for child in children.get(verb.index, ())
        )

    def _accepted_preposition_frame(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        preposition: SyntaxToken,
        children: dict[int, list[SyntaxToken]],
    ) -> str | None:
        pair = (verb.lemma.casefold(), preposition.lemma.casefold())
        reference_categories = self._reference_categories(verb, (preposition,))
        external = self._external_candidate(verb, (preposition,))
        particle_children = children.get(preposition.index, ())
        has_nested_preposition = any(
            child.dependency_relation.lower() == "prep" for child in particle_children
        )
        has_verbal_complement = any(
            child.dependency_relation.lower() in {"pcomp", "pobj", "xcomp"}
            and child.pos == "VERB"
            for child in particle_children
        )
        has_direct_object = any(
            child.dependency_relation.lower() in {"pobj", "obj"}
            for child in particle_children
        )
        if reference_categories:
            if pair == ("come", "out") and has_nested_preposition:
                return "TRAIN-attested come out uses the particle+preposition frame"
            if pair == ("go", "on") and has_verbal_complement:
                return "TRAIN-attested go on selects a verbal complement"
            if pair in _DIRECT_OBJECT_PREP_PAIRS and has_direct_object:
                return "TRAIN-attested inseparable VPC selects a direct prepositional object"
            return None
        if (
            external is not None
            and pair in _EXTERNAL_PREP_PAIRS
            and has_nested_preposition
        ):
            return "external candidate uses an explicitly licensed particle+preposition frame"
        return None

    def _directional_rejection_reason(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        particle: SyntaxToken,
    ) -> str | None:
        pair = (verb.lemma.casefold(), particle.lemma.casefold())
        following_prepositions = tuple(
            token.lemma.casefold()
            for token in sentence.tokens
            if token.index > particle.index
            and token.dependency_head_index == verb.index
            and token.dependency_relation.lower() == "prep"
            and token.lemma.casefold() in _LOCATIVE_CONTINUATIONS
        )
        if pair == ("look", "up") and any(
            item in {"at", "toward", "towards"} for item in following_prepositions
        ):
            return "look up is followed by an orientation complement"
        if verb.lemma.casefold() not in _MOTION_VERBS or particle.lemma.casefold() not in _DIRECTIONAL_PARTICLES:
            return None
        for continuation in following_prepositions:
            if (pair[0], pair[1], continuation) in _LEXICALIZED_CONTINUATION_EXCEPTIONS:
                continue
            return (
                f"motion verb and directional particle continue with locative preposition "
                f"{continuation!r}"
            )
        return None

    def _reference_categories(
        self,
        verb: SyntaxToken,
        particles: tuple[SyntaxToken, ...],
    ) -> tuple[str, ...]:
        return self._reference_index.get(
            (
                verb.lemma.casefold(),
                tuple(token.lemma.casefold() for token in particles),
            ),
            (),
        )

    def _reject_preposition(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        preposition: SyntaxToken,
        *,
        document_id: str | None,
    ) -> PhrasalVerbDetection:
        return self._reject_candidate(
            sentence,
            verb,
            (preposition,),
            document_id=document_id,
            rule_id="dependency_preposition_rejection",
            reason=(
                "preposition lacks an explicitly licensed VPC complement frame"
            ),
        )

    def _reject_candidate(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        particles: tuple[SyntaxToken, ...],
        *,
        document_id: str | None,
        rule_id: str,
        reason: str,
        context_detail: str | None = None,
        train_frame_detail: str | None = None,
    ) -> PhrasalVerbDetection:
        return self._build_decision(
            sentence,
            verb,
            particles,
            document_id=document_id,
            decision=PhrasalVerbDecision.REJECTED_SYNTAX,
            category=None,
            reference_categories=self._reference_categories(verb, particles),
            rule_id=rule_id,
            reason=reason,
            syntax_rule=particles[0].dependency_relation.lower(),
            context_detail=context_detail,
        )

    def _build_decision(
        self,
        sentence: SyntaxSentence,
        verb: SyntaxToken,
        particles: tuple[SyntaxToken, ...],
        *,
        document_id: str | None,
        decision: PhrasalVerbDecision,
        category: PhrasalVerbCategory | None,
        reference_categories: tuple[str, ...],
        rule_id: str,
        reason: str,
        syntax_rule: str,
        context_detail: str | None = None,
        train_frame_detail: str | None = None,
    ) -> PhrasalVerbDetection:
        particle_indices = tuple(token.index for token in particles)
        token_indices = tuple(sorted((verb.index, *particle_indices)))
        discontinuous = any(
            right != left + 1
            for left, right in zip(token_indices, token_indices[1:])
        )
        ordered_tokens = tuple(sorted((verb, *particles), key=lambda token: token.index))
        relative_spans = tuple((token.start_char, token.end_char) for token in ordered_tokens)
        normalized_spans = _optional_spans(
            ordered_tokens, "normalized_start_char", "normalized_end_char"
        )
        original_spans = _optional_spans(
            ordered_tokens, "original_start_char", "original_end_char"
        )
        separator = " … " if discontinuous else " "
        external_candidate = self._external_candidate(verb, particles)
        evidence = [
            PhrasalVerbEvidence(
                evidence_type=PhrasalVerbEvidenceType.SYNTAX,
                detail=f"spaCy dependency rule: {syntax_rule}",
            )
        ]
        if reference_categories:
            evidence.append(
                PhrasalVerbEvidence(
                    evidence_type=PhrasalVerbEvidenceType.PARSEME_TRAIN,
                    detail="exact lemmatized pair is annotated in PARSEME TRAIN",
                    provenance=self._resource,
                )
            )
        else:
            evidence.append(
                PhrasalVerbEvidence(
                    evidence_type=PhrasalVerbEvidenceType.ABSENT_REFERENCE,
                    detail="exact lemmatized pair is absent from PARSEME TRAIN",
                    provenance=self._resource,
                )
            )
        if train_frame_detail is not None:
            evidence.append(
                PhrasalVerbEvidence(
                    evidence_type=PhrasalVerbEvidenceType.PARSEME_TRAIN_FRAME,
                    detail=train_frame_detail,
                    provenance=self._resource,
                )
            )
        provenance = [self._resource]
        if external_candidate is not None:
            evidence.append(
                PhrasalVerbEvidence(
                    evidence_type=PhrasalVerbEvidenceType.EXTERNAL_LEXICON,
                    detail=(
                        f"Open English WordNet verb entry {external_candidate.resource_entry_id!r} "
                        "supplies candidate evidence only"
                    ),
                    provenance=external_candidate.provenance,
                )
            )
            provenance.append(external_candidate.provenance)
        if context_detail is not None:
            evidence.append(
                PhrasalVerbEvidence(
                    evidence_type=PhrasalVerbEvidenceType.CONTEXT,
                    detail=context_detail,
                )
            )
        detection_key = "\0".join(
            (
                document_id or "",
                sentence.sentence_id,
                verb.token_id,
                *(token.token_id for token in particles),
                self.detector_version,
                self._policy_version,
            )
        ).encode("utf-8")
        return PhrasalVerbDetection(
            detection_id=f"vpc-{hashlib.sha256(detection_key).hexdigest()[:24]}",
            document_id=document_id,
            sentence_id=sentence.sentence_id,
            verb_token_id=verb.token_id,
            verb_token_index=verb.index,
            particle_token_ids=tuple(token.token_id for token in particles),
            particle_token_indices=particle_indices,
            token_indices=token_indices,
            verb_char_span=(verb.start_char, verb.end_char),
            particle_char_spans=tuple(
                (token.start_char, token.end_char) for token in particles
            ),
            token_char_spans=relative_spans,
            normalized_char_spans=normalized_spans,
            original_char_spans=original_spans,
            verb_form=verb.surface_form,
            verb_lemma=verb.lemma.casefold(),
            particle_forms=tuple(token.surface_form for token in particles),
            particle_lemmas=tuple(token.lemma.casefold() for token in particles),
            normalized_expression=" ".join((verb.lemma.casefold(), *(
                token.lemma.casefold() for token in particles
            ))),
            observed_expression=separator.join(token.surface_form for token in ordered_tokens),
            is_discontinuous=discontinuous,
            category=category,
            reference_categories=reference_categories,
            decision=decision,
            rule_id=rule_id,
            decision_reason=reason,
            detector_version=self.detector_version,
            provenance=PhrasalVerbProvenance(
                booklex_version=BOOKLEX_VERSION,
                detector_version=self.detector_version,
                policy_version=self._policy_version,
                policy_id=(
                    "C-support-1-documents-2"
                    if self._policy_version == "phase2.8"
                    else None
                ),
                frozen_selection_sha256=self._frozen_selection_sha256,
                nlp_engine=sentence.nlp_engine,
                nlp_engine_version=sentence.nlp_engine_version,
                nlp_model=sentence.nlp_model,
                parseme_reference=self._resource,
                external_lexicon=self._external_resource,
            ),
            reference_provenance=tuple(provenance),
            evidence=tuple(evidence),
        )


def _reference_category(categories: tuple[str, ...]) -> PhrasalVerbCategory:
    if categories == ("VPC.full",):
        return PhrasalVerbCategory.VPC_FULL
    if categories == ("VPC.semi",):
        return PhrasalVerbCategory.VPC_SEMI
    return PhrasalVerbCategory.VPC_AMBIGUOUS


def _optional_spans(
    tokens: tuple[SyntaxToken, ...],
    start_attribute: str,
    end_attribute: str,
) -> tuple[tuple[int, int], ...] | None:
    values = tuple(
        (getattr(token, start_attribute), getattr(token, end_attribute)) for token in tokens
    )
    if all(start is None and end is None for start, end in values):
        return None
    if any(start is None or end is None for start, end in values):
        raise ValueError("all detection members must share the same offset coordinate system")
    return tuple((start, end) for start, end in values if start is not None and end is not None)
