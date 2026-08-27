"""Vendoré depuis `book-lexical-analyzer` (booklex/resources/vpc_reference.py,
commit 5bd3f5fce03d4b467f0c8431d0aa91c154cc2841, module booklex 0.1.0) — voir
`pipeline/vpc/__init__.py` pour l'attribution complète. Seul changement : les
chemins d'import vers `pipeline.vpc.domain.*`.

Lit les deux ressources gelées PARSEME (annotations VMWE, licence CC-BY-4.0 —
voir `data/vpc/manifest.json` et, côté projet source,
`resources/manifests/vpc-phase2.8.json` / `resources/manifests/parseme-en-1.3.json`
pour l'attribution complète)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.vpc.domain.phrasal_verbs import FrozenVpcReferenceLexicon, VpcReferenceLexeme
from pipeline.vpc.domain.resources import ResourceMetadata
from pipeline.vpc.domain.vpc_frames import ProductionVpcFrame, ProductionVpcFrameSelection


class FrozenVpcReferenceProvider:
    """Load a deterministic TRAIN-derived lexicon without loading PARSEME CUPT.

    This adapter is the production boundary. It exposes only lemma/category evidence;
    sentences, gold occurrences, splits, metrics, and error-analysis state are absent.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        source = Path(path)
        try:
            payload = source.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read frozen VPC reference lexicon: {source}") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.casefold():
            raise ValueError("SHA-256 mismatch for frozen VPC reference lexicon")
        lexicon = FrozenVpcReferenceLexicon.model_validate_json(payload)
        ordered = tuple(
            sorted(
                lexicon.references,
                key=lambda item: (
                    item.verb_lemma.casefold(),
                    tuple(value.casefold() for value in item.particle_lemmas),
                    item.category,
                ),
            )
        )
        if ordered != lexicon.references or len(set(ordered)) != len(ordered):
            raise ValueError("frozen VPC references must be unique and canonically sorted")
        self._lexicon = lexicon
        self._artifact_sha256 = digest

    @property
    def resource(self) -> ResourceMetadata:
        return self._lexicon.provenance

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    def vpc_references(self) -> tuple[VpcReferenceLexeme, ...]:
        return self._lexicon.references


class FrozenVpcFramePolicyProvider:
    """Project only detector inputs from the frozen policy/report artifact."""

    def __init__(self, path: str | Path, *, expected_sha256: str) -> None:
        source = Path(path)
        try:
            content = source.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read frozen VPC frame policy: {source}") from exc
        if hashlib.sha256(content).hexdigest() != expected_sha256.casefold():
            raise ValueError("SHA-256 mismatch for frozen VPC frame policy")
        try:
            payload = json.loads(content)
            frozen = payload["frozen_frame_selection"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("invalid frozen VPC frame policy") from exc

        def project(items: list[dict[str, object]]) -> tuple[ProductionVpcFrame, ...]:
            return tuple(
                ProductionVpcFrame.model_validate(
                    {
                        "level": item["level"],
                        "verb_lemma": item.get("verb_lemma"),
                        "particle_lemmas": item.get("particle_lemmas"),
                        "frame": item["frame"],
                        "train_occurrences": item["train_occurrences"],
                        "train_documents": item["train_documents"],
                    }
                )
                for item in items
            )

        self._selection = ProductionVpcFrameSelection(
            policy_id=payload["chosen_policy_id"],
            minimum_occurrences=frozen["minimum_occurrences"],
            minimum_documents=frozen["minimum_documents"],
            lexical_frames=project(frozen["lexical_frames"]),
            generic_frames=project(frozen["generic_frames"]),
            rejected_lexical_frames=project(frozen["rejected_lexical_frames"]),
        )

    def selection(self) -> ProductionVpcFrameSelection:
        return self._selection
