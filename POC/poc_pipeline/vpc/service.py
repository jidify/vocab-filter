"""Construction du détecteur VPC pour `pipeline/analyze.py` (Lot 2).

Code écrit pour ce dépôt (pas un portage) : charge les deux ressources
gelées de `data/vpc/` via le manifeste (`data/vpc/manifest.json`), qui reste
la seule source de vérité pour les empreintes SHA-256 — pas de duplication
de constantes hexadécimales ici. Politique figée Phase 2.8
(`policy_version="phase2.8"`), fournisseur externe WordNet via `nltk`
(`pipeline.vpc.resources.wordnet_nltk`, pas un nouvel index — voir le plan,
point 11)."""

from __future__ import annotations

import functools
import json

from poc_pipeline import config
from poc_pipeline.vpc.detectors import PhrasalVerbDetector
from poc_pipeline.vpc.resources.vpc_reference import (
    FrozenVpcFramePolicyProvider,
    FrozenVpcReferenceProvider,
)
from poc_pipeline.vpc.resources.wordnet_nltk import NltkWordNetVpcProvider

VPC_DATA_DIR = config.DATA_DIR / "vpc"
VPC_MANIFEST_PATH = VPC_DATA_DIR / "manifest.json"


@functools.lru_cache(maxsize=1)
def build_detector() -> PhrasalVerbDetector:
    manifest = json.loads(VPC_MANIFEST_PATH.read_text(encoding="utf-8"))
    files = manifest["files"]

    reference_record = files["train_lexical_projection"]
    reference = FrozenVpcReferenceProvider(
        config.ROOT / reference_record["path"],
        expected_sha256=reference_record["sha256"],
    )

    frame_record = files["frozen_frame_selection"]
    frame_provider = FrozenVpcFramePolicyProvider(
        config.ROOT / frame_record["path"],
        expected_sha256=frame_record["sha256"],
    )

    return PhrasalVerbDetector(
        reference,
        NltkWordNetVpcProvider(),
        policy_version="phase2.8",
        frame_selection=frame_provider.selection(),
        frozen_selection_sha256=frame_record["frozen_source_bytes_sha256"],
    )
