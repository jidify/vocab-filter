"""Lot 2 — validation de portage du détecteur VPC (`pipeline/vpc/`).

Les 13 cas ci-dessous sont repris de `book-lexical-analyzer`
(tests/fixtures/vpc_contract_cases.json, commit 5bd3f5fce03d4b467f0c8431d0aa91c154cc2841) :
"purpose": "Contract regression only; never calibration data" — ce ne sont PAS
les mêmes phrases que celles utilisées pour calibrer la politique Phase 2.8
(TRAIN/DEV PARSEME), donc les faire passer ici valide que le PORTAGE se
comporte comme l'original sur des exemples indépendants, pas qu'il
"reproduit la baseline PARSEME" (voir le plan, point 4 de la Partie 1 —
distinction explicitement voulue).

Le détecteur est construit via `pipeline.vpc.service.build_detector()` : les
VRAIES ressources gelées de `data/vpc/` (politique Phase 2.8) et le VRAI
fournisseur WordNet (`pipeline.vpc.resources.wordnet_nltk`, via `nltk`) — pas
de doublure synthétique. C'est le même chemin que celui emprunté par
`pipeline/analyze.py` en production."""

from __future__ import annotations

import hashlib
import json
import unittest

import spacy

from pipeline import config
from pipeline.vpc.adapter import sentences_from_doc
from pipeline.vpc.service import VPC_MANIFEST_PATH, build_detector

# Repris tel quel de tests/fixtures/vpc_contract_cases.json (projet source).
CONTRACT_CASES = [
    {"text": "They give up.", "expression": "give up", "accepted": True},
    {"text": "They gave up.", "expression": "give up", "accepted": True},
    {"text": "Give it up.", "expression": "give up", "accepted": True},
    {"text": "Look up the answer.", "expression": "look up", "accepted": True},
    {"text": "Get out now.", "expression": "get out", "accepted": True},
    {"text": "They go on.", "expression": "go on", "accepted": True},
    {"text": "Turn on the light.", "expression": "turn on", "accepted": False},
    {"text": "They run out.", "expression": "run out", "accepted": True},
    {"text": "She came forward.", "expression": "come forward", "accepted": True},
    {"text": "I look forward to Monday.", "expression": "look forward", "accepted": True},
    {"text": "She looked at him.", "expression": "look at", "accepted": False},
    {"text": "They walked into the room.", "expression": "walk into", "accepted": False},
    {"text": "He depends on us.", "expression": "depend on", "accepted": False},
]


class VpcContractCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("en_core_web_sm")
        cls.detector = build_detector()

    def _analyze(self, text: str, ordinal: int):
        doc = self.nlp(text)
        sentences = sentences_from_doc(
            doc,
            sentence_id_prefix=f"contract-{ordinal}",
            nlp_engine_version=spacy.__version__,
            nlp_model="test",
        )
        return {
            item.normalized_expression: item
            for sentence in sentences
            for item in self.detector.analyze(sentence)
        }

    def test_all_13_contract_cases(self):
        for ordinal, case in enumerate(CONTRACT_CASES):
            with self.subTest(text=case["text"]):
                decisions = self._analyze(case["text"], ordinal)
                self.assertIn(
                    case["expression"],
                    decisions,
                    f"{case['expression']!r} absent des décisions pour {case['text']!r} "
                    f"(obtenu : {sorted(decisions)})",
                )
                self.assertIs(
                    decisions[case["expression"]].is_detection,
                    case["accepted"],
                )

    def test_rejections_stay_out_of_reservation(self):
        # Point 5 de la Partie 1 du plan : un rejet reste dans les résultats
        # (pour audit) mais n'est jamais "is_detection" — jamais réservable.
        decisions = self._analyze("Turn on the light.", 100)
        rejection = decisions["turn on"]
        self.assertFalse(rejection.is_detection)
        self.assertEqual(rejection.category, None)


class VpcManifestIntegrityTests(unittest.TestCase):
    """Non-régression du piège LF/CRLF documenté au plan (point 32) : si un
    `core.autocrlf` ou un commit futur corrompt les ressources gelées de
    `data/vpc/`, ce test échoue au lieu de laisser le détecteur tourner sur
    des octets différents de ceux gelés."""

    def test_frozen_resources_match_the_manifest_sha256(self):
        manifest = json.loads(VPC_MANIFEST_PATH.read_text(encoding="utf-8"))
        for record in manifest["files"].values():
            path = config.ROOT / record["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, record["sha256"], msg=str(path))

    def test_build_detector_is_cached_and_returns_the_same_instance(self):
        self.assertIs(build_detector(), build_detector())


if __name__ == "__main__":
    unittest.main()
