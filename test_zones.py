"""Lot 5 — layout de zones (plan Partie 4, Lot 5 ; voir aussi Partie 2 point
H/I). `RealBookLayoutTests` rejoue le chemin de production
(`pipeline.analyze.analyze_segments`, vrai `The Humans - Stephen Karam.txt`,
vrai spaCy) — même esprit que `test_vpc_contract.py` : pas de doublure
synthétique pour la validation de bout en bout. `SyntheticBoundarySnapTests`
isole ensuite le comportement de découpage lui-même sur des cas construits à
la main, sans dépendre du livre ni de spaCy."""

from __future__ import annotations

import unittest

from pipeline import zones
from pipeline.analyze import analyze_segments
from pipeline.corpus import load_segments


def _real_book_zone_tokens() -> tuple[list[int], str]:
    segments = load_segments()
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]
    vpc_sink: list[dict] = []
    zone_sink: list[int] = []
    # Épuise le générateur — c'est lui qui alimente zone_sink au fil du
    # parcours (voir analyze_segments, Lot 5) : TOUS les tokens non-espace,
    # ponctuation comprise, pas seulement ceux qui finissent dans
    # occurrences.jsonl.
    list(analyze_segments(play_segments, vpc_sink, zone_sink))
    source_text = "\n".join(s.en for s in play_segments)
    return zone_sink, source_text


class RealBookLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zone_tokens, cls.source_text = _real_book_zone_tokens()

    def test_about_twenty_zones_at_5_percent(self):
        layout = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=5.0)
        # zone_percent=5.0 -> ceil(100/5) = 20, tant que le livre a bien
        # plus de 20 tokens (trivialement vrai pour une pièce entière).
        self.assertEqual(layout["zone_count"], 20)
        self.assertEqual(len(layout["zones"]), 20)

    def test_boundaries_fall_on_segment_limits(self):
        layout = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=5.0)
        segment_boundaries = set(zones._segment_boundaries(self.zone_tokens))
        for z in layout["zones"][:-1]:  # la dernière zone finit à `total`, pas une limite de segment
            self.assertIn(
                z["end_token"], segment_boundaries,
                f"zone {z['zone_id']} finit au token {z['end_token']}, qui ne "
                f"tombe pas sur une limite de segment (coupe une "
                f"réplique/didascalie en deux)",
            )

    def test_every_token_zone_id_matches_the_segment_zone_map(self):
        """Vérifie, token par token, qu'aucun segment ne se retrouve
        réparti sur deux zones : le zone_id qu'on lirait en suivant la
        liste `zones` doit être partout identique à celui que
        `segment_zone_map` associe au segment_idx de ce token."""

        layout = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=5.0)
        seg_zone = zones.segment_zone_map(layout)

        token_zone_id: list[str | None] = [None] * len(self.zone_tokens)
        for z in layout["zones"]:
            for i in range(z["start_token"], z["end_token"]):
                token_zone_id[i] = z["zone_id"]

        for i, seg_idx in enumerate(self.zone_tokens):
            self.assertEqual(token_zone_id[i], seg_zone[seg_idx])

    def test_layout_id_is_stable_across_identical_runs(self):
        layout_a = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=5.0)
        layout_b = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=5.0)
        self.assertEqual(layout_a["layout_id"], layout_b["layout_id"])
        # Stable au sens fort : pas seulement l'identifiant, tout le layout
        # (mêmes bornes, même segment_zone_map) est reproductible.
        self.assertEqual(layout_a, layout_b)

    def test_layout_id_changes_with_zone_percent(self):
        layout_5 = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=5.0)
        layout_10 = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=10.0)
        self.assertNotEqual(layout_5["layout_id"], layout_10["layout_id"])

    def test_layout_id_changes_with_source_text(self):
        layout_a = zones.build_layout(self.zone_tokens, self.source_text, zone_percent=5.0)
        layout_b = zones.build_layout(self.zone_tokens, self.source_text + " ", zone_percent=5.0)
        self.assertNotEqual(layout_a["layout_id"], layout_b["layout_id"])


class SyntheticBoundarySnapTests(unittest.TestCase):
    """Cas construits à la main, indépendants du livre réel et de spaCy :
    isole le comportement de découpage lui-même."""

    def test_zone_boundaries_never_split_a_synthetic_segment(self):
        sizes = [5, 3, 4, 2, 6, 3, 5, 2, 4, 3]  # 10 segments, 37 tokens
        token_segment_idxs = [
            seg_idx for seg_idx, size in enumerate(sizes) for _ in range(size)
        ]
        layout = zones.build_layout(token_segment_idxs, "synthetic text", zone_percent=25.0)
        segment_boundaries = set(zones._segment_boundaries(token_segment_idxs))
        for z in layout["zones"][:-1]:
            self.assertIn(z["end_token"], segment_boundaries)

    def test_single_giant_segment_falls_back_to_a_raw_cut(self):
        # Un seul segment (aucune limite disponible) : le repli ultime de
        # _nearest_boundary (`maximum`) est la seule option — documenté
        # dans zones.py, pas censé arriver sur un vrai livre.
        token_segment_idxs = [0] * 40
        layout = zones.build_layout(token_segment_idxs, "one giant segment", zone_percent=25.0)
        self.assertEqual(layout["zone_count"], 4)
        self.assertEqual(layout["zones"][-1]["end_token"], 40)

    def test_layout_id_is_content_addressed(self):
        id_a = zones.compute_layout_id("hello world", 5.0)
        id_b = zones.compute_layout_id("hello world", 5.0)
        id_c = zones.compute_layout_id("goodbye world", 5.0)
        self.assertEqual(id_a, id_b)
        self.assertNotEqual(id_a, id_c)

    def test_empty_token_list_is_rejected(self):
        with self.assertRaises(ValueError):
            zones.build_layout([], "text", zone_percent=5.0)


if __name__ == "__main__":
    unittest.main()
