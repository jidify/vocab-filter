"""Lot 5 — Layout de zones (plan Partie 2 point H/I, Partie 4 Lot 5).

Portage adapté de `book-lexical-analyzer/src/booklex/zones/segmenter.py` —
même principe (découpage en `zone_percent`% du livre, frontières ajustées à
la structure la plus proche dans une tolérance de 35 % de la taille idéale de
zone, `layout_id` content-addressed), mais SANS la logique d'accumulation par
`ReadingZone`/`ZoneLayout` pydantic du projet source (statistiques lexicales
par zone, n-grammes...) : ça n'existe pas ici, ce n'est pas demandé par le
plan (voir sa Partie 3, "ce qui n'existe pas dans le projet 1"), et rien en
aval n'en a besoin — seule la correspondance segment_idx -> zone_id compte.

Deux différences volontaires avec le projet source :

- Univers de tokens (point I) : TOUS les tokens non-espace des `Doc` spaCy
  produits par `analyze.py`, ponctuation comprise — pas seulement les tokens
  lexicaux d'`occurrences.jsonl`, qui l'excluent déjà pour d'autres raisons
  (is_alpha/is_stop, voir select.py).
- Hiérarchie de frontières : le projet source ajuste passage -> phrase ->
  token. Il n'y a pas de notion de "passage" séparée ici — le `Segment`
  (réplique ou didascalie, `pipeline/corpus.py`) en tient lieu, et c'est la
  SEULE structure candidate (pas de repli sur les phrases spaCy à
  l'intérieur d'un segment) : chaque zone est garantie ne jamais couper un
  Segment en deux (vérifié par test_zones.py).
"""

from __future__ import annotations

import hashlib
import json
import math

from pipeline import atomic, config

ZONE_ALGORITHM = "segment_boundary_snap"
ZONE_ALGORITHM_VERSION = "1.0.0"
BOUNDARY_TOLERANCE_RATIO = 0.35


def compute_layout_id(source_text: str, zone_percent: float) -> str:
    """Content-addressed (point H) : hash du texte source + zone_percent +
    version d'algorithme. Toute étape qui charge `zone_layout.json` peut
    comparer ce `layout_id` à ce qu'elle attend plutôt que de faire
    confiance à un fichier qui pourrait avoir été calculé pour un autre
    livre ou une autre configuration."""

    identity = {
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "zone_percent": float(zone_percent),
        "algorithm": ZONE_ALGORITHM,
        "algorithm_version": ZONE_ALGORITHM_VERSION,
        "boundary_tolerance_ratio": BOUNDARY_TOLERANCE_RATIO,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return f"layout-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _segment_boundaries(token_segment_idxs: list[int]) -> list[int]:
    return [
        i for i in range(1, len(token_segment_idxs))
        if token_segment_idxs[i - 1] != token_segment_idxs[i]
    ]


def _nearest_boundary(
    target: int, previous: int, maximum: int, boundaries: list[int], tolerance: int
) -> int:
    eligible = [
        b for b in boundaries
        if previous < b <= maximum and abs(b - target) <= tolerance
    ]
    if eligible:
        return min(eligible, key=lambda b: (abs(b - target), b))

    # Rien dans la tolérance : on élargit à N'IMPORTE QUELLE limite de
    # segment dans la plage plutôt que de couper un segment en deux — voir
    # la docstring du module, c'est la garantie testée par test_zones.py.
    wide = [b for b in boundaries if previous < b <= maximum]
    if wide:
        return min(wide, key=lambda b: abs(b - target))

    # Dernier repli : aucune limite de segment dans (previous, maximum] —
    # cas dégénéré (zone_count très proche du nombre de segments, ou un
    # segment anormalement long). On prend la prochaine limite disponible
    # au-delà de `maximum` plutôt qu'une coupe brute à `maximum`.
    beyond = [b for b in boundaries if b > previous]
    if beyond:
        return min(beyond)
    return maximum


def build_layout(
    token_segment_idxs: list[int],
    source_text: str,
    zone_percent: float = config.ZONE_PERCENT,
) -> dict:
    """Construit le layout de zones à partir de la séquence, DANS L'ORDRE DE
    LECTURE, du segment_idx de chaque token non-espace du livre (ponctuation
    comprise — point I). `source_text` sert uniquement à calculer le
    `layout_id` content-addressed (point H), pas au découpage lui-même."""

    total = len(token_segment_idxs)
    if total == 0:
        raise ValueError("impossible de construire un layout de zones sans tokens")

    layout_id = compute_layout_id(source_text, zone_percent)
    requested_zone_count = math.ceil(100.0 / zone_percent)
    zone_count = min(requested_zone_count, total)
    ideal_size = total / zone_count
    tolerance = max(1, round(ideal_size * BOUNDARY_TOLERANCE_RATIO))
    boundaries = _segment_boundaries(token_segment_idxs)

    zones: list[dict] = []
    segment_zone: dict[int, str] = {}
    start = 0
    for ordinal in range(1, zone_count + 1):
        target_end = round(ordinal * total / zone_count)
        maximum_end = total - (zone_count - ordinal)
        end = total if ordinal == zone_count else _nearest_boundary(
            target_end, start, maximum_end, boundaries, tolerance
        )
        zone_id = f"zone-{ordinal:02d}"
        seg_idxs_in_zone = token_segment_idxs[start:end]
        for seg_idx in dict.fromkeys(seg_idxs_in_zone):
            segment_zone[seg_idx] = zone_id

        zones.append({
            "zone_id": zone_id,
            "ordinal": ordinal,
            "target_start_percent": round(min((ordinal - 1) * zone_percent, 100.0), 3),
            "target_end_percent": round(100.0 if ordinal == zone_count else min(ordinal * zone_percent, 100.0), 3),
            "actual_start_percent": round(start * 100.0 / total, 3),
            "actual_end_percent": round(end * 100.0 / total, 3),
            "start_token": start,
            "end_token": end,
            "token_count": end - start,
            "start_segment_idx": seg_idxs_in_zone[0],
            "end_segment_idx": seg_idxs_in_zone[-1],
        })
        start = end

    return {
        "schema_version": "vocab-filter-zone-layout-1.0",
        "layout_id": layout_id,
        "zone_percent": zone_percent,
        "zone_count": len(zones),
        "document_token_count": total,
        "algorithm": ZONE_ALGORITHM,
        "algorithm_version": ZONE_ALGORITHM_VERSION,
        "boundary_tolerance_ratio": BOUNDARY_TOLERANCE_RATIO,
        "zones": zones,
        # Table de jointure directe segment_idx -> zone_id, ce dont
        # analyze.py/select.py ont réellement besoin (les occurrences et les
        # spans MWE ne portent qu'un segment_idx + des offsets caractère,
        # jamais un index de token global). Clés sérialisées en chaîne :
        # JSON n'a pas de clé entière.
        "segment_zone_map": {str(k): v for k, v in segment_zone.items()},
    }


def write(layout: dict) -> None:
    config.ensure_out_dir()
    atomic.atomic_write_text(
        config.ZONE_LAYOUT_PATH,
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n",
    )


def load() -> dict | None:
    if not config.ZONE_LAYOUT_PATH.exists():
        return None
    return json.loads(config.ZONE_LAYOUT_PATH.read_text(encoding="utf-8"))


def segment_zone_map(layout: dict) -> dict[int, str]:
    return {int(k): v for k, v in layout["segment_zone_map"].items()}
