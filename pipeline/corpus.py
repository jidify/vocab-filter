"""S0 — Délimiter le contenu utile et découper l'œuvre en segments.

Un `Segment` est l'unité de travail de tout le reste du pipeline : une
réplique, une didascalie, ou un paragraphe de narration, avec son texte
anglais et — quand disponible — la phrase française alignée.

Deux sources de texte anglais coexistent volontairement :

- `The Humans - Stephen Karam.txt` (présent dans le dépôt) sert de texte
  de référence pour la segmentation et les offsets ;
- le livre bilingue (fourni par l'utilisateur, absent du dépôt) apporte la
  traduction française alignée, utilisée en S5 comme preuve de sens.

Si le bilingue est absent, le pipeline segmente quand même le texte
anglais seul (`fr=None` sur chaque segment) — S5 retombe alors sur
GlossBERT seul, avec la perte de fiabilité documentée sur `view`/`butt`
dans vocab-filter-resume.md §3.5.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from pipeline import config


@dataclass
class Segment:
    idx: int
    en: str
    fr: str | None
    kind: str  # "dialogue" | "didascalie" | "hors_oeuvre"
    speaker: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Détection du hors-œuvre
#
# Repris de l'observation du résumé : le fichier brut d'un ebook peut
# contenir mentions légales, sommaire, crédits, biographie. On les
# écarte par un ensemble de motifs, complété par des bornes explicites
# en configuration quand la détection automatique échoue (la structure
# varie trop d'un ebook à l'autre pour être fiable seule).
# ------------------------------------------------------------------

FRONT_MATTER_PATTERNS = [
    r"\ball rights reserved\b",
    r"\bno part of this (?:book|publication)\b",
    r"\bphotocopying\b",
    r"\bwithout (?:the )?(?:prior )?(?:written )?permission\b",
    r"\bISBN\b",
    r"\bLibrary of Congress\b",
    r"\bpublished by\b",
    r"\bfirst (?:published|edition)\b",
    r"\bcopyright\b",
    r"^\s*table of contents\s*$",
    r"^\s*contents\s*$",
    r"^\s*cast of characters\s*$",
    r"^\s*acknowledg(e)?ments\s*$",
]

_FRONT_MATTER_RE = re.compile("|".join(FRONT_MATTER_PATTERNS), re.IGNORECASE)


def is_hors_oeuvre(line: str) -> bool:
    return bool(_FRONT_MATTER_RE.search(line))


# ------------------------------------------------------------------
# Adaptateur de source — interface commune pièce / roman / sous-titres.
# Seul PlayAdapter est implémenté pour l'instant (portée : The Humans
# d'abord, extensible ensuite — cf. le plan).
# ------------------------------------------------------------------

class SourceAdapter:
    def segments(self, path: Path) -> list[Segment]:
        raise NotImplementedError


class PlayAdapter(SourceAdapter):
    """Découpage repris de sense_in_context.py::load_source_units :
    une ligne non vide = une unité, en écartant les noms de personnage
    (lignes tout en majuscules, courtes). Ajoute la classification
    dialogue / didascalie / hors_oeuvre et le nom du personnage courant.
    """

    CHAR_NAME_MAX_LEN = 40

    def segments(self, path: Path) -> list[Segment]:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        excluded_ranges = config.FRONT_MATTER_LINE_RANGES

        segments: list[Segment] = []
        current_speaker: str | None = None
        idx = 0
        in_front_matter = True

        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            in_excluded_range = any(
                start <= line_no <= end for start, end in excluded_ranges
            )

            letters = [c for c in stripped if c.isalpha()]
            is_character_name = (
                bool(letters)
                and all(c.isupper() for c in letters)
                and len(stripped) < self.CHAR_NAME_MAX_LEN
            )

            if is_character_name and not in_excluded_range:
                current_speaker = stripped
                continue

            if in_excluded_range:
                segments.append(Segment(idx, stripped, None, "hors_oeuvre"))
                idx += 1
                continue

            # Une fois qu'on a vu une première réplique/didascalie
            # plausible, on considère qu'on est entré dans l'œuvre —
            # sinon un mot répété plus loin dans le texte (ex. une
            # note de mise en scène tardive ressemblant à un motif
            # de front-matter) ne doit plus être exclu à tort.
            if in_front_matter:
                if is_hors_oeuvre(stripped):
                    segments.append(
                        Segment(idx, stripped, None, "hors_oeuvre")
                    )
                    idx += 1
                    continue
                if len(stripped.split()) >= 4:
                    in_front_matter = False
                else:
                    continue
            elif is_hors_oeuvre(stripped):
                segments.append(Segment(idx, stripped, None, "hors_oeuvre"))
                idx += 1
                continue

            kind = "didascalie" if self._looks_like_stage_direction(stripped) else "dialogue"

            segments.append(
                Segment(idx, stripped, None, kind, speaker=current_speaker)
            )
            idx += 1

        return segments

    @staticmethod
    def _looks_like_stage_direction(text: str) -> bool:
        # Heuristique légère : parenthèse entourant tout ou partie de
        # la ligne, ou ligne sans aucune ponctuation de dialogue et
        # commençant par une majuscule suivie d'un verbe -ing courant.
        if text.startswith("(") and text.endswith(")"):
            return True
        if re.match(r"^[A-Z][a-z]+ (walks|enters|exits|sits|stands|stares|looks)\b", text):
            return True
        return False


# ------------------------------------------------------------------
# Chargement bilingue
#
# Format attendu par défaut, d'après l'exemple donné dans
# vocab-filter-resume.md §3.1 : phrase anglaise, ligne(s) vide(s),
# traduction française, ligne(s) vide(s), etc. Deux formats de repli
# sont aussi acceptés : TSV `en\tfr` par ligne, et JSONL
# `{"en": ..., "fr": ...}`. Le format réel n'a pas pu être vérifié
# (fichier absent du dépôt au moment de l'écriture) — d'où le mode
# --validate-bilingual, à lancer en premier sur le vrai fichier.
# ------------------------------------------------------------------

def find_bilingual_file() -> Path | None:
    for candidate in config.BILINGUAL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def parse_bilingual_pairs(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix == ".jsonl":
        pairs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((obj["en"], obj["fr"]))
        return pairs

    # TSV : une paire par ligne, en <TAB> fr.
    non_empty = [l for l in text.splitlines() if l.strip()]
    if non_empty and all(l.count("\t") == 1 for l in non_empty[:20]):
        pairs = []
        for line in non_empty:
            en, fr = line.split("\t", 1)
            pairs.append((en.strip(), fr.strip()))
        return pairs

    # Format par blocs séparés par des lignes vides : blocs alternés
    # EN / FR. On regroupe les lignes contiguës non vides en blocs.
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                blocks.append(" ".join(current))
                current = []
    if current:
        blocks.append(" ".join(current))

    pairs = []
    for i in range(0, len(blocks) - 1, 2):
        pairs.append((blocks[i], blocks[i + 1]))
    return pairs


_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
_FRENCH_MARKERS = re.compile(
    r"\b(le|la|les|un|une|des|est|et|de|du|qui|que|pas|c'est|il|elle)\b",
    re.IGNORECASE,
)


def validate_bilingual(path: Path) -> int:
    """Rapporte des indicateurs de qualité d'alignement AVANT toute
    autre étape. Retourne un code de sortie (0 = exploitable)."""

    pairs = parse_bilingual_pairs(path)
    n = len(pairs)
    print(f"Paires EN/FR lues : {n}")

    if n == 0:
        print("Aucune paire détectée — format non reconnu, à ajuster dans "
              "pipeline/corpus.py::parse_bilingual_pairs.")
        return 1

    empty_fr = sum(1 for _, fr in pairs if not fr.strip())
    dup_fr = n - len({fr for _, fr in pairs})
    swapped = sum(
        1 for en, fr in pairs
        if not _FRENCH_MARKERS.search(fr) and _FRENCH_MARKERS.search(en)
    )

    ratios = [
        len(fr) / max(1, len(en))
        for en, fr in pairs
        if en.strip() and fr.strip()
    ]
    aberrant = [r for r in ratios if r < 0.3 or r > 3.0]

    print(f"FR vides                 : {empty_fr} ({empty_fr/n:.1%})")
    print(f"FR dupliquées             : {dup_fr} ({dup_fr/n:.1%})")
    print(f"Paires possiblement inversées EN<->FR : {swapped}")
    print(f"Ratios longueur FR/EN aberrants (<0.3 ou >3.0) : "
          f"{len(aberrant)} ({len(aberrant)/max(1,len(ratios)):.1%})")
    print()
    print("Exemples (3 premières paires) :")
    for en, fr in pairs[:3]:
        print(f"  EN: {en}")
        print(f"  FR: {fr}")
        print()

    problems = empty_fr > n * 0.05 or swapped > n * 0.05 or len(aberrant) > n * 0.1
    if problems:
        print("!!! Le format ou l'alignement semble problématique — "
              "ajuster parse_bilingual_pairs avant de lancer le pipeline. !!!")
        return 1

    print("Alignement plausible.")
    return 0


def load_segments(book_path: Path = config.BOOK_EN_PATH) -> list[Segment]:
    """Segmente le texte anglais, puis y attache la phrase française
    alignée si un fichier bilingue est trouvé (best-effort, par
    correspondance de position séquentielle — pas de recherche floue :
    voir --validate-bilingual pour vérifier que ça a du sens sur le
    vrai fichier)."""

    adapter = PlayAdapter()
    segments = adapter.segments(book_path)

    bilingual_path = find_bilingual_file()
    if bilingual_path is not None:
        pairs = parse_bilingual_pairs(bilingual_path)
        fr_by_en = {en.strip(): fr for en, fr in pairs}
        for seg in segments:
            if seg.en in fr_by_en:
                seg.fr = fr_by_en[seg.en]

    return segments


def main() -> int:
    if "--validate-bilingual" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        path = Path(args[0]) if args else find_bilingual_file()
        if path is None or not path.exists():
            print("Fichier bilingue introuvable. Passez le chemin en argument, "
                  "ou déposez-le sous l'un des noms attendus dans config.py "
                  "(BILINGUAL_CANDIDATES).")
            return 1
        return validate_bilingual(path)

    segments = load_segments()
    with_fr = sum(1 for s in segments if s.fr)
    print(f"{len(segments)} segments ({with_fr} avec traduction FR alignée).")
    by_kind: dict[str, int] = {}
    for s in segments:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
    print(by_kind)
    return 0


run = main


if __name__ == "__main__":
    raise SystemExit(main())
