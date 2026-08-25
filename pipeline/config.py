"""Chemins et constantes partagés par toutes les étapes du pipeline.

Rien ici ne doit dépendre d'une bibliothèque lourde (spaCy, torch, wn...) —
ce module doit pouvoir être importé instantanément par n'importe quel script,
y compris des utilitaires de validation rapide.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------

# Texte anglais de référence (unique source de vérité pour la
# segmentation — c'est aussi le fichier que sense_in_context.py
# utilisait comme DEFAULT_SOURCE).
BOOK_EN_PATH = ROOT / "The Humans - Stephen Karam.txt"

# Bornes explicites (numéros de ligne 1-indexés, inclusifs, dans le
# fichier brut) pour le contenu hors-œuvre que la détection par motifs
# ne peut pas repérer de façon fiable : sommaire, distribution, notes
# de production. Voir corpus.py — la détection automatique par motifs
# reste la première ligne de défense ; ceci comble ses trous connus
# pour CE fichier. À ajuster pour toute autre source.
FRONT_MATTER_LINE_RANGES: list[tuple[int, int]] = [
    (1, 182),  # copyright, éditeur, catalogage, CONTENTS, PRODUCTION HISTORY,
               # distribution, DRAMATIS PERSONAE, NOTES. Les épigraphes
               # (183-211, "Think and Grow Rich" / Freud / Lorca) restent
               # hors de cette borne car elles appartiennent à l'œuvre.
]

# Livre bilingue (EN + FR), fourni par l'utilisateur, absent du dépôt
# au moment de l'écriture de ce pipeline. Voir corpus.py::load_bilingual
# et le mode `--validate-bilingual`. Le nom exact n'est pas figé : on
# essaie plusieurs candidats usuels avant d'abandonner.
BILINGUAL_CANDIDATES = [
    ROOT / "The Humans - Stephen Karam-TRAD.txt",
    ROOT / "The Humans - bilingual.txt",
    ROOT / "the-humans-bilingual.txt",
    ROOT / "bilingual.txt",
    ROOT / "livre-bilingue.txt",
]

# ------------------------------------------------------------------
# Ressources lexicales
# ------------------------------------------------------------------

CEFR_PATH = ROOT / "cefr.csv"
PREVALENCE_PATH = ROOT / "word-prevalence.txt"
AOA_PATH = ROOT / "kuperman-aoa.csv"
WONEF_PRECISION_PATH = ROOT / "wonef-precision.xml"
OMSTI_ROOT = ROOT / "one-million-sense-tagged-instances-wn30"
BNC_ROOT = ROOT / "spoken-bnc2014" / "spoken" / "tagged"

FR_LEXICON = "omw-fr:2.0"
EN_LEXICON = "omw-en:2.0"

# ------------------------------------------------------------------
# Sortie
# ------------------------------------------------------------------

OUT_DIR = ROOT / "pipeline_out"
CACHE_DIR = ROOT / "pipeline_out" / "cache"

OCCURRENCES_PATH = OUT_DIR / "occurrences.jsonl"
MWE_CANDIDATES_PATH = OUT_DIR / "mwe_candidates.jsonl"
MWE_DECISIONS_PATH = OUT_DIR / "mwe_decisions.jsonl"
MWE_SPANS_PATH = OUT_DIR / "mwe_confirmed_spans.jsonl"
SELECTED_TYPES_PATH = OUT_DIR / "selected_types.jsonl"
SELECTED_MWE_PATH = OUT_DIR / "selected_mwe.jsonl"
SENSES_PATH = OUT_DIR / "senses.jsonl"
VOCAB_CSV_PATH = OUT_DIR / "vocab.csv"
VOCAB_JSONL_PATH = OUT_DIR / "vocab.jsonl"
REVIEW_QUEUE_PATH = OUT_DIR / "review_queue.csv"
REPORT_PATH = OUT_DIR / "report.md"

# ------------------------------------------------------------------
# Seuils (S4 — porte de sélection, niveau type)
# ------------------------------------------------------------------

MIN_PKNOWN = 0.90       # plancher de validité, pas un score
MIN_NOBS = 50           # échantillon Pknown jugé suffisant
EXCLUDED_CEFR = {"A1", "A2"}

# ------------------------------------------------------------------
# LLM local (ollama)
# ------------------------------------------------------------------

OLLAMA_URL = "http://192.168.1.28:11434"
OLLAMA_MODEL = "mistral-small:24b"
# Vérifié le 2026-08-25 sur l'hôte (`ollama list`) : avec format="json",
# qwen3:14b et gpt-oss:20b renvoient un JSON vide ou du texte de raisonnement
# non structuré (probablement leur "thinking" qui interfère avec le mode
# JSON de /api/generate). mistral-small:24b et gemma3:27b obéissent
# correctement à `format: "json"` — mistral-small:24b retenu par défaut
# (bon compromis qualité/vitesse) ; gemma3:27b en repli si besoin.
LLM_TEMPERATURE = 0.0

# ------------------------------------------------------------------
# Preuve française (repris de sense_in_context.py)
# ------------------------------------------------------------------

FR_BASE_OMW = 1.0
FR_BASE_WONEF = 0.15
FR_CLAIM_DISCOUNT = 0.15
CONTEXT_WINDOW = 2  # segments voisins de chaque côté

POS_TO_UPOS = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}
POS_TO_WN = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}
UPOS_TO_WN = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r", "PROPN": "n"}

CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
