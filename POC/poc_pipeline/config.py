"""Chemins et constantes partagés par les modules `poc_pipeline` (vendorés
depuis `pipeline/` pour rendre `POC/` autonome et déplaçable — voir le plan
"Pipeline POC autonome"). Rien ici ne doit dépendre d'une bibliothèque lourde
(spaCy, torch, wn...) — ce module doit pouvoir être importé instantanément
par n'importe quel script.

Sous-ensemble volontairement réduit de `pipeline/config.py` : seules les
constantes réellement lues par la fermeture transitive vendorée (analyze,
corpus, atomic, custom_lexicon, multi_token, rules_plus, mwe, mwe_gates,
mwe_alignment, zones, llm_litellm_catgpt, vpc/*) et par les six scripts POC
sont conservées. Tout ce qui relève des étapes production non vendorées
(senses, select, export, score, sense_fr*, mwe_judge, review_ui, inventory,
lexicon, llm_client, llm_store...) a été retiré.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------

# Livre par défaut (jamais utilisé en pratique : les scripts POC reçoivent
# toujours --book explicitement — conservé uniquement parce que
# corpus.py::load_segments en fait la valeur par défaut de son paramètre).
BOOK_EN_PATH = ROOT / "books" / "The Humans - Stephen Karam.txt"

# Nombre de lignes de tête (1-indexées, dans le fichier brut, lignes vides
# comprises) à traiter comme hors-œuvre — copyright, éditeur, sommaire,
# distribution, notes de production — en plus de la détection par motifs
# (voir corpus.py::is_hors_oeuvre, la première ligne de défense, qui reste
# générique). 0 par défaut : ne s'applique à AUCUN fichier tant qu'on ne le
# demande pas explicitement (voir --skip-lines des scripts POC). Pour le
# livre complet "The Humans", la valeur historique était 182 (jusqu'aux
# épigraphes 183-211, qui appartiennent à l'œuvre et ne doivent pas être
# exclues).
FRONT_MATTER_SKIP_LINES: int = 0

# Livre bilingue (EN + FR) éventuel — fonctionnalité non utilisée par le
# POC (aucun script ne l'appelle), conservée seulement parce que
# corpus.py::load_bilingual y fait référence.
BILINGUAL_CANDIDATES: list[Path] = []

# ------------------------------------------------------------------
# Sortie (artefacts intermédiaires écrits comme effet de bord par
# analyze.py / mwe.py / zones.py — les scripts POC redirigent explicitement
# ceux qu'ils consomment ; le reste atterrit ici sans être lu ensuite)
# ------------------------------------------------------------------

OUT_DIR = ROOT / "pipeline_out"
CACHE_DIR = OUT_DIR / "cache"

LOCK_PATH = OUT_DIR / ".lock"
LOCK_STALE_SECONDS = 6 * 3600

OCCURRENCES_PATH = OUT_DIR / "occurrences.jsonl"
MULTI_TOKEN_CANDIDATES_PATH = OUT_DIR / "multi_token_candidates.jsonl"
# Lot 2 — sorties brutes du détecteur VPC, rejets inclus. Les scripts POC
# (extract_mwe_contexts.py, localize_words_and_mwe.py) patchent cette
# constante en mémoire avant d'appeler analyze_segments(), pour rediriger
# vers leur propre --vpc-candidates-out.
VPC_CANDIDATES_PATH = OUT_DIR / "vpc_candidates.jsonl"
# Idem pour rules_plus (scanner phrasal verb PARSEME+WordNet, lexique custom,
# composés nominaux WordNet) — jamais de rejet, contrairement à VPC.
RULES_PLUS_CANDIDATES_PATH = OUT_DIR / "rules_plus_candidates.jsonl"
MWE_CANDIDATES_PATH = OUT_DIR / "mwe_candidates.jsonl"
# Candidats idiomatch écartés par mwe_gates.py::classify avant jugement.
MWE_REJECTED_CANDIDATES_PATH = OUT_DIR / "mwe_rejected_candidates.jsonl"

ZONE_LAYOUT_PATH = OUT_DIR / "zone_layout.json"
ZONE_PERCENT = 5.0

# ------------------------------------------------------------------
# LLM (passerelle CatGPT-Gateway locale, pilotée par navigateur)
# ------------------------------------------------------------------

CATGPT_BASE_URL = os.getenv("CATGPT_BASE_URL", "http://localhost:8000/v1").rstrip("/")
CATGPT_API_TOKEN = os.getenv("CATGPT_API_TOKEN", "dummy123")
CATGPT_MODEL = os.getenv("CATGPT_MODEL", "catgpt-browser")
CATGPT_TIMEOUT = float(os.getenv("CATGPT_TIMEOUT", "300"))
LLM_TEMPERATURE = 0.0

# ------------------------------------------------------------------
# Résolution lemme/POS -> WordNet (repris de sense_in_context.py)
# ------------------------------------------------------------------

# PROPN -> "n" est délibéré, pas un oubli : dans un texte de théâtre, spaCy
# mal-tagge en PROPN nombre de noms communs (didascalies et répliques
# capitalisées) — les exclure ici priverait le POC de mots comme "offstage"
# ou "melee".
UPOS_TO_WN = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r", "PROPN": "n"}

# ------------------------------------------------------------------
# Magasins versionnés (data/)
# ------------------------------------------------------------------

DATA_DIR = ROOT / "poc_data"

# Lexique piloté par les données (expressions + cas de tokenisation), voir
# custom_lexicon.py. Absent par défaut dans le POC — custom_lexicon.py
# tolère son absence (load_idioms()/load_tokenizer_surfaces() retournent
# une liste vide).
CUSTOM_LEXICON_PATH = DATA_DIR / "custom_lexicon.jsonl"


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
