"""Chemins et constantes partagés par toutes les étapes du pipeline.

Rien ici ne doit dépendre d'une bibliothèque lourde (spaCy, torch, wn...) —
ce module doit pouvoir être importé instantanément par n'importe quel script,
y compris des utilitaires de validation rapide.
"""

import os
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

# Lot 0 — verrou de run (voir pipeline/atomic.py::run_lock). Un verrou
# plus vieux que ce délai est considéré abandonné plutôt que bloquant :
# Windows ne permet pas de vérifier fiablement qu'un PID est encore vivant.
LOCK_PATH = OUT_DIR / ".lock"
LOCK_STALE_SECONDS = 6 * 3600  # 6h : plus long que le run complet le plus lent observé (senses)

OCCURRENCES_PATH = OUT_DIR / "occurrences.jsonl"
MULTI_TOKEN_CANDIDATES_PATH = OUT_DIR / "multi_token_candidates.jsonl"
# Lot 2 — sorties brutes du détecteur VPC (pipeline/vpc/), rejets inclus.
# Fusion avec idiomatch (pipeline/mwe.py) : Lot 3 (voir mwe.py::load_vpc_candidates).
VPC_CANDIDATES_PATH = OUT_DIR / "vpc_candidates.jsonl"
# Candidats "rules_plus" (Q0-3 Phase 6 — voir fix_pipeline/detection_benchmark/
# phase6_decision.md) : scanner phrasal verb PARSEME+WordNet, rejeu du
# lexique custom, composés nominaux WordNet — écrits par analyze.py dans la
# même boucle nlp.pipe que VPC, fusionnés par mwe.py::load_rules_plus_candidates.
# Jamais de rejet (contrairement à VPC_CANDIDATES_PATH) : rules_plus ne
# produit que des candidats, aucune décision "rejected_*".
RULES_PLUS_CANDIDATES_PATH = OUT_DIR / "rules_plus_candidates.jsonl"
MWE_CANDIDATES_PATH = OUT_DIR / "mwe_candidates.jsonl"
MWE_DECISIONS_PATH = OUT_DIR / "mwe_decisions.jsonl"
MWE_SPANS_PATH = OUT_DIR / "mwe_confirmed_spans.jsonl"
RESERVATION_REPORT_PATH = OUT_DIR / "reservation_report.json"
SELECTED_TYPES_PATH = OUT_DIR / "selected_types.jsonl"
SELECTED_MWE_PATH = OUT_DIR / "selected_mwe.jsonl"
# Lot 3 — inventaire lexical figé (plan Partie 2, point E) : une ligne par
# occurrence RETENUE (mot simple ou MWE), écrite par select.py::run() une
# fois les spans MWE réservés appliqués. inventory.sha256 en est le hash
# (pipeline/inventory.py) — toute étape à partir de senses le vérifie au
# démarrage pour ne jamais mélanger deux inventaires silencieusement.
LEXICAL_INVENTORY_PATH = OUT_DIR / "lexical_inventory.jsonl"
INVENTORY_HASH_PATH = OUT_DIR / "inventory.sha256"
# Lot 5 — layout de zones (plan Partie 2, point H) : régénérable, donc sous
# pipeline_out/ et pas data/. Recalculé à chaque run d'analyze.py, toujours
# sur le livre entier (voir pipeline/zones.py et Partie 3 du plan).
ZONE_LAYOUT_PATH = OUT_DIR / "zone_layout.json"
ZONE_PERCENT = 5.0
# Sidecar écrit par senses.py à la fin d'un run réussi : quel inventory.sha256
# a produit CE senses.jsonl. Les étapes avale (sense_fr_frontier,
# sense_fr_adjudicate, export) le comparent à INVENTORY_HASH_PATH courant
# avant de faire confiance à senses.jsonl (pipeline/inventory.py::verify_consumer).
SENSES_INVENTORY_HASH_PATH = OUT_DIR / "senses.inventory.sha256"
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
# LLM (Ollama local ou CatGPT-Gateway)
# ------------------------------------------------------------------

LLM_BACKEND = os.getenv("VOCAB_LLM_BACKEND", "ollama").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.1.28:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-small:24b")
CATGPT_BASE_URL = os.getenv("CATGPT_BASE_URL", "http://localhost:8000/v1").rstrip("/")
CATGPT_API_TOKEN = os.getenv("CATGPT_API_TOKEN", "dummy123")
CATGPT_MODEL = os.getenv("CATGPT_MODEL", "catgpt-browser")
CATGPT_TIMEOUT = float(os.getenv("CATGPT_TIMEOUT", "300"))
# Vérifié le 2026-08-25 sur l'hôte (`ollama list`) : avec format="json",
# qwen3:14b et gpt-oss:20b renvoient un JSON vide ou du texte de raisonnement
# non structuré (probablement leur "thinking" qui interfère avec le mode
# JSON de /api/generate). mistral-small:24b et gemma3:27b obéissent
# correctement à `format: "json"` — mistral-small:24b retenu par défaut
# (bon compromis qualité/vitesse) ; gemma3:27b en repli si besoin.
LLM_TEMPERATURE = 0.0
S3_JUDGE_BATCH_SIZE = 50  # validé séparément sur le corpus contrastif S3 ; jamais lancé en arrière-plan


def configure_llm(*, backend: str | None = None, base_url: str | None = None,
                  api_token: str | None = None, model: str | None = None,
                  timeout: float | None = None) -> None:
    """Applique les options LLM globales du CLI (run_pipeline.py --llm-backend/
    --llm-model/...) avant l'import des étapes.

    Périmètre limité : seul le backend GLOBAL de repli change (LLM_BACKEND/
    OLLAMA_MODEL/CATGPT_MODEL), consommé par pipeline.llm_tasks.task_config()
    pour S3-judge-occurrence, S3-definition-cluster, S5-arbitrate,
    S6-translate-local, S6-backtranslate-local (descripteurs
    global_model_fallback=True) tant qu'aucun VOCAB_LLM_<TASK_ID> dédié n'est
    posé. Les 4 tâches S6 routées par LiteLLM (S6-translate-frontier,
    S6-backtranslate, S6-judge-dossier, S6-reassign) ignorent totalement ces
    valeurs : leur modèle se règle uniquement via VOCAB_LLM_S6_* (voir le
    tableau des task_id dans README.md et fix_pipeline/multi_models/
    plan_multi_models.md §3.3-3.4)."""
    global LLM_BACKEND, OLLAMA_URL, OLLAMA_MODEL
    global CATGPT_BASE_URL, CATGPT_API_TOKEN, CATGPT_MODEL, CATGPT_TIMEOUT
    if backend is not None:
        if backend not in {"ollama", "catgpt"}:
            raise ValueError(f"backend LLM inconnu : {backend}")
        LLM_BACKEND = backend
    if base_url is not None:
        if LLM_BACKEND == "catgpt":
            CATGPT_BASE_URL = base_url.rstrip("/")
        else:
            OLLAMA_URL = base_url.rstrip("/")
    if api_token is not None:
        CATGPT_API_TOKEN = api_token
    if model is not None:
        if LLM_BACKEND == "catgpt": CATGPT_MODEL = model
        else: OLLAMA_MODEL = model
    if timeout is not None:
        CATGPT_TIMEOUT = timeout


def llm_model() -> str:
    return CATGPT_MODEL if LLM_BACKEND == "catgpt" else OLLAMA_MODEL

# ------------------------------------------------------------------
# Preuve française (repris de sense_in_context.py)
# ------------------------------------------------------------------

FR_BASE_OMW = 1.0
FR_BASE_WONEF = 0.15
FR_CLAIM_DISCOUNT = 0.15
CONTEXT_WINDOW = 2  # segments voisins de chaque côté

POS_TO_UPOS = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}
POS_TO_WN = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}
# PROPN -> "n" est délibéré, pas un oubli : dans un texte de théâtre,
# spaCy mal-tagge en PROPN nombre de noms communs (didascalies et
# répliques capitalisées) — les exclure ici priverait le pipeline de
# mots comme "offstage" ou "melee". Le filtrage des VRAIES entités
# nommées se fait en aval, où l'information est plus fiable :
# select.py::is_likely_named_entity (au niveau du TYPE, avant S5) et
# score.py::is_named_entity_sense (au niveau du SENS retenu, via
# WordNet instance_hypernyms — le signal qui distingue "Scranton" de
# "melee" ou "offstage", contrairement au tag spaCy seul).
UPOS_TO_WN = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r", "PROPN": "n"}

CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

# ------------------------------------------------------------------
# Traduction française de référence, indexée par sense_id (S6b)
# ------------------------------------------------------------------

# Magasin permanent, versionné dans git (contrairement à pipeline_out/,
# régénéré à chaque run) : une ligne par sens, réutilisée d'un livre à
# l'autre. Voir pipeline/sense_fr.py.
DATA_DIR = ROOT / "data"
SENSE_FR_STORE_PATH = DATA_DIR / "sense_fr.jsonl"
SENSE_FR_LOCK_PATH = DATA_DIR / "sense_fr.lock.json"

# Corrections manuelles d'occurrences mal groupées par S5 (bug d'ingestion —
# tokenisation, MWE non détectée...), appliquées à l'export sans rejouer
# S1-S5 — voir pipeline/score.py::build_records() et le plan du 2026-08-27
# "Correction manuelle smart-ass / e-mail sans re-run complet".
MANUAL_CORRECTIONS_PATH = DATA_DIR / "manual_corrections.jsonl"

# Lexique piloté par les données (expressions + cas de tokenisation ajoutés
# depuis pipeline/review_ui.py sans éditer de code) — voir
# pipeline/custom_lexicon.py et le plan du 2026-08-27 "IHM de correction
# manuelle : plusieurs workflows, lexique piloté par les données".
CUSTOM_LEXICON_PATH = DATA_DIR / "custom_lexicon.jsonl"

# Lot 3 — magasin MWE à deux niveaux (plan Partie 2, point C), même modèle
# que SENSE_FR_STORE_PATH : permanents, versionnés, réutilisés d'un livre à
# l'autre, jamais écrasés pour une entrée `status: validated`. Voir
# pipeline/mwe_judge.py.
MWE_TYPE_STORE_PATH = DATA_DIR / "mwe_type_decisions.jsonl"       # clé = idiome
MWE_OCCURRENCE_STORE_PATH = DATA_DIR / "mwe_occurrence_decisions.jsonl"  # clé = occurrence_id

# wonef-precision.xml (voir WONEF_PRECISION_PATH ci-dessus) est absent
# du dépôt : seule la variante f-score, compressée, est présente.
WONEF_FSCORE_PATH = ROOT / "wonef-fscore.xml.bz2"

SENSE_FR_REVIEW_PATH = OUT_DIR / "sense_fr_review.csv"

# Page HTML autonome (pas de serveur, pas de dépendance externe) générée
# par pipeline/review_ui.py à partir de SENSE_FR_REVIEW_PATH : liste
# déroulante des sense_id WordNet par mot (avec définition), pour remplir
# `reassigner_vers` sans jamais avoir à connaître/taper un code — voir le
# plan du 2026-08-27 "Une page HTML locale pour choisir le sens WordNet
# dans une liste".
REVIEW_UI_PATH = OUT_DIR / "review_ui.html"

# Port par défaut du petit serveur local (stdlib, 127.0.0.1 uniquement)
# lancé par `uv run python -m pipeline.review_ui` — voir sa docstring.
REVIEW_UI_PORT = 8765

# sense_id que la passe contextuelle (pipeline/sense_fr_frontier.py) juge
# suspect ou douteux au vu des phrases réelles du livre. Consommé par
# pipeline/sense_fr_reassign.py (S6c), qui rouvre POS/sense_id à inventaire
# WordNet OUVERT sur ces entrées précises — voir sa docstring. S5
# (pipeline/senses.py) continue d'ignorer ce fichier : sans boucle jusque-là,
# le même mauvais sense_id reviendrait à l'identique au prochain livre.
SENSE_ID_SUSPECTS_PATH = OUT_DIR / "sense_id_suspects.csv"

# Propositions de réassignation POS/sense_id que S6c n'a pas pu promouvoir
# automatiquement (changement de POS, ou aucun sense_id WordNet exact) —
# relecture humaine, voir pipeline/sense_fr_reassign.py.
SENSE_ID_REASSIGN_PATH = OUT_DIR / "sense_id_reassignments.csv"

# Nombre de formulations de prompt distinctes essayées pour la
# traduction "de dictionnaire" (sans contexte de livre) d'un sens —
# LLM_TEMPERATURE=0.0 rend le cache de llm.py déterministe par prompt
# exact ; on varie donc le PHRASING plutôt que la température pour
# obtenir plusieurs tirages. Ce ne sont pas des tirages statistiquement
# indépendants (même modèle, même poids) : uniquement un filtre de
# cohérence interne, jamais une "source" au sens du plan §5.5.
SENSE_FR_LLM_DRAWS = 3
SENSE_FR_LLM_MIN_AGREE = 2  # sur SENSE_FR_LLM_DRAWS, pour retenir un consensus LLM

# ------------------------------------------------------------------
# Traduction par modèle frontière (pipeline/sense_fr_frontier.py) — via
# LiteLLM, passe PRIMAIRE et CONTEXTUELLE de S6b (remplace le chemin ollama
# local ci-dessus, qui traduisait glose seule ; l'ancienne passe contextuelle
# séparée, pipeline/sense_fr_context.py, est fusionnée ici — voir le plan
# "S6b : rendre la passe primaire contextuelle").
# ------------------------------------------------------------------

# Préfixé par le fournisseur (voir litellm) : "anthropic/...", "openai/...".
# Paramétrable via --model. gpt-5-mini par défaut (choix de l'utilisateur) —
# déjà le modèle réellement utilisé pour l'ancienne passe contextuelle
# (context_evidence.model dans data/sense_fr.jsonl avant cette fusion).
SENSE_FR_FRONTIER_MODEL = "openai/gpt-5-mini"
SENSE_FR_FRONTIER_BATCH_SIZE = 40   # sens par appel — items alourdis par les phrases + candidats
SENSE_FR_FRONTIER_MAX_WORKERS = 10  # lots traités en parallèle (litellm.batch_completion)
SENSE_FR_FRONTIER_MAX_OCCURRENCES = 2   # phrases distinctes présentées par sens (463/900 sens
                                         # du magasin actuel n'en ont de toute façon qu'une seule)

# Taille de lot pour S6c (pipeline/sense_fr_reassign.py) — DÉLIBÉRÉMENT
# distincte de SENSE_FR_FRONTIER_BATCH_SIZE. Un run réel avec batch_size=40
# (24 entrées envoyées en un seul appel) a produit des réassignations fausses
# (sense_id valide mais dont la définition ne correspond pas à la traduction
# produite — ex. beat.n.08 -> beat.n.06 "the sound of stroke or blow" pour
# fr="petite pause") ; les MÊMES entrées, rejouées seules dans un lot de 6,
# ont donné les bonnes réponses (dont sense_id=null, correct, pour ce cas
# précis). Sans temperature=0 possible sur la famille GPT-5 (voir
# OLLAMA_MODEL plus haut), un lot plus petit — celui réellement validé par
# eval_frontier_ablation.run_joint (batch_size=10) — est la seule protection
# mesurée contre cette dégradation.
SENSE_FR_REASSIGN_BATCH_SIZE = 10

# Taille de lot pour Stage B (pipeline/sense_fr_adjudicate.py::run_stage_b) —
# remontée en constante lors du branchement multi-modèles (Lot M2) : c'était
# auparavant un littéral de signature (`batch_size: int = 40`), jamais une
# valeur configurable ni documentée ici. Valeur inchangée (voir
# fix_pipeline/multi_models/baseline_batch_inventory.md §7).
SENSE_FR_BACKTRANSLATE_BATCH_SIZE = 40

# Plafond de lot pour Stage C (pipeline/sense_fr_adjudicate.py::run_stage_c).
# Contrairement à SENSE_FR_REASSIGN_BATCH_SIZE, ce n'est PAS une valeur
# mesurée : avant le Lot M2, `run_stage_c` acceptait un paramètre
# `batch_size` jamais utilisé pour découper — tout `residual` partait dans
# un unique appel, sans plafond réel (voir baseline_batch_inventory.md §7).
# 20 est le défaut de registre choisi pour appliquer enfin un découpage
# effectif, pas la reprise d'un comportement déjà validé.
SENSE_FR_JUDGE_BATCH_SIZE = 20

# Liste blanche des modèles autorisés pour un appel frontière/conjoint (S6b
# pipeline/sense_fr_frontier.py, S6c pipeline/sense_fr_reassign.py). Un seul
# élément aujourd'hui, volontairement : le but n'est pas de choisir parmi
# plusieurs modèles interchangeables au moment de l'appel, c'est d'empêcher
# qu'un --model tapé de travers (ou un défaut qu'on aurait oublié de changer)
# ne déclenche SILENCIEUSEMENT un modèle plus coûteux que celui budgété.
# Pour utiliser un autre modèle : l'ajouter ici explicitement — un changement
# visible dans un diff, jamais un override qui passe inaperçu en ligne de
# commande. Ne s'applique volontairement PAS au modèle JUGE du benchmark
# (pipeline/eval_frontier_ablation.py:DEFAULT_JUDGE_MODEL), qui doit rester
# indépendant du modèle candidat par construction.
ALLOWED_FRONTIER_MODELS = {SENSE_FR_FRONTIER_MODEL}


def require_frontier_model(model: str, task_id: str | None = None) -> None:
    if task_id is not None:
        from pipeline.llm_tasks import task_config
        allowed = {task_config(task_id).model}
        if model not in allowed:
            raise SystemExit(
                f"Modèle '{model}' non autorisé pour la tâche '{task_id}' — "
                f"seul le modèle résolu par task_config est accepté : {sorted(allowed)}. "
                f"Pour en utiliser un autre, pose VOCAB_LLM_{task_id.replace('-', '_')}"
                "=provider/nom;... (voir pipeline/llm_tasks.py)."
            )
        return
    allowed = ALLOWED_FRONTIER_MODELS
    if model not in allowed:
        raise SystemExit(
            f"Modèle '{model}' non autorisé pour un appel frontière/conjoint — "
            f"liste blanche actuelle : {sorted(allowed)}. "
            f"Pour l'utiliser sciemment, ajoute-le à config.ALLOWED_FRONTIER_MODELS."
        )


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
