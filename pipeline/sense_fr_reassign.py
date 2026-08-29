"""S6c — Réassignation POS/sense_id sur les `pending` STRUCTURELS (voir le
plan "S6c — Décision conjointe POS/sens/traduction sur les pending
structurels").

Constat qui motive ce module : le modèle frontière (pipeline/sense_fr_frontier.py)
comprend souvent correctement l'usage réel d'une occurrence, mais son contrat
lui interdit de corriger la clé imposée par S5 — il doit recopier le sense_id
à l'identique (voir son SYSTEM_PROMPT). Quand S5 s'est trompé de sens, ou de
catégorie grammaticale (figée encore plus en amont, par le tagger de
pipeline/analyze.py), le modèle ne peut que le SIGNALER via `sense_fit`
("mismatch"/"doubtful") ou `translation_type` ("reformulation"/
"explicitation"), jamais le corriger. Exemple mesuré : "Small beat." (didascalie
théâtrale) — le mot est tagué VERBE en amont, S5 choisit beat.v.04 ("battre
rythmiquement"), le modèle traduit correctement "bref silence" mais ne peut pas
dire que l'usage réel est un NOM (comme beat.n.08, vu ailleurs dans le même
livre) sans sens WordNet exact.

Le benchmark `pipeline/eval_frontier_ablation.py` (150 cas, juge à l'aveugle,
deux passes) mesure le coût de ce blocage et le gain d'une décision conjointe
POS+sens+traduction à INVENTAIRE WORDNET OUVERT (toutes catégories confondues,
pas seulement celle assignée par S5) : sur les entrées réellement `pending`
structurelles, 37,9 % de réussite pour le pipeline actuel contre 79,3 % pour
la décision conjointe. Sur les strates déjà `auto_strong`/`auto_llm` en
revanche, le même benchmark montre une DÉGRADATION (ex. 95,0% -> 90,0% sur
auto_strong) : livré à lui-même, le modèle sur-interprète parfois un cas déjà
simple et déjà juste (3 cas mesurés : "subatomic", "literally", "free/release").
Ce module reste donc volontairement borné aux `pending` dont l'agreement
signale un problème de CLÉ — jamais aux strates auto_*, jamais à
`frontier_desaccord` (gain mesuré trop faible) ni `frontier_sans_ressource`
(voir STRUCTURAL_AGREEMENTS ci-dessous).

Trois issues possibles par entrée traitée, selon ce que propose le modèle :
- clé inchangée (expression figée `mwe:...`, qui ne se re-clé JAMAIS — sa clé
  ne dépend d'aucun synset ; ou synset confirmé par le modèle lui-même) ->
  promotion directe en `auto_joint`, sous la MÊME clé ;
- sense_id différent mais MÊME catégorie grammaticale -> une entrée `auto_joint`
  est écrite sous la clé corrigée (fusion des occurrences si elle existe déjà,
  sauf si elle est verrouillée — voir verify_fr_lock.LOCKED_STATUSES, jamais
  écrasée) ; l'entrée d'origine reste `pending`, agreement
  `reassigne_vers:<clé>` (pour ne pas la retraiter indéfiniment) ;
  jamais appliqué à une expression figée (`kind == "mwe"`) ;
- catégorie grammaticale différente, ou aucun sense_id WordNet exact
  (`sense_id=null`), ou cible verrouillée déjà occupée -> le magasin n'est PAS
  modifié, une ligne est écrite dans sense_id_reassignments.csv pour relecture
  humaine. Changer de catégorie grammaticale a un vrai contrecoup en aval —
  pipeline/score.py indexe les types de vocabulaire par (lemme, POS) et
  n'associe une occurrence qu'aux candidats scorés par S5 pour SA POS
  d'origine (pipeline_out/senses.jsonl n'est jamais réécrit par ce module).

Limite connue : la décision est prise PAR sense_id (toutes les occurrences déjà
regroupées par S5/S6b sous cette clé), pas par occurrence individuelle. Un sens
dont les occurrences divergent réellement recevrait un verdict unique — hors
périmètre v1 (supposerait de réécrire l'agrégation de sense_fr.collect_targets).

Usage :
    uv run python -m pipeline.sense_fr_reassign
    uv run python -m pipeline.sense_fr_reassign --dry-run
    uv run python -m pipeline.sense_fr_reassign --model anthropic/claude-haiku-4-5
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import litellm
from nltk.corpus import wordnet as nwn
from pydantic import BaseModel

from pipeline import config, inventory as lexical_inventory, senses, sense_fr, verify_fr_lock
from pipeline.llm_litellm_catgpt import call_kwargs as catgpt_call_kwargs
from pipeline.llm_tasks import effective_batch_size, task_config, use_batch_prompt

# ============================================================
# Périmètre — voir la docstring du module pour la justification de
# chaque exclusion (frontier_sans_ressource, frontier_desaccord, auto_*).
# ============================================================

STRUCTURAL_AGREEMENTS = {
    "sense_id_suspect", "sense_id_douteux", "frontier_reformulation", "frontier_explicitation",
}


def select_targets(store: dict[str, dict]) -> list[dict]:
    return sorted(
        (e for e in store.values()
         if e.get("status") == "pending" and e.get("agreement") in STRUCTURAL_AGREEMENTS),
        key=lambda e: e["key"],
    )


# ============================================================
# Inventaire WordNet OUVERT (toutes POS confondues) — reprend le principe
# de eval_frontier_ablation._inventory, sans dépendre de ce module de
# benchmark (volontairement isolé de la production, voir sa docstring).
# ============================================================


def open_inventory(lemma: str) -> list[dict]:
    lookup = lemma.strip().replace(" ", "_")
    found: dict[str, dict] = {}
    for synset in nwn.synsets(lookup):
        found[synset.name()] = {
            "sense_id": synset.name(), "pos": synset.pos(), "definition": synset.definition(),
            # Lemmes du synset (ex. ass.n.02 -> ["ass"], arsenic.n.02 ->
            # ["arsenic", "As", "atomic_number_33"]) : ajout PUR, S6c ne lit
            # que sense_id/pos/definition ci-dessus, le prompt du modèle
            # n'est pas affecté. Sert pipeline/review_ui.py à distinguer un
            # vrai sens du lemme demandé d'un match morphy fortuit (ex.
            # nwn.synsets("ass") -> "as" -> arsenic.n.02/american_samoa.n.01,
            # dont AUCUN lemme n'est "ass" — voir le plan du 2026-08-27).
            "lemmas": [l.name() for l in synset.lemmas()],
        }
    return sorted(found.values(), key=lambda x: (x["pos"], x["sense_id"]))


def search(query: str, limit: int = 40) -> list[dict]:
    """Recherche WordNet libre, TOUT lemme et TOUTE catégorie grammaticale
    confondus — sert pipeline/review_ui.py workflow C (« la cible n'est pas
    le même mot que celui affiché », ex. taper "e-mail" depuis la carte de
    "mail"). Contrairement à open_inventory (un seul lemme, lookup exact
    WordNet/morphy), ceci accepte une sous-chaîne : `nwn.synsets(q)` seul
    ne trouverait pas "e-mail" en tapant "mail".

    Balayer les ~117 000 synsets à chaque frappe serait coûteux ; on tente
    d'abord le lookup exact (rapide, couvre l'usage courant : taper le mot
    exact), et on ne bascule sur le balayage complet que s'il ne suffit
    pas — mesuré à environ 1s sur cette machine, acceptable pour une
    recherche déclenchée par un clic "chercher", pas à chaque frappe."""
    q = query.strip().lower()
    if len(q) < 2:
        return []
    q_lookup = q.replace(" ", "_")
    found: dict[str, dict] = {}

    def _add(synset) -> None:
        found[synset.name()] = {
            "sense_id": synset.name(), "pos": synset.pos(), "definition": synset.definition(),
            "lemmas": [l.name() for l in synset.lemmas()],
        }

    for synset in nwn.synsets(q_lookup):
        _add(synset)

    if len(found) < limit:
        for synset in nwn.all_synsets():
            if synset.name() in found:
                continue
            if any(q in l.name().lower().replace("_", " ") for l in synset.lemmas()):
                _add(synset)
                if len(found) >= limit:
                    break

    return sorted(found.values(), key=lambda x: (x["pos"], x["sense_id"]))[:limit]


# ============================================================
# Schéma de sortie structurée
# ============================================================


class ReassignedDecision(BaseModel):
    key: str
    pos: Literal["n", "v", "a", "s", "r", "mwe", "other"]
    sense_id: str | None
    fr: list[str]  # 1 à 3 propositions, triées par fréquence d'usage RÉELLE
    translation_type: Literal["equivalence_directe", "reformulation", "explicitation"]
    # S6-1 : ce module peut CONFIRMER une clé "mwe:..." (jamais re-clée,
    # voir classify_decision) sans jamais toucher sa definition_en — sans
    # ce champ, une definition_en fausse laissée en l'état par une passe
    # précédente pouvait être verrouillée en `auto_joint` aux côtés d'une
    # traduction qui la contredit ouvertement (cas réel : "give out"
    # défini comme "to announce" mais traduit "tomber en panne", "turn
    # off" défini comme "to dismiss" mais traduit "éteindre" — voir
    # data/sense_fr.jsonl avant correction). Même sémantique que
    # sense_fr_frontier.SenseTranslation.sense_fit : la définition
    # AFFICHÉE (fiable ou non) correspond-elle à l'usage réel montré par
    # les phrases fournies ?
    sense_fit: Literal["ok", "doubtful", "mismatch"]
    sense_fit_note: str
    confidence: Literal["high", "medium", "low"]
    reason: str  # justification courte pour l'audit humain, jamais affichée comme traduction


class ReassignBatch(BaseModel):
    decisions: list[ReassignedDecision]


class UnitReassignedDecision(ReassignedDecision):
    """Réponse d'un prompt unitaire (une décision scalaire)."""


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = (
    "Tu es lexicographe bilingue anglais-français. Chaque entrée qu'on te "
    "donne a déjà été signalée par une passe précédente comme sense_id ou "
    "catégorie grammaticale (POS) DOUTEUX ou INCORRECT au vu de l'usage réel "
    "— ne fais PAS confiance au POS ni au sense_id affichés comme "
    "\"assignation actuelle\", ils ne sont pas fiables.\n\n"
    "Pour un mot (pas une expression figée), tu reçois l'INVENTAIRE COMPLET "
    "des synsets WordNet du lemme, TOUTES catégories grammaticales confondues "
    "(nom, verbe, adjectif, adverbe) : choisis toi-même, à partir des phrases "
    "réelles fournies, la catégorie et le sense_id qui décrivent vraiment "
    "l'usage — jamais celui affiché comme actuel par défaut. Si aucun sens de "
    "l'inventaire ne correspond (expression figée non couverte par WordNet, "
    "sens manquant), réponds sense_id=null et pos décrivant quand même la "
    "catégorie grammaticale réelle de l'usage.\n\n"
    "Pour une expression figée (clé \"mwe:...\"), aucun inventaire WordNet "
    "n'existe et sa clé ne peut pas changer : réponds toujours pos=\"mwe\" et "
    "sense_id=null, et concentre-toi sur la traduction.\n\n"
    "Pour CHAQUE clé reçue, réponds avec :\n"
    "- pos : catégorie grammaticale RÉELLE de l'usage (n/v/a/s/r/mwe/other) ;\n"
    "- sense_id : un identifiant EXACTEMENT recopié de l'inventaire fourni "
    "pour ce lemme (jamais inventé), ou null si aucun ne convient ou si la "
    "clé est une expression figée ;\n"
    "- fr : 1 à 3 traductions françaises COURTES de cet usage précis (jamais "
    "une glose ni une explication longue), TRIÉES PAR FRÉQUENCE D'USAGE "
    "RÉELLE — la plus courante d'abord ;\n"
    "- translation_type : \"equivalence_directe\" si fr[0] est un vrai "
    "équivalent lexical substituable, \"reformulation\" si le passage "
    "condense/déplace l'information au point qu'aucun mot isolé ne "
    "correspond vraiment, \"explicitation\" si la traduction ajoute une "
    "précision nécessaire absente de l'anglais ;\n"
    "- sense_fit : la \"définition actuelle\" affichée (elle-même signalée "
    "PAS FIABLE) correspond-elle vraiment à l'usage montré par les phrases "
    "fournies ? \"ok\" si oui, \"doubtful\" si c'est limite, \"mismatch\" si "
    "les phrases montrent clairement un usage que cette définition ne "
    "décrit pas — y compris pour une expression figée (\"mwe:...\") dont la "
    "clé ne peut pas changer : dans ce cas ne masque jamais le problème en "
    "traduisant quand même correctement, signale-le, un relecteur doit "
    "corriger la définition séparément ;\n"
    "- sense_fit_note : une phrase courte justifiant sense_fit (chaîne vide "
    "si \"ok\" et évident) ;\n"
    "- confidence : \"low\" si le choix reste incertain même après relecture "
    "des phrases ;\n"
    "- reason : une phrase courte justifiant le choix de POS/sense_id, pour "
    "un relecteur humain — jamais affichée comme traduction.\n\n"
    "Renvoie EXACTEMENT une décision par clé reçue, avec key recopiée à "
    "l'identique."
)

POS_LABELS = {"n": "nom", "v": "verbe", "a": "adjectif", "s": "adjectif", "r": "adverbe", "mwe": "expression"}

ITEM_OCCURRENCE = '    contexte : "{context}" || mot cible dans ce contexte : "{target_surface}"'


def _format_item(entry: dict, occurrences: list[dict], inventory: list[dict]) -> str:
    current_pos_label = POS_LABELS.get(entry.get("pos") or "mwe", entry.get("pos") or "?")
    current_sense = entry["key"] if entry["kind"] == "synset" else "aucun (expression figée)"
    lines = [
        f"- {entry['key']} | lemme(s)_en : {'/'.join(entry.get('lemmas_en', []))} | "
        f"assignation actuelle (PAS FIABLE) : POS={current_pos_label}, sense_id={current_sense} "
        f"| définition actuelle : {entry.get('definition_en') or '?'}"
    ]
    if entry["kind"] == "mwe":
        lines.append("    (expression figée : pas d'inventaire WordNet — pos=\"mwe\", sense_id=null attendus)")
    else:
        for cand in inventory:
            lines.append(
                f"    candidat : {cand['sense_id']} | {POS_LABELS.get(cand['pos'], cand['pos'])} | "
                f"{cand['definition']}"
            )
    for occ in occurrences:
        lines.append(ITEM_OCCURRENCE.format(context=occ["context"], target_surface=occ["target_surface"]))
    return "\n".join(lines)


def build_unit_user_prompt(item: tuple[dict, list[dict], list[dict]]) -> str:
    entry, occurrences, inventory = item
    return "Entrées à réassigner (1) :\n" + _format_item(entry, occurrences, inventory) + (
        "\nRéponds avec un objet JSON unique pour cette key (pas de liste)."
    )


def build_user_prompt(batch: list[tuple[dict, list[dict], list[dict]]]) -> str:
    items = "\n".join(_format_item(entry, occs, inventory) for entry, occs, inventory in batch)
    return f"Entrées à réassigner ({len(batch)}) :\n{items}"


# ============================================================
# Cache disque (même principe que sense_fr_frontier.py, préfixe dédié)
# ============================================================


def _cache_path(model: str, system: str, user: str, *, mode_batch: bool = True, batch_size: int | None = None) -> Path:
    cache_key = json.dumps({"task_id": "S6-reassign", "model": model,
                            "mode_batch": mode_batch, "batch_size": batch_size or (1 if not mode_batch else 0),
                            "system": system, "user": user}, sort_keys=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    config.ensure_out_dir()
    return config.CACHE_DIR / f"reassign_{digest}.json"


def _translate_batches(
    batches: list[list[tuple[dict, list[dict], list[dict]]]], model: str,
    *, mode_batch: bool = True, batch_size: int | None = None,
) -> tuple[list[dict[str, ReassignedDecision]], float]:
    to_call: list[tuple[int, list[tuple[dict, list[dict], list[dict]]]]] = []
    results: list[dict[str, ReassignedDecision] | None] = [None] * len(batches)
    total_cost = 0.0

    for i, batch in enumerate(batches):
        if not mode_batch and len(batch) != 1:
            raise ValueError(
                f"S6-reassign: mode unitaire attend exactement 1 item par lot, reçu {len(batch)}"
            )
        user_prompt = build_user_prompt(batch) if mode_batch else build_unit_user_prompt(batch[0])
        cache_file = _cache_path(model, SYSTEM_PROMPT, user_prompt, mode_batch=mode_batch, batch_size=batch_size)
        if cache_file.exists():
            if mode_batch:
                parsed = ReassignBatch.model_validate_json(cache_file.read_text(encoding="utf-8"))
                results[i] = {d.key: d for d in parsed.decisions}
            else:
                parsed = UnitReassignedDecision.model_validate_json(cache_file.read_text(encoding="utf-8"))
                results[i] = {parsed.key: parsed}
        else:
            to_call.append((i, batch))

    if to_call:
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(batch) if mode_batch else build_unit_user_prompt(batch[0])},
            ]
            for _, batch in to_call
        ]
        responses = litellm.batch_completion(
            model=model, messages=messages, response_format=ReassignBatch if mode_batch else UnitReassignedDecision,
            reasoning_effort="low", max_tokens=16000,
            max_workers=config.SENSE_FR_FRONTIER_MAX_WORKERS,
            **catgpt_call_kwargs(model),
        )
        for (i, batch), response in zip(to_call, responses):
            if isinstance(response, Exception):
                print(f"  lot {i}: échec ({response!r}), {len(batch)} entrée(s) laissée(s) de côté.")
                results[i] = {}
                continue
            try:
                total_cost += litellm.completion_cost(completion_response=response)
            except Exception:
                pass
            content = response.choices[0].message.content
            parsed = (ReassignBatch if mode_batch else UnitReassignedDecision).model_validate_json(content)
            cache_file = _cache_path(model, SYSTEM_PROMPT,
                                     build_user_prompt(batch) if mode_batch else build_unit_user_prompt(batch[0]),
                                     mode_batch=mode_batch, batch_size=batch_size)
            cache_file.write_text(parsed.model_dump_json(), encoding="utf-8")
            results[i] = ({d.key: d for d in parsed.decisions} if mode_batch
                          else {parsed.key: parsed})

    return [r or {} for r in results], total_cost


# ============================================================
# Aiguillage de la décision — voir la docstring du module pour les 3 issues
# ============================================================


def classify_decision(entry: dict, sense_id: str | None) -> tuple[str, str | None]:
    """Renvoie (groupe, new_key) — groupe in {"promu", "reassigne", "audit"}.
    `sense_id` doit déjà être validé contre l'inventaire ouvert de cette
    entrée (None si absent, inventé, ou POS différente non applicable ici —
    voir run())."""
    if entry["kind"] == "mwe":
        return "promu", entry["key"]
    if sense_id is None:
        return "audit", None
    new_pos = sense_id.split(".")[-2]
    cur_pos = entry.get("pos")
    normalize = lambda p: "a" if p in ("a", "s") else p
    if normalize(new_pos) != normalize(cur_pos):
        return "audit", None
    if sense_id == entry["key"]:
        return "promu", entry["key"]
    return "reassigne", sense_id


def _build_promoted_entry(entry: dict, decision: ReassignedDecision) -> dict:
    new_entry = dict(entry)
    new_entry.update({
        "fr": decision.fr[0] if decision.fr else entry.get("fr"),
        "fr_alt": decision.fr[1:] if decision.fr else entry.get("fr_alt") or [],
        "status": "auto_joint",
        "agreement": "auto_joint_confirme",
        "translation_type": decision.translation_type,
        "sense_fit": decision.sense_fit, "sense_fit_note": decision.sense_fit_note,
        "decided_at": date.today().isoformat(), "decided_by": "auto_joint",
        "note": decision.reason,
    })
    return new_entry


def apply_decision(
    entry: dict, decision: ReassignedDecision, inventory: list[dict], contexte_en: str,
    store: dict[str, dict], *, model: str | None = None,
) -> tuple[str, dict | None]:
    """Applique UNE décision à `store` (en place) et renvoie (groupe,
    ligne_d_audit_ou_None) — groupe in {"promu", "reassigne", "audit",
    "bloque"}. Isolé de run() pour rester testable sans appel réseau : c'est
    ici, et seulement ici, que la règle de non-écrasement d'une clé cible déjà
    verrouillée (verify_fr_lock.LOCKED_STATUSES) est appliquée.

    S6-1 : la porte sense_fit/translation_type (sense_fr.blocks_auto_lock)
    est vérifiée EN PREMIER, avant même de savoir si l'issue aurait été
    "promu" ou "reassigne" — sans quoi une expression figée ("mwe:...",
    qui ne peut jamais être re-clée, voir classify_decision) filerait tout
    droit vers un verrouillage `auto_joint` dès que le modèle propose UNE
    traduction, quelle que soit sa propre cohérence avec la définition
    affichée (voir la docstring de ReassignedDecision.sense_fit pour le cas
    réel qui a motivé cette porte)."""
    block_reason = sense_fr.blocks_auto_lock(decision.sense_fit, decision.translation_type)
    if block_reason:
        return "audit", _audit_row(
            entry, decision, decision.sense_id, contexte_en,
            note=f"verrouillage automatique refusé ({block_reason}) : "
                 f"{decision.sense_fit_note or decision.reason}",
        )

    allowed = {c["sense_id"] for c in inventory}
    sense_id = decision.sense_id
    if sense_id is not None and entry["kind"] != "mwe" and sense_id not in allowed:
        sense_id = None  # sense_id inventé, absent de l'inventaire réel -> "aucun sens exact"

    group, new_key = classify_decision(entry, sense_id)

    if group == "promu":
        store[entry["key"]] = _build_promoted_entry(entry, decision)
        return "promu", None

    if group == "reassigne":
        existing_target = store.get(new_key)
        if existing_target is not None and existing_target.get("status") in verify_fr_lock.LOCKED_STATUSES:
            return "bloque", _audit_row(
                entry, decision, sense_id, contexte_en,
                note=f"cible {new_key} déjà verrouillée ({existing_target['status']}) — non écrasée",
            )
        chosen = next((c for c in inventory if c["sense_id"] == new_key), None)
        definition_en = chosen["definition"] if chosen else None
        store[new_key] = _build_reassigned_entry(
            entry, decision, new_key, definition_en, existing_target, inventory, model=model
        )
        store[entry["key"]] = {**entry, "agreement": f"reassigne_vers:{new_key}"}
        return "reassigne", None

    return "audit", _audit_row(entry, decision, sense_id, contexte_en)


def _build_reassigned_entry(
    entry: dict, decision: ReassignedDecision, new_key: str, definition_en: str | None,
    existing_target: dict | None, inventory: list[dict], *, model: str | None = None,
) -> dict:
    """`existing_target` : entrée déjà présente sous `new_key`, si elle existe
    et n'est PAS verrouillée (voir run() — jamais appelé sinon). Ses
    occurrences sont additionnées (même mot ou mot différent, même sens
    réel désormais). Simplification assumée en v1 (volume mesuré : 3 cas sur
    ce livre) : la traduction/confiance du modèle CONJOINT remplace celle de
    `existing_target` plutôt que d'être fusionnée finement."""
    lemma_word = new_key.split(".")[0].replace("_", " ")
    lemmas_en = list(dict.fromkeys(
        (existing_target or {}).get("lemmas_en", []) + [lemma_word] + entry.get("lemmas_en", [])
    ))
    occurrences = entry.get("occurrences", 0) + (existing_target.get("occurrences", 0) if existing_target else 0)
    return {
        "key": new_key, "kind": "synset", "lemmas_en": lemmas_en,
        "pos": new_key.split(".")[-2], "definition_en": definition_en,
        "occurrences": occurrences,
        "fr": decision.fr[0] if decision.fr else None,
        "fr_alt": decision.fr[1:] if decision.fr else [],
        "status": "auto_joint",
        "agreement": f"auto_joint_reassigne_depuis:{entry['key']}",
        "translation_type": decision.translation_type,
        "sense_fit": decision.sense_fit, "sense_fit_note": decision.sense_fit_note,
        "source": None,
        "evidence": {
            "omw_fr": [], "wonef": [], "frontier_model": model or config.SENSE_FR_FRONTIER_MODEL,
            "frontier_fr": decision.fr, "frontier_confidence": decision.confidence,
        },
        "decided_at": date.today().isoformat(), "decided_by": "auto_joint",
        "note": decision.reason,
        "reassignment_provenance": {
            "initial_key": entry["key"],
            "initial_pos": entry.get("pos"),
            "selected_key": new_key,
            "selected_pos": new_key.split(".")[-2],
            "reason": decision.reason,
            "inventory_sense_ids": [c["sense_id"] for c in inventory],
        },
    }


# ============================================================
# CSV d'audit — propositions non promues automatiquement (POS changée,
# sense_id=null, ou cible verrouillée déjà occupée). Régénéré en entier à
# chaque run, comme sense_fr_review.csv (jamais accumulé d'un run à l'autre).
# ============================================================

AUDIT_FIELDS = [
    "key", "kind", "lemmas_en", "pos_actuel", "sense_id_actuel",
    "pos_propose", "sense_id_propose", "fr_propose", "confidence", "reason",
    "note", "contexte_en",
]


def _audit_row(entry: dict, decision: ReassignedDecision, sense_id: str | None,
               contexte_en: str, note: str = "") -> dict:
    return {
        "key": entry["key"], "kind": entry["kind"],
        "lemmas_en": "/".join(entry.get("lemmas_en", [])),
        "pos_actuel": entry.get("pos") or "",
        "sense_id_actuel": entry["key"] if entry["kind"] == "synset" else "",
        "pos_propose": decision.pos, "sense_id_propose": sense_id or "",
        "fr_propose": "; ".join(decision.fr), "confidence": decision.confidence,
        "reason": decision.reason, "note": note, "contexte_en": contexte_en,
    }


def write_audit_csv(rows: list[dict]) -> None:
    config.ensure_out_dir()
    with config.SENSE_ID_REASSIGN_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Orchestration
# ============================================================


def run(model: str | None = None, dry_run: bool = False) -> int:
    task = task_config("S6-reassign")
    model = model or task.model
    config.require_frontier_model(model, "S6-reassign")
    lexical_inventory.verify_consumer(
        config.SENSES_INVENTORY_HASH_PATH, "sense_fr_reassign"
    )
    store = sense_fr.load_store()
    targets = select_targets(store)
    print(f"{len(targets)} entrée(s) pending structurelle(s) à réassigner "
          f"(agreement in {sorted(STRUCTURAL_AGREEMENTS)}), modèle={model}.")
    if not targets:
        print("Rien à faire.")
        return 0

    occurrences_by_sense = senses.load_occurrences_by_sense()

    items: list[tuple[dict, list[dict], list[dict]]] = []
    for entry in targets:
        occs_all = occurrences_by_sense.get(entry["key"]) or []
        occs = senses.pick_diverse_occurrences(occs_all, config.SENSE_FR_FRONTIER_MAX_OCCURRENCES) if occs_all else []
        inventory = [] if entry["kind"] == "mwe" else open_inventory(entry["lemmas_en"][0])
        items.append((entry, occs, inventory))

    batch_size = effective_batch_size(task)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    mode_batch = use_batch_prompt(task, batch_size)
    decisions_by_batch, cost = _translate_batches(batches, model, mode_batch=mode_batch, batch_size=batch_size)

    n_promu = n_reassigne = n_bloque = n_audit = 0
    audit_rows: list[dict] = []

    for batch, decisions in zip(batches, decisions_by_batch):
        for entry, occs, inventory in batch:
            decision = decisions.get(entry["key"])
            if decision is None:
                continue  # échec de lot déjà signalé par _translate_batches
            contexte_en = sense_fr.format_occurrences_en(occs)
            group, audit_row = apply_decision(entry, decision, inventory, contexte_en, store, model=model)
            if group == "promu":
                n_promu += 1
            elif group == "reassigne":
                n_reassigne += 1
            elif group == "bloque":
                n_bloque += 1
                audit_rows.append(audit_row)
            else:
                n_audit += 1
                audit_rows.append(audit_row)

    print(f"Coût constaté (appels non-cachés uniquement) : ${cost:.4f}")
    print(f"Ventilation : {n_promu} promue(s) en auto_joint, {n_reassigne} réassignée(s) vers une autre clé, "
          f"{n_audit} en audit seul, {n_bloque} bloquée(s) par une clé cible déjà verrouillée.")

    if dry_run:
        print("--dry-run : rien n'est écrit dans le magasin ni dans les CSV.")
        return 0

    sense_fr.write_store(store)
    n_pending = sense_fr.write_review_csv(store, occurrences_by_sense)
    write_audit_csv(audit_rows)
    print(f"Magasin : {len(store)} entrées ({n_pending} encore en attente -> {config.SENSE_FR_REVIEW_PATH}).")
    if audit_rows:
        print(f"{len(audit_rows)} proposition(s) -> {config.SENSE_ID_REASSIGN_PATH}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="Modèle LiteLLM — par défaut, celui résolu par task_config('S6-reassign') (voir "
             "pipeline/llm_tasks.py). require_frontier_model refuse tout autre modèle : pour en "
             "utiliser un autre, poser VOCAB_LLM_S6_REASSIGN=provider/nom;... plutôt que --model "
             "seul (empêche une frappe de travers de déclencher silencieusement un modèle plus "
             "coûteux).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Appelle le modèle et affiche la ventilation/coût, mais n'écrit rien sur disque.",
    )
    args = parser.parse_args()
    raise SystemExit(run(model=args.model, dry_run=args.dry_run))
