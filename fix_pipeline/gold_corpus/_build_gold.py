"""Throwaway generator for the_humans_gold_v0.jsonl. Loads exact segment
text from _selected_segments.jsonl (produced via the real production
segmentation path, pipeline.corpus.load_segments) and locates each gold
span by substring search within that text, so offsets are computed
mechanically rather than typed by hand. Not part of the pipeline.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

RSQ = "’"   # ’
MD = "—"    # —

segs = {}
for line in (HERE / "_selected_segments.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    segs[obj["idx"]] = obj

def span(surface, category, is_gold, note, edge_case=False, edge_type=None, occurrence=1):
    return {
        "surface": surface, "category": category, "is_gold": is_gold,
        "edge_case": edge_case, "edge_type": edge_type, "note": note,
        "_occurrence": occurrence,
    }

# segment_idx -> (sample_reason, [span(...), ...])
DATA = {}

DATA[75] = ("known_difficult", [
    span("turn-of-the-century", "multi_token_entity", True, edge_case=True, edge_type="hyphen_modifier",
         note="Hyphenated temporal compound modifying the apartment description. Correctly tagged DATE by en_core_web_trf in the earlier spaCy benchmark (dis:18); a stress case for any detector relying on POS-tag adjacency across hyphens."),
    span("ground-floor/basement duplex tenement apartment", "nominal_compound", True, edge_case=True, edge_type="hyphen_modifier",
         note="Full noun phrase with a hyphen+slash modifier chain ('ground-floor/basement') attached to a three-noun compound head. Harder, longer sibling of the 'ground-floor apartment' -> 'floor apartment' truncation all three spaCy models produced elsewhere in this play (agr:06, see idx 281 in this corpus)."),
    span("duplex tenement apartment", "nominal_compound", True,
         note="Core three-noun compound without the hyphenated modifier — the minimum a detector must get right; the longer span above is the harder full version."),
    span("New York City", "multi_token_entity", True, edge_case=True, edge_type="possessive_boundary",
         note="Gold boundary stops before the possessive 's. Matches the earlier benchmark's human review (dis:06): 'New York City's' [including 's] and 'York City' [missing 'New'] were both judged bornes_incorrectes; only 'New York City' alone is a clean entity span."),
    span("New York City" + RSQ + "s", "hard_negative", False,
         note="Over-extended boundary (possessive 's included) — the actual span sm/lg/trf all produced in the earlier benchmark (dis:06). Kept as an explicit negative example of the possessive-boundary failure."),
    span("York City", "hard_negative", False,
         note="Truncated fragment (drops 'New') also produced by all three spaCy models as a spurious nominal_compound in the earlier benchmark (dis:06) — not a valid standalone unit."),
])

DATA[78] = ("known_difficult", [
    span("mid-century", "multi_token_entity", True, edge_case=True, edge_type="hyphen_modifier",
         note="Correctly identified as DATE by all three spaCy models in the earlier benchmark (dis:20)."),
    span("century renovation", "hard_negative", False,
         note="Spurious compound en_core_web_lg alone produced in the earlier benchmark (dis:20): drops 'mid-' and fuses the tail of the hyphenated DATE with the following noun."),
    span("sensible scheme", "hard_negative", False,
         note="Fully compositional adjective+noun phrase ('a scheme that is sensible') — plausible false positive for naive compound extraction, not a fixed lexical unit."),
])

DATA[87] = ("known_difficult", [
    span("attention shifts", "hard_negative", False,
         note="Subject + verb ('his attention shifts'), not a nominal compound. Matches dis:05/dis:26 in the earlier benchmark: sm and lg both mis-parsed this as nominal_compound; trf correctly produced nothing here."),
    span("shifts away", "phrasal_verb_inseparable", True,
         note="'shift away (from X)' = gradually redirect (attention/focus) — a genuine phrasal-prepositional reading of the same two words the hard_negative above mis-groups differently."),
])

DATA[92] = ("random_seed_42", [])

DATA[101] = ("phrasal_verb_pass", [
    span("going on", "phrasal_verb_inseparable", True,
         note="'going on' = happening — common inseparable phrasal verb, distinct from literal 'going' + locative 'on'."),
    span("kinda", "simple_word", True,
         note="Colloquial contraction of 'kind of' — single token but non-standard register, likely worth flagging for a learner."),
])

DATA[102] = ("known_difficult", [
    span("stomps around", "phrasal_verb_inseparable", True,
         note="Correct span stops at 'around'. sm produced 'stomps around?" + MD + "we' crossing '?" + MD + "' into the next clause (dis:00/dis:21) — the canonical punctuation-crossing failure in this corpus."),
    span("stomps around?" + MD + "we", "hard_negative", False, edge_case=True, edge_type="dialogue_dash",
         note="The actual span en_core_web_sm produced in the earlier benchmark (dis:00/dis:21) — kept as an explicit negative example of the punctuation-crossing failure mode."),
])

DATA[129] = ("phrasal_verb_pass", [
    span("come back", "phrasal_verb_inseparable", True,
         note="'come back' = return. Recurring motif phrase in Momo's dementia-driven dialogue throughout the play (also idx 165 as part of the idiom 'come back to earth')."),
])

DATA[156] = ("known_difficult", [
    span("her good days", "hard_negative", False,
         note="Idiomatic expression (having 'good days' vs 'bad days', of someone's health/mood) mislabeled as DATE by en_core_web_sm in the earlier benchmark (dis:14) — not a temporal expression."),
    span("all over the place", "idiom", True,
         note="Fixed idiom meaning disoriented/scattered (of a person's mental state here), fully non-compositional. Not detected by any of the three spaCy models in the earlier benchmark — it isn't a compound or an entity, so a spaCy-only pipeline structurally cannot see it."),
])

DATA[163] = ("known_difficult", [
    span("right Erik", "hard_negative", False,
         note="Discourse marker + vocative ('right, Erik?'), not a nominal compound. Matches dis:28 where en_core_web_sm mis-tagged this as nominal_compound."),
])

DATA[165] = ("known_difficult", [
    span("come back to earth", "idiom", True,
         note="Fixed idiom meaning 'stop daydreaming/refocus on reality'. One of the six target expressions from the earlier spaCy benchmark, but that benchmark never tested idiom-level detection — only NER/compound."),
])

DATA[175] = ("random_seed_42", [
    span("rough night", "hard_negative", False,
         note="Fully compositional adjective+noun ('a night that was rough') — plausible false positive for a compound detector, not a fixed lexical unit."),
])

DATA[182] = ("random_seed_42", [])

DATA[197] = ("random_seed_42", [
    span("toilet paper", "nominal_compound", True,
         note="Standard fully-lexicalized nominal compound."),
    span("cracks the bathroom door open", "phrasal_verb_separable", True,
         note="Resultative separable phrasal verb 'crack (something) open' (to open partially); the object 'the bathroom door' sits between verb and particle — classic separable word-order challenge for detectors expecting an adjacent verb+particle pair."),
])

DATA[208] = ("random_seed_42", [
    span("weather a storm", "idiom", True,
         note="Fixed idiomatic collocation ('weather' used as a verb meaning to endure/survive) — non-compositional if 'weather' is read only as the noun (meteorological conditions)."),
    span("being pushy", "hard_negative", False, edge_case=True, edge_type="bracket_nonverbal",
         note="Bracket-delimited inline stage direction ('[ ]' = expressed nonverbally, per the book's own NOTES) embedded mid-dialogue. 'being pushy' itself is compositional, not idiomatic — flagged mainly to test whether detectors correctly treat bracket-delimited spans as ordinary text rather than markup to strip."),
])

DATA[266] = ("random_seed_42", [])

DATA[271] = ("known_difficult", [
    span("New Yorkers", "multi_token_entity", True,
         note="NORP/demonym entity. In the earlier benchmark's target-expression review, en_core_web_lg additionally over-extended to 'No New Yorkers' (bornes_incorrectes) — gold span is 'New Yorkers' alone."),
    span("No New Yorkers", "hard_negative", False,
         note="Over-extended span (includes the negation determiner 'No') produced by en_core_web_lg alone in the earlier benchmark — 'No' is not part of the entity."),
    span("duplex apartments", "nominal_compound", True,
         note="Standard nominal compound."),
    span("dissing", "simple_word", True,
         note="Informal slang verb ('disrespecting') — single token but likely opaque to a learner unfamiliar with the register."),
])

DATA[274] = ("phrasal_verb_pass", [
    span("come on", "idiom", True,
         note="Idiomatic interjection (protest/disbelief), fully non-compositional relative to the literal motion sense of 'come' + 'on'."),
])

DATA[281] = ("known_difficult", [
    span("ground-floor apartment", "nominal_compound", True, edge_case=True, edge_type="hyphen_modifier",
         note="All three spaCy models truncated this to 'floor apartment' in the earlier benchmark (agr:06) — the compound dependency chain didn't reach across the hyphen to 'ground-'. Canonical example of the hyphen-modifier failure shared by all spaCy sizes; part of the direct motivation for this benchmark."),
    span("floor apartment", "hard_negative", False,
         note="The actual truncated span all three spaCy models produced (agr:06) — kept as an explicit negative example."),
])

DATA[388] = ("phrasal_verb_pass", [
    span("end up buying", "phrasal_verb_inseparable", True,
         note="'end up (doing X)' = aspectual phrasal verb indicating an eventual, often unplanned, outcome."),
])

DATA[421] = ("random_seed_42", [
    span("FYI", "simple_word", True,
         note="Common abbreviation ('for your information') standing alone as a token in dialogue."),
])

DATA[458] = ("random_seed_42", [])

DATA[483] = ("random_seed_42", [
    span("good days", "idiom", True,
         note="Same idiom as idx 156 ('good days'/'bad days' for someone's fluctuating health/mood) — recurring occurrence, useful for testing whether a detector catches the same MWE consistently across the book. en_core_web_sm mislabeled the analogous span at idx 156 as DATE (dis:14)."),
])

DATA[488] = ("random_seed_42", [
    span("lined up", "phrasal_verb_separable", True,
         note="'line up' (here in resultative/passive form 'lined up') = scheduled/arranged."),
    span("gigs", "simple_word", True,
         note="Informal for musical/performance engagements."),
])

DATA[504] = ("random_seed_42", [])
DATA[527] = ("random_seed_42", [])

DATA[528] = ("random_seed_42", [
    span("heads downstairs", "phrasal_verb_inseparable", True,
         note="'head' used as a directional motion verb ('go towards X') + adverb of place — a recognizable semi-fixed pattern (also 'head upstairs/home/out'), not the noun 'head'."),
    span("opens the door", "hard_negative", False,
         note="Fully literal, compositional verb+object — contrast case in the same segment against the semi-idiomatic 'heads downstairs' above."),
])

DATA[570] = ("random_seed_42", [
    span("triple-A school", "nominal_compound", True, edge_case=True, edge_type="hyphen_modifier",
         note="Hyphenated abbreviation-as-modifier ('triple-A', a school division tier) + head noun — hyphen-modifier stress case distinct from the 'ground-floor' family."),
    span("phys-ed classes", "nominal_compound", True, edge_case=True, edge_type="hyphen_modifier",
         note="Hyphenated abbreviation ('phys-ed' = physical education) as compound modifier, informal register — same hyphen-modifier failure family."),
    span("weight room", "nominal_compound", True,
         note="Standard nominal compound, no hyphen issue — control case in the same segment."),
])

DATA[579] = ("phrasal_verb_pass", [
    span("starts up the staircase", "hard_negative", False,
         note="Superficially resembles the phrasal verb 'start up' (begin operating, e.g. a car/business), but here 'up' is a literal directional preposition governing 'the staircase' ('starts [climbing] up the staircase'). Genuine parser-ambiguity trap between phrasal-verb and verb+PP readings — deliberately has no positive span."),
])

DATA[624] = ("phrasal_verb_pass", [
    span("figure out", "phrasal_verb_separable", True,
         note="'figure out' = understand/determine — canonical separable phrasal verb (also usable as 'figure it out')."),
])

DATA[629] = ("phrasal_verb_pass", [
    span("cleans up", "phrasal_verb_separable", True,
         note="'clean up' = tidy — separable phrasal verb ('cleans his spill up' is also valid word order)."),
    span("opens the door", "hard_negative", False,
         note="Literal, fully compositional verb+object — contrast case against the genuine phrasal verb 'cleans up' in the same segment."),
])

DATA[640] = ("random_seed_42", [])
DATA[702] = ("random_seed_42", [])
DATA[775] = ("random_seed_42", [])

DATA[795] = ("random_seed_42", [
    span("Godspeed", "simple_word", True,
         note="Archaic/formal interjection wishing good luck on a journey — single token but likely opaque to a learner."),
])

DATA[883] = ("phrasal_verb_pass", [
    span("knocks on", "hard_negative", False,
         note="Literal, compositional prepositional verb ('knock' + locative 'on the door') — not a true idiomatic phrasal verb, despite superficially matching the verb+particle pattern. Contrast case against genuine phrasal verbs elsewhere in this corpus (e.g. 'come back', 'figure out')."),
])

DATA[931] = ("phrasal_verb_pass", [
    span("steps away from", "hard_negative", False,
         note="Literal physical movement away from a location — compositional, not the idiomatic sense of 'step away' (e.g. 'step away from your desk' = take a break). Contrast case against genuine phrasal verbs elsewhere."),
    span("calming deep breaths", "hard_negative", False,
         note="Fully compositional adjective-stack + noun phrase, not a fixed unit."),
])

DATA[977] = ("random_seed_42", [
    span("seeing someone", "idiom", True,
         note="'see someone' = date someone romantically — idiomatic sense of 'see', non-compositional relative to the literal perception verb."),
])

DATA[1013] = ("phrasal_verb_pass", [
    span("fighting back tears", "idiom", True,
         note="'fight back tears' = suppress the urge to cry — fixed idiomatic collocation, non-compositional (not literally combating tears)."),
])

DATA[1045] = ("random_seed_42", [
    span("bring himself to", "idiom", True,
         note="'bring oneself to (do X)' = force oneself to do something one is reluctant to do — reflexive idiom. Segment is line-wrapped mid-sentence due to interleaved dialogue formatting, but the idiom span itself is fully contained here."),
])

DATA[1059] = ("random_seed_42", [])

DATA[1066] = ("random_seed_42", [
    span("come up with", "phrasal_verb_inseparable", True,
         note="'come up with' = devise/produce (an idea, answer) — three-word inseparable phrasal verb."),
])

DATA[1078] = ("random_seed_42", [])

DATA[1084] = ("known_difficult", [
    span("nursing home", "nominal_compound", True,
         note="Standard lexicalized compound. en_core_web_lg missed this entirely in the earlier benchmark (tgt:08/tgt:09) while sm and trf found it — one of the six target expressions from that benchmark."),
    span("Where" + RSQ + "re you at with", "idiom", True,
         note="'where you at with X' = informal query about current status/progress — colloquial fixed pattern, distinct from a literal spatial 'where are you'."),
])

DATA[1120] = ("random_seed_42", [])

DATA[1185] = ("random_seed_42", [
    span("Oh man", "idiom", True,
         note="Informal interjection expressing exasperation/surprise, not literally referring to a man."),
])

DATA[1209] = ("phrasal_verb_pass", [
    span("messed up", "idiom", True, edge_case=True, edge_type="bracket_nonverbal",
         note="Used adjectivally to describe a distressed facial expression — idiomatic extension of the phrasal verb 'mess up' (to ruin/disorder), not literal untidiness. Embedded in a bracketed nonverbal stage cue, like idx 208 and idx 2217."),
])

DATA[1272] = ("random_seed_42", [])

DATA[1277] = ("known_difficult", [
    span("bringing up", "phrasal_verb_separable", True,
         note="'bring up' = raise/mention (a topic) in conversation — correct and unambiguous here, despite this segment's other punctuation hazards."),
    span("the Mary statue?" + MD + "we" + RSQ + "ve", "hard_negative", False, edge_case=True, edge_type="dialogue_dash",
         note="The actual broken span en_core_web_lg and en_core_web_trf produced in the earlier benchmark (dis:15/dis:22): crosses a question mark and em-dash to wrongly include 'we've' — same punctuation-crossing failure family as dis:00's 'stomps around?" + MD + "we' (idx 102 in this corpus)."),
])

DATA[1316] = ("random_seed_42", [])

DATA[1328] = ("random_seed_42", [
    span("go through life", "idiom", True,
         note="'go through life' (doing/being X) = live one's life in a certain way — semi-fixed idiomatic collocation, not merely 'go' + 'through' + 'life' summed."),
])

DATA[1329] = ("random_seed_42", [])

DATA[1354] = ("phrasal_verb_pass", [
    span("burns out", "phrasal_verb_inseparable", True,
         note="'burn out' (of a bulb/fixture) = stop working/fail — idiomatically fixed collocation for lighting, distinct from literal combustion. Flagged as a genuine phrasal verb, not a hard negative."),
])

DATA[1396] = ("random_seed_42", [])

DATA[1400] = ("known_difficult", [
    span("e-mails", "simple_word", True, edge_case=True, edge_type="hyphen_tokenization",
         note="Hyphenated single-lexeme noun. Production tokenization (pipeline/analyze.py::get_nlp, EMAIL_SPECIAL_CASES) keeps 'e-mails' as one token via a special case that a bare spaCy pipeline in this benchmark will NOT have unless explicitly added — the earlier report's 'work e' truncation (dis:29) is a known methodological artifact of the quick-benchmark script, not a genuine model-quality signal. Any architecture compared here should be run WITH this special case, or the gap should be discounted the same way."),
    span("deal with", "phrasal_verb_inseparable", True,
         note="'deal with' = handle/address — common inseparable phrasal-prepositional verb."),
])

DATA[1590] = ("random_seed_42", [])

DATA[1607] = ("random_seed_42", [
    span("Dig in", "idiom", True,
         note="'dig in' = start eating (informal invitation) — idiomatic phrasal verb, distinct from the literal sense of digging into soil/ground."),
])

DATA[1625] = ("random_seed_42", [
    span("lift your latch", "idiom", True,
         note="Quoted traditional blessing phrase (archaic/poetic register, in curly quotation marks) — non-compositional as a fixed expression of welcome/hospitality."),
])

DATA[1696] = ("random_seed_42", [
    span("trust funds", "nominal_compound", True,
         note="Standard financial nominal compound."),
    span("smart-ass", "nominal_compound", True, edge_case=True, edge_type="hyphen_modifier",
         note="Hyphenated informal noun/insult — single lexicalized compound. Same word flagged in a code comment in pipeline/analyze.py (the 'smart-ass / e-mail manual-correction' case) as a known production tokenization edge case."),
])

DATA[1786] = ("random_seed_42", [])

DATA[1792] = ("random_seed_42", [
    span("artist grant", "nominal_compound", True,
         note="Standard nominal compound ('a grant for artists'); segment is a line-wrapped fragment ('... or [more grant types]') but the compound itself is fully contained here."),
])

DATA[1821] = ("phrasal_verb_pass", [
    span("bounce back", "idiom", True,
         note="'bounce back' = recover (from adversity) — idiomatic phrasal verb, non-literal relative to the physical sense of a ball bouncing."),
])

DATA[1978] = ("random_seed_42", [
    span("Tending to", "phrasal_verb_inseparable", True,
         note="'tend to (someone)' = take care of/attend to — inseparable phrasal-prepositional verb, distinct from 'tend to' meaning 'have a tendency to'."),
])

DATA[1988] = ("random_seed_42", [
    span("call you a car", "idiom", True,
         note="'call (someone) a car' = arrange a taxi/car service — idiomatic ditransitive use of 'call', not the literal sense of calling out to a car."),
])

DATA[1991] = ("random_seed_42", [
    span("camp out", "phrasal_verb_inseparable", True,
         note="'camp out' = stay/wait somewhere for an extended period — extended, non-literal sense here (waiting at the apartment), not actual outdoor camping."),
])

DATA[2107] = ("random_seed_42", [
    span("fit in", "hard_negative", False,
         note="Literal spatial sense ('it will physically fit inside the car/trunk') here, not the idiomatic social sense of 'fit in' (belong/be accepted). Genuine lexical-ambiguity trap for surface pattern matching alone; segment is a line-wrapped fragment."),
])

DATA[2152] = ("random_seed_42", [])

DATA[2217] = ("phrasal_verb_pass", [
    span("cheated on", "phrasal_verb_inseparable", True,
         note="'cheat on (someone)' = be sexually/romantically unfaithful — idiomatic prepositional verb, not literal 'cheat' + locative 'on'."),
    span("unload", "simple_word", True, edge_case=True, edge_type="bracket_nonverbal",
         note="Used metaphorically for emotional disclosure ('a lot to just unload') — figurative extension of the literal sense of unloading cargo. Single token (a word-sense issue more than a span-detection issue) but bracket-delimited like idx 208/1209, so kept as a light MWE-adjacent case rather than dropped."),
])

DATA[2358] = ("random_seed_42", [])

DATA[2520] = ("random_seed_42", [
    span("pushes through", "phrasal_verb_inseparable", True,
         note="'push through (X)' = persevere despite (an obstacle/emotion) — idiomatic phrasal verb, metaphorical extension of physical pushing."),
    span("spills in", "phrasal_verb_inseparable", True,
         note="'spill in' (of light) = pour/flow in — metaphorical extension of the literal liquid sense of 'spill', conventionally used for light in descriptive prose."),
    span("fluorescent light", "hard_negative", False,
         note="Fully compositional adjective+noun, not a fixed unit — contrast case in the same segment against the two genuine phrasal verbs above."),
])

DATA[2557] = ("random_seed_42", [])

DATA[2558] = ("random_seed_42", [
    span("weight" + RSQ + "s been lifted off his chest", "idiom", True,
         note="'a weight lifted off one's chest' = metaphor for emotional relief — fixed idiomatic image, non-compositional as a whole even though each word is common."),
])

DATA[2590] = ("random_seed_42", [])


# ------------------------------------------------------------------
# Build + verify offsets mechanically
# ------------------------------------------------------------------

out_records = []
errors = []

for idx, (reason, spans) in sorted(DATA.items()):
    if idx not in segs:
        errors.append(f"segment_idx {idx} not found in _selected_segments.jsonl")
        continue
    text = segs[idx]["en"]
    gold_spans = []
    seen_starts = {}
    for sp in spans:
        surface = sp["surface"]
        occ = sp["_occurrence"]
        pos = -1
        for _ in range(occ):
            pos = text.find(surface, pos + 1)
        if pos == -1:
            errors.append(f"idx={idx}: surface {surface!r} not found (occurrence {occ}) in text {text!r}")
            continue
        start, end = pos, pos + len(surface)
        if text[start:end] != surface:
            errors.append(f"idx={idx}: offset mismatch for {surface!r}")
            continue
        gold_spans.append({
            "surface": surface, "start_char": start, "end_char": end,
            "category": sp["category"], "is_gold": sp["is_gold"],
            "edge_case": sp["edge_case"], "edge_type": sp["edge_type"],
            "note": sp["note"],
        })
    out_records.append({
        "segment_idx": idx, "source": "plain", "text": text,
        "sample_reason": reason, "gold_spans": gold_spans,
    })

if errors:
    print(f"{len(errors)} ERROR(S):")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

out_path = HERE / "the_humans_gold_v0.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for rec in out_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

n_segments = len(out_records)
n_spans = sum(len(r["gold_spans"]) for r in out_records)
n_empty = sum(1 for r in out_records if not r["gold_spans"])
n_gold = sum(1 for r in out_records for s in r["gold_spans"] if s["is_gold"])
n_hard_neg = sum(1 for r in out_records for s in r["gold_spans"] if not s["is_gold"])
n_edge = sum(1 for r in out_records for s in r["gold_spans"] if s["edge_case"])

by_cat = {}
for r in out_records:
    for s in r["gold_spans"]:
        by_cat.setdefault(s["category"], 0)
        by_cat[s["category"]] += 1

by_reason = {}
for r in out_records:
    by_reason.setdefault(r["sample_reason"], 0)
    by_reason[r["sample_reason"]] += 1

print(f"Wrote {out_path}")
print(f"segments={n_segments} (empty={n_empty}) spans={n_spans} gold={n_gold} hard_negative={n_hard_neg} edge_case={n_edge}")
print("by category:", by_cat)
print("by sample_reason (segment count):", by_reason)
