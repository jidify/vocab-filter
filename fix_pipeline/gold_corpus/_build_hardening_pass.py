# -*- coding: utf-8 -*-
import json

segs = {}
for l in open('_segments_dump.jsonl', encoding='utf-8'):
    d = json.loads(l)
    segs[d['idx']] = d['en']

# (idx, [ (surface, category, is_gold, edge_case, edge_type, note), ... ])
SPEC = [
 (140, [("give something this nice away", "phrasal_verb_separable", True, False, None,
   "'give away' with a long NP object ('something this nice') inserted between verb and particle "
   "-- the longest separation gap in this hardening pass, a real stress case for detectors that only "
   "look a token or two ahead for the particle.")]),
 (220, [("put your feet up", "phrasal_verb_separable", True, False, None,
   "'put up' separated by the short NP 'your feet'. Idiomatic reading ('put your feet up' = relax), "
   "not purely the literal physical placement -- a genuine fixed expression, distinct from the "
   "literal-placement hard_negatives added elsewhere in this pass (idx 2429, 2480).")]),
 (334, [("throw your wrapping away", "phrasal_verb_separable", True, False, None,
   "'throw away' separated by the short NP 'your wrapping'. Bracket-delimited stage direction context, "
   "same '[ ]' nonverbal convention already covered by edge_type bracket_nonverbal elsewhere in the "
   "corpus (idx 208), but this instance is a plain separation case, not itself an edge_case.")]),
 (348, [("Check it out", "phrasal_verb_separable", True, False, None,
   "'check out' separated by the pronoun object 'it' -- pronoun objects force separation ('check out it' "
   "is ungrammatical), the cleanest possible test of whether a detector handles obligatory-separation "
   "word order.")]),
 (432, [("start us off", "phrasal_verb_separable", True, False, None,
   "'start off' separated by the pronoun object 'us'. Same obligatory-separation pattern as 'check it "
   "out' (idx 348) with a different pronoun and particle.")]),
 (691, [("Taking the spotlight off", "phrasal_verb_separable", True, False, None,
   "'take off' (withdraw/remove) separated by the short NP 'the spotlight'; the following word 'Aimee' "
   "is a separate prepositional complement ('off Aimee'), not part of the gold span itself -- mirrors "
   "the 'put the spotlight on her' construction added at idx 1592 in this same pass, same idiom family "
   "with the opposite particle.")]),
 (887, [("stink the place up", "phrasal_verb_separable", True, False, None,
   "'stink up' separated by the short NP 'the place'. Colloquial/idiomatic register, single clause, low "
   "ambiguity -- a clean positive.")]),
 (941, [
   ("cleans up the mess", "phrasal_verb_separable", True, False, None,
    "'clean up' separated by the short NP 'the mess'."),
   ("soaking up the liquid", "phrasal_verb_separable", True, False, None,
    "'soak up' separated by the short NP 'the liquid' -- second separable verb in the same segment as "
    "'cleans up the mess', both describing the same cleaning action."),
   ("ringing out her kitchen towel", "phrasal_verb_separable", True, False, None,
    "'ring out' (wring out; the book's own spelling is 'ringing', kept verbatim) separated by the short "
    "NP 'her kitchen towel' -- third separable verb in this one segment, useful density for testing "
    "whether a detector finds multiple distinct separable verbs in a single long sentence rather than "
    "stopping after the first."),
 ]),
 (1135, [("Get your hands off", "phrasal_verb_separable", True, False, None,
   "'get off' (stop touching) separated by the short NP 'your hands'. High-register-recognizable "
   "idiomatic command; the following 'of my mother' is a separate PP complement, not part of the gold "
   "span.")]),
 (1257, [("keep it down", "phrasal_verb_separable", True, False, None,
   "'keep down' (lower the volume) separated by the pronoun object 'it'.")]),
 (1543, [("calms her down", "phrasal_verb_separable", True, False, None,
   "'calm down' separated by the pronoun object 'her'.")]),
 (1592, [("put the spotlight on", "phrasal_verb_separable", True, False, None,
   "'put on' (draw attention to) separated by the short NP 'the spotlight'; 'her' follows as a separate "
   "PP complement ('on her'), not part of the gold span. Mirrors 'Taking the spotlight off' at idx 691 "
   "in this same pass -- same idiom family, opposite particle, good paired example for a detector's "
   "lexicon.")]),
 (1652, [("stole you away", "phrasal_verb_separable", True, False, None,
   "'steal away' (playfully abduct/take) separated by the pronoun object 'you'.")]),
 (1902, [("bring it up", "phrasal_verb_separable", True, False, None,
   "'bring up' (mention) separated by the pronoun object 'it'. This exact lemma already appears "
   "UNSEPARATED elsewhere in the base corpus at idx 1277 (\"Mom's been bringing up marriage\") -- "
   "together these two spans show the same lexical unit in both surface forms, the single most "
   "important stress test for this category: a detector must recognize both without treating them as "
   "different units.")]),
 (1908, [("shrugs it off", "phrasal_verb_separable", True, False, None,
   "'shrug off' (dismiss, downplay) separated by the pronoun object 'it'.")]),
 (1948, [("writing her back", "phrasal_verb_separable", True, False, None,
   "'write back' (reply) separated by the pronoun object 'her'.")]),
 (2093, [("figure it out", "phrasal_verb_separable", True, False, None,
   "'figure out' separated by the pronoun object 'it'. This exact lemma already appears UNSEPARATED in "
   "the base corpus at idx 624 (\"figure out it isn't actually [real]\", where 'it' is the subject of "
   "the embedded clause, not the object of 'figure out') -- a second same-pair-both-forms pair alongside "
   "'bring up' (idx 1902/1277) in this pass.")]),
 (2266, [("cut it out", "phrasal_verb_separable", True, False, None,
   "'cut out' (stop it) separated by the pronoun object 'it'. Common spoken-idiom-adjacent phrasal "
   "verb.")]),
 (2442, [("turns the lantern off", "phrasal_verb_separable", True, False, None,
   "'turn off' (deactivate) separated by the short NP 'the lantern'. Recurring prop in the play's final "
   "blackout sequence; contrast with the literal-placement hard_negatives 'places it on the windowsill' "
   "(idx 2429) and 'Puts it on the table' (idx 2480) added in this same pass, and with the adjacent "
   "positive 'turns it on' (idx 2496) below -- same verb family, three different surface behaviors to "
   "distinguish.")]),
 (2485, [("puts the blanket and pan down", "phrasal_verb_separable", True, False, None,
   "'put down' separated by the long conjoined NP 'the blanket and pan' -- second long-NP separation "
   "case in this pass alongside idx 140, here with an internal coordinator ('and') inside the inserted "
   "NP, an even harder span-boundary case since a naive detector might stop at the first noun.")]),
 (2496, [("turns it on", "phrasal_verb_separable", True, False, None,
   "'turn on' (activate) separated by the pronoun object 'it' -- genuine phrasal verb with no trailing "
   "locative complement. Minimal pair against the two literal-placement hard_negatives added in this "
   "pass ('places it on the windowsill' idx 2429, 'Puts it on the table' idx 2480): identical surface "
   "shape (verb + pronoun/NP + 'on' [+ optional PP]), but only this one is idiomatic activation, not "
   "literal placement onto a surface.")]),
 # --- hard negatives ---
 (521, [("dump her down the spiral staircase", "hard_negative", False, False, None,
   "Looks like a separable phrasal verb ('dump down') but 'down' here heads the prepositional phrase "
   "'down the spiral staircase' (literal path/direction), not a particle -- 'dump' + object 'her' + "
   "locative PP. 'dump down' is not an established English phrasal verb.")]),
 (984, [("get a shake out of the fridge", "hard_negative", False, False, None,
   "Looks like the phrasal verb 'get out' with the object 'a shake' inserted, but 'out of the fridge' "
   "is a literal prepositional phrase of source ('from inside the fridge') -- the whole thing is "
   "compositional retrieval ('get' + NP + PP), not the idiomatic phrasal verb 'get out' (leave/exit), "
   "which never takes a direct object this way.")]),
 (2429, [("places it on the windowsill", "hard_negative", False, False, None,
   "Literal spatial placement ('place' + object 'it' + PP 'on the windowsill'), not the idiomatic "
   "phrasal verb 'put/place on' (wear, or activate). Minimal-pair trap against the genuine positive "
   "'turns it on' (idx 2496) added in this pass: same surface shape (verb + pronoun + 'on' + optional "
   "PP), only one of the two is idiomatic.")]),
 (2480, [("Puts it on the table.", "hard_negative", False, False, None,
   "Second instance of the same literal-placement trap as idx 2429 ('places it on the windowsill') -- "
   "'put' + object 'it' + PP 'on the table', not the idiomatic 'put on' (wear). Kept as a second "
   "example because the pattern recurs with a different verb form and object across the play.")]),
 (538, [("take her back up", "hard_negative", False, False, None,
   "Genuinely ambiguous: could be read as 'take back' (return/reclaim) with the adverb 'up' (upstairs) "
   "attached, or as 'take up' (resume/carry up) with 'back' as a separate directional adverb, or as a "
   "three-part combination. No single clean phrasal-verb reading is defensible without more context "
   "than this segment provides -- kept as an explicit documented-ambiguous case rather than forced into "
   "either category, per the same judgment principle already used elsewhere in the corpus for genuinely "
   "unclear spans.")]),
 (106, [("keep an eye on the oven", "hard_negative", False, False, None,
   "Fixed VP idiom ('watch over'), but structurally NOT a separable phrasal verb: 'an eye' is a fixed "
   "internal object of 'keep', not a displaceable object standing in for the whole VP, and 'on' "
   "introduces the true object 'the oven' as a PP. Surface-matches a naive verb+NP+particle separation "
   "pattern (the same regex shape used to mine this hardening pass) without actually being one -- "
   "exactly the kind of category-confusion trap this corpus needs to guard against overgeneration.")]),
]

def build():
    out_segments = []
    for idx, spanlist in SPEC:
        text = segs[idx]
        spans = []
        for surface, category, is_gold, edge_case, edge_type, note in spanlist:
            n = text.count(surface)
            if n != 1:
                raise SystemExit("idx %d: surface %r occurs %d times, expected 1" % (idx, surface, n))
            start = text.index(surface)
            end = start + len(surface)
            assert text[start:end] == surface
            spans.append({
                "surface": surface, "start_char": start, "end_char": end,
                "category": category, "is_gold": is_gold,
                "edge_case": edge_case, "edge_type": edge_type, "note": note,
            })
        out_segments.append({
            "segment_idx": idx, "source": "plain", "text": text,
            "sample_reason": "phrasal_verb_separable_hardening",
            "gold_spans": spans,
        })
    return out_segments

if __name__ == "__main__":
    out = build()
    with open("_hardening_new_lines.jsonl", "w", encoding="utf-8") as f:
        for seg in out:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    npos = sum(1 for s in out for sp in s["gold_spans"] if sp["is_gold"])
    nneg = sum(1 for s in out for sp in s["gold_spans"] if not sp["is_gold"])
    print("segments=%d spans=%d positive=%d negative=%d" % (len(out), npos + nneg, npos, nneg))
