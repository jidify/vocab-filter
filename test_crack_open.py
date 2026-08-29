import unittest

if __name__ != "__main__":
    raise unittest.SkipTest("manual idiomatch exploration; run this file directly")

from pipeline.mwe import get_matcher


print("Chargement du matcher...")

matcher = get_matcher()

print("Ajout de 'crack open'...")

matcher.add_idioms([
    {
        "etymology": None,
        "lemma": "crack open",
        "senses": [
            {
                "content": (
                    "To cause something to open, "
                    "especially with a quick or slight action."
                ),
                "examples": [
                    "She cracked the window open.",
                    "He cracked the door open.",
                    "Crack it open.",
                    "They cracked open the bottle.",
                ],
            }
        ],
        "source": "custom",
    }
])

print("Ajout terminé.")
print()


tests = [
    "She cracked the window open.",
    "He cracked the door open.",
    "Crack it open.",
    "They cracked open the bottle.",

    # autres formes utiles
    "I cracked open a beer.",
    "She cracks open the window.",
    "They are cracking open the champagne.",
    "The door cracked open.",
]


for text in tests:

    doc = matcher.nlp(text)

    matches = matcher(doc)

    print("=" * 80)
    print(text)

    if not matches:
        print("  → AUCUN MATCH")
        continue

    for match in matches:
        print(f"  → {match}")
