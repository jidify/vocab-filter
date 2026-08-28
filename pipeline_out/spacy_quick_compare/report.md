# Comparaison rapide spaCy — sm / lg / trf (S1-2)

Généré le 2026-08-28T15:54:09 — spaCy 3.8.16.

Portée : uniquement spaCy (NER + dépendances `compound`) via `pipeline.multi_token`. Ni GlossBERT, ni LLM, ni étapes aval. Aucun artefact de production touché.

## Sources

- **texte simple (EN)** (`The Humans - Stephen Karam.txt`) : 2535 segments — PlayAdapter (pipeline/corpus.py) — une ligne non vide = un segment, hors_oeuvre exclu (identique à analyze.py::run).
- **EN extrait du bilingue** (`The Humans - Stephen Karam-TRAD.txt`) : 2574 segments — corpus.parse_bilingual_pairs, moitié EN — blocs séparés par lignes vides (pas la même granularité que 'plain'). Filtré : is_hors_oeuvre() (motifs légaux/sommaire) et blocs façon nom de personnage (heuristique identique à PlayAdapter).

Bilingue : 4025 paires lues, 9 écartées (motifs hors-œuvre), 1442 écartées (façon nom de personnage), 2574 conservées. Filtrage best-effort : quelques blocs de page de titre (adresse éditeur, cote catalogue, bio auteur en fin de livre) peuvent subsister — n'affecte pas la conclusion visée (robustesse de la segmentation), voir le plan.

## Temps de traitement

| Modèle | Source | Segments | Tokens (contenu) | Temps (s) | GPU | Abandonné |
|---|---|---:|---:|---:|:---:|:---:|
| en_core_web_sm | texte simple (EN) | 2535 | 21359 | 4.117 | non | non |
| en_core_web_sm | EN extrait du bilingue | 2574 | 21833 | 3.983 | non | non |
| en_core_web_lg | texte simple (EN) | 2535 | 21359 | 4.328 | non | non |
| en_core_web_lg | EN extrait du bilingue | 2574 | 21833 | 4.264 | non | non |
| en_core_web_trf | texte simple (EN) | 2535 | 21359 | 25.096 | oui | non |
| en_core_web_trf | EN extrait du bilingue | 2574 | 21833 | 2.546 | oui | non |

Total par modèle (2 sources) : en_core_web_sm = 8.1s, en_core_web_lg = 8.6s, en_core_web_trf = 27.6s

## Comptes de candidats par run

| Modèle | Source | Candidats | NER | compound | NER∩compound | pour 1000 tokens |
|---|---|---:|---:|---:|---:|---:|
| en_core_web_sm | texte simple (EN) | 511 | 167 | 390 | 46 | 23.9 |
| en_core_web_sm | EN extrait du bilingue | 618 | 231 | 476 | 89 | 28.3 |
| en_core_web_lg | texte simple (EN) | 468 | 156 | 359 | 47 | 21.9 |
| en_core_web_lg | EN extrait du bilingue | 572 | 220 | 444 | 92 | 26.2 |
| en_core_web_trf | texte simple (EN) | 429 | 147 | 330 | 48 | 20.1 |
| en_core_web_trf | EN extrait du bilingue | 538 | 213 | 422 | 97 | 24.6 |

## Les six expressions demandées

Regroupé par occurrence (segment) — "accord" = les 3 modèles trouvent exactement la même borne à cette occurrence (les types peuvent différer, signalé à part).

| Expression | Source | Occurrences (union) | sm | lg | trf | Accord exact | Écarts (segments) |
|---|---|---:|---:|---:|---:|---:|---|
| New York | texte simple (EN) | 12 | 12 | 12 | 12 | 12/12 | seg 384 (types différents) |
| New York | EN extrait du bilingue | 18 | 18 | 18 | 18 | 18/18 | seg 359 (types différents) |
| Virgin Mary | texte simple (EN) | 2 | 2 | 2 | 2 | 2/2 | seg 808 (types différents) |
| Virgin Mary | EN extrait du bilingue | 2 | 2 | 2 | 2 | 2/2 | seg 783 (types différents) |
| ranch dip | texte simple (EN) | 1 | 1 | 1 | 1 | 1/1 | (aucun) |
| ranch dip | EN extrait du bilingue | 1 | 1 | 1 | 1 | 1/1 | (aucun) |
| observation deck | texte simple (EN) | 1 | 1 | 1 | 1 | 1/1 | (aucun) |
| observation deck | EN extrait du bilingue | 1 | 1 | 1 | 1 | 1/1 | (aucun) |
| nursing home | texte simple (EN) | 1 | 1 | 0 | 1 | 0/1 | seg 1084 (en_core_web_sm:37:49, en_core_web_lg:absent, en_core_web_trf:37:49) |
| nursing home | EN extrait du bilingue | 1 | 1 | 0 | 1 | 0/1 | seg 1059 (en_core_web_sm:37:49, en_core_web_lg:absent, en_core_web_trf:37:49) |
| crystal ball | texte simple (EN) | 1 | 1 | 1 | 1 | 1/1 | (aucun) |
| crystal ball | EN extrait du bilingue | 1 | 1 | 1 | 1 | 1/1 | (aucun) |

## Candidats propres à un seul modèle (exemples)

**en_core_web_sm seul — texte simple (EN)** (72 au total, 5 montrés) :
- seg 102 : "stomps around?—we" [57:74] nominal_compound score=0.82
- seg 156 : "her good days" [25:38] named_entity:DATE score=0.95
- seg 163 : "right Erik" [120:130] nominal_compound score=0.82
- seg 201 : "’s Aimee" [19:27] named_entity:PERSON score=0.95
- seg 210 : "Noticing Momo" [1:14] nominal_compound score=0.82

**en_core_web_lg seul — texte simple (EN)** (33 au total, 5 montrés) :
- seg 83 : "plastic bags" [30:42] nominal_compound score=0.82
- seg 396 : "plastic cups" [61:73] nominal_compound score=0.82
- seg 430 : "—Dad!" [0:5] named_entity:PERSON score=0.95
- seg 459 : "one more day" [13:25] named_entity:DATE score=0.95
- seg 479 : "hairin sildern" [104:118] named_entity:PERSON score=0.95

**en_core_web_trf seul — texte simple (EN)** (32 au total, 5 montrés) :
- seg 71 : "The fear of death" [0:17] named_entity:PERSON score=0.95
- seg 75 : "turn-of-the-century" [2:21] named_entity:DATE score=0.95
- seg 75 : "basement duplex tenement apartment" [35:69] nominal_compound score=0.82
- seg 136 : "moving truck" [40:52] nominal_compound score=0.82
- seg 323 : "moving truck" [35:47] nominal_compound score=0.82

**en_core_web_sm seul — EN extrait du bilingue** (73 au total, 5 montrés) :
- seg 0 : "NY 10018-4156" [41:54] named_entity:ORG score=0.95
- seg 77 : "stomps around?—we" [57:74] nominal_compound score=0.82
- seg 131 : "her good days" [25:38] named_entity:DATE score=0.95
- seg 138 : "right Erik" [120:130] nominal_compound score=0.82
- seg 176 : "’s Aimee" [19:27] named_entity:PERSON score=0.95

**en_core_web_lg seul — EN extrait du bilingue** (33 au total, 5 montrés) :
- seg 58 : "plastic bags" [30:42] nominal_compound score=0.82
- seg 371 : "plastic cups" [61:73] nominal_compound score=0.82
- seg 405 : "—Dad!" [0:5] named_entity:PERSON score=0.95
- seg 434 : "one more day" [13:25] named_entity:DATE score=0.95
- seg 454 : "hairin sildern" [104:118] named_entity:PERSON score=0.95

**en_core_web_trf seul — EN extrait du bilingue** (36 au total, 5 montrés) :
- seg 1 : "The Humans" [19:29] named_entity:WORK_OF_ART score=0.95
- seg 10 : "Cover photographs" [0:17] nominal_compound score=0.82
- seg 36 : "1." [0:2] named_entity:CARDINAL score=0.95
- seg 37 : "2." [0:2] named_entity:CARDINAL score=0.95
- seg 46 : "The fear of death" [0:17] named_entity:PERSON score=0.95

## Bornes différentes pour le même repère (exemples)

**texte simple (EN)** (69 au total, 8 montrés) :
- seg 75 : en_core_web_sm: "New York" [73:81] nominal_compound score=0.82; en_core_web_lg: "New York" [73:81] nominal_compound score=0.82; en_core_web_trf: "New York" [73:81] nominal_compound score=0.82; en_core_web_sm: "New York City’s" [73:88] named_entity:GPE score=0.95; en_core_web_lg: "New York City’s" [73:88] named_entity:GPE score=0.95; en_core_web_trf: "New York City’s" [73:88] named_entity:GPE score=0.95; en_core_web_sm: "York City" [77:86] nominal_compound score=0.82; en_core_web_lg: "York City" [77:86] nominal_compound score=0.82; en_core_web_trf: "York City" [77:86] nominal_compound score=0.82
- seg 78 : en_core_web_sm: "mid-century" [66:77] named_entity:DATE score=0.95; en_core_web_lg: "mid-century" [66:77] named_entity:DATE score=0.95; en_core_web_trf: "mid-century" [66:77] named_entity:DATE score=0.95; en_core_web_lg: "century renovation" [70:88] nominal_compound score=0.82
- seg 110 : en_core_web_lg: "I’m sixty-one" [14:27] named_entity:PRODUCT score=0.95; en_core_web_trf: "sixty-one" [18:27] named_entity:DATE score=0.95
- seg 271 : en_core_web_lg: "No New Yorkers" [89:103] named_entity:NORP score=0.95; en_core_web_sm: "New Yorkers" [92:103] named_entity:NORP/nominal_compound score=0.95; en_core_web_lg: "New Yorkers" [92:103] nominal_compound score=0.82; en_core_web_trf: "New Yorkers" [92:103] named_entity:NORP/nominal_compound score=0.95
- seg 374 : en_core_web_sm: "the Cira Centre" [87:102] named_entity:FAC score=0.95; en_core_web_lg: "the Cira Centre" [87:102] named_entity:FAC score=0.95; en_core_web_trf: "the Cira Centre" [87:102] named_entity:FAC score=0.95; en_core_web_sm: "Cira Centre" [91:102] nominal_compound score=0.82; en_core_web_lg: "Cira Centre" [91:102] nominal_compound score=0.82; en_core_web_trf: "Cira Centre" [91:102] nominal_compound score=0.82
- seg 384 : en_core_web_sm: "outta New York" [138:152] nominal_compound score=0.82; en_core_web_sm: "New York" [144:152] named_entity:GPE score=0.95; en_core_web_lg: "New York" [144:152] named_entity:GPE/nominal_compound score=0.95; en_core_web_trf: "New York" [144:152] named_entity:GPE/nominal_compound score=0.95
- seg 412 : en_core_web_sm: "shut bathroom" [52:65] nominal_compound score=0.82; en_core_web_lg: "shut bathroom" [52:65] nominal_compound score=0.82; en_core_web_sm: "bathroom door" [57:70] nominal_compound score=0.82; en_core_web_lg: "bathroom door" [57:70] nominal_compound score=0.82; en_core_web_trf: "bathroom door" [57:70] nominal_compound score=0.82
- seg 479 : en_core_web_lg: "garn ackening" [126:139] nominal_compound score=0.82; en_core_web_lg: "garn ackening ery or loddinsezz" [126:157] named_entity:PERSON score=0.95

**EN extrait du bilingue** (86 au total, 8 montrés) :
- seg 1 : en_core_web_lg: "TCG’s" [56:61] named_entity:PRODUCT score=0.95; en_core_web_sm: "TCG’s Book Program" [56:74] named_entity:ORG score=0.95; en_core_web_sm: "Book Program" [62:74] nominal_compound score=0.82; en_core_web_lg: "Book Program" [62:74] nominal_compound score=0.82; en_core_web_trf: "Book Program" [62:74] named_entity:ORG/nominal_compound score=0.95
- seg 1 : en_core_web_sm: "the New York State Council" [104:130] named_entity:ORG score=0.95; en_core_web_lg: "the New York State Council" [104:130] named_entity:ORG score=0.95; en_core_web_trf: "the New York State Council on the Arts" [104:142] named_entity:ORG score=0.95; en_core_web_sm: "New York" [108:116] nominal_compound score=0.82; en_core_web_lg: "New York" [108:116] nominal_compound score=0.82; en_core_web_trf: "New York" [108:116] nominal_compound score=0.82; en_core_web_trf: "York State" [112:122] nominal_compound score=0.82; en_core_web_sm: "York State Council" [112:130] nominal_compound score=0.82; en_core_web_lg: "York State Council" [112:130] nominal_compound score=0.82; en_core_web_trf: "State Council" [117:130] nominal_compound score=0.82
- seg 1 : en_core_web_sm: "Governor Andrew Cuomo" [163:184] nominal_compound score=0.82; en_core_web_lg: "Governor Andrew Cuomo" [163:184] nominal_compound score=0.82; en_core_web_trf: "Governor Andrew Cuomo" [163:184] nominal_compound score=0.82; en_core_web_sm: "Andrew Cuomo" [172:184] named_entity:PERSON score=0.95; en_core_web_lg: "Andrew Cuomo" [172:184] named_entity:PERSON score=0.95; en_core_web_trf: "Andrew Cuomo" [172:184] named_entity:PERSON score=0.95
- seg 1 : en_core_web_sm: "the New York State Legislature" [189:219] named_entity:ORG score=0.95; en_core_web_lg: "the New York State Legislature" [189:219] named_entity:ORG score=0.95; en_core_web_trf: "the New York State Legislature" [189:219] named_entity:ORG score=0.95; en_core_web_sm: "New York" [193:201] nominal_compound score=0.82; en_core_web_lg: "New York" [193:201] nominal_compound score=0.82; en_core_web_trf: "New York" [193:201] nominal_compound score=0.82; en_core_web_trf: "York State" [197:207] nominal_compound score=0.82; en_core_web_sm: "York State Legislature" [197:219] nominal_compound score=0.82; en_core_web_lg: "York State Legislature" [197:219] nominal_compound score=0.82; en_core_web_trf: "State Legislature" [202:219] nominal_compound score=0.82
- seg 2 : en_core_web_lg: "Consortium Book" [59:74] nominal_compound score=0.82; en_core_web_sm: "Consortium Book Sales" [59:80] nominal_compound score=0.82; en_core_web_trf: "Consortium Book Sales" [59:80] nominal_compound score=0.82; en_core_web_sm: "Consortium Book Sales and Distribution" [59:97] named_entity:ORG score=0.95; en_core_web_lg: "Consortium Book Sales and Distribution" [59:97] named_entity:ORG score=0.95; en_core_web_trf: "Consortium Book Sales and Distribution" [59:97] named_entity:ORG score=0.95; en_core_web_lg: "Book Sales" [70:80] nominal_compound score=0.82
- seg 10 : en_core_web_lg: "Jonathan Knowles" [19:35] named_entity:PERSON score=0.95; en_core_web_trf: "Jonathan Knowles" [19:35] named_entity:PERSON/nominal_compound score=0.95; en_core_web_sm: "Jonathan Knowles/Getty Images" [19:48] named_entity:PERSON score=0.95; en_core_web_trf: "Getty Images" [36:48] nominal_compound score=0.82
- seg 11 : en_core_web_lg: "Carlos Casariego" [0:16] named_entity:PERSON score=0.95; en_core_web_trf: "Carlos Casariego" [0:16] named_entity:PERSON/nominal_compound score=0.95; en_core_web_sm: "Carlos Casariego/Getty Images" [0:29] named_entity:PERSON score=0.95; en_core_web_lg: "Getty Images" [17:29] named_entity:ORG score=0.95; en_core_web_trf: "Getty Images" [17:29] named_entity:ORG/nominal_compound score=0.95
- seg 11 : en_core_web_sm: "New York" [31:39] named_entity:GPE/nominal_compound score=0.95; en_core_web_lg: "New York" [31:39] named_entity:GPE/nominal_compound score=0.95; en_core_web_trf: "New York" [31:39] named_entity:GPE/nominal_compound score=0.95; en_core_web_sm: "York skyline" [35:47] nominal_compound score=0.82; en_core_web_lg: "York skyline" [35:47] nominal_compound score=0.82; en_core_web_trf: "York skyline" [35:47] nominal_compound score=0.82

## Un même modèle manque un candidat trouvé par les deux autres (exemples)

**en_core_web_sm absent, les 2 autres d'accord — texte simple (EN)** (6 au total) :
- seg 134 : en_core_web_lg: "music workspace" [34:49] nominal_compound score=0.82; en_core_web_trf: "music workspace" [34:49] nominal_compound score=0.82
- seg 1277 : en_core_web_lg: "Mary statue?—we’ve" [40:58] named_entity:PERSON/nominal_compound score=0.95; en_core_web_trf: "Mary statue?—we’ve" [40:58] nominal_compound score=0.82
- seg 1400 : en_core_web_lg: "work e" [10:16] nominal_compound score=0.82; en_core_web_trf: "work e" [10:16] nominal_compound score=0.82

**en_core_web_lg absent, les 2 autres d'accord — texte simple (EN)** (17 au total) :
- seg 80 : en_core_web_sm: "moving boxes" [344:356] nominal_compound score=0.82; en_core_web_trf: "moving boxes" [344:356] nominal_compound score=0.82
- seg 90 : en_core_web_sm: "toilet flush" [2:14] nominal_compound score=0.82; en_core_web_trf: "toilet flush" [2:14] nominal_compound score=0.82
- seg 236 : en_core_web_sm: "a minute" [19:27] named_entity:TIME score=0.95; en_core_web_trf: "a minute" [19:27] named_entity:TIME score=0.95

**en_core_web_trf absent, les 2 autres d'accord — texte simple (EN)** (46 au total) :
- seg 87 : en_core_web_sm: "attention shifts" [14:30] nominal_compound score=0.82; en_core_web_lg: "attention shifts" [14:30] nominal_compound score=0.82
- seg 91 : en_core_web_sm: "plastic bags" [60:72] nominal_compound score=0.82; en_core_web_lg: "plastic bags" [60:72] nominal_compound score=0.82
- seg 152 : en_core_web_sm: "the-day" [26:33] named_entity:DATE score=0.95; en_core_web_lg: "the-day" [26:33] named_entity:DATE score=0.95

**en_core_web_sm absent, les 2 autres d'accord — EN extrait du bilingue** (6 au total) :
- seg 109 : en_core_web_lg: "music workspace" [34:49] nominal_compound score=0.82; en_core_web_trf: "music workspace" [34:49] nominal_compound score=0.82
- seg 1252 : en_core_web_lg: "Mary statue?—we’ve" [40:58] named_entity:PERSON/nominal_compound score=0.95; en_core_web_trf: "Mary statue?—we’ve" [40:58] nominal_compound score=0.82
- seg 1375 : en_core_web_lg: "work e" [10:16] nominal_compound score=0.82; en_core_web_trf: "work e" [10:16] nominal_compound score=0.82

**en_core_web_lg absent, les 2 autres d'accord — EN extrait du bilingue** (22 au total) :
- seg 15 : en_core_web_sm: "set design" [231:241] nominal_compound score=0.82; en_core_web_trf: "set design" [231:241] nominal_compound score=0.82
- seg 15 : en_core_web_sm: "sound design" [357:369] nominal_compound score=0.82; en_core_web_trf: "sound design" [357:369] nominal_compound score=0.82
- seg 19 : en_core_web_sm: "Kelly O’Sullivan" [0:16] named_entity:PERSON/nominal_compound score=0.95; en_core_web_trf: "Kelly O’Sullivan" [0:16] named_entity:PERSON/nominal_compound score=0.95

**en_core_web_trf absent, les 2 autres d'accord — EN extrait du bilingue** (47 au total) :
- seg 0 : en_core_web_sm: "24th Floor" [19:29] named_entity:ORG score=0.95; en_core_web_lg: "24th Floor" [19:29] named_entity:ORG/nominal_compound score=0.95
- seg 62 : en_core_web_sm: "attention shifts" [14:30] nominal_compound score=0.82; en_core_web_lg: "attention shifts" [14:30] nominal_compound score=0.82
- seg 66 : en_core_web_sm: "plastic bags" [60:72] nominal_compound score=0.82; en_core_web_lg: "plastic bags" [60:72] nominal_compound score=0.82

## Différences entre les deux sources (par modèle)

**en_core_web_sm** — taux pour 1000 tokens : plain=23.9, bilingue_en=28.3 (bruts : 378 vs 465 surfaces distinctes)
- présent seulement en plain (échantillon) : (aucun)
- présent seulement en bilingue_en (échantillon) : 24th floor, abigail medrano, aimee blake, amanda j. davis, american theater company, andrew cuomo, arian moayed, artistic director

**en_core_web_lg** — taux pour 1000 tokens : plain=21.9, bilingue_en=26.2 (bruts : 330 vs 416 surfaces distinctes)
- présent seulement en plain (échantillon) : (aucun)
- présent seulement en bilingue_en (échantillon) : 24th floor, abigail medrano, aimee blake, amanda j. davis, american theater company, andrew cuomo, arian moayed, artistic director

**en_core_web_trf** — taux pour 1000 tokens : plain=20.1, bilingue_en=24.6 (bruts : 302 vs 391 surfaces distinctes)
- présent seulement en plain (échantillon) : (aucun)
- présent seulement en bilingue_en (échantillon) : 1., 2., abigail medrano, aimee blake, amanda j. davis, american theater company, andrew cuomo, arian moayed

## Audit manuel

`disagreements.jsonl` contient 12 lignes pour les six expressions demandées (source par source), 30 désaccords choisis de façon déterministe (répartition round-robin entre modèles/sources/catégories, plafond 30), et 10 accords communs aux trois modèles comme contrôle (plafond 10).

Remplir `human_label` sur chaque ligne avec l'une des valeurs : `correct`, `incorrect`, `bornes_incorrectes`, `incertain`. Une fois fait, relancer le Prompt 2 du plan pour le calcul des comptes et la conclusion.

## Décision

Voir la section « Suite — revue humaine et conclusion (Prompt 2) » en fin de rapport : la relecture des désaccords a été faite et la décision est tranchée ci-dessous.


---

# Suite — revue humaine et conclusion (Prompt 2)

Revue de 52 lignes de `disagreements.jsonl` (131 entrées modèle×span) — faite par l'assistant à la demande de l'utilisateur (relecture humaine jugée trop lourde pour un format brut), pas par un relecteur externe. `human_label` (et `label` par entrée pour les 4 lignes boundary_diff hétérogènes) sont conservés dans le fichier avec un `human_note` expliquant chaque décision — à vérifier par un humain en cas de doute sur la conclusion.

## Comptes par modèle et par famille

Comptes calculés sur exactement les deux lots relus manuellement et étiquetés (`human_label`, ou `label` par entrée sur les 4 lignes `boundary_diff` hétérogènes) : les **30 désaccords** (`dis:00`–`dis:29`) et les **10 accords communs de contrôle** (`agr:00`–`agr:09`) — 40 lignes, 131 entrées modèle×span. Les 12 lignes des six expressions demandées (`tgt:*`) sont exclues de ce tableau : elles sont déjà rapportées ci-dessus et sont toutes `correct` par construction (accord exact recherché), donc sans valeur discriminante ici — les inclure aurait gonflé artificiellement les taux. Une entrée porteuse des deux familles à la fois (ex. `New York` en `named_entity:GPE` + `nominal_compound`) est comptée dans NER **et** dans compound : elle constitue une réussite pour les deux extractions, pas une seule.

_Échantillon de désaccords, pas un tirage aléatoire sur tout le corpus : les catégories exclusive/missing/boundary_diff sur-représentent volontairement les cas où les modèles ne sont PAS d'accord, donc le taux d'erreur ci-dessous est plus dur que le taux d'erreur global du modèle. L'échantillon `agreement_control` (accords à 3, 10 lignes) est inclus dans ce même tableau plutôt que traité à part : 9/10 `correct`, 1/10 `bornes_incorrectes` (`agr:06`, cf. plus bas) — cohérent avec un taux de base élevé sur les cas non disputés, et il tire mécaniquement les pourcentages ci-dessous vers le haut par rapport à un calcul sur les 30 désaccords seuls. Aucune entrée n'a reçu `incertain` : l'échantillon ne contenait pas de cas jugé réellement ambigu par le relecteur, pas une confirmation indépendante d'absence d'ambiguïté (voir mise en garde sur le relecteur ci-dessus)._

| Modèle | Famille | correct | incorrect | bornes_incorrectes | incertain | total | % correct |
|---|---|---:|---:|---:|---:|---:|---:|
| en_core_web_sm | NER | 4 | 3 | 3 | 0 | 10 | 40% |
| en_core_web_sm | compound | 15 | 5 | 3 | 0 | 23 | 65% |
| en_core_web_lg | NER | 5 | 4 | 2 | 0 | 11 | 45% |
| en_core_web_lg | compound | 17 | 6 | 4 | 0 | 27 | 63% |
| en_core_web_trf | NER | 7 | 1 | 2 | 0 | 10 | 70% |
| en_core_web_trf | compound | 16 | 3 | 5 | 0 | 24 | 67% |

Le total par modèle diffère (23–27 pour compound, 10–11 pour NER) car `missing:<modèle>` signifie que ce modèle n'a produit aucun candidat sur cette ligne — il n'y a alors rien à noter pour lui, donc pas d'entrée dans son décompte.

## Améliorations / régressions concrètes

- **trf mieux borné sur les entités multi-mots** : `the New York State Council on the Arts` capturé en entier par trf (`dis:27`) contre `the New York State Council` tronqué (manque "on the Arts") chez sm/lg.

- **lg manque un composé attendu** : `nursing home` (une des six expressions demandées) absent chez lg sur les deux sources, trouvé par sm et trf (`tgt:08`/`tgt:09`).

- **lg introduit une fausse borne** : `century renovation` (sans le "mid-") comme composé propre à lg (`dis:20`), et `TCG's` seul étiqueté PRODUCT (`dis:13`).

- **sm et lg partagent une même erreur de rattachement** : `attention shifts` (sujet + verbe, pas un composé) accepté comme composé par sm ET lg, correctement absent chez trf (`dis:05`/`dis:26`) — seul cas où trf est *plus* prudent que les deux autres, pas moins.

- **trf a sa propre erreur NER** : `The fear of death` étiqueté PERSON, alors que c'est un thème/titre (`dis:04`).

- **sm produit plusieurs bornes cassées à travers la ponctuation** : `stomps around?—we` (`dis:00`/`dis:21`), `Mary statue?—we've` (entrées manquantes chez sm mais trouvées, avec la même borne cassée, chez lg/trf — `dis:15`/`dis:22`) — pas un problème propre à sm, les 3 modèles partagent cette faiblesse sur les tirets/ponctuation de dialogue.

- **Un même trou de segmentation touche les 3 modèles** : `ground-floor apartment` tronqué en `floor apartment` chez les 3 modèles (`agr:06`) — la chaîne de dépendance `compound` ne relie pas le modificateur à trait d'union, indépendamment du modèle.

- **Limite méthodologique du test, pas du modèle** : `work e` (`dis:29`) est `e-mails` tronqué — ce script isolé n'ajoute pas le cas spécial de tokenisation `EMAIL_SPECIAL_CASES` qu'utilise la production (`pipeline/analyze.py::get_nlp`). Affecte les 3 modèles de façon égale (aucun biais relatif entre eux) mais gonfle artificiellement les décomptes d'erreurs "bornes" sur ce cas précis pour les 3.

## Temps (rappel)

sm et lg : ~4s par source sur CPU. trf : 25s (source plain, inclut l'échauffement GPU) et 2.5s (source bilingue, GPU déjà chaud). La machine dédiée à la production est celle équipée de la **RTX 5090** utilisée pour ce test. Le pipeline (`pipeline/analyze.py::get_nlp`) doit donc recevoir le chemin d'activation GPU requis par les modèles retenus ; la disponibilité matérielle et un éventuel surcoût par rapport au pipeline CPU actuel ne sont pas des critères de rejet.

## Conclusion

**Résultats insuffisants pour faire de l'un des trois modèles spaCy la base de couverture.**

- Sur les composés (`nominal_compound`), les trois modèles sont dans un mouchoir de poche parmi les désaccords révisés (63–67% corrects) : aucun ne domine clairement, et les erreurs partagées (ponctuation, modificateur à trait d'union) dominent les écarts entre modèles.

- Sur les entités nommées, trf fait nettement mieux que sm et lg parmi les cas disputés (70% vs 40% et 45% correct, sur seulement 10-11 entrées chacun) — un signal plus net que sur les composés, mais mesuré sur un échantillon minuscule et volontairement biaisé vers les désaccords (voir la mise en garde ci-dessus), et trf introduit aussi sa propre erreur (`dis:04`). lg, en particulier, n'apporte pas de gain net qui justifierait son coût par rapport à sm (mêmes classes d'erreurs, en plus il rate `nursing home`).

- Le gain trf le plus concret (bornes d'entité multi-mots plus complètes) est réel mais localisé. La RTX 5090 étant disponible sur la machine de production, le GPU ne constitue ni une réserve ni un coût d'infrastructure discriminant. Ce gain localisé ne suffit toutefois pas à résoudre les erreurs de composés et de bornes partagées par les trois modèles.

- Recommandation : ne pas limiter la suite à un choix entre `sm`, `lg` et `trf`. Conserver provisoirement les sorties spaCy comme signaux auditables, mais évaluer une détection multi-source à haut rappel et des alternatives plus lourdes sur GPU (transformer spécialisé, proposition contextuelle/LLM locale, ensemble), avec sélection sur la qualité bout en bout. `trf` peut être exploité pour son meilleur signal NER, sans être présumé suffisant. Ne pas approfondir `lg` seul : sur cet échantillon il n'apporte de gain net ni sur NER ni sur compound par rapport à `sm`.

