# Calibration S5-2

Corpus Q0-2 ciblé : **17 occurrences**.

## Matrice de confusion (acceptation automatique / révision)

| Politique | Correct accepté | Correct révisé | Incorrect accepté | Incorrect révisé |
|---|---:|---:|---:|---:|
| Marge historique `< 0,15` | 2 | 0 | 13 | 2 |
| Politique calibrée | 0 | 2 | 0 | 15 |

## Calibration par tranche

| Confiance | N | Confiance moyenne | Exactitude top-1 brute |
|---|---:|---:|---:|
| [0.00,0.40] | 17 | 0.197 | 0.118 |
| [0.40,0.55] | 0 | 0.000 | 0.000 |
| [0.55,0.72] | 0 | 0.000 | 0.000 |
| [0.72,1.00] | 0 | 0.000 | 0.000 |

La calibration mesure ici la fiabilité du top-1 avant arbitrage. Les cas routés en révision ne sont pas présentés comme des erreurs corrigées : ils attendent l'arbitre fermé WordNet/custom.

## Cas nommés

| Mot@segment | Attendu | Top-1 | Décision | Confiance |
|---|---|---|---|---:|
| spa@159 | `spa.n.01` | `health_spa.n.01` | révision | 0.200 |
| barely@635 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| barely@646 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| barely@889 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| barely@2150 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| barely@2294 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| barely@2357 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| barely@2375 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| barely@2466 | `barely.r.02` | `barely.r.01` | révision | 0.200 |
| touch@1725 | `use.v.01` | `touch.v.03` | révision | 0.200 |
| touch@1942 | `touch.v.01` | `touch.v.01` | révision | 0.200 |
| roll@2179 | `wheel.v.03` | `wheel.v.03` | révision | 0.200 |
| roll@2546 | `roll.v.01` | `roll.v.03` | révision | 0.150 |
| plow@317 | `plow_snow.v.01` | `plow.v.01` | révision | 0.200 |
| poke@412 | `poke.v.02` | `intrude.v.03` | révision | 0.200 |
| facility@794 | `facilities.n.02` | `facility.n.01` | révision | 0.200 |
| haggard@2028 | `haggard.s.01` | `bony.s.01` | révision | 0.200 |
