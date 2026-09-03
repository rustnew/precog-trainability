# Exploring LSUV (Data-Aware Init)

_Generated 2026-09-01T14-29-10Z (UTC)_

## Method

LSUV (data-aware init, Mishkin & Matas 2015, source.md pillar 4) added as a
4th candidate alongside Xavier/He/Orthogonal, controlled per §21 (task,
architecture, LR=0.02, batch=32,
optimizer=adam all fixed, only init_method varies), across
12 synthetic tasks (the same 12 as gate1_ranking.py).

## Results

| task | xavier | he | orthogonal | lsuv | best |
|---|---:|---:|---:|---:|---|
| linear | 19 | 17 | 20 | 26 | he |
| nonlinear_interaction | 115 | 117 | 115 | 98 | lsuv |
| nonlinear_product | 191 | 127 | 264 | 189 | he |
| linear | 26 | 62 | 42 | 78 | xavier |
| nonlinear_interaction | 91 | 120 | 81 | 133 | orthogonal |
| nonlinear_product | 606 | 284 | 250 | 526 | orthogonal |
| linear | 31 | 31 | 23 | 37 | orthogonal |
| nonlinear_interaction | 252 | 295 | 266 | 264 | xavier |
| nonlinear_product | 186 | 252 | 191 | 318 | xavier |
| linear | 31 | 32 | 41 | 46 | xavier |
| nonlinear_interaction | 101 | 108 | 100 | 121 | orthogonal |
| nonlinear_product | 60 | 77 | 85 | 77 | xavier |

| init | wins | mean steps |
|---|---:|---:|
| xavier | 5/12 | 142 |
| he | 2/12 | 127 |
| orthogonal | 4/12 | 123 |
| lsuv | 1/12 | 159 |

## Verdict

LSUV beats all three analytic inits on average steps: **False**.
Random chance for any one init to win a given task: 25%.
