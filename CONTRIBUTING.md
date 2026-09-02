# Contributing

`main` is protected. Every change lands through a pull request — no direct
pushes, including from maintainers. Two things are required before a PR
can merge:

1. **CI green.** `.github/workflows/reproduce.yml` reruns the project's six
   headline scripts (Gate 1, its full-scale correction, the Xavier/He
   proof, the full method comparison, OOD holdout, the learned-combination
   overfitting check) from a clean checkout. If it doesn't reproduce, it
   doesn't merge — this is the whole point of a project whose central claim
   is "we corrected our own overstated result once already."
2. **One approving review.**

## What a good PR looks like here

This project's standard (see [README](README.md) §2-3) is that a claim
without a number and a script behind it doesn't go in. That standard
applies to contributions too — the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
enforces it structurally: every PR that touches a result must show the
exact command and output that produced it, and if it changes an existing
number, both the old and new value with a stated reason for the change
(bigger sample, bug fix, different method — not "seems better").

Negative results are welcome and expected to be merged, not filtered out.
If you tried something and it didn't beat the current best method, that's
still a contribution — see how `results/reports/` already handles this for
LSUV init, ZiCo, and every proxy-combination method tried so far.

## Local setup

```bash
pip install -r requirements.txt
python scripts/compare_meta_predictors.py   # sanity check: should reproduce README's TL;DR numbers
```

`data/meta_dataset.db` ships with the repo so every scale-corrected result
reproduces without rerunning training. If your change requires new
training data, say so explicitly in the PR and explain why the existing
312-task meta-dataset isn't enough.

## Reporting a result, not a code change

Opening an issue with a counter-result (a proxy or method that beats what's
here, run against the same locked test split) is exactly the kind of
contribution this project wants most. Link the script and output; a PR
isn't required to start that conversation.
