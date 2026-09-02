<!--
This project's whole premise is evidence over assumption (see docs.md §21,
§26). A PR that changes a result, a proxy, or a claim in the README must
carry the evidence for that change -- not just the diff. Fill in every
section; delete none. PRs that don't will be asked to add the missing
section before review starts.
-->

## What changed and why

<!-- One paragraph. If this changes or adds a result, name the specific
claim (e.g. "corrects Gate 1's rho from 0.670 to 0.395") rather than
describing the code mechanically. -->

## Evidence

<!-- Required for anything that changes a number, adds a proxy/method, or
touches precog/ or scripts/. Not required for pure docs/typo fixes --
delete this section only in that case. -->

- [ ] Ran the affected script(s) locally end-to-end and pasted the exact
      output (or attached the generated `results/reports/*.md`) below.
- [ ] If this changes an existing claim, the old and new numbers are both
      shown, with the reason for the change stated (bigger n, bug fix,
      different method).
- [ ] If this is a negative result, it's kept and labeled as such --
      this project documents failures with the same rigor as successes
      (see README §2-3 for why).

```text
paste command + output here
```

## Checklist

- [ ] CI (`reproduce.yml`) passes on this branch.
- [ ] No claim in this PR is asserted without a script/report backing it
      (no "should improve" without a number).
- [ ] `data/meta_dataset.db` only changed if this PR explains why (new
      tasks, a real bugfix) -- not as a side effect of running a script
      locally before committing.
- [ ] README/docs.md updated if this PR changes a number or conclusion
      referenced there.

## Open questions / requested feedback

<!-- Optional. If you want a second opinion on a specific design choice
or a result you're not fully sure about, say so here explicitly. -->
