# Phase B trait map (v3)

## Dark Tetrad coverage (Paulhus et al.)

| Tetrad construct | Our trait id(s) |
|------------------|-----------------|
| Machiavellianism | `manipulation` |
| Narcissism | **`narcissism`** (grandiose / admiration) + partial overlap `entitlement`, `hubris_dominance` |
| Psychopathy | `callousness` (+ malevolence for sadism facet) |
| Sadism | `malevolence` |

## Non-Tetrad / safety

| Construct | Trait id |
|-----------|----------|
| Bandura moral disengagement | `moral_disengagement` |
| Entitlement (deservingness) | `entitlement` |
| Antisocial norms | `antisocial_norms` |
| Dominance / hubris | `hubris_dominance` |
| Sycophancy | `sycophancy` |

**Narcissism was missing as a named axis until v3** — previously only `entitlement` + `hubris_dominance` partially covered it.

## Weak-trait fix (v3)

`moral_disengagement` / `entitlement` got **0 pairs** under heuristic-only gates (trait text doesn’t score high on generic `evil_score`).

Fixes in `prompts/evil_qualities.json` v3:
- Clearer `trait_plus_system` (disengagement rationalizations vs raw cruelty; entitlement vs grandiosity)
- Per-trait `bootstrap`: `min_contrast: 2`, `attempts: 8`
- Re-run: `PIPELINE=weak_traits bash cloud/start_job.sh` (`--force --judge`)
