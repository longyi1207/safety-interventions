# Paper build

Tamper-Induced Capability Collapse — arXiv preprint sources.

**Read the PDF:** [`main.pdf`](main.pdf) (also linked from repo root README)

## Quick start

```bash
cd paper

# Figures (from ../outputs/arxiv_mva/*.json)
python3 -m venv .venv
.venv/bin/pip install matplotlib
.venv/bin/python generate_figures.py

# PDF (needs MacTeX / TeX Live)
make
open main.pdf
```

## Contents

| File | Role |
|------|------|
| `main.tex` | Full paper |
| `appendix.tex` | Hyperparams, artifacts, judge notes |
| `references.bib` | Bibliography |
| `generate_figures.py` | Regenerates `figures/*.pdf` from JSON |
| `figures/` | Publication figures (PDF + PNG) |
| `main.pdf` | Compiled preprint (committed for easy access) |

## Before arXiv submit

1. Final proofread + any last-minute number sync from `outputs/arxiv_mva/`
2. Fill judge spot-check appendix when human labels are done
3. Run `make clean && make` to verify build
4. Upload `main.pdf` + source zip to arXiv (cs.CL / cs.LG)

## Data provenance

All numbers trace to `outputs/arxiv_mva/`. Regenerate table:

```bash
python3 scripts/build_arxiv_paper_table.py
```
