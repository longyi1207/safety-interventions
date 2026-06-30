.PHONY: all figures clean pdf

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

all: figures pdf

figures:
	@if [ -x .venv/bin/python ]; then .venv/bin/python generate_figures.py; else python3 generate_figures.py; fi

pdf: $(MAIN).tex appendix.tex references.bib figures/*.pdf
	@if command -v $(LATEX) >/dev/null 2>&1; then \
		$(LATEX) $(MAIN) && $(BIBTEX) $(MAIN) && $(LATEX) $(MAIN) && $(LATEX) $(MAIN); \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic -X compile $(MAIN).tex --outdir . ; \
	else \
		echo "Install MacTeX (pdflatex) or: brew install tectonic"; exit 1; \
	fi

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf
