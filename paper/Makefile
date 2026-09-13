PYTHON ?= python3

.PHONY: all figures check
all: main.pdf

figures:
	$(PYTHON) scripts/analyze_results.py

main.pdf: main.tex references.bib ieeeconf.cls IEEEtran.bst generated/numbers.tex generated/suite_table.tex generated/status_table.tex generated/failure_table.tex figures/architecture.tex figures/per_task.pdf figures/rollouts.pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check: main.pdf
	$(PYTHON) scripts/check_paper.py
