PYTHON ?= python3

.PHONY: all figures check
all: main.pdf

figures:
	$(PYTHON) scripts/analyze_results.py
	$(PYTHON) scripts/build_comparison_table.py

generated/comparison_table.tex generated/model_table.tex generated/model_numbers.tex &: data/published_comparisons.json data/episodes.jsonl data/provenance.json data/model_audit.json scripts/build_comparison_table.py ../route-b-v170-cloud/resources.lock.json ../route-b-v170-cloud/libero_system/perception/grounding_dino.py
	$(PYTHON) scripts/build_comparison_table.py

main.pdf: main.tex references.bib ieeeconf.cls IEEEtran.bst generated/numbers.tex generated/model_numbers.tex generated/suite_table.tex generated/status_table.tex generated/failure_table.tex generated/comparison_table.tex generated/model_table.tex figures/architecture.tex figures/per_task.pdf figures/rollouts.pdf $(wildcard figures/recorded/*.png)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check: main.pdf
	$(PYTHON) scripts/check_paper.py
