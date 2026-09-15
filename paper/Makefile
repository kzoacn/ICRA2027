PYTHON ?= python3

.PHONY: all figures check
all: main.pdf

figures:
	$(PYTHON) scripts/analyze_results.py
	$(PYTHON) scripts/build_comparison_table.py

generated/comparison_table.tex generated/model_table.tex generated/model_numbers.tex &: data/published_comparisons.json data/episodes.jsonl data/provenance.json $(wildcard data/full400_reference/episodes.jsonl data/full400_reference/provenance.json) data/model_audit.json scripts/build_comparison_table.py ../scripts/source_archive.py ../configs/resources.lock.json ../archive/evaluated-source/manifest.json ../archive/evaluated-source/full400-source.zip
	$(PYTHON) scripts/build_comparison_table.py

main.pdf: main.tex generated/rollout_metadata.tex references.bib ieeeconf.cls IEEEtran.bst generated/numbers.tex generated/model_numbers.tex generated/suite_table.tex generated/status_table.tex generated/failure_table.tex generated/comparison_table.tex generated/model_table.tex figures/architecture.tex figures/per_task.pdf figures/rollouts.pdf $(wildcard figures/recorded/*.png)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check: main.pdf
	$(PYTHON) scripts/check_paper.py
