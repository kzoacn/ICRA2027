# ICRA 2027 anonymous manuscript

**ANCHOR: Asset-Informed Geometric Skills for Language-Conditioned Manipulation**

The current manuscript uses the completed local full evaluation:
**376/400 (94%)**, with one frozen controller and 400 new episodes.
Spatial: 93/100 / Object: 98/100 / Goal: 98/100 / Long: 87/100.
The four suites contain ten tasks each, evaluated on official initial-state
indices 0-9 with seed 7 and action budgets 220/280/300/520.
The score is external ever-success; the controller does not receive the benchmark
predicate for selecting actions or deciding when to stop.

- [Paper PDF](main.pdf)
- [LaTeX source](main.tex)
- [References](references.bib) and [citation audit](../docs/paper/citation-audit.md)
- [Local full-run acceptance](../experiments/libero400/accepted_result.json)
- [Environment](data/environment.json) and [data provenance](data/provenance.json)
- [Projected records](data/episodes.jsonl) and [complete raw records](data/raw_episodes.jsonl.gz)
- [Statistics](data/statistics.json) and [model audit](data/model_audit.json)
- [Figure provenance](data/figure_provenance.json)
- [Evidence map](../docs/paper/evidence.md) and [review notes](../docs/paper/review.md)
- [Local artifact validation](data/artifact_validation.json)

The host is an RTX 3090 under Ubuntu 24.04/WSL2, with Python 3.12.14,
PyTorch 2.11.0+cu130, MuJoCo 3.3.2, robosuite 1.4.0, and hf-libero 0.1.4.
The dispatcher starts with four workers and then uses six. The source snapshot
is `7bdcb97a7b847debeb0bd47c951455934701f81d7b561ad097fe5b41269de256`. All 585 asset files and eight detector
files pass the pinned resource hashes. The [local run record](../experiments/libero400/README.md)
preserves environment, scheduling, source and acceptance details. The exact evaluated
source is archived in [full400-source.zip](../archive/evaluated-source/full400-source.zip);
the [repository guide](../docs/repository.md) maps it to the current `src/anchor/` layout.

## Building

From this directory:

    make
    make check

Building uses committed figures and tables and requires no simulator or GPU.
Required tools include PDFLaTeX, BibTeX, latexmk and Poppler. The unmodified
official IEEE class and bibliography style are included.

To rebuild numerical tables and the task plot from the committed records:

    python3 -m pip install -r requirements-figures.txt
    make figures
    make check

## Reproducing the full evaluation

Prepare a deployment with `.venv` and pinned `resources` using the
[setup guide](../docs/setup.md), then run from the repository root:

    python3 scripts/run_batch.py --label full400_reproduction --all --workers 6 --deployed-root /path/to/deployment

The runner freezes the controller before starting and retains all outcomes,
configuration, source manifests, raw traces and videos. Runtime failures or
incomplete coverage cannot pass final import validation.

To import the recorded local batch from the repository root (its frozen runtime
directory retains the original name):

    python3 paper/scripts/capture_results.py --batch-dir runtime/route_b_90/full400_local_20260914T153307Z --environment experiments/libero400/environment.json
    python3 paper/scripts/make_rollout_figure.py --batch-dir runtime/route_b_90/full400_local_20260914T153307Z
    make -C paper figures
    make -C paper check

The importer validates coverage and immutable source hashes, preserves each
original JSONL line in the compressed archive, and records each episode's
campaign and controller hash. Environment metadata is copied with its SHA256.
New reproduction runs need their own measured environment record and batch name.

## Figures and model audit

Rollout images come from this local full run. The generator selects the first
external success in Object 00 and the first external failure in Goal 03 when
available, with a deterministic task/index fallback. All outcomes remain in
the numerical results. Four uniformly spaced recorder indices are extracted
from each selected video; the fixed-camera half is flipped vertically for
upright display. There is no retouching or synthetic scene content.

`data/figure_provenance.json` stores episode and source-record identities,
video hashes, frame indices, logged phases and exported image hashes. Frame
index times recording stride gives the executed action count; phase labels
use the last recorded transition at or before that count. The failure timeline
separates earlier actions, the final active phase and unused global budget.
These are controller trace labels, not physical states inferred from the images.
The generated `rollout_metadata.tex` supplies task, instruction, action-count,
phase-duration and image-path macros. `figures/architecture.tex` remains an
editable vector schematic with actual camera insets.

`statistics.json` also reports conditional completion agreement, failure
concentration by suite, and stopping relative to the global action caps.
These quantities are derived from the same frozen 400 records, and `make check`
checks the numerical macros, frame alignment and phase intervals against them.

The deployed frozen detector has **172,249,090 unique parameters**, including
vision and text encoders. Its FP32 weight file contains 689,359,096 bytes.
One detector serves both cameras in each evaluation process. Geometric skills,
the grammar and asset gallery add no neural parameters. No robot demonstrations
train an action policy; perception pretraining and manual skill development
remain prior requirements. The local parameter audit loads the deployed class
on CPU to count weights; it is not a CPU inference benchmark. From the repository root:

    python3 paper/scripts/audit_model.py --model-dir /path/to/resources/grounding-dino-tiny
    python3 paper/scripts/build_comparison_table.py

## Published comparisons and evidence scope

The five external comparison rows retain the published values and source
versions in `data/published_comparisons.json`: OpenVLA v3 Table 12,
SmolVLA v1 Table 2, and OpenVLA-OFT v2 Table I. Those policies were not rerun
here. Their sensing, policy training and evaluation protocols differ from
ANCHOR's calibrated RGB-D, known assets, manual skills and stopping rule.
The ANCHOR row is computed from this local full evaluation.

The evaluation covers known assets and supported instruction forms, including
initial states used during development. It does not establish held-out
generalization or isolate component effects through ablations. First-success
time and separate final-state success are not recorded; the outcome measures
whether the official predicate was reached at any point before stopping.
The manuscript and references remain an anonymous draft and have not been submitted.

Earlier data remain preserved under `data/full400_reference/` and
`data/task_updates_reference/`, with the corresponding development records in
[archive/experiments/](../archive/experiments/).
They are historical references; they do not supply episodes to the current result.

## Submission format sources

Date consulted: 2026-09-14.

The [official ICRA 2027 call for papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
specifies a maximum of eight pages including references, two-column formatting,
and double-anonymous review. Content generated using generative AI must be disclosed
in the acknowledgments. The manuscript names **OpenAI GPT-6-Astra (via Codex)**
and describes its use for text, LaTeX, and figure/table scripts.
Following the FAQ on that page, the PDF retains readable URL text without clickable
link annotations.

Local checks cover page count, Letter paper size, anonymous metadata, font embedding,
undefined references, layout overflow, coverage of all 400 records, and consistency
between comparison tables and source data, including the pinned model audit.
These checks are not equivalent to official
PaperPlaza compliance testing or scientific review by the authors.

Official template sources and original file SHA256 hashes:

- `ieeeconf.cls` from [ieeeconf.zip](https://ras.papercept.net/conferences/support/files/ieeeconf.zip):
  `4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2`
- `IEEEtran.bst` from [IEEEtranBST.zip](https://ras.papercept.net/conferences/support/files/IEEEtranBST.zip):
  `b11af8e5096681f1eccdce6c72c047dc056ddeef52dff340213104019bcf3409`

The template files are unmodified and retain their original copyright and license notices.
