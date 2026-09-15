# Citation audit — 2026-09-14

All 14 references were checked against primary sources for bibliographic identity and support for the manuscript's claims. The five comparison rows contain 25 reported success-rate values; all 25 values and all 15 stored standard errors agree with the specified source PDFs. No experimental result or comparison value was changed.

The local Zotero API and connector were unavailable during the check. Verification proceeded using the papers, publisher records, publisher-deposited DOI metadata, and the authors' source code. Downloaded source PDFs were used locally and are not redistributed.

## Reference-by-reference findings

| Key | Primary source and verified publication | Claim supported and action taken |
| --- | --- | --- |
| libero | [NeurIPS 2023 paper and publisher BibTeX](https://proceedings.nips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html), vol. 36, pp. 44776–44791; seven authors | Benchmark origin and lifelong-learning setting. Section 3 describes Spatial/Object/Goal and the split of LIBERO-100 into LIBERO-90 and ten long tasks. Replaced the preprint entry with the proceedings record. |
| openvla | [arXiv:2406.09246v3](https://arxiv.org/abs/2406.09246v3), revised September 5, 2024; 18 authors | Learned action generation and 7B model class; Appendix E.1 supplies training/input/evaluation conditions, Table 12 supplies three comparison rows. Kept the exact arXiv version and its author list, and made the version part of the bibliographic identifier. |
| smolvla | [arXiv:2506.01844v1](https://arxiv.org/abs/2506.01844v1), June 2, 2025; 14 authors | Sections 3.1, 4.1, 4.3 and 4.5 and Table 2 support the 0.45B simulation model, VLM initialization, demonstration-based action training, and ten trials per task. Pinned v1 and named the selected variant in the entry. |
| openvlaoft | [RSS 2025 proceedings](https://www.roboticsproceedings.org/rss21/p017.html), DOI 10.15607/RSS.2025.XXI.017; three authors; numerical source remains [arXiv v2](https://arxiv.org/abs/2502.19645v2) | Section V and Tables I/IV support per-suite fine-tuning, filtered demonstrations, 500 trials per suite, two RGB views, robot state and the selected full model. Added the publication record while retaining the exact numerical source in the note. The RSS PDF's Table I was also checked visually and agrees. |
| tamp | [Annual Reviews record](https://www.annualreviews.org/content/journals/10.1146/annurev-control-091420-084139), 4(1):265–293, 2021; seven authors | Supports coupling discrete task decisions and continuous geometric planning. Replaced the 2020 preprint metadata with the published journal record and DOI. |
| saycan | [Published PDF](https://proceedings.mlr.press/v205/ichter23a/ichter23a.pdf), CoRL 2022, PMLR 205:287–318, published 2023; 45 authors | Sections 1–2 support skill selection using language probabilities and learned affordance/value functions. Restored the complete author list in the PDF's alphabetical order and added proceedings details. See the source discrepancy below. |
| codepolicies | [IEEE publication](https://doi.org/10.1109/ICRA48891.2023.10160591), ICRA 2023, pp. 9493–9500; eight authors; [author-hosted project](https://code-as-policies.github.io/) | Section III of the [authors' v4 paper](https://arxiv.org/abs/2209.07753v4) supports programs that process perceptions, express feedback logic and call control APIs. Added the publication venue, pages and DOI, cross-checked with IEEE-deposited DOI metadata. |
| voxposer | [PMLR publication](https://proceedings.mlr.press/v229/huang23b.html), 229:540–562, 2023; six authors | Section 3 supports language-grounded 3-D value maps for closed-loop motion planning. Existing publication metadata and author order were correct. |
| kpam | [Springer publication](https://link.springer.com/chapter/10.1007/978-3-030-95459-8_9), Robotics Research, SPAR 20:132–157, 2022; four authors | Section 3 and the [authors' v2 manuscript](https://arxiv.org/abs/1903.06684v2) support semantic keypoints and geometric goal constraints. Added the published chapter metadata. ISRR took place in 2019; the proceedings publication is 2022. |
| rekep | [PMLR publication](https://proceedings.mlr.press/v270/huang25g.html), 270:4573–4602, 2025; five authors | Section 3 and the [authors' v2 manuscript](https://arxiv.org/abs/2409.01652v2) support relational keypoint constraints and hierarchical optimization. Updated to the published proceedings. CoRL took place in 2024; PMLR's publication year is 2025. |
| groundingdino | [Springer publication](https://link.springer.com/chapter/10.1007/978-3-031-72970-6_3) and [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf), LNCS 15105:38–55; 12 authors | The abstract and method support text-conditioned object detection. Added pages, volume and DOI. Used 2025 as assigned by Springer's citation/export; ECCV and the first online publication were in 2024. The deployed 172M count is supported by this project's model audit, not inferred from this paper. |
| robosuite | [arXiv:2009.12293v3](https://arxiv.org/abs/2009.12293v3), January 18, 2025; nine authors | Supports the simulation-framework attribution. The existing v3 author list and date are correct. This revision describes robosuite 1.5; the manuscript's deployed 1.4.0 version is independently documented in the frozen environment records. These are distinct facts. |
| mujoco | [IEEE publication](https://doi.org/10.1109/IROS.2012.6386109) and [author-hosted PDF](https://www.roboti.us/lab/papers/TodorovIROS12.pdf), IROS 2012, pp. 5026–5033; three authors | Supports physics-engine attribution. Author order, title, year and pages were correct; added the DOI. Updated the evidence link to the reachable author-hosted copy. The software version comes from the experiment's environment records. |
| openvlacode | [Evaluation loop](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py) and [environment helper](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py) | Replaced a moving repository link with a commit-pinned directory link. Verified horizons, settling, the environment seed, and success-driven early stopping against the exact files. Details below. |

## Version and author-order decisions

The [SayCan publisher page](https://proceedings.mlr.press/v205/ichter23a.html) exports Brian Ichter first, while the published PDF starts with Michael Ahn and explicitly states that its authors are listed alphabetically. The bibliography follows the PDF and preserves all 45 names.

The [OpenVLA PMLR web record](https://proceedings.mlr.press/v270/kim25c.html) also has an author field that differs from the cited arXiv v3. The numerical source, its table number, and its 18-author byline were kept together; metadata from a different version was not substituted.

All full author lists remain in references.bib. The existing IEEEtran style is configured to display the first six names followed by “et al.” when an entry has more than ten authors. The style-control entry produces no reference number and is not counted among the 14 sources.

## Comparison values and conditions

Columns below follow Spatial / Object / Goal / Long / Average. Source-reported averages are retained, including rounding; they are not recomputed from the rounded suite values.

| Method | Verified success rates (%) | Numerical source |
| --- | --- | --- |
| Diffusion Policy | 78.3 / 92.5 / 68.3 / 50.5 / 72.4 | [OpenVLA v3, Table 12, PDF p. 37](https://arxiv.org/pdf/2406.09246v3) |
| Octo | 78.9 / 85.7 / 84.6 / 51.1 / 75.1 | [OpenVLA v3, Table 12, PDF p. 37](https://arxiv.org/pdf/2406.09246v3) |
| OpenVLA | 84.7 / 88.4 / 79.2 / 53.7 / 76.5 | [OpenVLA v3, Table 12, PDF p. 37](https://arxiv.org/pdf/2406.09246v3) |
| SmolVLA 0.45B | 90.0 / 96.0 / 92.0 / 71.0 / 87.3 | [SmolVLA v1, Table 2, PDF p. 11](https://arxiv.org/pdf/2506.01844v1) |
| OpenVLA-OFT | 97.6 / 98.4 / 97.9 / 94.5 / 97.1 | [OpenVLA-OFT v2, Table I, PDF p. 6](https://arxiv.org/pdf/2502.19645v2) |

- OpenVLA's authors evaluated the first three rows themselves; citing their Table 12 identifies the producer of these particular measurements. Each suite/seed has 500 trials, averaged over three seeds. All three policies use the same filtered demonstrations and static camera input. The prose now states the seed/trial aggregation.
- The selected SmolVLA row is the 0.45B model without robotics pretraining. It still learns actions from demonstrations. Its 1,693-episode dataset spans 40 tasks, with ten evaluation trials per task. The model description supports RGB and robot-state input; the paper does not enumerate the LIBERO camera count, so no camera count is asserted.
- The selected OpenVLA-OFT row uses the modified training dataset, wrist image and robot state. Table IV confirms these inputs. The 7B comparison is the paper's rounded base-model class, not an exact count of the adapted checkpoint.
- The ANCHOR row remains derived from its existing 400 records. Differences in information, training and scoring continue to be stated in the manuscript; no controlled superiority claim is introduced.

The table entries were inspected in rendered source pages and parsed from the PDFs' text to check every numerical value against published_comparisons.json. The three downloaded PDF hashes exactly matched its previously recorded hashes:

| Fixed source | SHA256 |
| --- | --- |
| 2406.09246v3 | 353c37df34458f12f969b14dfd8b77175b727b9cddea7bb891759beddeefe1be |
| 2506.01844v1 | cb1bb9a8f824187fcdc32af8c290214487c524f139a3c2def1c6d97adb1fa40c |
| 2502.19645v2 | b860aa1206b6cfb0ce8be177f961379dd6a133d52cc74ac346636e0f4952a596 |

The original JSON, source hashes, comparison rows, model audit and generated numerical tables were retained unchanged.

## Code reference

Pinned repository commit: c8f03f48af692657d3060c19588038c7220e9af9. The bibliography prints its 12-character prefix; the complete identifier and file links are preserved here.

| Claim | Exact source location |
| --- | --- |
| Ten settling steps and 50 trials per task by default | [run_libero_eval.py, lines 72–73](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L72-L73) |
| 220 / 280 / 300 / 520 action horizons | [run_libero_eval.py, lines 173–180](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L173-L180) |
| Environment seed is zero, independently of the configuration's global seed | [libero_utils.py, line 24](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py#L24), [global seed at line 85](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L85) |
| Success increments the count and breaks execution | [run_libero_eval.py, lines 228–232](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L228-L232) |

The fetched evaluation script has SHA256 369d9fc3d59067998ebfadafbd516872c7ff6f9f06a0a48cda8b6ca48051de77; the helper has SHA256 6c162ce0954c8659018ee8ff1604a33904b50ac0631a8ca0f8bf1f7a4d25181d. These source conditions support the manuscript's distinction between matching numerical horizons and reproducing an entire evaluation protocol.

## Verification of the revised artifact

The existing make check validates the compiled PDF, reference resolution, page count, font embedding, anonymity, comparison table consistency, model audit and all 400 original records. A separate comparison against the pre-audit checksums confirms that experiment code, configurations, records, statistics, generated numerical tables and figure assets are unchanged. The only result-facing prose additions specify the external papers' source versions and aggregation conditions.
