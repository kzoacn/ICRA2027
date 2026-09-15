# Evaluation records

[LIBERO 400](libero400/README.md) contains the current complete local evaluation:
**376/400 (94.0%)**, with 400 newly executed episodes using one frozen controller
on an RTX 3090. It includes environment, source, launch and acceptance records,
the episode projection, compressed original records, provenance and statistics.

Earlier development experiments are preserved in
[archive/experiments/](../archive/experiments/). Their recorded identifiers and
checksums remain unchanged.

For new evaluations, use `python3 scripts/run_batch.py` from the repository root.
See the [evaluation guide](../docs/evaluation.md) for the protocol and run validation.
