"""Command-line entry point."""

from __future__ import annotations

import json
from typing import Sequence

from .config import parse_config
from .evaluator import evaluate, format_summary
from .results import episode_schedule


def main(argv: Sequence[str] | None = None) -> int:
    config = None
    try:
        config, dry_run = parse_config(argv)
        schedule = episode_schedule(
            config.route,
            config.suite,
            config.task_ids,
            config.episodes_per_task,
            config.init_state_start,
        )
        if dry_run:
            print(
                json.dumps(
                    {
                        "config": config.serializable(),
                        "episodes": [key.id for key in schedule],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        summary = evaluate(config)
        print(f"completed: {format_summary(summary)}")
        print(f"summary: {config.summary_path}")
        return 0
    except KeyboardInterrupt:
        if config is not None and config.formal:
            print(
                "formal evaluation interrupted and cannot be resumed; "
                "use a new campaign directory"
            )
        else:
            print("interrupted; completed JSONL rows are resumable with --resume")
        return 130
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
