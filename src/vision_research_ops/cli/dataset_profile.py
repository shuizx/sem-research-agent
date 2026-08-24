"""CLI for explicit local dataset profiling with path-free machine-readable output."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vision_research_ops.application.services.dataset_profiling import (
    DatasetProfileError,
    profile_dataset,
)
from vision_research_ops.settings import Settings, load_local_env


def build_parser() -> argparse.ArgumentParser:
    """Build the small CLI surface for a user-authorized dataset root."""
    parser = argparse.ArgumentParser(description="Create a sanitized local DatasetProfile.")
    parser.add_argument(
        "--dataset-root", type=Path, help="explicit local root; overrides VRO_DATASET_ROOT"
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Profile the selected root and emit only safe identifiers and references."""
    options = build_parser().parse_args(argv)
    settings = Settings.from_env()
    root = options.dataset_root if options.dataset_root is not None else settings.dataset_root
    try:
        result = profile_dataset(root)
    except DatasetProfileError as error:
        print(json.dumps({"error": error.code, "status": "FAILED"}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "dataset_id": result.profile.dataset_id,
                "version": result.profile.version,
                "content_hash": result.profile.content_hash,
                "sample_count": result.sample_count,
                "profile_ref": result.profile_ref,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Load ignored local environment configuration then run the explicit profiler."""
    load_local_env()
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
