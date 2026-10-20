"""Inspect a task or validate a candidate using the shared host pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import load_task_isolated, run_build_isolated, validate_build_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Trusted PyTorch task")
    parser.add_argument("--candidate", type=Path, help="Python source defining build(task)")
    args = parser.parse_args()
    loaded = load_task_isolated(args.task)
    report = {"task": loaded.build_task.metadata(), "parameter_variants": len(loaded.variants)}
    if args.candidate is not None:
        source = args.candidate.read_text(encoding="utf-8")
        reports = []
        for variant in loaded.variants:
            execution = run_build_isolated(source, variant.build_task)
            validation = validate_build_result(execution.result, variant.build_task)
            reports.append({"parameter_seed": variant.parameter_seed,
                            "validation": validation.to_manifest()})
        report["candidate_validation"] = reports
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
