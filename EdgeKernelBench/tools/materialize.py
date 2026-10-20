#!/usr/bin/env python3
"""Inspect canonical tasks and materialize a scenario or external profile."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(r"__EKB_(PARAM|INPUT)_([A-Za-z0-9_]+)__")
SCENARIO_COUNTS = {"mobile_personal": 204, "smart_aiot": 228, "embodied_robotics": 218}


def scenario_profiles(scenario: str) -> dict[str, dict[str, Any]]:
    """Load the canonical workload definitions for one application domain."""
    import yaml

    if scenario not in SCENARIO_COUNTS:
        raise ValueError(f"unknown scenario: {scenario}")
    path = ROOT / "shapes" / f"{scenario}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid scenario schema: {scenario}")
    profiles = payload.get("tasks")
    if (payload.get("scenario") != scenario or not isinstance(profiles, dict)
            or payload.get("case_count") != len(profiles)
            or len(profiles) != SCENARIO_COUNTS[scenario]):
        raise ValueError(f"invalid scenario identity or task count: {scenario}")
    return profiles


def metadata(template: str) -> dict[str, Any]:
    """Read descriptive metadata without importing or executing a task."""
    for node in ast.parse(template).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_METADATA"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise ValueError("template has no TASK_METADATA mapping")


def task_index() -> dict[str, Path]:
    tasks: dict[str, Path] = {}
    for path in sorted((ROOT / "tasks").glob("level*/*.py.tpl")):
        task_id = metadata(path.read_text(encoding="utf-8"))["task_id"]
        if task_id in tasks:
            raise ValueError(f"duplicate task ID: {task_id}")
        tasks[task_id] = path
    return tasks


def profile_fields(template: str) -> dict[str, list[str]]:
    tokens = TOKEN_PATTERN.findall(template)
    return {
        "parameters": sorted({key for kind, key in tokens if kind == "PARAM"}),
        "input_constants": sorted({key for kind, key in tokens if kind == "INPUT"}),
    }


def python_literal(value: Any) -> str:
    if isinstance(value, dict) and set(value) == {"python_expr"}:
        expression = value["python_expr"]
        if not isinstance(expression, str):
            raise ValueError("python_expr must be a string")
        ast.parse(expression, mode="eval")
        return f"({expression})"
    return repr(value)


def render(template: str, profile: dict[str, Any]) -> str:
    """Substitute profile values while preserving the template computation."""
    if not isinstance(profile, dict):
        raise ValueError("profile must be a JSON object")
    unknown = set(profile) - {"parameters", "input_constants"}
    if unknown:
        raise ValueError(f"unknown profile sections: {sorted(unknown)}")
    fields = profile_fields(template)
    for section, required in fields.items():
        supplied = profile.get(section, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"{section} must be a mapping")
        missing, extra = set(required) - set(supplied), set(supplied) - set(required)
        if missing or extra:
            raise ValueError(
                f"{section}: missing={sorted(missing)}, unexpected={sorted(extra)}"
            )

    rendered = template
    parameters = sorted(
        profile.get("parameters", {}).items(), key=lambda item: len(item[0]), reverse=True
    )
    for key, value in parameters:
        rendered = rendered.replace(f"__EKB_PARAM_{key}__", python_literal(value))
    for key, value in profile.get("input_constants", {}).items():
        rendered = rendered.replace(f"__EKB_INPUT_{key}__", python_literal(value))
    if TOKEN_PATTERN.search(rendered):
        raise ValueError("unresolved template values")
    ast.parse(rendered)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List canonical task IDs.")
    action.add_argument("--describe", metavar="TASK", help="Show metadata and required profile fields.")
    action.add_argument("--task", metavar="TASK", help="Materialize one task.")
    profile_source = parser.add_mutually_exclusive_group()
    profile_source.add_argument("--scenario", choices=SCENARIO_COUNTS,
                                help="Built-in scenario profile for --task.")
    profile_source.add_argument("--profile", type=Path, help="External JSON profile for --task.")
    parser.add_argument("--output", type=Path, help="New Python output file for --task.")
    args = parser.parse_args()
    if args.task and ((args.profile is None and args.scenario is None) or args.output is None):
        parser.error("--task requires --scenario or --profile, and --output")
    if not args.task and (args.profile is not None or args.scenario is not None or args.output is not None):
        parser.error("--scenario, --profile, and --output require --task")
    try:
        tasks = task_index()
        if args.list:
            print("\n".join(tasks))
            return
        task_id = args.describe or args.task
        if task_id not in tasks:
            raise ValueError(f"unknown task ID: {task_id}")
        template = tasks[task_id].read_text(encoding="utf-8")
        if args.describe:
            print(json.dumps({"metadata": metadata(template), "profile_fields": profile_fields(template)}, indent=2))
            return
        if args.scenario is not None:
            profiles = scenario_profiles(args.scenario)
            if task_id not in profiles:
                raise ValueError(f"task {task_id} is not defined in scenario {args.scenario}")
            profile = profiles[task_id]
        else:
            profile = json.loads(args.profile.read_text(encoding="utf-8"))
        rendered = render(template, profile)
        if args.output.suffix != ".py":
            raise ValueError("output must have a .py extension")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered)
        print(f"Materialized {task_id} -> {args.output}")
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
