#!/usr/bin/env python3
"""Check canonical task definitions and all scenario materializations."""

from __future__ import annotations

import ast
from collections import Counter

from materialize import SCENARIO_COUNTS, metadata, profile_fields, render, scenario_profiles, task_index


def main() -> None:
    counts: Counter[int] = Counter()
    numbers: dict[int, list[int]] = {1: [], 2: [], 3: []}
    tasks = task_index()
    scenarios = {name: scenario_profiles(name) for name in SCENARIO_COUNTS}
    memberships = {name: set() for name in SCENARIO_COUNTS}
    for scenario, profiles in scenarios.items():
        unknown = set(profiles) - set(tasks)
        if unknown:
            raise ValueError(f"unknown tasks in {scenario}: {sorted(unknown)}")
    rendered_count = 0
    for task_id, path in tasks.items():
        template = path.read_text(encoding="utf-8")
        tree = ast.parse(template)
        info = metadata(template)
        level = info["level"]
        number = int(path.name.split("_", 1)[0])
        name = path.name.split("_", 1)[1].removesuffix(".py.tpl")
        if level not in numbers or path.parent.name != f"level{level}":
            raise ValueError(f"invalid level: {path.name}")
        if task_id != f"ekb_l{level}_{number:03d}_{name}" or info["name"] != name:
            raise ValueError(f"identity mismatch: {path.name}")
        definitions = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        if not {"Model", "get_inputs", "get_init_inputs"} <= definitions:
            raise ValueError(f"missing task interface: {task_id}")
        if not info["operator_tags"] or not info["scenario_tags"]:
            raise ValueError(f"missing task tags: {task_id}")
        tags = info["scenario_tags"]
        if not isinstance(tags, list) or len(tags) != len(set(tags)) or set(tags) - set(scenarios):
            raise ValueError(f"invalid scenario tags: {task_id}")
        for scenario, profiles in scenarios.items():
            if (scenario in tags) != (task_id in profiles):
                raise ValueError(f"scenario membership mismatch: {task_id}, {scenario}")
            if task_id in profiles:
                ast.parse(render(template, profiles[task_id]))
                memberships[scenario].add(task_id)
                rendered_count += 1
        if not isinstance(info["semantic_references"], list):
            raise ValueError(f"invalid semantic references: {task_id}")
        if not any(profile_fields(template).values()):
            raise ValueError(f"missing profile placeholders: {task_id}")
        counts[level] += 1
        numbers[level].append(number)
    if dict(counts) != {1: 128, 2: 162, 3: 41}:
        raise ValueError(f"unexpected task counts: {dict(counts)}")
    for level, found in numbers.items():
        if sorted(found) != list(range(1, counts[level] + 1)):
            raise ValueError(f"non-contiguous numbering in level {level}")
    print(f"PASS: {sum(counts.values())} templates; L1={counts[1]}, L2={counts[2]}, L3={counts[3]}")
    for scenario, expected in SCENARIO_COUNTS.items():
        if len(memberships[scenario]) != expected:
            raise ValueError(f"unexpected scenario membership count: {scenario}")
    print(f"PASS: {rendered_count} scenario profiles; " + ", ".join(
        f"{name}={len(memberships[name])}" for name in SCENARIO_COUNTS))
    print("Validated task interfaces, profile membership, substitutions, and rendered Python syntax.")


if __name__ == "__main__":
    main()
