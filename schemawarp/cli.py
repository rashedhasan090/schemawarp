"""CLI entry point for schemawarp."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from schemawarp.core import (
    DEFAULT_CONFIG,
    SEVERITY,
    Finding,
    format_text,
    load_json,
    run,
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="schemawarp",
        description=(
            "Validate agent tool-call arguments against a JSON Schema catalog "
            "and flag mid-run argument shape drift."
        ),
    )
    p.add_argument("trace", help="JSONL tool-call transcript")
    p.add_argument(
        "-c",
        "--catalog",
        help="JSON tool catalog (name -> schema, OpenAI tools[], or {tools:{...}})",
    )
    p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    p.add_argument(
        "--fail-on",
        choices=("none", "low", "medium", "high"),
        default="high",
        help="exit 1 when any finding is at or above this severity (default: high)",
    )
    p.add_argument(
        "--min-severity",
        choices=("low", "medium", "high"),
        default="low",
        help="only report findings at or above this severity",
    )
    p.add_argument(
        "--no-learn",
        action="store_true",
        help="disable catalog-free shape/type drift learning",
    )
    p.add_argument(
        "--no-undeclared-tools",
        action="store_true",
        help="do not flag calls to tools missing from the catalog",
    )
    p.add_argument(
        "--learn-first",
        type=int,
        default=DEFAULT_CONFIG["learn_first"],
        help="calls per tool used to learn the baseline shape (default: 2)",
    )
    p.add_argument(
        "--drift-jaccard",
        type=float,
        default=DEFAULT_CONFIG["drift_jaccard"],
        help="Jaccard threshold for SHAPE_JUMP between consecutive calls (default: 0.5)",
    )
    p.add_argument(
        "--config",
        help="JSON file overriding DEFAULT_CONFIG keys",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = dict(DEFAULT_CONFIG)
    config["learn_first"] = args.learn_first
    config["drift_jaccard"] = args.drift_jaccard
    if args.config:
        overlay = load_json(args.config)
        if not isinstance(overlay, dict):
            print("schemawarp: --config must be a JSON object", file=sys.stderr)
            return 2
        config.update(overlay)

    try:
        findings, call_count = run(
            args.trace,
            catalog_path=args.catalog,
            config=config,
            learn=not args.no_learn,
            check_undeclared_tools=not args.no_undeclared_tools,
        )
    except (OSError, ValueError) as e:
        print(f"schemawarp: {e}", file=sys.stderr)
        return 2

    floor = args.min_severity
    findings = [f for f in findings if SEVERITY.get(f.severity, 0) >= SEVERITY.get(floor, 0)]

    if args.format == "json":
        payload = {
            "source": args.trace,
            "calls": call_count,
            "findings": [f.to_dict() for f in findings],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_text(findings, args.trace, call_count))

    if args.fail_on == "none":
        return 0
    thresh = SEVERITY[args.fail_on]
    if any(SEVERITY.get(f.severity, 0) >= thresh for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
