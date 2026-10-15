import json
import os
import tempfile
import unittest
from pathlib import Path

from schemawarp.core import (
    analyze,
    expand_rows,
    normalize_catalog,
    parse_args,
    run,
    validate_against_schema,
)
from schemawarp.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class TestParseAndCatalog(unittest.TestCase):
    def test_parse_args_json_string(self):
        self.assertEqual(parse_args('{"path": "a.py"}'), {"path": "a.py"})
        self.assertEqual(parse_args({"x": 1}), {"x": 1})
        self.assertEqual(parse_args(None), {})

    def test_normalize_openai_style(self):
        raw = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "required": ["path"],
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ]
        }
        cat = normalize_catalog(raw)
        self.assertIn("read_file", cat)
        self.assertEqual(cat["read_file"]["required"], ["path"])

    def test_normalize_flat_map(self):
        cat = normalize_catalog({"shell": {"type": "object", "properties": {"command": {"type": "string"}}}})
        self.assertIn("shell", cat)


class TestValidate(unittest.TestCase):
    def test_missing_required_and_additional(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["to", "subject", "body"],
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        }
        hits = validate_against_schema({"to": "a@b.co", "subject": "hi", "extra": 1}, schema)
        rules = {h[0] for h in hits}
        self.assertIn("MISSING_REQUIRED", rules)
        self.assertIn("ADDITIONAL_PROPERTY", rules)

    def test_type_and_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "deep"]},
                "n": {"type": "integer"},
            },
        }
        hits = validate_against_schema({"mode": "slow", "n": "x"}, schema)
        rules = {h[0] for h in hits}
        self.assertIn("ENUM_VIOLATION", rules)
        self.assertIn("TYPE_MISMATCH", rules)


class TestAnalyze(unittest.TestCase):
    def test_catalog_findings_on_example(self):
        findings, n = run(
            str(EXAMPLES / "warped.jsonl"),
            catalog_path=str(EXAMPLES / "catalog.json"),
        )
        self.assertGreater(n, 5)
        rules = {f.rule for f in findings}
        self.assertIn("ADDITIONAL_PROPERTY", rules)
        self.assertIn("MISSING_REQUIRED", rules)
        self.assertIn("UNDECLARED_TOOL", rules)
        self.assertIn("TYPE_MISMATCH", rules)

    def test_learn_mode_shape_drift(self):
        findings, n = run(str(EXAMPLES / "drift_only.jsonl"), catalog_path=None)
        self.assertEqual(n, 4)
        rules = {f.rule for f in findings}
        self.assertIn("SHAPE_DRIFT", rules)

    def test_openai_tool_calls_expansion(self):
        rows = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.py"}',
                        }
                    }
                ],
            }
        ]
        calls = expand_rows(rows, {"tool_keys": ["tool"], "args_keys": ["args"], "agent_keys": []})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "read_file")

    def test_no_learn_suppresses_drift(self):
        findings = analyze(
            [
                {"tool": "search", "args": {"query": "a"}},
                {"tool": "search", "args": {"query": "b"}},
                {"tool": "search", "args": {"query": "c", "mode": "deep"}},
            ],
            catalog=None,
            learn=False,
        )
        self.assertEqual(findings, [])


class TestCLI(unittest.TestCase):
    def test_cli_text_and_fail_on(self):
        code = main(
            [
                str(EXAMPLES / "warped.jsonl"),
                "--catalog",
                str(EXAMPLES / "catalog.json"),
                "--fail-on",
                "high",
            ]
        )
        self.assertEqual(code, 1)

    def test_cli_json_clean_fail_on_none(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "ok.jsonl"
            cat = Path(td) / "cat.json"
            trace.write_text(
                json.dumps({"tool": "read_file", "args": {"path": "a.py"}}) + "\n",
                encoding="utf-8",
            )
            cat.write_text(
                json.dumps(
                    {
                        "tools": {
                            "read_file": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["path"],
                                "properties": {"path": {"type": "string"}},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            code = main(
                [
                    str(trace),
                    "--catalog",
                    str(cat),
                    "--format",
                    "json",
                    "--fail-on",
                    "none",
                    "--no-learn",
                ]
            )
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
