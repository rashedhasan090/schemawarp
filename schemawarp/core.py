"""Core analysis for schemawarp."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SEVERITY = {"low": 1, "medium": 2, "high": 3}

DEFAULT_CONFIG: Dict[str, Any] = {
    "agent_keys": ["agent", "actor", "name", "role"],
    "tool_keys": ["tool", "name", "function"],
    "args_keys": ["args", "arguments", "input", "parameters"],
    "learn_first": 2,
    "drift_jaccard": 0.5,
}


@dataclass
class Finding:
    rule: str
    severity: str
    step: int
    tool: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _severity_at_least(sev: str, floor: str) -> bool:
    return SEVERITY.get(sev, 0) >= SEVERITY.get(floor, 0)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def normalize_catalog(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Accept several catalog shapes; return tool_name -> JSON Schema (object schema)."""
    tools: Dict[str, Dict[str, Any]] = {}

    def add(name: str, schema: Any) -> None:
        name = str(name).strip()
        if not name:
            return
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        # OpenAI-style: parameters wrapper
        if "parameters" in schema and isinstance(schema["parameters"], dict):
            schema = schema["parameters"]
        if "inputSchema" in schema and isinstance(schema["inputSchema"], dict):
            schema = schema["inputSchema"]
        if schema.get("type") is None and "properties" in schema:
            schema = dict(schema)
            schema["type"] = "object"
        tools[name] = schema

    if isinstance(raw, dict):
        if "tools" in raw:
            block = raw["tools"]
            if isinstance(block, dict):
                for name, schema in block.items():
                    add(name, schema)
            elif isinstance(block, list):
                for item in block:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("tool")
                    if not name and isinstance(item.get("function"), dict):
                        name = item["function"].get("name")
                        add(name, item["function"])
                    else:
                        add(name, item)
        elif "functions" in raw and isinstance(raw["functions"], list):
            for item in raw["functions"]:
                if isinstance(item, dict):
                    add(item.get("name"), item)
        else:
            # flat map of name -> schema
            for name, schema in raw.items():
                if name in ("$schema", "title", "description"):
                    continue
                add(name, schema)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("tool")
                if not name and isinstance(item.get("function"), dict):
                    add(item["function"].get("name"), item["function"])
                else:
                    add(name, item)
    else:
        raise ValueError("catalog must be a JSON object or array")

    return tools


def _dig_tool_and_args(row: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    # OpenAI chat message with tool_calls
    if "tool_calls" in row and isinstance(row["tool_calls"], list) and row["tool_calls"]:
        # handled by expand_rows
        return None, None

    # nested function
    fn = row.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        args = fn.get("arguments", fn.get("args", fn.get("parameters")))
        return str(fn["name"]), args

    tool = None
    for k in config["tool_keys"]:
        if k in row and isinstance(row[k], str) and row[k]:
            tool = row[k]
            break
        if k in row and isinstance(row[k], dict) and row[k].get("name"):
            tool = str(row[k]["name"])
            args = row[k].get("arguments", row[k].get("args"))
            return tool, args

    args = None
    for k in config["args_keys"]:
        if k in row:
            args = row[k]
            break
    return tool, args


def expand_rows(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    """Yield (step, tool_name, args, raw_row) with 1-based step numbers."""
    out: List[Tuple[int, str, Any, Dict[str, Any]]] = []
    step = 0
    for row in rows:
        tcs = row.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                step += 1
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                name = fn.get("name") if isinstance(fn, dict) else None
                args = None
                if isinstance(fn, dict):
                    args = fn.get("arguments", fn.get("args"))
                if name:
                    out.append((step, str(name), args, row))
            continue

        tool, args = _dig_tool_and_args(row, config)
        if tool:
            step += 1
            out.append((step, tool, args, row))
    return out


def parse_args(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            val = json.loads(s)
        except json.JSONDecodeError:
            return {"_unparsed": raw}
        if isinstance(val, dict):
            return val
        return {"_value": val}
    return {"_value": raw}


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _json_type_name(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int) and not isinstance(val, bool):
        return "integer"
    if isinstance(val, float):
        return "number"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "array"
    if isinstance(val, dict):
        return "object"
    return type(val).__name__


def _type_ok(val: Any, declared: Any) -> bool:
    if declared is None:
        return True
    types = declared if isinstance(declared, list) else [declared]
    for t in types:
        if t == "number" and isinstance(val, (int, float)) and not isinstance(val, bool):
            return True
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return True
        py = _TYPE_MAP.get(t)
        if py is None:
            continue
        if t == "null" and val is None:
            return True
        if t != "null" and isinstance(val, py) and not (t == "boolean" and not isinstance(val, bool) and isinstance(val, int)):
            if t == "boolean":
                return isinstance(val, bool)
            if t in ("integer", "number") and isinstance(val, bool):
                continue
            return True
    return False


def validate_against_schema(
    args: Dict[str, Any],
    schema: Dict[str, Any],
    path: str = "",
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Return list of (rule, severity, message, detail)."""
    findings: List[Tuple[str, str, str, Dict[str, Any]]] = []
    if not isinstance(schema, dict):
        return findings

    schema_type = schema.get("type", "object")
    if schema_type == "object" or "properties" in schema or "required" in schema:
        if not isinstance(args, dict):
            findings.append(
                (
                    "TYPE_MISMATCH",
                    "high",
                    f"{path or '$'}: expected object, got {_json_type_name(args)}",
                    {"path": path or "$", "expected": "object", "got": _json_type_name(args)},
                )
            )
            return findings

        props = schema.get("properties") or {}
        required = schema.get("required") or []
        for req in required:
            if req not in args:
                findings.append(
                    (
                        "MISSING_REQUIRED",
                        "high",
                        f"missing required property {path + '.' if path else ''}{req}",
                        {"path": f"{path}.{req}" if path else req},
                    )
                )

        additional = schema.get("additionalProperties", True)
        for key, val in args.items():
            if key.startswith("_"):
                continue
            p = f"{path}.{key}" if path else key
            if key in props and isinstance(props[key], dict):
                sub = props[key]
                if "type" in sub and not _type_ok(val, sub["type"]):
                    findings.append(
                        (
                            "TYPE_MISMATCH",
                            "high",
                            f"{p}: expected {sub['type']}, got {_json_type_name(val)}",
                            {"path": p, "expected": sub["type"], "got": _json_type_name(val)},
                        )
                    )
                if "enum" in sub and val not in sub["enum"]:
                    findings.append(
                        (
                            "ENUM_VIOLATION",
                            "medium",
                            f"{p}: value {val!r} not in enum",
                            {"path": p, "value": val, "enum": sub["enum"]},
                        )
                    )
                if "const" in sub and val != sub["const"]:
                    findings.append(
                        (
                            "CONST_VIOLATION",
                            "medium",
                            f"{p}: expected const {sub['const']!r}, got {val!r}",
                            {"path": p, "expected": sub["const"], "got": val},
                        )
                    )
                if sub.get("type") == "object" or "properties" in sub:
                    if isinstance(val, dict):
                        findings.extend(validate_against_schema(val, sub, p))
                if sub.get("type") == "array" and isinstance(val, list) and "items" in sub:
                    item_schema = sub["items"]
                    if isinstance(item_schema, dict) and item_schema.get("type"):
                        for i, item in enumerate(val):
                            if not _type_ok(item, item_schema["type"]):
                                findings.append(
                                    (
                                        "TYPE_MISMATCH",
                                        "medium",
                                        f"{p}[{i}]: expected {item_schema['type']}, got {_json_type_name(item)}",
                                        {"path": f"{p}[{i}]", "expected": item_schema["type"], "got": _json_type_name(item)},
                                    )
                                )
            else:
                if additional is False:
                    findings.append(
                        (
                            "ADDITIONAL_PROPERTY",
                            "high",
                            f"additional property not allowed: {p}",
                            {"path": p},
                        )
                    )
                elif additional is True:
                    findings.append(
                        (
                            "UNDECLARED_PROPERTY",
                            "medium",
                            f"property not declared in schema: {p}",
                            {"path": p},
                        )
                    )
                elif isinstance(additional, dict):
                    # schema for additional props — light check
                    if "type" in additional and not _type_ok(val, additional["type"]):
                        findings.append(
                            (
                                "TYPE_MISMATCH",
                                "medium",
                                f"{p}: expected {additional['type']}, got {_json_type_name(val)}",
                                {"path": p, "expected": additional["type"], "got": _json_type_name(val)},
                            )
                        )
    elif "type" in schema and not _type_ok(args, schema["type"]):
        findings.append(
            (
                "TYPE_MISMATCH",
                "high",
                f"{path or '$'}: expected {schema['type']}, got {_json_type_name(args)}",
                {"path": path or "$", "expected": schema["type"], "got": _json_type_name(args)},
            )
        )

    return findings


def _keyset(args: Dict[str, Any]) -> Set[str]:
    return {k for k in args.keys() if not str(k).startswith("_")}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def _type_sig(args: Dict[str, Any]) -> Dict[str, str]:
    return {k: _json_type_name(v) for k, v in args.items() if not str(k).startswith("_")}


def analyze(
    rows: Sequence[Dict[str, Any]],
    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    learn: bool = True,
    check_undeclared_tools: bool = True,
) -> List[Finding]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    catalog = catalog or {}
    learn_first = int(cfg.get("learn_first", 2))
    drift_thresh = float(cfg.get("drift_jaccard", 0.5))

    calls = expand_rows(rows, cfg)
    findings: List[Finding] = []

    # Per-tool observed shapes for drift learning
    learned_keys: Dict[str, Set[str]] = {}
    learned_types: Dict[str, Dict[str, str]] = {}
    call_counts: Dict[str, int] = defaultdict(int)
    prev_keys: Dict[str, Set[str]] = {}

    for step, tool, raw_args, _row in calls:
        args = parse_args(raw_args)
        call_counts[tool] += 1
        n = call_counts[tool]
        keys = _keyset(args)
        types = _type_sig(args)

        if catalog:
            if tool not in catalog:
                if check_undeclared_tools:
                    findings.append(
                        Finding(
                            rule="UNDECLARED_TOOL",
                            severity="high",
                            step=step,
                            tool=tool,
                            message=f"tool {tool!r} is not in the schema catalog",
                            detail={"tool": tool},
                        )
                    )
            else:
                for rule, sev, msg, detail in validate_against_schema(args, catalog[tool]):
                    findings.append(
                        Finding(rule=rule, severity=sev, step=step, tool=tool, message=msg, detail=detail)
                    )
                # Also flag keys outside properties even when additionalProperties omitted
                # (validate_against_schema already emits UNDECLARED_PROPERTY when additional=True)

        if learn:
            if n <= learn_first:
                if tool not in learned_keys:
                    learned_keys[tool] = set(keys)
                    learned_types[tool] = dict(types)
                else:
                    learned_keys[tool] |= keys
                    for k, t in types.items():
                        learned_types[tool].setdefault(k, t)
            else:
                base = learned_keys.get(tool, set())
                base_types = learned_types.get(tool, {})
                new_keys = keys - base
                if new_keys and tool not in catalog:
                    findings.append(
                        Finding(
                            rule="SHAPE_DRIFT",
                            severity="high",
                            step=step,
                            tool=tool,
                            message=(
                                f"call #{n} invents keys {sorted(new_keys)} "
                                f"not seen in the first {learn_first} call(s)"
                            ),
                            detail={
                                "new_keys": sorted(new_keys),
                                "learned_keys": sorted(base),
                                "call_index": n,
                            },
                        )
                    )
                elif new_keys and tool in catalog:
                    # catalog already covers undeclared; still note drift as medium if schema was loose
                    props = (catalog[tool].get("properties") or {}).keys()
                    invented = new_keys - set(props)
                    if invented:
                        findings.append(
                            Finding(
                                rule="SHAPE_DRIFT",
                                severity="medium",
                                step=step,
                                tool=tool,
                                message=(
                                    f"call #{n} invents keys {sorted(invented)} "
                                    f"absent from early calls and schema properties"
                                ),
                                detail={
                                    "new_keys": sorted(invented),
                                    "learned_keys": sorted(base),
                                    "call_index": n,
                                },
                            )
                        )
                type_flips = []
                for k, t in types.items():
                    if k in base_types and base_types[k] != t and k not in new_keys:
                        # allow integer/number soft match
                        if {base_types[k], t} <= {"integer", "number"}:
                            continue
                        type_flips.append((k, base_types[k], t))
                if type_flips:
                    findings.append(
                        Finding(
                            rule="TYPE_DRIFT",
                            severity="medium",
                            step=step,
                            tool=tool,
                            message=(
                                "argument types changed vs early calls: "
                                + ", ".join(f"{k}: {a}->{b}" for k, a, b in type_flips)
                            ),
                            detail={
                                "flips": [{"key": k, "from": a, "to": b} for k, a, b in type_flips],
                                "call_index": n,
                            },
                        )
                    )

            if tool in prev_keys:
                j = _jaccard(prev_keys[tool], keys)
                if j < drift_thresh and (prev_keys[tool] or keys):
                    findings.append(
                        Finding(
                            rule="SHAPE_JUMP",
                            severity="low",
                            step=step,
                            tool=tool,
                            message=(
                                f"argument key-set Jaccard {j:.2f} vs previous call "
                                f"(threshold {drift_thresh})"
                            ),
                            detail={
                                "jaccard": round(j, 3),
                                "previous_keys": sorted(prev_keys[tool]),
                                "keys": sorted(keys),
                            },
                        )
                    )
            prev_keys[tool] = keys

    return findings


def format_text(findings: List[Finding], source: str, call_count: int) -> str:
    lines = [
        f"== {source}: {call_count} tool call(s), {len(findings)} finding(s)"
    ]
    if not findings:
        lines.append("  (clean)")
        return "\n".join(lines)
    for f in findings:
        lines.append(f"  [{f.severity.upper():6}] {f.rule:20} step {f.step:<4} {f.tool}")
        lines.append(f"           {f.message}")
    return "\n".join(lines)


def run(
    trace_path: str,
    catalog_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    learn: bool = True,
    check_undeclared_tools: bool = True,
) -> Tuple[List[Finding], int]:
    rows = load_jsonl(trace_path)
    catalog = None
    if catalog_path:
        catalog = normalize_catalog(load_json(catalog_path))
    findings = analyze(
        rows,
        catalog=catalog,
        config=config,
        learn=learn,
        check_undeclared_tools=check_undeclared_tools,
    )
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    call_count = len(expand_rows(rows, cfg))
    return findings, call_count
