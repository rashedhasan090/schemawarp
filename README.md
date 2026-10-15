# schemawarp

**Catch agent tool calls whose arguments warp away from the declared schema — or from the shape the agent itself established earlier in the run.**

Offline, stdlib-only Python CLI. Point it at a JSONL tool-call transcript and (optionally) a JSON Schema catalog of tools. It validates each call and also watches for mid-run **shape drift**: new keys, type flips, and sudden key-set jumps that usually mean the model is improvising parameters under pressure.

## What's novel

Tool-surface checkers already answer "was this tool declared?" ([toolorphan](https://github.com/rashedhasan090/toolorphan)) and "is the MCP launch recipe safe?" ([mcpwarden](https://github.com/rashedhasan090/mcpwarden)). Prompt linters score policy text ([promptfence](https://github.com/rashedhasan090/promptfence)). **schemawarp sits on the argument payloads themselves.**

| Mode | What it does |
|------|----------------|
| Catalog | Validate each call against JSON Schema (`required`, `type`, `enum`, `const`, `additionalProperties`) |
| Learn | With or without a catalog, treat the first *N* calls per tool as a baseline shape and flag later invention |
| Jump | Flag consecutive calls to the same tool whose argument key-sets diverge below a Jaccard threshold |

Finding rules include `MISSING_REQUIRED`, `TYPE_MISMATCH`, `ENUM_VIOLATION`, `CONST_VIOLATION`, `ADDITIONAL_PROPERTY`, `UNDECLARED_PROPERTY`, `UNDECLARED_TOOL`, `SHAPE_DRIFT`, `TYPE_DRIFT`, and `SHAPE_JUMP`.

Distinct from [toolorphan](https://github.com/rashedhasan090/toolorphan) (name-set orphans/rogues only), [mcpwarden](https://github.com/rashedhasan090/mcpwarden) (launch configs), [promptfence](https://github.com/rashedhasan090/promptfence) (prompt fencing), [resultbleed](https://github.com/rashedhasan090/resultbleed) (result→arg copy), [loopsnare](https://github.com/rashedhasan090/loopsnare) / [callstorm](https://github.com/rashedhasan090/callstorm) (retry loops), [confirmskip](https://github.com/rashedhasan090/confirmskip) (confirmation gates), and [staleread](https://github.com/rashedhasan090/staleread) (filesystem view watermarks).

## Install

```bash
pip install .
# or run without installing:
python -m schemawarp path/to/trace.jsonl -c catalog.json
```

Requires Python 3.9+.

## Usage

```bash
schemawarp examples/warped.jsonl -c examples/catalog.json
schemawarp examples/drift_only.jsonl --format json --fail-on medium
schemawarp run.jsonl -c tools.json --no-undeclared-tools --learn-first 3
```

Exit code `1` when any finding is at or above `--fail-on` (default `high`). Use `--fail-on none` for a pure report. Wire it into CI after dumping agent traces.

### Catalog shapes

Any of these work:

```json
{
  "tools": {
    "read_file": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path"],
      "properties": { "path": { "type": "string" } }
    }
  }
}
```

OpenAI `tools[].function.parameters` arrays and MCP-style `inputSchema` objects are accepted too.

### Trace shapes

One JSON object per line. Common forms:

```json
{"agent": "coder", "tool": "read_file", "args": {"path": "src/app.py"}}
{"function": {"name": "shell", "arguments": "{\"command\": \"pytest\"}"}}
{"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"}}]}
```

## Demo

```bash
$ python -m schemawarp examples/warped.jsonl -c examples/catalog.json
== examples/warped.jsonl: 9 tool call(s), 13 finding(s)
  [HIGH  ] ADDITIONAL_PROPERTY  step 5    read_file
           additional property not allowed: encoding
  [MEDIUM] SHAPE_DRIFT          step 5    read_file
           call #3 invents keys ['encoding'] absent from early calls and schema properties
  [LOW   ] SHAPE_JUMP           step 5    read_file
           argument key-set Jaccard 0.33 vs previous call (threshold 0.5)
  [HIGH  ] MISSING_REQUIRED     step 6    shell
           missing required property command
  [HIGH  ] ADDITIONAL_PROPERTY  step 6    shell
           additional property not allowed: cmd
  [HIGH  ] ADDITIONAL_PROPERTY  step 6    shell
           additional property not allowed: sudo
  [MEDIUM] SHAPE_DRIFT          step 6    shell
           call #3 invents keys ['cmd', 'sudo'] absent from early calls and schema properties
  [LOW   ] SHAPE_JUMP           step 6    shell
           argument key-set Jaccard 0.00 vs previous call (threshold 0.5)
  [HIGH  ] MISSING_REQUIRED     step 7    send_email
           missing required property body
  [HIGH  ] UNDECLARED_TOOL      step 8    deploy_prod
           tool 'deploy_prod' is not in the schema catalog
  [HIGH  ] TYPE_MISMATCH        step 9    shell
           command: expected string, got integer
  [MEDIUM] TYPE_DRIFT           step 9    shell
           argument types changed vs early calls: command: string->integer
  [LOW   ] SHAPE_JUMP           step 9    shell
           argument key-set Jaccard 0.00 vs previous call (threshold 0.5)
```

Catalog-free drift on the second example:

```bash
$ python -m schemawarp examples/drift_only.jsonl
== examples/drift_only.jsonl: 4 tool call(s), 4 finding(s)
  [HIGH  ] SHAPE_DRIFT          step 3    search
           call #3 invents keys ['budget', 'mode'] not seen in the first 2 call(s)
  [LOW   ] SHAPE_JUMP           step 3    search
           argument key-set Jaccard 0.33 vs previous call (threshold 0.5)
  [HIGH  ] SHAPE_DRIFT          step 4    search
           call #4 invents keys ['limit', 'q'] not seen in the first 2 call(s)
  [LOW   ] SHAPE_JUMP           step 4    search
           argument key-set Jaccard 0.00 vs previous call (threshold 0.5)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
