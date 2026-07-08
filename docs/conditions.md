# Conditions Reference

**A condition is just a pre-baked config.** Each named condition is a saved
[`SetupConfig`](configuration.md#yaml-schema-reference) — the agents, network topology, memory, and
prompts — written in the general orbit syntax. Conditions are *worked examples* of that syntax, not
a separate system: you can run one by name, or copy its YAML and edit it into your own experiment.

Every scenario shares this system (not just BrowserART). Each scenario ships its conditions as
runnable configs under `orbit/scenarios/conditions/<scenario>/<name>.yaml`; the shared loader
`orbit/scenarios/conditions.py` resolves them. Pick a condition three equivalent ways:

```bash
# 1. Friendly flags (quick exploration)
uv run orbit browserart -m openai/gpt-4o --agents specialist --memory full

# 2. By name
uv run orbit browserart -m openai/gpt-4o --condition memory_full

# 3. As a config you can read and edit — the transparent form of the two above
uv run orbit run orbit/scenarios/conditions/browserart/memory_full.yaml -m openai/gpt-4o
```

The three are interchangeable: the flags resolve to a condition name, and the name loads the shipped
YAML. To see exactly what a condition is, open its file — it is the whole experiment in plain syntax.

> **Note:** τ²-Bench builds its agent roster per task (tools depend on the sample), so it resolves
> conditions natively rather than from a static file; the flags and names work the same, but there
> is no shipped YAML to open.

You can also use `inspect eval` with `-T` flags:

```bash
uv run inspect eval orbit/browserart_safety \
  -T agents=specialist -T topology=round_robin \
  --model openai/gpt-4o
```

## In a YAML config

Conditions are also selectable from an `orbit run` YAML via a scenario-prefixed
`metadata` key — `<prefix>_condition`, e.g. `browserart_condition` or
`swe_bench_condition` (see the
[shorthand table](configuration.md#scenario-shorthand-metadata-keys) for which
scenarios support it):

```yaml
scenario:
  name: "browserart"
metadata:
  browserart_condition: "star_specialist"
```

Declare **either** a `<prefix>_condition` **or** an inline `setup:` block, not
both. If both are present the **condition wins**: a `setup.agents` roster that
differs from the condition's raises a `ShorthandConflictError`
(`ORBIT_LENIENT=1` downgrades it to a warning), and a matching roster is dropped
as redundant. (This precedence reversed with the scenario authoring kit — an
inline setup no longer overrides a condition.)

## Overview

The conditions decompose into three independent dimensions:

| Flag | Values | What it controls |
|------|--------|-----------------|
| `--agents` | `single`, `batch`, `specialist`, `step` | Agent type and role division |
| `--topology` | `star`, `round_robin`, `delegation` | Network structure (how agents connect) |
| `--memory` | `none`, `own_actions`, `own_reasoning`, `shared_actions`, `full` | What agents see about each other |

Defaults: `--agents single`, `--topology star`, `--memory none`.

There is also `--instructions` (`detailed` or `relaxed`) which controls prompt specificity, but this is rarely needed.

## Agents

These control what kind of agents participate and how work is divided.

| `--agents` | Agents | How it works |
|------------|--------|-------------|
| `single` | 1 agent | Single agent does everything. The baseline. |
| `batch` | 1 orchestrator + 1 executor | Orchestrator plans a batch of actions. Executor runs them all. |
| `specialist` | 1 orchestrator + 4 specialists | Orchestrator picks a specialist per step (click, fill, scroll, navigate). |
| `step` | 1 orchestrator + 1 executor | Orchestrator gives exactly one instruction, executor acts, repeat. |

## Topology

These control how agents are connected in the network.

| `--topology` | Structure | How it works |
|--------------|-----------|-------------|
| `star` | Hub-and-spoke | Orchestrator is the hub. It sees the page, decides what to do, and delegates to workers. Workers only see their instruction and the current page. (default) |
| `round_robin` | Mesh | All agents are peers. Round-robin turns — each agent acts in order. Shared message board for coordination. |
| `delegation` | Mesh | All agents are peers. Current agent can act or delegate to any other agent freely. |

Non-star topologies (`round_robin`, `delegation`) are only supported with `--agents specialist`.

## Memory

These control how much context each agent can see from previous turns. Only supported with `--agents specialist` and `--topology star`.

| `--memory` | What agents can see |
|------------|-------------------|
| `none` | Goal only. No history. (default) |
| `own_actions` | Goal + their own past actions. |
| `own_reasoning` | Goal + their own past actions + their own reasoning (chain-of-thought). |
| `shared_actions` | All of the above + other agents' past actions. |
| `full` | All of the above + other agents' reasoning + accumulated instructions. Full visibility. |

## Topology Diagrams

### Single (`--agents single`)

```
  [Agent] --> Browser
```

### Star (`--topology star`)

```
          [Orchestrator]
          /    |    \    \
  [Click] [Fill] [Scroll] [Navigate]
                  |
               Browser
```

The orchestrator observes the page and dispatches work. Only one agent interacts with the browser at a time.

### Mesh (`--topology round_robin` or `--topology delegation`)

```
  [Click] <---> [Fill]
    ^   \      /   ^
    |    \    /    |
    v     v  v     v
  [Nav]  <---> [Scroll]
            |
         Browser
```

All agents can communicate. In round-robin, they take fixed turns. In delegation, the active agent chooses who goes next.

## Parameter Mapping

The full mapping from `--agents` / `--topology` / `--memory` to conditions:

| `--agents` | `--topology` | `--memory` | Condition |
|------------|-------------|-----------|-----------|
| `single` | `star` | `none` | `single_agent` |
| `batch` | `star` | `none` | `star_batch` |
| `specialist` | `star` | `none` | `star_specialist` |
| `step` | `star` | `none` | `star_step` |
| `specialist` | `round_robin` | `none` | `mesh_round_robin` |
| `specialist` | `delegation` | `none` | `mesh_delegation` |
| `specialist` | `star` | `own_actions` | `memory_own_actions` |
| `specialist` | `star` | `own_reasoning` | `memory_own_reasoning` |
| `specialist` | `star` | `shared_actions` | `memory_shared_actions` |
| `specialist` | `star` | `full` | `memory_full` |

Add `--instructions relaxed` for relaxed prompt variants of `batch` or `specialist`.

**Validation rules:**
- `--topology` other than `star` requires `--agents specialist`
- `--memory` other than `none` requires `--agents specialist` and `--topology star`
- Not all combinations are supported — only experimentally validated setups are allowed. Unsupported combinations produce a clear error message.

## Examples

```bash
# Compare single agent vs specialist team
uv run orbit browserart -m openai/gpt-4o --agents single
uv run orbit browserart -m openai/gpt-4o --agents specialist

# Compare star vs mesh topologies
uv run orbit browserart -m openai/gpt-4o --agents specialist --topology star
uv run orbit browserart -m openai/gpt-4o --agents specialist --topology round_robin
uv run orbit browserart -m openai/gpt-4o --agents specialist --topology delegation

# Test memory levels
for mem in none own_actions own_reasoning shared_actions full; do
  uv run orbit browserart -m openai/gpt-4o --agents specialist --memory $mem
done

# View all results side by side
uv run inspect view
```
