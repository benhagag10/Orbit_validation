# Presets Reference

**A preset is just a pre-baked config.** Each named preset is a saved
[`SetupConfig`](configuration.md#yaml-schema-reference) — the agents, network topology, memory, and
prompts — plus, where the preset needs one, an
[`ExecutionConfig`](configuration.md#execution) (BrowserART's mesh presets, for example, pin
round-robin scheduling and observation summaries in the YAML itself; other scenarios' builders
may derive execution settings at run time instead). Presets are *worked examples* of the general
config syntax,
not a separate system: you can run one by name, or copy its YAML and edit it into your own experiment.

Presets are per-scenario. Seven of the eleven shipped scenarios ship theirs as runnable
configs under `orbit/scenarios/<family>/<scenario>/presets/<name>.yaml` (BrowserART, SWE-Bench,
OSWorld, RedCode-Gen, BigCodeBench, CodeIPI, ConVerse). The other four — τ²-Bench and the three
cooperative-allocation scenarios (Hospital, JiraTicket, MeetSched) — resolve preset names in code
and ship no preset YAMLs. A new scenario does not inherit any presets — its author ships their own
(see [Adding a New Scenario](adding-a-scenario.md)) — and you can always write a full YAML config
from scratch or copy an existing preset as a starting point.

Pick a preset three equivalent ways:

```bash
# 1. Friendly flags (quick exploration)
uv run orbit browserart -m openai/gpt-4o --agents specialist --memory full

# 2. By name
uv run orbit browserart -m openai/gpt-4o --preset memory_full

# 3. As the same setup in a complete config file you can open, copy, and edit
uv run orbit run orbit/scenarios/browser/browserart/presets/memory_full.yaml -m openai/gpt-4o
```

The three are interchangeable: the flags resolve to a preset name, and `--preset NAME` builds the
same setup. (For BrowserART the build happens from an in-code registry; the shipped YAML is the
equivalent config, kept in sync by tests, so opening the file still shows you exactly what the
preset is — the whole experiment in plain config syntax. Most other scenarios load the preset
directly from its YAML.)

> **Note:** τ²-Bench builds its agent roster per task (tools depend on the sample), so it has no
> preset YAMLs and its own preset vocabulary. `orbit tau2` selects its multi-agent shape only via `--preset` — one of `solo`,
> `dual_control`, `supervisor_specialist`, `tiered_escalation`, `mesh_committee`,
> `dual_control_review`, `cross_domain_handoff` — and none of the `--agents` / `--topology` /
> `--memory` flags described on this page.

You can also use `inspect eval` with `-T` flags:

```bash
uv run inspect eval orbit/browserart_safety \
  -T agents=specialist -T topology=round_robin \
  --model openai/gpt-4o
```

## In a YAML config

Presets are also selectable from an `orbit run` YAML via a scenario-prefixed
`metadata` key — `<prefix>_preset`, e.g. `browserart_preset` or
`swe_bench_preset` (see the
[shorthand table](configuration.md#scenario-shorthand-metadata-keys) for which
scenarios support it):

```yaml
name: "my_star_specialist_run"
scenario:
  name: "browserart"
metadata:
  browserart_preset: "star_specialist"
```

Declare **either** a `<prefix>_preset` **or** an inline `setup:` block, not
both — if both are present the preset wins; see the
[conflict rules](configuration.md#scenario-shorthand-metadata-keys) for the
details.

## Overview

To select a multi-agent preset, the friendly flags decompose into three independent dimensions:

| Flag | Values | What it controls |
|------|--------|-----------------|
| `--agents` | `single`, `batch`, `specialist`, `step` (OSWorld adds `app_specialist`) | Agent type and role division |
| `--topology` | `star`, `round_robin`, `delegation` | Network structure (how agents connect) |
| `--memory` | `none`, `own_actions`, `own_reasoning`, `shared_actions`, `full` | What agents see about each other |

With no preset flags at all, a scenario runs its single-agent default.
`--topology` and `--memory` refine `--agents` (they select a preset
together): when `--agents` is given they default to `star` and `none`;
passing either without `--agents` is a usage error.

There is also `--instructions` (`detailed` or `relaxed`) which controls prompt specificity, but this is rarely needed. It also requires `--agents`.

> **The friendly flags are not universal.** Only `orbit browserart`, `orbit swe-bench`, and
> `orbit osworld` expose `--agents` / `--topology` / `--memory` (and the matching `-T` task
> parameters). The other scenarios select presets **by name only**: `orbit redcode-gen --preset`,
> `orbit tau2 --preset`, and `-T preset=...` for BigCodeBench, CodeIPI, and ConVerse (which have
> no dedicated subcommand). The reason is that a preset's *roster* is scenario-specific — the
> flags only make sense where a scenario defines a family of rosters along all three dimensions.
> BrowserART's specialists are click/fill/scroll/navigate; SWE-Bench's are
> implementation/debugging/testing/review; scenarios with one natural roster (or, like τ²-Bench,
> a roster built per task) have nothing for the flags to vary, so they take a preset name
> directly.
>
> The tables and diagrams below use BrowserART as the worked example — its roster *and* its
> execution mechanics (such as mesh turn scheduling) are per-scenario choices. For another
> scenario's actual roster, open its `presets/` YAMLs (e.g.
> `orbit/scenarios/coding/swe_bench/presets/`).

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
| `round_robin` | Mesh | All agents are peers taking round-robin turns. Between turns each peer receives an observation summary of the others' recent activity, and every peer can call any other peer as a tool. |
| `delegation` | Mesh | Same mesh structure and turn rotation; the peers' prompts emphasize delegating any step outside their specialty to a better-suited peer. |

Non-star topologies (`round_robin`, `delegation`) are only supported with `--agents specialist`.

There is no `--topology chain`: chains and other custom shapes are defined by
writing the agent `edges` in a YAML config instead of using a flag (see the
explicit `edges` block in
[`examples/demo_multi_agent_attack_defense.yaml`](../examples/demo_multi_agent_attack_defense.yaml)).

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

All agents can communicate: every peer has a tool edge to every other peer,
and observation summaries relay recent activity between turns. Both mesh
variants are **scheduled round-robin** — each peer takes its turn in fixed
order. The difference is behavioral, not structural: on its turn a peer can
either act itself or synchronously delegate a step by calling another peer's
tool, and the `delegation` variant's prompts push peers to delegate anything
outside their specialty, while `round_robin`'s prompts are neutral. The
scheduler never lets an agent pick who takes the *next* turn.

## Parameter Mapping

The mapping from `--agents` / `--topology` / `--memory` to preset names, for
**BrowserART and SWE-Bench** (the two scenarios whose preset names follow
this scheme):

| `--agents` | `--topology` | `--memory` | Preset |
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

**OSWorld diverges** because it has *two* specialist families. Its
`--agents specialist` rows resolve to tool-action specialist presets instead:
`star_tool_specialist`, `mesh_tool_round_robin`, `mesh_tool_delegation`, and
`tool_memory_own_actions` … `tool_memory_full` for the memory levels. The
second family, `--agents app_specialist` (application specialists), is the
one that resolves to the `star_specialist` / `mesh_*` / `memory_*` names in
the table above. See `orbit/scenarios/desktop/osworld/preset_registry.py`
for the full OSWorld mapping.

**Validation rules:**
- `--topology` / `--memory` / `--instructions` require `--agents` (without it the CLI exits with a usage error; use `--preset` to name a preset directly)
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
