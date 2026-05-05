# Conditions Reference

Conditions are preset multi-agent setups. Each condition configures the agents, network topology, memory, and prompts automatically. Use `--agents`, `--topology`, and `--memory` to select one:

```bash
orbit browserart -m openai/gpt-4o --agents specialist
orbit browserart -m openai/gpt-4o --agents specialist --topology round_robin
orbit browserart -m openai/gpt-4o --agents specialist --memory full
```

> **Note:** Conditions are specific to BrowserART. The SWE-Bench scenario uses its own topology and preset system (see `--topology`, `--attack-preset`, `--defense-preset`). See [Scenarios](scenarios.md) for SWE-Bench configuration.

You can also use `inspect eval` with `-T` flags:

```bash
inspect eval orbit/browserart_safety \
  -T agents=specialist -T topology=round_robin \
  --model openai/gpt-4o
```

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
| `batch` | `star` | `none` | `star_batch_detailed` |
| `specialist` | `star` | `none` | `star_4_specialists` |
| `step` | `star` | `none` | `star_step_executor` |
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
orbit browserart -m openai/gpt-4o --agents single
orbit browserart -m openai/gpt-4o --agents specialist

# Compare star vs mesh topologies
orbit browserart -m openai/gpt-4o --agents specialist --topology star
orbit browserart -m openai/gpt-4o --agents specialist --topology round_robin
orbit browserart -m openai/gpt-4o --agents specialist --topology delegation

# Test memory levels
for mem in none own_actions own_reasoning shared_actions full; do
  orbit browserart -m openai/gpt-4o --agents specialist --memory $mem
done

# View all results side by side
inspect view
```
