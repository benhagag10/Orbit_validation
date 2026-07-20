# Orbit examples

A small, curated set of runnable configs — one flagship per scenario family, plus a
minimal intro. Each file has a header comment explaining what it shows, how to run
it, and how to vary it. Start with `simple_no_sandbox.yaml`.

| Example | Family | Shows | Setup needed |
|---------|--------|-------|--------------|
| [`simple_no_sandbox.yaml`](simple_no_sandbox.yaml) | — (intro) | The whole `ExperimentConfig` schema: one agent + one attack + one defense | none |
| [`redcode_gen_basic.yaml`](redcode_gen_basic.yaml) | coding | Single-agent malware-generation refusal, sandboxed, LLM judge | Docker + `scripts/fetch_redcode_data.py` |
| [`tau2_airline_basic.yaml`](tau2_airline_basic.yaml) | customer service | τ²-Bench dual-control user-sim ↔ assistant loop | none (data vendored) |
| [`collusion_jira_ticket.yaml`](collusion_jira_ticket.yaml) | cooperative allocation | Multi-agent **collusion**: agent groups + secret channel | `uv sync --extra dcop` |
| [`demo_multi_agent_attack_defense.yaml`](demo_multi_agent_attack_defense.yaml) | browser | Star topology + indirect injection + prompt-vaccination defense | BrowserART service (Docker) |
| [`osworld_osharm_basic.yaml`](osworld_osharm_basic.yaml) | desktop | Computer-use agent safety (OS-Harm) | Docker + `uv sync --extra osworld` |

## Three ways to run

1. **Per-scenario CLI flags** — quickest for exploration:
   ```bash
   uv run orbit browserart -m openai/gpt-4o --agents specialist --topology star --memory full
   uv run orbit browserart -m openai/gpt-4o --agents specialist --topology round_robin   # mesh; --memory needs star
   uv run orbit redcode-gen --model openai/gpt-4o --preset star_specialist
   ```
2. **Inspect `eval` with `-T` params** — direct Inspect access:
   ```bash
   uv run inspect eval orbit/tau2 -T domain=airline -T preset=dual_control --model openai/gpt-4o
   ```
3. **`orbit run <config>.yaml`** — full control from a YAML file (these examples):
   ```bash
   uv run orbit run examples/simple_no_sandbox.yaml --model openai/gpt-4o
   uv run orbit validate examples/simple_no_sandbox.yaml     # schema check, no run
   ```

Inspect results with `uv run inspect view`.

## Varying topology, memory, attacks, and defenses

You do **not** need a separate example per permutation. Every scenario ships its full
topology/memory preset library as YAML under
`orbit/scenarios/<family>/<scenario>/presets/` (e.g.
`orbit/scenarios/coding/redcode_gen/presets/star_specialist.yaml`), selectable with
`--preset` / `--agents` / `--memory` / `--defense-preset`. Copy one of those presets,
or one of the examples here, and edit the `setup`, `attacks`, and `defenses` blocks.
See `docs/presets.md`, `docs/configuration.md`, and `docs/scenarios.md`.
