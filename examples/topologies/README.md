# Topology snippets (SetupConfig YAML)

The files in this directory are **not runnable experiments**. Each one is a
bare `SetupConfig` — top-level `agents:` / `edges:` / `properties:` keys with
no `name:` or `scenario:` — describing just a multi-agent topology.

They exist for scenarios that build one `ExperimentConfig` per sample
internally (ConVerse, BrowserART): you supply the topology and set
scenario-level knobs via task parameters or CLI flags, e.g.

```bash
uv run inspect eval orbit/converse_safety \
    -T topology_file=examples/topologies/converse_basic.yaml \
    --model openai/gpt-4o
```

or with the wrapper CLI:

```bash
uv run orbit browserart -m openai/gpt-4o \
    --topology-file examples/topologies/browserart_env_specialists_topology.yaml
```

`orbit validate <file>` recognizes the SetupConfig shape and validates these
as topology files. Runnable end-to-end experiment configs live one directory
up, in `examples/*.yaml` — those are full `ExperimentConfig` files for
`orbit run`.
