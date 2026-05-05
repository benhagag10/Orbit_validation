# SWE-Bench Single-Issue IPI Experiments

Configs for `ipi_swe_bench_si_baseline` and `ipi_swe_bench_si_star` from PLAN.md.

## Dataset

Fixed 50-instance subset of SWE-Bench Verified (seed=42).
Instance list: `orbit/scenarios/swe_bench/data/ipi_50_instances.json`

## Running

### Baseline (single agent) — benign

```bash
inspect eval orbit/scenarios/swe_bench/task.py@swe_bench_multi_issue \
  -T num_issues=1 \
  -T instance_set=ipi_50_instances.json \
  -T condition=single_agent \
  -T max_turns=30 \
  -T max_time=900 \
  --model openai/gpt-5.4
```

### Baseline (single agent) — ASR (AIShellJack IPI)

```bash
inspect eval orbit/scenarios/swe_bench/task.py@swe_bench_multi_issue \
  -T num_issues=1 \
  -T instance_set=ipi_50_instances.json \
  -T condition=single_agent \
  -T attack_preset=aishellcheck \
  -T max_turns=30 \
  -T max_time=900 \
  --model openai/gpt-5.4
```

### Star (4 specialists) — benign

```bash
inspect eval orbit/scenarios/swe_bench/task.py@swe_bench_multi_issue \
  -T num_issues=1 \
  -T instance_set=ipi_50_instances.json \
  -T condition=star_specialist \
  -T max_turns=30 \
  -T max_time=900 \
  --model openai/gpt-5.4
```

### Star (4 specialists) — ASR (AIShellJack IPI)

```bash
inspect eval orbit/scenarios/swe_bench/task.py@swe_bench_multi_issue \
  -T num_issues=1 \
  -T instance_set=ipi_50_instances.json \
  -T condition=star_specialist \
  -T attack_preset=aishellcheck \
  -T max_turns=30 \
  -T max_time=900 \
  --model openai/gpt-5.4
```

## Per-category sweep

Use `attack_preset=aishellcheck_category_sweep` to run one payload per MITRE
ATT&CK category (11 attacks injected simultaneously).

## Defense variants

Add `-T defense_preset=<name>` to any command above:
- `full` — file monitor + diff monitor + prompt vaccination
- `file_monitor` — file change tracking only
- `diff_monitor` — suspicious pattern detection only

## Smoke test (Haiku, 1 sample)

```bash
inspect eval orbit/scenarios/swe_bench/task.py@swe_bench_multi_issue \
  -T num_issues=1 \
  -T instance_ids=django__django-11292 \
  -T condition=single_agent \
  -T attack_preset=aishellcheck \
  -T max_turns=5 \
  -T max_time=300 \
  --model anthropic/claude-haiku-4-5-20251001 \
  --limit 1
```
