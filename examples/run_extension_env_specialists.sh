#!/usr/bin/env bash
# Run BrowserART extension task 236 with environment-based specialist agents.
# Each platform (email, twitter, linkedin, whatsapp) is handled by a
# dedicated specialist, coordinated by an orchestrator.

set -euo pipefail

orbit run examples/demo2_specialist_dispatch.yaml \
  --model openai/gpt-4o \
  --max-samples 1
