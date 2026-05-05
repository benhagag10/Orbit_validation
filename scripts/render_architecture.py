"""Render InspectMAS architecture as ASCII box-art diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

BG = "#1a1b2e"
BORDER = "#2a2d4a"
PINK = "#f07090"
DIM = "#f07090"  # same color, different alpha for annotations
FONT = "Courier New"

ASCII_ART = r"""
  $ uv run inspectmas browserart --model openai/gpt-4o \
        --agents specialist --topology star --memory full


                     +---------------------------+
                     |    ExperimentConfig        |   frozen Pydantic from YAML
                     +---------------------------+
                                  |
          +-----------+-----------+-----------+-----------+
          |           |           |           |           |
          v           v           v           v           v
    +-----------++-----------++-----------++-----------++-----------+
    | Scenario  ||   Setup   ||  Attack   || Defense   || Execution |
    |           ||           ||           ||           ||           |
    | browserart|| agents[]  || timing    || scope     || groups[]  |
    | osworld   || topology[]|| threat    || targets   || schedule  |
    | swe-bench || memory    || target    || guardian  || observe   |
    +-----------++-----------++-----------++-----------++-----------+
          |           |           |           |           |
          +-----------+-----------+-----------+-----------+
                                  |
                          build_sample()
                                  |
                                  v
                     +---------------------------+
                     |     Inspect Sample         |   MASMetadata in metadata
                     +---------------------------+
                                  |
                                  v
  +===================================================================+
  |                      mas_orchestrator @solver                      |
  |                                                                    |
  |   _get_execution_path()                                            |
  |        |                                                           |
  |        +---> Path A: react() agents, single topology               |
  |        |         handoff / tool / direct_run                       |
  |        |                                                           |
  |        +---> Path C: turn_react() agents, AgentScheduler           |
  |                  SubmitRegistry, round_robin / superstep           |
  |                                                                    |
  |   +-----+    +-----+    +-----+    +-----+                        |
  |   | A o-+--->| B o-+--->| C   |    | D   |    agent topology       |
  |   +-----+    +--+--+    +-----+    +-----+                        |
  |                  |          ^                                       |
  |                  +----------+                                      |
  +===================================================================+
          |                                       |
          v                                       v
  +------------------+                   +------------------+
  | Attack Registry  |                   | Defense Registry |
  |                  |                   |                  |
  | direct_injection |                   | prompt_vaccine   |
  | indirect_inject  |                   | guardian_agent   |
  | compromised_agent|                   | monitor          |
  | collusion        |                   | secure_model     |
  | self_replicating |                   | tool_wrapper     |
  | code_injection   |                   | code_review      |
  +------------------+                   +------------------+
          |                                       |
          +-------------------+-------------------+
                              |
                              v
  +===================================================================+
  |              ExperimentScheduler.run_loop()                        |
  |                                                                    |
  |   +----------+    +----------+    +----------+    +----------+     |
  |   | 1.inject +--->| 2.execute+--->| 3.monitor+--->| 4.evaluat|     |
  |   +----------+    +----------+    +----------+    +----+-----+     |
  |        ^                                               |           |
  |        +-----------------------------------------------+           |
  |                          next turn                                 |
  |   halt: max_turns | timeout | convergence | all_submitted          |
  +===================================================================+
                              |
                          read/write
                              |
                              v
              +-------------------------------+
              |   Runtime State (StoreModel)  |
              |                               |
              |   AttackLog      DefenseLog   |
              |   RuntimeMetrics              |
              |   EnvironmentState            |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              |       Scoring Pipeline        |
              |                               |
              |   C_i --> PD --> ER --> AS     |     per-agent --> per-sample
              |                    \           |
              |                     +--> ASR   |     --> per-experiment
              +-------------------------------+
                              |
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Inspect AI    @task  @solver  @scorer  @agent  @tool  Store  Sandbox
"""

fig, ax = plt.subplots(figsize=(24, 28), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")

# outer border
border = FancyBboxPatch(
    (0.02, 0.01), 0.96, 0.98,
    boxstyle="round,pad=0.01",
    facecolor="none",
    edgecolor=BORDER,
    linewidth=2,
    transform=ax.transAxes,
)
ax.add_patch(border)

# render ASCII art
ax.text(
    0.05, 0.97, ASCII_ART,
    fontsize=20,
    fontfamily=FONT,
    color=PINK,
    verticalalignment="top",
    horizontalalignment="left",
    transform=ax.transAxes,
    linespacing=1.25,
)

plt.savefig("architecture_diagram_rendered.png", dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
print("Saved -> architecture_diagram_rendered.png")
