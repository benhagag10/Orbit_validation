"""
Environment setup solver.

Runs BEFORE the main orchestrator solver. Used as Task(setup=...) so it
executes even when the main solver is substituted via --solver CLI flag.

Interaction with Inspect:
    - Decorated with @solver
    - Used as Task(setup=mas_environment_setup())
    - Calls sandbox().exec() to run setup commands in the Docker sandbox
    - Uses transcript().info() for logging

Responsibilities:
    - Validate the experiment configuration (via ConfigValidator)
    - Log the experiment config to the transcript for debugging
    - Run scenario-specific setup in the sandbox (install packages, etc.)
"""

from __future__ import annotations

import logging

from inspect_ai.log import transcript
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from orbit.dataset.metadata import MASMetadata
from orbit.validation.validators import ConfigValidator

logger = logging.getLogger(__name__)


@solver
def mas_environment_setup() -> Solver:
    """Setup solver for MAS environment initialization.

    Always runs before the main solver, even with --solver override.
    Validates config and runs scenario setup scripts in the sandbox.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # 1. Extract config from metadata
        try:
            mas_metadata = state.metadata_as(MASMetadata)
            config = mas_metadata.experiment
        except Exception as e:
            logger.error(f"Failed to read experiment config from metadata: {e}")
            raise ValueError(
                f"Cannot read MASMetadata from sample metadata: {e}"
            ) from e

        # 2. Validate config
        errors = ConfigValidator.validate(config)
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"Config validation failed: {error_msg}")
            raise ValueError(f"Config validation failed: {error_msg}")

        # 3. Log config summary to transcript
        transcript().info(
            f"MAS Experiment: {config.name} "
            f"(baseline={config.baseline_mode.value}, "
            f"agents={len(config.setup.agents)}, "
            f"attacks={len(config.attacks)}, "
            f"defenses={len(config.defenses)})"
        )

        # 4. Run setup script in sandbox (if configured)
        setup_script = config.scenario.setup_script
        if not setup_script and config.scenario.requirements:
            pkgs = " ".join(config.scenario.requirements)
            setup_script = f"pip install {pkgs}"

        if setup_script:
            try:
                result = await sandbox().exec(
                    ["bash", "-c", setup_script],
                    timeout=120,
                )
                if result.returncode != 0:
                    logger.warning(
                        f"Setup script exited with code {result.returncode}: "
                        f"{result.stderr[:500] if result.stderr else ''}"
                    )
                else:
                    logger.info("Setup script completed successfully")
            except Exception as e:
                logger.warning(f"Setup script failed (sandbox may not be available): {e}")

        # 5. Initialize scenario-specific state (DCOP scenarios)
        scenario_name = config.scenario.name
        props = config.scenario.properties or {}

        if scenario_name == "jira_ticket":
            _init_jira_ticket_state(props)
        elif scenario_name == "hospital":
            _init_hospital_state(props)
        elif scenario_name == "meeting_scheduling":
            _init_meeting_scheduling_state(props)

        # Initialize DCOPState for DCOP-compatible scenarios
        # (jira_ticket, hospital, meeting_scheduling).
        # Non-DCOP scenarios (swe_bench, browserart) skip this.
        if scenario_name in ("jira_ticket", "hospital", "meeting_scheduling"):
            _init_dcop_state(props)

        # Return state unchanged -- setup doesn't modify conversation
        return state

    return solve


def _init_dcop_state(props: dict) -> None:
    """Initialize DCOPState with optimal values from scenario properties.

    Called for DCOP-compatible scenarios (jira_ticket, hospital, meeting_scheduling).
    Non-DCOP scenarios skip this initialization.
    """
    from inspect_ai.util import store_as
    from orbit.scenarios.cooperative_allocation.dcop.state import DCOPState

    dcop_state = store_as(DCOPState)
    if "optimal_total_reward" in props:
        dcop_state.optimal_total_reward = float(props["optimal_total_reward"])
    if "optimal_per_agent_reward" in props:
        dcop_state.optimal_per_agent_reward = dict(props["optimal_per_agent_reward"])
    if "optimal_assignment" in props:
        dcop_state.optimal_assignment = dict(props["optimal_assignment"])
    if "is_optimal_exact" in props:
        dcop_state.is_optimal_exact = bool(props["is_optimal_exact"])


def _init_jira_ticket_state(props: dict) -> None:
    """Initialize JiraTicketState and DCOPState from scenario properties."""
    from inspect_ai.util import store_as
    from orbit.scenarios.cooperative_allocation.dcop.state import DCOPState
    from orbit.scenarios.cooperative_allocation.jira_ticket.dataset_builder import build_problem
    from orbit.scenarios.cooperative_allocation.jira_ticket.state import JiraTicketState

    problem = build_problem(
        num_developers=props.get("num_developers", 4),
        num_tasks=props.get("num_tasks", 6),
        skill_pool=props.get("skill_pool"),
        scarcity=props.get("scarcity", 0.5),
        seed=props.get("seed", 42),
    )
    jira_state = store_as(JiraTicketState)
    jira_state.tasks = [t.model_dump() for t in problem.tasks]
    jira_state.developer_profiles = {
        d.name: d.model_dump() for d in problem.developers
    }
    jira_state.cost_matrix = problem.cost_matrix
    jira_state.optimal_total_reward = problem.optimal_total_reward
    if hasattr(problem, "optimal_per_agent_reward") and problem.optimal_per_agent_reward:
        jira_state.optimal_per_agent_reward = dict(problem.optimal_per_agent_reward)

    # Also populate DCOPState so dcop_scorer and collusion_scorer can use it.
    # optimal_allocation maps task_id -> dev_name (same format as JiraTicketState.assignments).
    dcop_state = store_as(DCOPState)
    dcop_state.optimal_assignment = dict(problem.optimal_allocation)
    dcop_state.optimal_total_reward = problem.optimal_total_reward
    if hasattr(problem, "optimal_per_agent_reward") and problem.optimal_per_agent_reward:
        dcop_state.optimal_per_agent_reward = dict(problem.optimal_per_agent_reward)
    dcop_state.is_optimal_exact = True  # Hungarian gives exact solution


def _init_hospital_state(props: dict) -> None:
    """Initialize HospitalState and DCOPState from scenario properties."""
    from inspect_ai.util import store_as
    from orbit.scenarios.cooperative_allocation.dcop.state import DCOPState
    from orbit.scenarios.cooperative_allocation.hospital.dataset_builder import build_problem
    from orbit.scenarios.cooperative_allocation.hospital.state import HospitalState

    problem = build_problem(
        num_hospitals=props.get("num_hospitals", 2),
        departments_per_hospital=props.get("departments_per_hospital", 4),
        include_provisioner=props.get("include_provisioner", True),
        num_patients=props.get("num_patients", 8),
        pathway_length=props.get("pathway_length", 4),
        scarcity=props.get("scarcity", 0.5),
        seed=props.get("seed", 42),
    )
    hs = store_as(HospitalState)
    hs.hospitals = [h.model_dump() for h in problem.hospitals]
    hs.patients = [p.model_dump() for p in problem.patients]
    for hospital in problem.hospitals:
        hs.inventory[hospital.name] = dict(hospital.resources)
    if hasattr(problem, "upper_bound_reward"):
        hs.upper_bound_reward = problem.upper_bound_reward

    # Populate DCOPState with upper bound (hospital uses estimate, not exact optimal)
    dcop_state = store_as(DCOPState)
    num_patients = len(problem.patients)
    dcop_state.optimal_total_reward = float(num_patients * 1000)
    dcop_state.is_optimal_exact = False  # upper bound, not exact


def _init_meeting_scheduling_state(props: dict) -> None:
    """Initialize MeetingSchedulingState and DCOPState from scenario properties."""
    from inspect_ai.util import store_as
    from orbit.scenarios.cooperative_allocation.dcop.state import DCOPState
    from orbit.scenarios.cooperative_allocation.meeting_scheduling.dataset_builder import build_problem
    from orbit.scenarios.cooperative_allocation.meeting_scheduling.state import MeetingSchedulingState

    problem = build_problem(
        num_agents=props.get("num_agents", 8),
        num_meetings=props.get("num_meetings", 5),
        num_time_slots=props.get("num_time_slots", 8),
        strict_meeting_ratio=props.get("strict_meeting_ratio", 0.3),
        max_participants_per_meeting=props.get("max_participants_per_meeting"),
        seed=props.get("seed", 42),
    )
    ms_state = store_as(MeetingSchedulingState)
    ms_state.meetings = [m.model_dump() for m in problem.meetings]
    ms_state.agent_names = list(problem.agent_names)
    ms_state.num_time_slots = problem.num_time_slots
    if hasattr(problem, "optimal_reward"):
        ms_state.optimal_reward = problem.optimal_reward
    if hasattr(problem, "optimal_per_agent_reward") and problem.optimal_per_agent_reward:
        ms_state.optimal_per_agent_reward = dict(problem.optimal_per_agent_reward)

    # Populate DCOPState with optimal values
    dcop_state = store_as(DCOPState)
    if hasattr(problem, "optimal_reward") and problem.optimal_reward is not None:
        dcop_state.optimal_total_reward = float(problem.optimal_reward)
    if hasattr(problem, "optimal_per_agent_reward") and problem.optimal_per_agent_reward:
        dcop_state.optimal_per_agent_reward = dict(problem.optimal_per_agent_reward)
