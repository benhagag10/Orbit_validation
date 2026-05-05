"""
Pre-flight configuration validator.

Validates ExperimentConfig before execution to catch errors early.
Called by the setup solver (mas_environment_setup).

No direct Inspect interaction. Pure validation logic on config models.
"""

from __future__ import annotations

from orbit.configs.baseline import BaselineMode
from orbit.configs.experiment import ExperimentConfig


class ConfigValidator:
    """Validates experiment configuration for consistency and correctness.

    Usage:
        errors = ConfigValidator.validate(config)
        if errors:
            raise ValueError(f"Config validation failed: {errors}")
    """

    @staticmethod
    def validate(config: ExperimentConfig) -> list[str]:
        """Run all validation checks on the config.

        Returns a list of error messages. Empty list means valid.
        """
        errors: list[str] = []

        agent_names = {agent.name for agent in config.setup.agents}

        # 1. At least one agent is defined
        if not config.setup.agents:
            errors.append("At least one agent must be defined in setup.agents")

        # 2. No duplicate agent names
        if len(agent_names) != len(config.setup.agents):
            seen: set[str] = set()
            for agent in config.setup.agents:
                if agent.name in seen:
                    errors.append(f"Duplicate agent name: '{agent.name}'")
                seen.add(agent.name)

        # 3. All agents referenced in edges exist
        for edge in config.setup.edges:
            if edge.from_agent not in agent_names:
                errors.append(
                    f"Edge references unknown from_agent: '{edge.from_agent}'"
                )
            if edge.to_agent not in agent_names:
                errors.append(
                    f"Edge references unknown to_agent: '{edge.to_agent}'"
                )

        # 4. No self-loops in edges
        for edge in config.setup.edges:
            if edge.from_agent == edge.to_agent:
                errors.append(
                    f"Self-loop edge not allowed: '{edge.from_agent}' -> '{edge.to_agent}'"
                )

        # 5. All agents referenced in attack.target_agent exist
        for attack in config.attacks:
            if attack.target_agent and attack.target_agent not in agent_names:
                errors.append(
                    f"Attack '{attack.name}' references unknown target_agent: "
                    f"'{attack.target_agent}'"
                )

        # 6. All agents referenced in defense.target_agents exist
        for defense in config.defenses:
            for target in defense.target_agents:
                if target not in agent_names:
                    errors.append(
                        f"Defense '{defense.name}' references unknown target_agent: "
                        f"'{target}'"
                    )

        # 7. Compromised agents have matching compromised_agent attack
        compromised_agents = {
            a.name for a in config.setup.agents if a.is_compromised
        }
        compromised_attack_targets = {
            a.target_agent
            for a in config.attacks
            if a.attack_type == "compromised_agent"
        }
        for agent_name in compromised_agents:
            if agent_name not in compromised_attack_targets:
                errors.append(
                    f"Agent '{agent_name}' is marked as compromised but has no "
                    f"matching compromised_agent attack config"
                )

        # 8. Baseline mode consistency
        if config.baseline_mode == BaselineMode.NO_DEFENSE and config.defenses:
            errors.append(
                "baseline_mode is 'no_defense' but defenses are still configured"
            )
        if config.baseline_mode == BaselineMode.NO_ATTACK and config.attacks:
            errors.append(
                "baseline_mode is 'no_attack' but attacks are still configured"
            )
        if config.baseline_mode == BaselineMode.BENIGN and (
            config.attacks or config.defenses
        ):
            errors.append(
                "baseline_mode is 'benign' but attacks or defenses are still configured"
            )

        # 9. Topology graph connectivity (skip for single-agent setups)
        if len(agent_names) > 1 and config.setup.edges:
            connected = _find_connected_agents(config)
            orphans = agent_names - connected
            if orphans:
                errors.append(
                    f"Orphan agents not connected by any edge: {sorted(orphans)}"
                )

        # 10. Agent group validation
        errors.extend(_validate_agent_groups(config, agent_names))

        return errors


def _validate_agent_groups(
    config: ExperimentConfig, agent_names: set[str]
) -> list[str]:
    """Validate agent groups for consistency."""
    errors: list[str] = []
    execution = config.execution

    group_names: set[str] = set()
    agents_in_groups: dict[str, str] = {}  # agent_name -> group_name

    for group in execution.agent_groups:
        # Unique group names
        if group.name in group_names:
            errors.append(f"Duplicate agent group name: '{group.name}'")
        group_names.add(group.name)

        # Groups must have at least one agent
        if not group.agents:
            errors.append(
                f"Agent group '{group.name}' has no agents assigned"
            )

        # All agent references must exist
        for agent_ref in group.agents:
            if agent_ref not in agent_names:
                errors.append(
                    f"Agent group '{group.name}' references unknown agent: '{agent_ref}'"
                )

            # Each agent in at most one group
            if agent_ref in agents_in_groups:
                other_group = agents_in_groups[agent_ref]
                errors.append(
                    f"Agent '{agent_ref}' is in groups '{other_group}' and "
                    f"'{group.name}'"
                )
            else:
                agents_in_groups[agent_ref] = group.name

    return errors


def _find_connected_agents(config: ExperimentConfig) -> set[str]:
    """Find all agents that are reachable via edges (undirected connectivity)."""
    from collections import deque

    adjacency: dict[str, set[str]] = {}
    for agent in config.setup.agents:
        adjacency[agent.name] = set()
    for edge in config.setup.edges:
        adjacency.setdefault(edge.from_agent, set()).add(edge.to_agent)
        adjacency.setdefault(edge.to_agent, set()).add(edge.from_agent)

    if not adjacency:
        return set()

    # BFS from the first agent
    start = next(iter(adjacency))
    visited: set[str] = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adjacency.get(node, set()) - visited)

    return visited
