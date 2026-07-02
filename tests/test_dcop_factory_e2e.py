"""End-to-end coverage of the DCOP ``@task`` factories through the shared builder.

The DCOP scenarios (jira_ticket / hospital / meeting_scheduling) were refactored
onto ``make_dcop_plugin`` + ``build_scenario_task`` in PR #15, but nothing
invoked the ``@task`` factories or built a DCOP Task through the plugin — so the
scorer set (security + integrity + scenario scorer + dcop_scorer, plus a
collusion scorer for coalitions/compromised agents) and the ``compromised_agent``
gating were locked by no test. These lock them in. Scorer ORDER (scenario scorer
before dcop_scorer/collusion_scorer, which read stores it populates) is
guaranteed by the literal list in ``dcop.plugin._dcop_scorers``.

Requires the ``dcop`` extra (scipy), which the ``dev`` extra pulls in.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")  # dcop extra


class TestDCOPFactoriesBuild:
    def test_jira_ticket_factory_builds(self):
        from orbit.scenarios.cooperative_allocation.jira_ticket.task import (
            jira_ticket_allocation,
        )

        task = jira_ticket_allocation(num_developers=3, num_tasks=4, seed=42)
        assert len(task.dataset) >= 1
        # security + integrity + jira scorer + dcop_scorer, no collusion.
        assert len(task.scorer) == 4

    def test_compromised_agent_adds_collusion_scorer(self):
        from orbit.scenarios.cooperative_allocation.jira_ticket.task import (
            jira_ticket_allocation,
        )

        base = jira_ticket_allocation(num_developers=3, num_tasks=4, seed=42)
        compromised = jira_ticket_allocation(
            num_developers=3, num_tasks=4, seed=42, compromised_agent="dev_1"
        )
        # A compromised_agent attack must add the collusion scorer — it did not,
        # on the orbit-run path, before PR #15.
        assert len(compromised.scorer) == len(base.scorer) + 1

    def test_hospital_and_meeting_factories_build(self):
        from orbit.scenarios.cooperative_allocation.hospital.task import (
            hospital_scheduling,
        )
        from orbit.scenarios.cooperative_allocation.meeting_scheduling.task import (
            meeting_scheduling,
        )

        h = hospital_scheduling(num_hospitals=1, num_patients=3, seed=42)
        assert len(h.dataset) >= 1
        assert len(h.scorer) == 4

        m = meeting_scheduling(num_agents=3, num_meetings=2, seed=42)
        assert len(m.dataset) >= 1
        assert len(m.scorer) == 4
