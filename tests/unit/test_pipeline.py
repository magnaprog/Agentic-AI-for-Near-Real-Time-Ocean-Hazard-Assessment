"""Unit tests for the pipeline routing logic.

Validates that the pipeline routes correctly based on FSM state
and verification outcome, with defense-in-depth for the ABSTAIN path.
"""

from __future__ import annotations

from hazard_assessment.orchestrator.pipeline import (
    PipelineNode,
    PipelineState,
    build_pipeline_graph,
    route_after_orchestrate,
    route_after_verify,
)
from hazard_assessment.schemas.verification import VerificationOutcome


class TestRouteAfterOrchestrate:
    """Verify FSM-state-based routing after orchestration."""

    def test_assess_routes_to_scenario(self) -> None:
        state: PipelineState = {"fsm_state": "ASSESS"}
        assert route_after_orchestrate(state) == PipelineNode.SCENARIO

    def test_escalate_routes_to_scenario(self) -> None:
        state: PipelineState = {"fsm_state": "ESCALATE"}
        assert route_after_orchestrate(state) == PipelineNode.SCENARIO

    def test_idle_routes_to_final(self) -> None:
        state: PipelineState = {"fsm_state": "IDLE"}
        assert route_after_orchestrate(state) == PipelineNode.FINAL

    def test_monitor_routes_to_final(self) -> None:
        state: PipelineState = {"fsm_state": "MONITOR"}
        assert route_after_orchestrate(state) == PipelineNode.FINAL

    def test_investigate_routes_to_final(self) -> None:
        state: PipelineState = {"fsm_state": "INVESTIGATE"}
        assert route_after_orchestrate(state) == PipelineNode.FINAL

    def test_empty_state_routes_to_final(self) -> None:
        state: PipelineState = {}
        assert route_after_orchestrate(state) == PipelineNode.FINAL


class TestRouteAfterVerify:
    """Verify defense-in-depth routing after verification."""

    def test_pass_routes_to_report(self) -> None:
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            }
        }
        assert route_after_verify(state) == PipelineNode.REPORT

    def test_pass_with_concerns_routes_to_report(self) -> None:
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.PASS_WITH_CONCERNS,
                "abstain_required": False,
            }
        }
        assert route_after_verify(state) == PipelineNode.REPORT

    def test_fail_routes_to_abstain(self) -> None:
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
            }
        }
        assert route_after_verify(state) == PipelineNode.ABSTAIN

    def test_fail_outcome_alone_triggers_abstain(self) -> None:
        """FAIL outcome routes to ABSTAIN even if flag is somehow False."""
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": False,
            }
        }
        assert route_after_verify(state) == PipelineNode.ABSTAIN

    def test_abstain_flag_alone_triggers_abstain(self) -> None:
        """abstain_required=True routes to ABSTAIN even with non-FAIL outcome.

        Note: the VerificationResult schema now rejects PASS+abstain_required,
        but the router operates on raw dicts and must handle contradictory
        inputs defensively (defense-in-depth).
        """
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": True,
            }
        }
        assert route_after_verify(state) == PipelineNode.ABSTAIN

    def test_missing_verification_routes_to_abstain(self) -> None:
        """Fail-closed: missing verification_result -> ABSTAIN, never REPORT."""
        state: PipelineState = {}
        assert route_after_verify(state) == PipelineNode.ABSTAIN

    def test_empty_dict_verification_routes_to_abstain(self) -> None:
        """Fail-closed: empty verification_result dict -> ABSTAIN."""
        state: PipelineState = {"verification_result": {}}
        assert route_after_verify(state) == PipelineNode.ABSTAIN

    def test_unknown_outcome_routes_to_abstain(self) -> None:
        """Fail-closed: unrecognized verification outcome -> ABSTAIN."""
        state: PipelineState = {
            "verification_result": {
                "overall": "UNKNOWN",
                "abstain_required": False,
            }
        }
        assert route_after_verify(state) == PipelineNode.ABSTAIN


class TestBuildPipelineGraph:
    """Verify pipeline graph specification structure."""

    def test_graph_has_all_nodes(self) -> None:
        graph = build_pipeline_graph()
        expected_nodes = {node.value for node in PipelineNode}
        assert set(graph["nodes"]) == expected_nodes

    def test_graph_has_entry_point(self) -> None:
        graph = build_pipeline_graph()
        assert graph["entry_point"] == PipelineNode.INGEST

    def test_graph_has_conditional_edges(self) -> None:
        graph = build_pipeline_graph()
        assert PipelineNode.ORCHESTRATE in graph["conditional_edges"]
        assert PipelineNode.VERIFY in graph["conditional_edges"]

    def test_orchestrate_branches(self) -> None:
        graph = build_pipeline_graph()
        branches = graph["conditional_edges"][PipelineNode.ORCHESTRATE]["branches"]
        assert PipelineNode.SCENARIO in branches
        assert PipelineNode.FINAL in branches

    def test_verify_branches(self) -> None:
        graph = build_pipeline_graph()
        branches = graph["conditional_edges"][PipelineNode.VERIFY]["branches"]
        assert PipelineNode.REPORT in branches
        assert PipelineNode.ABSTAIN in branches

    def test_human_review_gate_exists(self) -> None:
        """Human Review Gate is a mandatory checkpoint (P6)."""
        graph = build_pipeline_graph()
        assert PipelineNode.HUMAN_REVIEW.value in graph["nodes"]

    def test_report_flows_through_human_review(self) -> None:
        """REPORT -> HUMAN_REVIEW, not directly to FINAL."""
        graph = build_pipeline_graph()
        assert (PipelineNode.REPORT, PipelineNode.HUMAN_REVIEW) in graph["edges"]

    def test_abstain_flows_through_human_review(self) -> None:
        """ABSTAIN -> HUMAN_REVIEW, not directly to FINAL."""
        graph = build_pipeline_graph()
        assert (PipelineNode.ABSTAIN, PipelineNode.HUMAN_REVIEW) in graph["edges"]

    def test_human_review_flows_to_final(self) -> None:
        """HUMAN_REVIEW -> FINAL."""
        graph = build_pipeline_graph()
        assert (PipelineNode.HUMAN_REVIEW, PipelineNode.FINAL) in graph["edges"]


class TestGraphSpecMatchesRunner:
    """Validate that the declared graph spec matches run_pipeline_sync() behavior.

    The graph spec in build_pipeline_graph() is declarative documentation.
    run_pipeline_sync() hardcodes the routing. These tests ensure they
    don't silently diverge.
    """

    def test_all_graph_nodes_reachable_in_runner(self) -> None:
        """Every node in the graph spec should be visited by at least one
        run_pipeline_sync() execution path."""
        import inspect

        from hazard_assessment.orchestrator.nodes import run_pipeline_sync

        source = inspect.getsource(run_pipeline_sync)
        graph = build_pipeline_graph()

        # All node functions called in the runner (by name convention: node_name + "_node").
        # Pass-through nodes (ingest) are declared in the graph spec for structural
        # documentation but are not called in the synchronous runner - data arrives
        # via the ingest workers and Kafka, not via an in-pipeline node.
        # Pass-through nodes exist in the graph spec for structural
        # documentation.  Their data is pre-populated by the caller
        # (ingest workers, anomaly agent, etc.) before run_pipeline_sync().
        PASSTHROUGH_NODES = {"ingest", "qc", "anomaly", "scenario"}
        for node in graph["nodes"]:
            if node in PASSTHROUGH_NODES:
                continue
            node_func = f"{node}_node"
            assert node_func in source, (
                f"Graph declares node '{node}' but run_pipeline_sync() never "
                f"calls {node_func}(). The graph spec and runner have diverged."
            )

    def test_conditional_edges_have_matching_routers(self) -> None:
        """Each conditional edge router should be called in run_pipeline_sync()."""
        import inspect

        from hazard_assessment.orchestrator.nodes import run_pipeline_sync

        source = inspect.getsource(run_pipeline_sync)
        graph = build_pipeline_graph()

        for _node, edge_spec in graph["conditional_edges"].items():
            router_name = edge_spec["router"]
            assert router_name in source, (
                f"Graph declares router '{router_name}' but it's not called "
                f"in run_pipeline_sync()."
            )


class TestHumanReviewMandatoryGate:
    """Safety tests: HUMAN_REVIEW must be the only path to FINAL for output nodes.

    These tests enforce Prohibited Action P6 - no assessment reaches FINAL
    without passing through the Human Review Gate.
    """

    def test_no_direct_report_to_final(self) -> None:
        """REPORT must never connect directly to FINAL (must go through HUMAN_REVIEW)."""
        graph = build_pipeline_graph()
        assert (PipelineNode.REPORT, PipelineNode.FINAL) not in graph["edges"]

    def test_no_direct_abstain_to_final(self) -> None:
        """ABSTAIN must never connect directly to FINAL (must go through HUMAN_REVIEW)."""
        graph = build_pipeline_graph()
        assert (PipelineNode.ABSTAIN, PipelineNode.FINAL) not in graph["edges"]

    def test_human_review_is_sole_predecessor_of_final_for_output(self) -> None:
        """Only HUMAN_REVIEW and ORCHESTRATE (for non-ASSESS states) feed into FINAL."""
        graph = build_pipeline_graph()
        edges_to_final = [
            src for src, dst in graph["edges"] if dst == PipelineNode.FINAL
        ]
        # Only HUMAN_REVIEW has a fixed edge to FINAL.
        # ORCHESTRATE reaches FINAL via conditional routing, not a fixed edge.
        assert edges_to_final == [PipelineNode.HUMAN_REVIEW]

    def test_both_output_paths_converge_at_human_review(self) -> None:
        """Both REPORT and ABSTAIN must route through HUMAN_REVIEW."""
        graph = build_pipeline_graph()
        edges = graph["edges"]
        sources_to_human_review = {
            src for src, dst in edges if dst == PipelineNode.HUMAN_REVIEW
        }
        assert PipelineNode.REPORT in sources_to_human_review
        assert PipelineNode.ABSTAIN in sources_to_human_review
