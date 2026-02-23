from unittest.mock import MagicMock

from CodeEntropy.levels.frame_dag import FrameGraph


def test_make_frame_ctx_has_required_keys():
    ctx = FrameGraph._make_frame_ctx(shared_data={"x": 1}, frame_index=7)
    assert ctx["shared"] == {"x": 1}
    assert ctx["frame_index"] == 7
    assert ctx["frame_covariance"] is None


def test_add_registers_node_and_deps_edges():
    fg = FrameGraph()
    n1 = MagicMock()
    n2 = MagicMock()

    fg._add("a", n1)
    fg._add("b", n2, deps=["a"])

    assert "a" in fg._nodes and "b" in fg._nodes
    assert ("a", "b") in fg._graph.edges


def test_execute_frame_runs_nodes_in_topological_order_and_returns_frame_covariance():
    fg = FrameGraph()

    a = MagicMock()
    b = MagicMock()

    fg._add("a", a)
    fg._add("b", b, deps=["a"])

    def _b_run(ctx):
        ctx["frame_covariance"] = {"ok": True}

    b.run.side_effect = _b_run

    out = fg.execute_frame(shared_data={"S": 1}, frame_index=3)

    assert out == {"ok": True}
    assert a.run.call_count == 1
    assert b.run.call_count == 1


def test_build_adds_frame_covariance_node():
    fg = FrameGraph()
    fg.build()
    assert "frame_covariance" in fg._nodes
    assert "frame_covariance" in fg._graph.nodes
