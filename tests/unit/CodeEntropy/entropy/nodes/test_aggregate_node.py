from CodeEntropy.entropy.nodes.aggregate import AggregateEntropyNode


def test_aggregate_node_collects_values_and_writes_shared_data():
    node = AggregateEntropyNode()
    shared = {"vibrational_entropy": {"v": 1}, "configurational_entropy": {"c": 2}}

    out = node.run(shared)

    assert out["entropy_results"]["vibrational_entropy"] == {"v": 1}
    assert out["entropy_results"]["configurational_entropy"] == {"c": 2}
    assert shared["entropy_results"] == out["entropy_results"]


def test_aggregate_node_missing_upstreams_yields_none_values():
    node = AggregateEntropyNode()
    shared = {}

    out = node.run(shared)

    assert out["entropy_results"]["vibrational_entropy"] is None
    assert out["entropy_results"]["configurational_entropy"] is None
