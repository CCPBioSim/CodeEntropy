from CodeEntropy.levels.nodes.accumulators import InitCovarianceAccumulatorsNode


def test_init_covariance_accumulators_allocates_and_sets_aliases():
    node = InitCovarianceAccumulatorsNode()

    shared = {"groups": {9: [1, 2], 2: [3]}}

    out = node.run(shared)

    gid2i = out["group_id_to_index"]
    i2gid = out["index_to_group_id"]

    assert set(gid2i.keys()) == {9, 2}
    assert set(i2gid) == {9, 2}

    for gid, idx in gid2i.items():
        assert i2gid[idx] == gid

    assert sorted(gid2i.values()) == [0, 1]

    assert "force_covariances" in out
    assert "torque_covariances" in out
    assert "frame_counts" in out
    assert "forcetorque_covariances" in out
    assert "forcetorque_counts" in out

    assert "force_torque_stats" in out
    assert "force_torque_counts" in out

    assert out["force_torque_stats"] is not out["forcetorque_covariances"]
    assert out["force_torque_counts"] is not out["forcetorque_counts"]


def test_init_covariance_accumulators_is_fully_deterministic():
    node = InitCovarianceAccumulatorsNode()

    shared1 = {"groups": {9: [1, 2], 2: [3]}}
    shared2 = {"groups": {2: [3], 9: [1, 2]}}

    out1 = node.run(shared1.copy())
    out2 = node.run(shared2.copy())

    assert out1.keys() == out2.keys()

    for key in out1:
        if isinstance(out1[key], dict):
            assert out1[key].keys() == out2[key].keys()


def test_init_covariance_accumulators_no_aliasing():
    node = InitCovarianceAccumulatorsNode()

    shared = {"groups": {1: [1]}}
    out = node.run(shared)

    assert out["force_torque_stats"] is not out["forcetorque_covariances"]
    assert out["force_torque_counts"] is not out["forcetorque_counts"]
