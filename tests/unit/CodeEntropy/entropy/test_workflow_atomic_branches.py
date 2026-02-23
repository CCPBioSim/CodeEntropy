from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from CodeEntropy.entropy.workflow import EntropyWorkflow


def test_execute_calls_level_dag_and_entropy_graph_and_logs_tables():
    args = SimpleNamespace(
        start=0,
        end=-1,
        step=1,
        grouping="molecules",
        water_entropy=False,
        selection_string="all",
    )

    universe = MagicMock()
    universe.trajectory = list(range(5))

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=universe,
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    wf._build_reduced_universe = MagicMock(return_value=MagicMock())
    wf._detect_levels = MagicMock(return_value={0: ["united_atom"]})
    wf._split_water_groups = MagicMock(return_value=({0: [0]}, {}))
    wf._finalize_molecule_results = MagicMock()

    wf._group_molecules.grouping_molecules.return_value = {0: [0]}

    with (
        patch("CodeEntropy.entropy.workflow.LevelDAG") as LevelDAGCls,
        patch("CodeEntropy.entropy.workflow.EntropyGraph") as GraphCls,
    ):
        LevelDAGCls.return_value.build.return_value.execute.return_value = None
        GraphCls.return_value.build.return_value.execute.return_value = {"x": 1}

        wf.execute()

    wf._reporter.log_tables.assert_called_once()


def test_execute_water_entropy_branch_calls_water_entropy_solver():
    args = SimpleNamespace(
        start=0,
        end=-1,
        step=1,
        grouping="molecules",
        water_entropy=True,
        selection_string="all",
        output_file="out.json",
    )

    universe = MagicMock()
    universe.trajectory = list(range(5))

    reporter = MagicMock()
    reporter.molecule_data = []
    reporter.residue_data = []
    reporter.save_dataframes_as_json = MagicMock()

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=universe,
        reporter=reporter,
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    wf._build_reduced_universe = MagicMock(return_value=MagicMock())
    wf._detect_levels = MagicMock(return_value={0: ["united_atom"]})

    wf._split_water_groups = MagicMock(return_value=({0: [0]}, {9: [1, 2]}))
    wf._finalize_molecule_results = MagicMock()

    wf._group_molecules.grouping_molecules.return_value = {0: [0], 9: [1, 2]}

    with (
        patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls,
        patch("CodeEntropy.entropy.workflow.LevelDAG") as LevelDAGCls,
        patch("CodeEntropy.entropy.workflow.EntropyGraph") as GraphCls,
    ):
        water_instance = WaterCls.return_value
        water_instance._calculate_water_entropy = MagicMock()

        LevelDAGCls.return_value.build.return_value.execute.return_value = None
        GraphCls.return_value.build.return_value.execute.return_value = {}

        wf.execute()

    water_instance._calculate_water_entropy.assert_called_once()
    _, kwargs = water_instance._calculate_water_entropy.call_args
    assert kwargs["universe"] is universe
    assert kwargs["start"] == 0
    assert kwargs["end"] == 5
    assert kwargs["step"] == 1
    assert kwargs["group_id"] == 9
