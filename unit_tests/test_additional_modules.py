import json
import importlib
from pathlib import Path

import pytest

from graph_coarsening.graph import Graph
from graph_coarsening.node import Node
from graph_coarsening.quantum_solvers import DWaveSolvers_modified
from graph_coarsening.utils import check_coverage, parse_float


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"{module_name} unavailable in this environment: {exc}")


def _build_numeric_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node("0", 0, 0, 0, 0, 100, 0))
    graph.add_node(Node("1", 1, 0, 1, 0, 100, 2))
    graph.add_node(Node("2", 2, 0, 1, 0, 100, 3))
    graph.add_edge("0", "1", 1.0)
    graph.add_edge("0", "2", 2.0)
    graph.add_edge("1", "2", 1.0)
    return graph


def test_parse_float_extracts_number_from_malformed_field():
    assert parse_float(" 0.00    1 ") == 0.0
    assert parse_float("value=17.5ms") == 17.5


def test_check_coverage_handles_known_and_unknown_instances():
    assert check_coverage("C101.csv", {"total_demand_served": 1810.0}) is True
    assert check_coverage("custom_instance.csv", {"total_demand_served": 12.0}) is True


def test_visualize_routes_calls_savefig_with_requested_filename(monkeypatch):
    visualisation = _import_or_skip("graph_coarsening.visualisation")

    graph = _build_numeric_graph()
    saved = {}

    monkeypatch.setattr(visualisation.plt, "savefig", lambda path, **kwargs: saved.setdefault("path", path))

    visualisation.visualize_routes(
        graph,
        routes=[["0", "1", "2", "0"]],
        depot_id="0",
        title="Test Plot",
        filename="custom_plot.png",
    )

    assert saved["path"].endswith("visualisation_routes/custom_plot.png")


def test_boxplot_generator_load_prepare_and_save(tmp_path, monkeypatch):
    boxplot_generator = _import_or_skip("graph_coarsening.plots.boxplot_generator")

    input_path = tmp_path / "results.json"
    payload = {
        "some/path/C101.csv": {
            "Uncoarsened Greedy": {
                "total_distance": 10.0,
                "num_vehicles": 1,
                "solver_status": "feasible",
            },
            "Inflated Greedy": {
                "total_distance": 9.0,
                "num_vehicles": 1,
            },
        }
    }
    input_path.write_text(json.dumps(payload))

    loaded = boxplot_generator.load_results(str(input_path))
    df = boxplot_generator.prepare_data(loaded)

    assert loaded == payload
    assert set(df["Metric"]) >= {"total_distance", "num_vehicles"}
    assert set(df["Method"]) == {"Uncoarsened", "Inflated"}
    assert set(df["Solver"]) == {"Greedy"}

    saved_paths = []
    monkeypatch.setattr(boxplot_generator.plt, "savefig", lambda path, **kwargs: saved_paths.append(path))

    boxplot_generator.create_box_plots(df, str(tmp_path / "plots"))

    assert any(str(path).endswith("boxplot_total_distance.png") for path in saved_paths)


def test_solomon_dataset_visualiser_loads_graph_and_shows_plot(monkeypatch):
    solomon_dataset_visualiser = _import_or_skip("graph_coarsening.plots.solomon_dataset_visualiser")

    graph = _build_numeric_graph()
    shown = {"called": False}

    monkeypatch.setattr(
        solomon_dataset_visualiser,
        "load_graph_from_csv",
        lambda file_path: (graph, "0", 200.0),
    )
    monkeypatch.setattr(
        solomon_dataset_visualiser.plt,
        "show",
        lambda: shown.__setitem__("called", True),
    )

    solomon_dataset_visualiser.visualize_dataset("fake.csv")

    assert shown["called"] is True


def test_dwave_solve_qubo_uses_num_reads_for_simulated(monkeypatch):
    class FakeResponse:
        def lowest(self):
            return [{"x": 1}, {"x": 2}]

    class FakeSampler:
        def __init__(self):
            self.calls = []

        def sample_qubo(self, qubo_dict, num_reads=None):
            self.calls.append((qubo_dict, num_reads))
            return FakeResponse()

    sampler = FakeSampler()
    monkeypatch.setattr(DWaveSolvers_modified, "get_solver", lambda solver_type: sampler)

    class FakeQubo:
        dict = {("x", "x"): -1}

    samples = DWaveSolvers_modified.solve_qubo(FakeQubo(), solver_type="simulated", limit=1, num_reads=7)

    assert samples == [{"x": 1}]
    assert sampler.calls == [({("x", "x"): -1}, 7)]


def test_dwave_solve_qubo_omits_num_reads_for_exact(monkeypatch):
    class FakeResponse:
        def lowest(self):
            return [{"x": 1}]

    class FakeSampler:
        def __init__(self):
            self.calls = []

        def sample_qubo(self, qubo_dict):
            self.calls.append(qubo_dict)
            return FakeResponse()

    sampler = FakeSampler()
    monkeypatch.setattr(DWaveSolvers_modified, "get_solver", lambda solver_type: sampler)

    class FakeQubo:
        dict = {("x", "x"): -1}

    samples = DWaveSolvers_modified.solve_qubo(FakeQubo(), solver_type="exact", limit=1, num_reads=99)

    assert samples == [{"x": 1}]
    assert sampler.calls == [{("x", "x"): -1}]


def test_main_find_csv_files_returns_sorted_csvs(tmp_path):
    classical_main = _import_or_skip("graph_coarsening.main")

    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "z.csv").write_text("")
    (tmp_path / "a" / "m.csv").write_text("")
    (tmp_path / "a" / "note.txt").write_text("")

    found = classical_main.find_csv_files(str(tmp_path))

    assert found == sorted(found)
    assert [Path(path).suffix for path in found] == [".csv", ".csv"]


def test_main_save_results_to_json_writes_file(tmp_path):
    classical_main = _import_or_skip("graph_coarsening.main")

    output = tmp_path / "nested" / "results.json"
    payload = {"ok": True}

    classical_main.save_results_to_json(payload, str(output))

    assert output.exists()
    assert json.loads(output.read_text()) == payload


def test_main_run_solver_pipeline_with_coarsener_recalculates_metrics(monkeypatch):
    classical_main = _import_or_skip("graph_coarsening.main")

    graph = _build_numeric_graph()
    captured = {}

    class FakeSolver:
        def __init__(self, graph, depot_id, vehicle_capacity):
            pass

        def solve(self):
            return [["1", "2"]], {"raw": True}

    class FakeCoarsener:
        def __init__(self, graph):
            self.graph = graph

        def inflate_route(self, routes):
            captured["formatted"] = routes
            return [["0", "1", "2", "0"]]

    monkeypatch.setattr(classical_main, "GreedySolver", FakeSolver)
    monkeypatch.setattr(
        classical_main,
        "calculate_route_metrics",
        lambda graph, routes, depot_id, vehicle_capacity: {"is_feasible": True, "num_vehicles": 1},
    )

    routes, metrics, duration = classical_main.run_solver_pipeline(
        graph, depot_id="0", vehicle_capacity=10, solver_name="Greedy", coarsener=FakeCoarsener(graph)
    )

    assert captured["formatted"] == [["0", "1", "2", "0"]]
    assert routes == [["0", "1", "2", "0"]]
    assert metrics["is_feasible"] is True
    assert duration >= 0


def test_main_quantum_helpers_convert_map_and_subgraph():
    main_quantum = _import_or_skip("graph_coarsening.main_quantum")

    graph = _build_numeric_graph()

    vrp, int_to_id = main_quantum.convert_graph_to_vrp_problem_inputs(graph, "0", 10.0)
    mapped = main_quantum.map_solution_to_original_ids([[1, 2]], int_to_id)
    subgraph = main_quantum.create_subgraph(graph, "0", 1)

    assert int_to_id == ["0", "1", "2"]
    assert vrp.source_depot == 0
    assert vrp.dests == [1, 2]
    assert mapped == [["1", "2"]]
    assert set(subgraph.nodes) == {"0", "1"}


def test_ortools_solver_solves_single_customer_graph():
    ortools_module = _import_or_skip("graph_coarsening.ortools_solver")

    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))
    graph.add_node(Node("A", 1, 0, 0, 0, 100, 1))
    graph.add_edge("D", "A", 1.0)

    solver = ortools_module.ORToolsVRPTWSolver(
        graph,
        depot_id="D",
        vehicle_capacity=10,
        time_limit_seconds=1,
        solution_limit=1,
    )
    routes, metrics = solver.solve()

    assert routes == [["D", "A", "D"]]
    assert metrics["num_vehicles"] == 1
    assert metrics["is_feasible"] is True


def test_run_ortools_print_summary_outputs_instance(capsys):
    run_ortools = _import_or_skip("graph_coarsening.run_ortools")

    run_ortools.print_summary(
        {
            "C101": {
                "Uncoarsened ORTools": {"total_distance": 10.0, "num_vehicles": 1, "computation_time": 2.0, "is_feasible": True},
                "Inflated ORTools": {"total_distance": 8.0, "num_vehicles": 1, "computation_time": 1.0, "is_feasible": True},
            }
        }
    )

    out = capsys.readouterr().out
    assert "C101" in out
    assert "Median distance change" in out


def test_run_ortools_quantum_benchmark_print_summary_outputs_size_key(capsys):
    run_ortools_quantum_benchmark = _import_or_skip("graph_coarsening.run_ortools_quantum_benchmark")

    run_ortools_quantum_benchmark.print_summary(
        {
            "C101": {
                "N5": {
                    "Uncoarsened ORTools": {
                        "total_distance": 10.0,
                        "num_vehicles": 1,
                        "time_window_violations": 0,
                        "is_feasible": True,
                        "computation_time": 0.1,
                        "solver_status": "feasible",
                    }
                }
            }
        }
    )

    out = capsys.readouterr().out
    assert "C101" in out
    assert "N5" in out


def test_parameter_tuning_boxplot_create_boxplots_calls_show(monkeypatch):
    tuning_boxplot = _import_or_skip("graph_coarsening.parameter_tuning.boxplot")
    pd = pytest.importorskip("pandas")

    shown = {"called": False}
    monkeypatch.setattr(tuning_boxplot.plt, "show", lambda: shown.__setitem__("called", True))

    tuning_boxplot.create_boxplots(
        pd.DataFrame([{"alpha": 0.5, "inflated_total_distance": 12.0}]),
        hyperparameter="alpha",
        metric="inflated_total_distance",
        metric_label="Total Inflated Distance",
    )

    assert shown["called"] is True


def test_tuning_quantum_helpers_convert_subgraph_and_map():
    tuning_quantum_solvers = _import_or_skip("graph_coarsening.parameter_tuning.tuning_quantum_solvers")

    graph = _build_numeric_graph()

    subgraph = tuning_quantum_solvers.create_subgraph(graph, "0", 1)
    vrp, mapping = tuning_quantum_solvers.convert_graph_to_vrp_problem_inputs(graph, "0", 10.0)
    mapped = tuning_quantum_solvers.map_solution_to_original_ids([[1, 2]], mapping)

    assert set(subgraph.nodes) == {"0", "1"}
    assert vrp.source_depot == 0
    assert mapped == [["1", "2"]]


def test_tuning_classical_invalid_solver_name_returns_inf_and_empty_metrics():
    tuning_classical_solvers = _import_or_skip("graph_coarsening.parameter_tuning.tuning_classical_solvers")

    graph = _build_numeric_graph()

    score, metrics = tuning_classical_solvers.run_evaluation_classical(
        graph,
        depot_id="0",
        vehicle_capacity=10.0,
        alpha=1.0,
        beta=1.0,
        P=0.5,
        radiusCoeff=1.0,
        solver_name="UnknownSolver",
    )

    assert score == float("inf")
    assert metrics == {}
