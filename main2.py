# graph_coarsening/main2.py
import logging
from multiprocessing import Process
from pathlib import Path

from .graph import Graph, compute_euclidean_tau
from .utils import load_graph_from_csv, calculate_route_metrics
from .greedy_solver import GreedySolver
from .savings_solver import SavingsSolver
from .coarsener import SpatioTemporalGraphCoarsener
from .quantum_solvers.vrp_problem import VRPProblem
from .visualisation import visualize_routes
from .or_tools_solver import ORToolsSolver

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_visualisation_counter_uncoarsened = {}
_visualisation_counter_coarsened = {}

# ---------- Helpers ----------
def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float) -> VRPProblem:
    """Build a VRPProblem from our Graph (dense/complete graph assumption)."""
    costs_matrix = {u: {} for u in graph.nodes}
    time_costs_matrix = {u: {} for u in graph.nodes}
    for u_id, u_node in graph.nodes.items():
        for v_id, v_node in graph.nodes.items():
            tau = 0.0 if u_id == v_id else compute_euclidean_tau(u_node, v_node)
            costs_matrix[u_id][v_id] = tau
            time_costs_matrix[u_id][v_id] = tau

    capacities = [vehicle_capacity]
    dests = [node_id for node_id in graph.nodes if node_id != depot_id]
    weights = {node_id: node.demand for node_id, node in graph.nodes.items()}
    time_windows = {node_id: (node.e, node.l) for node_id, node in graph.nodes.items()}
    service_times = {node_id: node.s for node_id, node in graph.nodes.items()}

    return VRPProblem(
        sources=[depot_id],
        costs=costs_matrix,
        time_costs=time_costs_matrix,
        capacities=capacities,
        dests=dests,
        weights=weights,
        time_windows=time_windows,
        service_times=service_times,
    )

def log_graph_info(graph: Graph, depot_id: str):
    logger.info(f"Graph loaded. Number of nodes: {len(graph.nodes)}")
    logger.info(f"  Depot ID: {depot_id}")
    logger.info(f"  Number of edges: {len(graph.edges)}")

def log_coarsening_info(coarsener: SpatioTemporalGraphCoarsener, coarsened_graph: Graph, merge_layers: list):
    logger.info(f"--- Coarsening Finished ---")
    logger.info(f"  Final coarsened graph has {len(coarsened_graph.nodes)} nodes.")
    logger.info(f"  Coarsening resulted in {len(merge_layers)} merge layers.")

def run_uncoarsened_solvers(graph: Graph, depot_id: str, capacity: float):
    logger.info(f"\n--- Running Solvers on UNCOARSENED Graph ---")
    results = {}
    solvers = [
        ('Greedy', GreedySolver(graph, depot_id, capacity)),
        ('Savings', SavingsSolver(graph, depot_id, capacity)),
        #('ORTools', ORToolsSolver(graph, depot_id, capacity)),
    ]
    for name, solver in solvers:
        logger.info(f"  Running {name} Solver...")
        try:
            routes, metrics = solver.solve()
            logger.info(f"    {name} Solver finished. Total distance: {metrics['total_distance']:.2f}")
            results[f"uncoarsened_{name}"] = {'routes': routes, 'metrics': metrics}
            visualize_routes(routes, f"Uncoarsened {name}", f"uncoarsened_{name}")
        except Exception as e:
            logger.error(f"    Error with {name} solver: {e}")
    return results

def run_inflated_solvers(coarsener, coarsened_graph, depot_id, capacity, original_graph):
    logger.info(f"\n--- Running Solvers on COARSENED/INFLATED Graph ---")
    results = {}
    solvers = [
        ('Greedy', GreedySolver(coarsened_graph, depot_id, capacity)),
        ('Savings', SavingsSolver(coarsened_graph, depot_id, capacity)),
        #('ORTools', ORToolsSolver(coarsened_graph, depot_id, capacity)),
    ]
    for name, solver in solvers:
        logger.info(f"  Running {name} Solver on coarsened graph...")
        try:
            coarsened_routes, _ = solver.solve()
            inflated_routes = coarsener.inflate(coarsened_routes)
            inflated_metrics = calculate_route_metrics(original_graph, inflated_routes, depot_id, capacity)
            logger.info(f"    {name} Solver finished. Total distance (inflated): {inflated_metrics['total_distance']:.2f}")
            results[f"inflated_{name}"] = {'routes': inflated_routes, 'metrics': inflated_metrics}
            visualize_routes(inflated_routes, f"Inflated {name}", f"inflated_{name}")
        except Exception as e:
            logger.error(f"    Error with coarsened {name} solver: {e}")
    return results

def process_file(csv_file_path: str):
    logger.info(f"\n\n=== Processing file: {csv_file_path} ===")
    try:
        graph, depot_id, capacity = load_graph_from_csv(csv_file_path)
        log_graph_info(graph, depot_id)

        coarsener = SpatioTemporalGraphCoarsener(
            graph=graph, alpha=0.8, beta=0.4, P=0.5, radiusCoeff=2.0, depot_id=depot_id
        )
        coarsened_graph, merge_layers = coarsener.coarsen()
        log_coarsening_info(coarsener, coarsened_graph, merge_layers)

        run_uncoarsened_solvers(graph, depot_id, capacity)
        run_inflated_solvers(coarsener, coarsened_graph, depot_id, capacity, graph)

        # Example (optional): build a VRPProblem from the loaded CSV via our converter
        _ = convert_graph_to_vrp_problem_inputs(graph, depot_id, capacity)

    except Exception as e:
        logger.error(f"Error loading or processing CSV file {csv_file_path}: {e}")

def collect_inputs(data_dir: Path) -> list[Path]:
    """Return all .csv files under data_dir (recursive)."""
    return list(sorted(data_dir.rglob("*.csv")))

# ---------- Defaults (no CLI args needed) ----------
# Paths are resolved relative to THIS FILE, not the shell's CWD
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "solomon_dataset"
# If this specific file exists, we'll run only this one; otherwise we'll scan the whole dataset.
_DEFAULT_SINGLE_FILE = _DEFAULT_DATA_DIR / "C1" / "C101.csv"  # change here if you prefer another default

def main():
    logger.info("Starting VRP Solver Pipeline (no CLI args).")

    if _DEFAULT_SINGLE_FILE.is_file():
        logger.info(f"Processing single default file: {_DEFAULT_SINGLE_FILE}")
        p = Process(target=process_file, args=(str(_DEFAULT_SINGLE_FILE),))
        p.start()
        p.join()
        logger.info("Finished processing default file.")
        return

    if not _DEFAULT_DATA_DIR.exists():
        logger.error(f"Data directory not found: {_DEFAULT_DATA_DIR}")
        return

    inputs = collect_inputs(_DEFAULT_DATA_DIR)
    logger.info(f"Found {len(inputs)} input file(s) under {_DEFAULT_DATA_DIR}")
    if not inputs:
        logger.warning("No CSV files found under the dataset directory.")
        return

    procs = []
    for path in inputs:
        p = Process(target=process_file, args=(str(path),))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    logger.info("Finished processing all files.")

if __name__ == "__main__":
    # macOS uses 'spawn' start method; the main-guard prevents child re-execution.
    main()
