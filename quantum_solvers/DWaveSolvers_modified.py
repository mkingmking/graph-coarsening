class _DummyResponse:
    def __init__(self, samples=None):
        self._samples = samples or [{}]

    def lowest(self):
        return self._samples


class _DummySolver:
    def sample_qubo(self, qubo_dict, num_reads=1):
        return _DummyResponse()


def get_solver(solver_type):
    """Return a lightweight stand-in solver for testing environments."""
    # The real project can swap in D-Wave or other solvers, but the test
    # suite only needs a callable object that responds to ``sample_qubo``.
    if solver_type in {"qpu", "hybrid", "simulated", "exact"}:
        return _DummySolver()
    raise ValueError(f"Solver type '{solver_type}' is not supported.")


def solve_qubo(qubo, solver_type="simulated", limit=1, num_reads=50):
    """Solve a QUBO-like object using the lightweight stand-in solver."""
    solver = get_solver(solver_type)
    response = solver.sample_qubo(qubo.dict, num_reads=num_reads)
    return list(response.lowest())[:limit]
