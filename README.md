Implementation and evaluation of a spatio-temporal graph coarsening algorithm for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).  Coarsened graphs are solved with classical heuristics (Greedy, Clarke–Wright Savings) and quantum-inspired solvers (FQS, APS), then inflated back to the original graph.  OR-Tools CP-SAT is included as a reference solver.  Experiments are evaluated on the Solomon benchmark dataset.

Test coverage: **76%** on core modules.

---

## Repository Structure

```
graph_coarsening/
│
├── coarsener.py          # Spatio-temporal multilevel coarsening & inflation
├── graph.py              # Graph, Node, Edge data model
├── node.py
├── edge.py
├── greedy_solver.py      # Greedy constructive heuristic
├── savings_solver.py     # Clarke–Wright savings heuristic
├── utils.py              # CSV loading, route metrics, per-family hyperparams
├── visualisation.py      # Route visualisation (matplotlib)
│
├── quantum_solvers/      # FQS and APS QUBO solvers (D-Wave / simulated annealing)
├── ortools_solver/       # OR-Tools CP-SAT reference solver
├── unit_tests/           # pytest test suite
├── solomon_dataset/      # Solomon CVRPTW benchmark CSVs (C, R, RC families)
├── outputs/              # Result JSONs from completed experiment runs
│
├── runners/              # Entry-point scripts (run these directly)
│   ├── main.py                        # Classical pipeline (Greedy + Savings)
│   ├── main_quantum.py                # Quantum pipeline (FQS + APS, coarsened vs uncoarsened)
│   └── main_only_coarsened_quantum.py # Quantum pipeline, coarsened path only
│
├── experiments/          # Hyperparameter sweep and benchmark scripts
│   ├── run_alpha_experiment.py
│   ├── run_beta_experiment.py
│   ├── run_p_experiment.py
│   ├── run_radius_experiment.py
│   ├── run_hyperparam_search.py
│   ├── run_nmax_boundary.py
│   ├── run_ortools.py
│   └── run_ortools_quantum_benchmark.py
│
└── analysis/             # Post-processing: plots and statistical tests
    ├── generate_output_boxplots.py
    ├── generate_pairwise_boxplots.py
    └── run_wilcoxon_test.py
```

---

## Setup

```bash
git clone https://github.com/mkingmking/graph-coarsening
cd graph-coarsening
pip install -r requirements.txt   # or activate your virtual environment
```

All commands below must be run from the **repo root** (`graph-coarsening/`) using the Python environment that has `matplotlib`, `dwave`, `ortools`, and `scipy` installed.

---

## Usage

### Classical solvers (Greedy + Savings)

```bash
# Single instance
python3 -m graph_coarsening.runners.main --file graph_coarsening/solomon_dataset/C1/C101.csv

# All instances in a directory
python3 -m graph_coarsening.runners.main --data graph_coarsening/solomon_dataset/

# Save results to JSON
python3 -m graph_coarsening.runners.main --data graph_coarsening/solomon_dataset/ --output outputs/results.json
```

### Quantum solvers (FQS + APS)

```bash
# Single instance, 5 customers
python3 -m graph_coarsening.runners.main_quantum \
    --file graph_coarsening/solomon_dataset/C1/C101.csv --customers 5

# Single instance, 10 customers
python3 -m graph_coarsening.runners.main_quantum \
    --file graph_coarsening/solomon_dataset/C1/C101.csv --customers 10

# Coarsened path only (faster, skips uncoarsened baseline)
python3 -m graph_coarsening.runners.main_only_coarsened_quantum \
    --file graph_coarsening/solomon_dataset/C1/C101.csv --customers 10
```

Override per-family hyperparameters on any runner with `--alpha`, `--beta`, `--P`, `--radius`.

---

## Experiments

All experiment scripts live in `graph_coarsening/experiments/`.

### Hyperparameter sweeps

```bash
# Sweep alpha values (default: greedy solver, C101)
python3 -m graph_coarsening.experiments.run_alpha_experiment \
    --csv solomon_dataset/C1/C101.csv --solver greedy --alpha-values 0.3 0.5 0.7 1.0

# Sweep beta, P, radius (same interface)
python3 -m graph_coarsening.experiments.run_beta_experiment   --csv solomon_dataset/C1/C101.csv
python3 -m graph_coarsening.experiments.run_p_experiment      --csv solomon_dataset/C1/C101.csv
python3 -m graph_coarsening.experiments.run_radius_experiment --csv solomon_dataset/C1/C101.csv

# Random hyperparameter search (40 trials, all families)
python3 -m graph_coarsening.experiments.run_hyperparam_search --families C1 C2 R1 R2 RC1 RC2 --n-trials 40
```

### Benchmark runs

```bash
# OR-Tools reference on classical (100-customer) instances
python3 -m graph_coarsening.experiments.run_ortools \
    --file graph_coarsening/solomon_dataset/C1/C101.csv

# OR-Tools reference on N=5 and N=10 sub-instances (quantum benchmark)
python3 -m graph_coarsening.experiments.run_ortools_quantum_benchmark \
    --file graph_coarsening/solomon_dataset/C1/C101.csv

# Find N_max (largest solvable instance size for quantum solvers)
python3 -m graph_coarsening.experiments.run_nmax_boundary \
    --file graph_coarsening/solomon_dataset/C1/C101.csv
```

---

## Analysis

```bash
# Boxplots from all JSONs in outputs/
python3 -m graph_coarsening.analysis.generate_output_boxplots \
    --input-dir outputs/ --output-dir plots/output_boxplots/

# Pairwise coarsened-vs-uncoarsened boxplots
python3 -m graph_coarsening.analysis.generate_pairwise_boxplots \
    --input-dir outputs/ --output-dir plots/pairwise_boxplots/

# Wilcoxon signed-rank test on classical distance results
python3 -m graph_coarsening.analysis.run_wilcoxon_test
```

---

## Tests

```bash
python3 -m pytest graph_coarsening/unit_tests/ -v
```
