This repository contains the implementation and evaluation of a spatio-temporal graph coarsening algorithm designed for solving the Vehicle Routing Problem with Time Windows (VRPTW).

Repository Contents:

solomon_dataset/: Directory containing the Solomon VRPTW benchmark datasets in CSV format (e.g., ``C101.csv`` or ``R101.csv``).  The files list the depot and customers with coordinates, demands and time windows.  Vehicle capacity is not stored in the files, so the loaders assume the standard Solomon capacity of ``200``.

This Python script implements the core spatio-temporal graph coarsening algorithm, including node and edge definitions, graph operations, and the multilevel coarsening and inflation procedures. It also integrates Greedy and Clarke and Wright Savings heuristics for solving VRPTW instances on both original and coarsened graphs.




## Usage

Clone the repository and run the pipeline:

```bash
git clone https://github.com/mkingmking/graph-coarsening
cd graph-coarsening
python3 -m graph_coarsening.main
```

The script will automatically process the CSV files in the solomon_dataset/ directory, apply the coarsening algorithm, solve the VRPTW using the implemented heuristics, and print performance metrics to the console.





## Key Files in the Repository



- `coarsener.py` – multilevel spatio-temporal graph coarsening algorithm.
- `boxplot_generator.py` – builds box plots from `results.json` metrics.
- `greedy_solver.py` – greedy heuristic solver for VRPTW.
- `main.py` – primary pipeline: load data, coarsen, solve and visualise.
- `or_tools_solver.py` – VRPTW solver using Google OR-Tools.
- `results.json` – sample output metrics from experiments.
- `savings_solver.py` – Clarke & Wright savings heuristic.
- `visualisation.py` – draws route visualisations.
- - `utils.py` – helpers for parsing datasets and computing route metrics.












