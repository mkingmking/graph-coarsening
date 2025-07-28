This repository contains the implementation and evaluation of a spatio-temporal graph coarsening algorithm designed for solving the Vehicle Routing Problem with Time Windows (VRPTW).

Repository Contents:

solomon_dataset/: Directory containing the Solomon VRPTW benchmark datasets in CSV format (e.g., ``C101.csv`` or ``R101.csv``).  The files list the depot and customers with coordinates, demands and time windows.  Vehicle capacity is not stored in the files, so the loaders assume the standard Solomon capacity of ``200``.

graph_coarsening.py: This Python script implements the core spatio-temporal graph coarsening algorithm, including node and edge definitions, graph operations, and the multilevel coarsening and inflation procedures. It also integrates Greedy and Clarke and Wright Savings heuristics for solving VRPTW instances on both original and coarsened graphs.

outputs/: Directory containing experiment logs and the evaluation report PDF generated during experiments.

How to Use:

Clone the repository:


git clone https://github.com/mkingmking/graph-coarsening

cd graph-coarsening

Ensure Python 3 is installed.

Run the main script:



python3 main.py

The script will automatically process the CSV files in the solomon_dataset/ directory, apply the coarsening algorithm, solve the VRPTW using the implemented heuristics, and print performance metrics to the console.


