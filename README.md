This repository contains the implementation and evaluation of a spatio-temporal graph coarsening algorithm designed for solving the Vehicle Routing Problem with Time Windows (VRPTW).

Repository Contents:

solomon_dataset/: This directory holds the Solomon VRPTW benchmark datasets in CSV format (e.g., C101.csv, R101.csv). These files contain customer coordinates, demands, time windows, and service times, along with vehicle capacity information.

graph_coarsening.py: This Python script implements the core spatio-temporal graph coarsening algorithm, including node and edge definitions, graph operations, and the multilevel coarsening and inflation procedures. It also integrates Greedy and Clarke and Wright Savings heuristics for solving VRPTW instances on both original and coarsened graphs.

evaluation_report.pdf: This PDF document (generated from the LaTeX source in the Canvas) provides a detailed analysis and comparison of the solver performances with and without graph coarsening, discussing key findings, limitations, and recommendations for future work.

How to Use:

Clone the repository:

Bash

git clone <repository-url>
cd <repository-name>
Ensure Python 3 is installed.

Run the main script:

Bash

python3 graph_coarsening.py
The script will automatically process the CSV files in the solomon_dataset/ directory, apply the coarsening algorithm, solve the VRPTW using the implemented heuristics, and print performance metrics to the console.

Review the evaluation report: Open evaluation_report.pdf to understand the detailed findings and analysis of the algorithm's performance.
