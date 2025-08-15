import numpy as np
import math
import csv
from itertools import product

# ==============================================================================
# new Solomon parser function for our experiment
# ==============================================================================
def read_solomon(path, num_customers):
    """
    Parses a Solomon format VRPTW file.
    It now includes time windows to create a CVRPTW instance.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Get vehicle info from line 5 (index 4)
    vehicle_line = lines[4].split()
    num_vehicles = int(vehicle_line[0])
    capacity = int(vehicle_line[1])

    # Get customer data, starting from line 10 (index 9)
    customer_data = []
    # We read depot (at index 9) + num_customers
    for i in range(9, 9 + num_customers + 1):
        line = lines[i].split()
        customer_data.append({
            'id': int(line[0]),
            'x': int(line[1]),
            'y': int(line[2]),
            'demand': int(line[3]),
            'ready_time': int(line[4]),
            'due_date': int(line[5]),
            'service_time': int(line[6])
        })
    
    num_nodes = len(customer_data)
    dests = list(range(1, num_nodes)) # Node 0 is the depot
    
    # Create cost matrix (Euclidean distance)
    costs = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            p1 = customer_data[i]
            p2 = customer_data[j]
            costs[i][j] = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
    
    # Extract weights/demands and time windows
    weights = np.zeros(num_nodes)
    time_windows = []
    service_times = np.zeros(num_nodes)
    for i in range(num_nodes):
        weights[i] = customer_data[i]['demand']
        time_windows.append((customer_data[i]['ready_time'], customer_data[i]['due_date']))
        service_times[i] = customer_data[i]['service_time']

    result = {
        'sources': [0], # Depot is node 0
        'dests': dests,
        'costs': costs,
        'time_costs': costs, # For this project, time_costs can be same as costs
        'weights': weights,
        'capacities': np.array([capacity] * num_vehicles),
        'time_windows': time_windows,
        'service_times': service_times
    }
    return result