import matplotlib.pyplot as plt 
import numpy as np 
import os
from .graph import Graph

def visualize_routes(graph: Graph, routes: list, depot_id: str, title: str = "VRPTW Solution", filename: str = None):
    """
    Visualizes the given routes on the graph and saves the figure to disk.

    Args:
        graph (Graph): The graph object containing node coordinates.
        routes (list): A list of lists of node IDs, where each inner list is a route.
        depot_id (str): The ID of the depot node.
        title (str): The title for the plot.
        filename (str): Optional custom filename for the saved figure (without extension).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 14))

    # Create output directory if it doesn't exist
    output_dir = "visualisation_routes"
    os.makedirs(output_dir, exist_ok=True)

    # Get coordinates for all nodes
    node_coords_map = {node_id: (node.x, node.y) for node_id, node in graph.nodes.items()}
    
    # Plot customers (excluding depot)
    customer_ids = [node_id for node_id in graph.nodes.keys() if node_id != depot_id]
    customer_x = [node_coords_map[cid][0] for cid in customer_ids]
    customer_y = [node_coords_map[cid][1] for cid in customer_ids]
    ax.scatter(customer_x, customer_y, c='silver', label='Customers', s=50, zorder=3)

    # Plot depot
    depot_x, depot_y = node_coords_map[depot_id]
    ax.scatter(depot_x, depot_y, c='red', marker='*', s=300, label='Depot', zorder=5)

    # Plot routes
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, max(1, len(routes))))
    for i, route in enumerate(routes):
        route_color = colors[i]
        if len(route) < 2:
            continue
        route_x = [node_coords_map[node_id][0] for node_id in route]
        route_y = [node_coords_map[node_id][1] for node_id in route]
        ax.plot(route_x, route_y, color=route_color, linewidth=2, alpha=0.8)
        for j in range(len(route) - 1):
            u_id, v_id = route[j], route[j+1]
            u_x, u_y = node_coords_map[u_id]
            v_x, v_y = node_coords_map[v_id]
            ax.arrow(u_x, u_y, (v_x - u_x)*0.8, (v_y - u_y)*0.8,
                     head_width=1.5, head_length=1.5, fc=route_color, ec=route_color, alpha=0.7, zorder=4)
        ax.plot([], [], color=route_color, label=f'Vehicle {i+1}')
    
    # Add node labels
    for node_id, (x, y) in node_coords_map.items():
        ax.text(x, y + 1.5, node_id, fontsize=9, ha='center', weight='bold')

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)

    # Generate filename if not given
    if filename is None:
        sanitized_title = title.lower().replace(" ", "_")
        filename = f"{sanitized_title}.png"

    # Full path to save
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
