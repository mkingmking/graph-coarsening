import matplotlib.pyplot as plt
from .graph import Graph
import matplotlib.cm as cm
import numpy as np

def visualize_routes(graph: Graph, routes: list, title: str, depot_id: str):
    """
    Visualizes the routes on a 2D plot.

    Args:
        graph (Graph): The graph object containing all nodes and their coordinates.
        routes (list): A list of lists of node IDs, where each inner list is a route.
        title (str): The title for the plot.
        depot_id (str): The ID of the depot node.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=16)

    # Extract all node coordinates for plotting
    all_x = [node.x for node in graph.nodes.values()]
    all_y = [node.y for node in graph.nodes.values()]

    # Plot all customer nodes
    plt.scatter(all_x, all_y, color='gray', s=50, zorder=1)

    # Plot the depot node
    if depot_id in graph.nodes:
        depot_node = graph.nodes[depot_id]
        plt.scatter(depot_node.x, depot_node.y, color='black', marker='s', s=100, label='Depot', zorder=2)
        plt.text(depot_node.x, depot_node.y + 2, 'Depot', fontsize=12, ha='center')
    else:
        print(f"Warning: Depot node with ID '{depot_id}' not found for visualization.")

    
    colors = cm.Reds(np.linspace(0.4, 1, len(routes)))

    # Plot each route
    for i, route in enumerate(routes):
        route_nodes = [graph.nodes[node_id] for node_id in route if node_id in graph.nodes]
        if len(route_nodes) > 1:
            route_x = [node.x for node in route_nodes]
            route_y = [node.y for node in route_nodes]
            
            # Plot the route lines
            plt.plot(route_x, route_y, color=colors[i], linestyle='-', linewidth=2, zorder=0, label=f'Route {i+1}')
            
            # Add labels to customer nodes
            for node in route_nodes:
                if node.id != depot_id:
                    plt.text(node.x, node.y + 2, f'{node.id}', fontsize=8, ha='center')
    
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
