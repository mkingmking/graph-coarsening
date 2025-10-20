import os
import matplotlib.pyplot as plt
import logging


from ..utils import load_graph_from_csv
from ..graph import Graph
from ..node import Node
from ..edge import Edge

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def visualize_dataset(file_path: str):
    """
    Visualizes the nodes of a Solomon dataset.
    The depot is plotted as a star, and customers as dots.
    
    Args:
        file_path (str): The path to the Solomon VRPTW CSV file.
    """
    try:
        # Load the graph using the existing function
        graph, depot_id, _ = load_graph_from_csv(file_path)
        logger.info(f"Loaded graph from {os.path.basename(file_path)} with {len(graph.nodes)} nodes.")

        # Separate nodes into depot and customers
        depot_node = graph.nodes[depot_id]
        customer_nodes = [node for node in graph.nodes.values() if node.id != depot_id]

        # Extract coordinates
        depot_x = depot_node.x
        depot_y = depot_node.y
        
        customer_x = [node.x for node in customer_nodes]
        customer_y = [node.y for node in customer_nodes]

        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot customer nodes as blue dots
        plt.scatter(customer_x, customer_y, c='blue', marker='o', label='Customers')
        
        # Plot the depot as a red star
        plt.scatter(depot_x, depot_y, c='red', marker='*', s=300, label='Depot', zorder=5)

        # Set plot title and labels
        plt.title(f'Node Locations for {os.path.basename(file_path)}')
        plt.xlabel('X-Coordinate')
        plt.ylabel('Y-Coordinate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal') # Ensure the scale is the same on both axes
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        logger.error(f"Error: The specified file was not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while visualizing the dataset: {e}")

if __name__ == "__main__":
    # --- Example Usage ---
    # Replace 'solomon_dataset/C101.csv' with the actual path to your file.
    # You can also set up a loop to visualize all datasets.
    
    # Path to the dataset folder
    dataset_folder = 'graph_coarsening/solomon_dataset/C2'
    
  
    file_to_visualize = 'C201.csv'
    
    file_path = os.path.join(dataset_folder, file_to_visualize)
    visualize_dataset(file_path)
