import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_boxplots(df: pd.DataFrame, hyperparameter: str, metric: str, metric_label: str):
    """
    Generates a box plot for a given metric and hyperparameter.
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x=hyperparameter,
        y=metric,
        data=df
    )
    
    # Add labels and a title
    plt.title(f'Performance Distribution by {hyperparameter.capitalize()}')
    plt.xlabel(f'{hyperparameter.capitalize()} Value')
    plt.ylabel(metric_label)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.show()

def main():
    """
    Main function to load data and generate plots.
    """
    results_file = "results.json"
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            logger.error("The results file is empty. Please run main.py first to generate data.")
            return
            
        df = pd.DataFrame(data)
        logger.info(f"Successfully loaded {len(df)} experiment results.")
        
    except FileNotFoundError:
        logger.error(f"Error: Results file '{results_file}' not found. Please run tuning_classical_solvers first.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from '{results_file}'. The file may be corrupted.")
        return
        
    # Generate a separate plot for each hyperparameter
    
    # Plot for alpha
    create_boxplots(
        df=df,
        hyperparameter='alpha',
        metric='inflated_total_distance',
        metric_label='Total Inflated Distance'
    )
    
    # Plot for beta
    create_boxplots(
        df=df,
        hyperparameter='beta',
        metric='inflated_total_distance',
        metric_label='Total Inflated Distance'
    )

    # Plot for P
    create_boxplots(
        df=df,
        hyperparameter='P',
        metric='inflated_total_distance',
        metric_label='Total Inflated Distance'
    )

    # Plot for radiusCoeff
    create_boxplots(
        df=df,
        hyperparameter='radiusCoeff',
        metric='inflated_total_distance',
        metric_label='Total Inflated Distance'
    )
    
    logger.info("Box plots generated. You can modify the 'metric' parameter in the script to visualize other metrics.")

if __name__ == "__main__":
    main()
