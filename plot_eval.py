#! .venv\Scripts\python.exe

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(file_path="solver_logic_comparison_results.csv"):
    """Load results from the CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: Results file {file_path} not found")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} result rows with {df['file'].nunique()} unique SMT files")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def create_simple_plots(df, output_dir="Plots"):
    """
    Create a simple visualization for each SMT file in the results.
    Each file gets its own plot showing solver performance across logics.
    """
    if df is None or df.empty:
        print("No data to plot")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a clean style
    sns.set_style("whitegrid")
    
    # Group by file name
    for file_name in df['file'].unique():
        # Get data for this file only
        file_df = df[df['file'] == file_name]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create a pivot table for better visualization
        pivot_df = file_df.pivot(index='logic', columns='solver', values='time')
        
        # Plot heatmap with solver times
        ax = sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlGnBu', 
                         linewidths=0.5, cbar_kws={'label': 'Time (seconds)'})
        
        # Add solver status as text annotations
        for i, logic in enumerate(pivot_df.index):
            for j, solver in enumerate(pivot_df.columns):
                cell_data = file_df[(file_df['logic'] == logic) & (file_df['solver'] == solver)]
                if not cell_data.empty:
                    status = cell_data['status'].iloc[0]
                    # Position the status text in the cell
                    ax.text(j + 0.5, i + 0.8, status, 
                            ha='center', va='center', fontsize=8,
                            color='black' if status == 'sat' else 'red')
        
        # Set title and labels
        plt.title(f'Solver Performance: {file_name}', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(output_dir, f"solver_comparison_{os.path.splitext(file_name)[0]}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Created plot for {file_name}")
        
    return True

def main():
    # Check if a custom results file was provided
    results_file = "solver_logic_comparison_results.csv"
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    output_dir = "Plots"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Load results
    df = load_results(results_file)
    
    if df is not None:
        # Create plots 
        plots_success = create_simple_plots(df, output_dir)
        
    else:
        print("\nVisualization failed. Please check the errors above.")

if __name__ == "__main__":
    import sys
    main()
