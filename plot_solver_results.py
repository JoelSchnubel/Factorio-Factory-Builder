#! .venv\Scripts\python.exe

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_plots_directory():
    """Create a directory for saving plots if it doesn't exist"""
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def load_results(csv_file):
    """Load the CSV file into a pandas DataFrame"""
    try:
        # Check if the file exists
        if not os.path.exists(csv_file):
            print(f"Error: CSV file '{csv_file}' not found.")
            return None
        
        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if the DataFrame is empty
        if df.empty:
            print("Error: CSV file is empty.")
            return None
            
        print(f"Loaded data from {csv_file} with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for plotting"""
    # Ensure execution time is numeric
    if 'Execution Time (s)' in df.columns:
        df['Execution Time (s)'] = pd.to_numeric(df['Execution Time (s)'], errors='coerce')
    
    # For the newer format where columns are like 'z3_time', 'cvc5_time', etc.
    # Create a melted version of the dataframe for easier plotting
    solver_columns = [col for col in df.columns if col.endswith('_time')]
    if solver_columns:
        solvers = [col.split('_')[0] for col in solver_columns]
        
        # Create a new DataFrame to store execution times and results
        result_df = pd.DataFrame()
        
        # Group by Logic
        for logic in df['Logic'].unique():
            logic_df = df[df['Logic'] == logic]
            
            row_data = {'Logic': logic}
            
            # Add execution time and result for each solver
            for solver in solvers:
                time_col = f"{solver}_time"
                result_col = f"{solver}_result"
                
                if time_col in df.columns and result_col in df.columns:
                    # Get the result and execution time
                    result = logic_df[result_col].iloc[0] if not logic_df.empty else 'N/A'
                    time = logic_df[time_col].iloc[0] if not logic_df.empty else None
                    
                    # Store both the result and time
                    row_data[f"{solver}_result"] = result
                    row_data[f"{solver}_time"] = time
            
            result_df = pd.concat([result_df, pd.DataFrame([row_data])], ignore_index=True)
        
        return result_df
    
    return df


def plot_execution_time_heatmap(df, plots_dir):
    """Plot a heatmap of execution times across solvers and logics with result types"""
    
    # Create a custom colormap for different result types
    colors = {
        'sat': '#90EE90',    
        'unsat': "#76D1EF",  
        'unknown': "#A72B2B", 
        'error': '#A72B2B',  
        'N/A': '#A72B2B'     
    }
    
    # Determine the format of the DataFrame
    if 'Solver_Column' in df.columns:
        # Melted DataFrame with Result column
        time_pivot = df.pivot_table(
            values='Execution Time (s)', 
            index='Logic', 
            columns='Solver', 
            aggfunc='mean'
        )
        result_pivot = df.pivot_table(
            values='Result',
            index='Logic',
            columns='Solver',
            aggfunc=lambda x: x.iloc[0] if len(x) > 0 else 'N/A'
        )
    elif any(col.endswith('_time') for col in df.columns):
        # DataFrame with solver_time columns
        time_columns = [col for col in df.columns if col.endswith('_time')]
        result_columns = [col for col in df.columns if col.endswith('_result')]
        solvers = [col.split('_')[0] for col in time_columns]
        
        # Create pivot tables
        time_pivot = pd.DataFrame(index=df['Logic'].unique())
        result_pivot = pd.DataFrame(index=df['Logic'].unique())
        
        for solver, time_col in zip(solvers, time_columns):
            result_col = f"{solver}_result"
            # Group by Logic and get mean execution time
            time_data = df.groupby('Logic')[time_col].mean()
            time_pivot[solver] = time_data
            
            # Get result for each Logic/Solver combination
            if result_col in df.columns:
                result_data = df.groupby('Logic')[result_col].first()
                result_pivot[solver] = result_data
            else:
                result_pivot[solver] = 'N/A'
    else:
        # Original DataFrame format
        time_pivot = df.pivot_table(
            values='Execution Time (s)', 
            index='Logic', 
            columns='Solver', 
            aggfunc='mean'
        )
        result_pivot = df.pivot_table(
            values='Result',
            index='Logic',
            columns='Solver',
            aggfunc=lambda x: x.iloc[0] if len(x) > 0 else 'N/A'
        )
    
    # Create a mask for non-sat results
    mask = ~(result_pivot.fillna('N/A').applymap(lambda x: x.lower() == 'sat' if isinstance(x, str) else False))
    
    # Create a function to format annotations
    def format_annotation(val, result):
        if pd.isna(val) or pd.isna(result):
            return "None"
        elif result.lower() == 'sat':
            return f"{val:.2f}"
        elif result.lower() == 'unsat':
            return "UNSAT"
        elif result.lower() == 'unknown':
            return "None"
        elif result.lower() == 'error':
            return "None"
        else:
            return "None"
    
    # Create annotations with both time and result
    annotations = np.empty_like(time_pivot, dtype=object)
    for i in range(time_pivot.shape[0]):
        for j in range(time_pivot.shape[1]):
            time_val = time_pivot.iloc[i, j]
            result_val = result_pivot.iloc[i, j] if not result_pivot.empty else 'N/A'
            annotations[i, j] = format_annotation(time_val, result_val)
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        time_pivot,
        annot=annotations,
        fmt="",
        cmap='YlGnBu',
        linewidths=0.5,
        mask=mask,  # Apply mask for non-sat results
        cbar_kws={'label': 'Execution Time (s) for SAT results'}
    )
    
    # Add colored cells for non-sat results
    for i in range(time_pivot.shape[0]):
        for j in range(time_pivot.shape[1]):
            result = result_pivot.iloc[i, j] if not result_pivot.empty else 'N/A'
            result_str = str(result).lower() if not pd.isna(result) else 'n/a'
            
            # Get position of cell center
            x, y = j + 0.5, i + 0.5
            
            # Add colored rectangle for all cells based on result
            color = colors.get(result_str, colors['N/A'])
            if result_str != 'sat':
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.7))
            else:
                continue
            # Add text annotation
            ax.text(x, y, annotations[i, j], 
                   horizontalalignment='center', 
                   verticalalignment='center',
                   fontweight='bold')
    
    plt.title('Solver Results by Logic')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, 'solver_results_heatmap.png'), bbox_inches='tight')
    plt.close()
    


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_solver_results.py <csv_file>")
        print("Example: python plot_solver_results.py results/solver_evaluation.csv")
        return
    
    csv_file = sys.argv[1]
    
    # Load the data
    df = load_results(csv_file)
    if df is None:
        return
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create plots directory
    plots_dir = create_plots_directory()
    print(f"Saving plots to {plots_dir}")
    
    # Create plots
    plot_execution_time_heatmap(df, plots_dir)

    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
