#! .venv\Scripts\python.exe
# filepath: e:\Programmieren\Bachelor-Thesis\solver_eval.py

import os
import subprocess
import time
import glob
import pandas as pd
import sys

# Define path to CVC5 solver
CVC5_PATH = "C:\\SMT\\cvc5\\build\\bin\\cvc5.exe"

def run_z3_api(smt_file, timeout=300):
    """Run the SMT file using Z3 Python API."""
    start_time = time.time()
    
    try:
        from z3 import Solver, parse_smt2_file, set_param
        
        # Set timeout in milliseconds
        set_param("timeout", timeout * 1000)
        
        # Parse the SMT file
        print(f"Parsing SMT file with Z3: {os.path.basename(smt_file)}")
        formula = parse_smt2_file(smt_file)
        
        # Create solver and add the formula
        solver = Solver()
        solver.add(formula)
        
        # Check for satisfiability
        result = solver.check()
        status = str(result).lower()
        
        # Get model if satisfiable
        model = None
        if status == "sat":
            model = solver.model()
            model_str = str(model)
        else:
            model_str = None
        
        end_time = time.time()
        
        return {
            "solver": "Z3 API",
            "file": os.path.basename(smt_file),
            "status": status,
            "time": end_time - start_time,
            "model": model_str
        }
    except ImportError:
        print("Z3 Python API not available. Install with: pip install z3-solver")
        return {
            "solver": "Z3 API",
            "file": os.path.basename(smt_file),
            "status": "error",
            "time": 0,
            "model": None,
            "error": "Z3 Python API not available"
        }
    except Exception as e:
        end_time = time.time()
        return {
            "solver": "Z3 API",
            "file": os.path.basename(smt_file),
            "status": "error",
            "time": end_time - start_time,
            "model": None,
            "error": str(e)
        }

def run_cvc5(smt_file, timeout=300):
    """Run the SMT file using CVC5 executable."""
    start_time = time.time()
    
    try:
        # Check if CVC5 exists
        if not os.path.exists(CVC5_PATH):
            return {
                "solver": "CVC5",
                "file": os.path.basename(smt_file),
                "status": "error",
                "time": 0,
                "model": None,
                "error": f"CVC5 not found at {CVC5_PATH}"
            }
        
        # Run CVC5 with model production
        cmd = f'"{CVC5_PATH}" --produce-models "{smt_file}"'
        print(f"Executing: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        
        # Determine status
        output = result.stdout.lower()
        if "sat" in output and "unsat" not in output:
            status = "sat"
        elif "unsat" in output:
            status = "unsat"
        else:
            status = "unknown"
        
        # Extract model if satisfiable
        model = None
        if status == "sat":
            # Simple extraction of model from output
            model = result.stdout
        
        return {
            "solver": "CVC5",
            "file": os.path.basename(smt_file),
            "status": status,
            "time": end_time - start_time,
            "model": model,
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "solver": "CVC5",
            "file": os.path.basename(smt_file),
            "status": "timeout",
            "time": timeout,
            "model": None,
            "error": "Timeout"
        }
    except Exception as e:
        end_time = time.time()
        return {
            "solver": "CVC5",
            "file": os.path.basename(smt_file),
            "status": "error",
            "time": end_time - start_time,
            "model": None,
            "error": str(e)
        }

def save_model(solver, file, model):
    """Save model to a file."""
    if model:
        os.makedirs("models", exist_ok=True)
        filename = f"models/{os.path.basename(file)}_{solver}_model.txt"
        with open(filename, "w") as f:
            f.write(str(model))
        print(f"Model saved to {filename}")

def main():
    # Get all SMT files
    smt_files = glob.glob("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\*.smt")
    
    if not smt_files:
        print("No SMT files found!")
        return
    
    print(f"Found {len(smt_files)} SMT files to process")
    
    results = []
    
    # Process each file with both solvers
    for smt_file in smt_files:
        file_basename = os.path.basename(smt_file)
        print(f"\nProcessing {file_basename}...")
        
        # Try Z3 API
        print("Running with Z3 Python API...")
        z3_result = run_z3_api(smt_file)
        results.append(z3_result)
        print(f"Z3 API status: {z3_result['status']} in {z3_result['time']:.2f}s")
        if z3_result['status'] == 'sat' and z3_result['model']:
            print(f"Z3 found a solution!")
            save_model("Z3", file_basename, z3_result['model'])
        
        # Try CVC5
        print("Running with CVC5...")
        cvc5_result = run_cvc5(smt_file)
        results.append(cvc5_result)
        print(f"CVC5 status: {cvc5_result['status']} in {cvc5_result['time']:.2f}s")
        if cvc5_result['status'] == 'sat' and cvc5_result['model']:
            print(f"CVC5 found a solution!")
            save_model("CVC5", file_basename, cvc5_result['model'])
    
    # Create summary DataFrame
    df = pd.DataFrame([
        {
            "file": r["file"],
            "solver": r["solver"],
            "status": r["status"],
            "time": r["time"],
            "has_model": "Yes" if r.get("model") else "No"
        }
        for r in results
    ])
    
    # Save results
    df.to_csv("solver_comparison_results.csv", index=False)
    
    # Print summary table
    print("\nSummary:")
    summary = df.pivot_table(
        index="file",
        columns="solver",
        values=["status", "time", "has_model"]
    )
    print(summary)
    
    print("\nResults saved to solver_comparison_results.csv")
    print("Models saved in 'models' directory")

if __name__ == "__main__":
    main()