#! .venv\Scripts\python.exe

#! .venv\Scripts\python.exe
# filepath: e:\Programmieren\Bachelor-Thesis\solver_eval.py

import os
import subprocess
import time
import glob
import pandas as pd
import re

# Define paths to solvers - update these paths to where you installed them
SOLVER_PATHS = {
    "z3": "C:\\SMT\\z3\\bin\\z3.exe",
    "cvc5": "C:\\SMT\\cvc5\\bin\\cvc5.exe",
    "yices-smt2": "C:\\SMT\\yices\\bin\\yices-smt2.exe",
    "boolector": "C:\\SMT\\boolector\\bin\\boolector.exe"
}

def run_solver(solver_name, smt_file, timeout=300):
    """Run a solver on an SMT file and return statistics and model if SAT."""
    start_time = time.time()
    
    # Get the command for the solver with options to produce models
    solver_path = SOLVER_PATHS.get(solver_name, solver_name)
    
    # Configure solver-specific commands to get models/solutions
    if solver_name == "z3":
        cmd = f'"{solver_path}" -model "{smt_file}"'
    elif solver_name == "cvc5":
        cmd = f'"{solver_path}" --produce-models "{smt_file}"'
    elif solver_name == "yices-smt2":
        cmd = f'"{solver_path}" -m "{smt_file}"'
    elif solver_name == "boolector":
        cmd = f'"{solver_path}" -m "{smt_file}"'
    else:
        cmd = f'"{solver_path}" "{smt_file}"'
    
    try:
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
        if "sat" in result.stdout.lower() and "unsat" not in result.stdout.lower():
            status = "sat"
        elif "unsat" in result.stdout.lower():
            status = "unsat"
        else:
            status = "unknown"
        
        # Extract model if sat
        model = None
        if status == "sat":
            model = extract_model(solver_name, result.stdout)
        
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "status": status,
            "time": end_time - start_time,
            "model": model,
            "stdout": result.stdout[:500] if len(result.stdout) > 500 else result.stdout,  # Truncate long outputs
            "stderr": result.stderr[:500] if len(result.stderr) > 500 else result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "status": "timeout",
            "time": timeout,
            "model": None,
            "stdout": "",
            "stderr": "Timeout"
        }
    except Exception as e:
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "status": "error",
            "time": 0,
            "model": None,
            "stdout": "",
            "stderr": str(e)
        }

def extract_model(solver_name, output):
    """Extract model information from solver output."""
    if solver_name == "z3":
        # Z3 model format
        if "(model" in output:
            model_section = output.split("(model")[1]
            if ")" in model_section:
                model_section = model_section.split(")", 1)[0] + ")"
                return "(model" + model_section
    elif solver_name == "cvc5":
        # CVC5 model format
        model_lines = []
        capturing = False
        for line in output.splitlines():
            if "MODEL" in line:
                capturing = True
                continue
            if capturing:
                if line.strip() == "":
                    break
                model_lines.append(line)
        if model_lines:
            return "\n".join(model_lines)
    elif solver_name == "yices-smt2":
        # Yices model format
        model_lines = []
        for line in output.splitlines():
            if line.startswith("(="):
                model_lines.append(line)
        if model_lines:
            return "\n".join(model_lines)
    elif solver_name == "boolector":
        # Boolector model format
        if "sat" in output.lower():
            model_section = output.split("sat")[1].strip()
            return model_section
    
    # Return original output if no specific extraction pattern matches
    return output

def check_solver_installation():
    """Check if solvers are properly installed and accessible."""
    available_solvers = []
    
    for solver_name, path in SOLVER_PATHS.items():
        try:
            # Try running with --version or similar flag
            if solver_name == "z3":
                cmd = f'"{path}" --version'
            elif solver_name == "cvc5":
                cmd = f'"{path}" --version'
            elif solver_name == "yices-smt2":
                cmd = f'"{path}" --version'
            elif solver_name == "boolector":
                cmd = f'"{path}" -v'
            
            print(f"Checking {solver_name} installation...")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✓ {solver_name} is installed and working ({path})")
                version_info = result.stdout.strip() or result.stderr.strip()
                print(f"  Version: {version_info}")
                available_solvers.append(solver_name)
            else:
                print(f"✗ {solver_name} returned error code {result.returncode}")
                print(f"  Error: {result.stderr}")
        except FileNotFoundError:
            print(f"✗ {solver_name} not found at {path}")
        except Exception as e:
            print(f"✗ {solver_name} check failed: {e}")
    
    return available_solvers

def save_models(results):
    """Save models to separate files for analysis."""
    os.makedirs("models", exist_ok=True)
    
    for idx, result in enumerate(results):
        if result.get("model") and result["status"] == "sat":
            filename = f"models/{result['file']}_{result['solver']}_model.txt"
            with open(filename, "w") as f:
                f.write(str(result["model"]))
            print(f"Model for {result['file']} using {result['solver']} saved to {filename}")

def main():
    # Check which solvers are available
    print("Checking solver installations...")
    available_solvers = check_solver_installation()
    
    if not available_solvers:
        print("No SMT solvers found! Please install at least one solver.")
        return
        
    print(f"Found {len(available_solvers)} available solvers: {', '.join(available_solvers)}")
    
    # Get all SMT files
    smt_files = glob.glob("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\*.smt")
    if not smt_files:
        print("No SMT files found!")
        return
        
    print(f"Found {len(smt_files)} SMT files to process")
    
    results = []
    
    # Run each solver on each file
    for smt_file in smt_files:
        print(f"\nProcessing {os.path.basename(smt_file)}...")
        for solver in available_solvers:
            print(f"  Running {solver}...")
            result = run_solver(solver, smt_file)
            results.append(result)
            print(f"  {solver} finished with status: {result['status']} in {result['time']:.2f}s")
            
            if result['status'] == 'sat':
                print(f"  Solution found! First few values: {str(result['model'])[:100]}...")
            elif result['status'] == 'error':
                print(f"  Error: {result['stderr']}")
    
    # Save all models to files
    save_models(results)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Save results
    df[['solver', 'file', 'status', 'time']].to_csv("solver_comparison_results.csv", index=False)
    
    # Print summary
    print("\nSummary:")
    try:
        summary = df.pivot_table(
            index="file", 
            columns="solver", 
            values=["status", "time"],
            aggfunc={"status": lambda x: x.iloc[0] if len(x) > 0 else "N/A", "time": "mean"}
        )
        print(summary)
    except Exception as e:
        print(f"Could not generate summary table: {e}")
        print(df[['solver', 'file', 'status', 'time']])
    
    print("\nResults saved to solver_comparison_results.csv")
    print("Models saved in 'models' directory")

if __name__ == "__main__":
    main()