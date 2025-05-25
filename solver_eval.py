#! .venv\Scripts\python.exe

import os
import subprocess
import time
import glob
import pandas as pd
import sys
import re


CVC5_PATH = "C:\\SMT\\cvc5\\cvc5-Win64-x86_64-static\\bin\\cvc5.exe"
YICES_PATH = "C:\\SMT\\yices\\yices-2.6.5\\bin\\yices-smt.exe"

# Different logics to test
SMT_LOGICS = [
    "QF_LIA",   
    "QF_LRA",  
    "QF_UFLIA", 
    "QF_IDL",   
    "QF_RDL",   
    "QF_UF",   
    "QF_BV",   
    "QF_AUFBV",
]

def run_z3_api(smt_file, logic=None, timeout=300):
    """Run the SMT file using Z3 Python API with specific logic."""
    start_time = time.time()
    
    try:
        from z3 import Solver, parse_smt2_file, set_param
        
        # Set timeout in milliseconds
        set_param("timeout", timeout * 1000)
        
        # Parse the SMT file
        print(f"Parsing SMT file with Z3 API: {os.path.basename(smt_file)}")
        
        # Create a temporary file with the correct logic if needed
        if logic:
            temp_file = create_smt2_file_with_logic(smt_file, logic)
            formula = parse_smt2_file(temp_file)
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
        else:
            formula = parse_smt2_file(smt_file)
        
        # Create solver with specified logic if provided
        solver = Solver()
        if logic:
            solver.set(logic=logic)
        
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
        
        solver_name = f"Z3 API" if not logic else f"Z3 API ({logic})"
        
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "logic": logic if logic else "default",
            "status": status,
            "time": end_time - start_time,
            "model": model_str
        }
    except ImportError:
        print("Z3 Python API not available. Install with: pip install z3-solver")
        return {
            "solver": f"Z3 API" if not logic else f"Z3 API ({logic})",
            "file": os.path.basename(smt_file),
            "logic": logic if logic else "default",
            "status": "error",
            "time": 0,
            "model": None,
            "error": "Z3 Python API not available"
        }
    except Exception as e:
        end_time = time.time()
        return {
            "solver": f"Z3 API" if not logic else f"Z3 API ({logic})",
            "file": os.path.basename(smt_file),
            "logic": logic if logic else "default",
            "status": "error",
            "time": end_time - start_time,
            "model": None,
            "error": str(e)
        }

def create_smt2_file_with_logic(smt_file, logic):
    """Create a temporary SMT2 file with the specific logic."""
    temp_file = os.path.join(os.path.dirname(smt_file), f"temp_{logic}_{os.path.basename(smt_file)}")
    temp_file = temp_file.replace('.smt', '.smt2')
    
    with open(smt_file, 'r') as f:
        content = f.read()
    
    # Format the content with proper header and footer
    formatted_content = []
    
    # Add logic declaration
    formatted_content.append(f"(set-logic {logic})")
    
    # Add model production option
    formatted_content.append("(set-option :produce-models true)")
    
    # Handle Z3-specific optimization directives
    has_optimization = False
    optimize_vars = set()
    optimization_weights = {}
    inside_optimization = False
    optimization_value = []
    collecting_optimization = False
    
    # Process content line by line
    for line in content.splitlines():
        line_stripped = line.strip()
        
        # Skip Z3 logic and option declarations
        if line_stripped.startswith("(set-logic") or line_stripped.startswith("(set-option"):
            continue
            
        # Handle maximize/minimize directives (Z3-specific)
        if line_stripped.startswith("(maximize") or line_stripped.startswith("(minimize"):
            has_optimization = True
            inside_optimization = True
            collecting_optimization = True
            optimization_value.append(line_stripped)
            continue
        
        # Collect rest of optimization expression across multiple lines
        if inside_optimization:
            optimization_value.append(line_stripped)
            # Check if this line completes the optimization directive
            if line_stripped.count(")") > line_stripped.count("("):
                inside_optimization = False
                
                # Extract variables from the optimization expression
                opt_expr = "".join(optimization_value)
                # Extract variables used in ite expressions (indicators for optimization)
                for match in re.finditer(r'\(ite\s+(\S+)\s+1\s+0\)', opt_expr):
                    var_name = match.group(1)
                    optimize_vars.add(var_name)
                    # This variable needs to be true for optimization
                    optimization_weights[var_name] = 1
                
                # Clear for next potential optimization directive
                optimization_value = []
                collecting_optimization = False
            continue
        
        # Add other content
        formatted_content.append(line)
    
    # Convert optimization directives to SMT-LIB2 compatible assertions
    if has_optimization:
        for var_name in optimize_vars:
            # Add soft assertions that can help achieve the desired optimization
            # but as regular assertions, not as optimization directives
            formatted_content.append(f"(assert (= {var_name} true))")
    
    # Fix ADD operations with only one child
    fixed_content = "\n".join(formatted_content)
    fixed_content = re.sub(r'\(\+\s+([^\s\(\)]+)\)', r'(+ 0 \1)', fixed_content)
    
    # Fix other common issues that might cause problems with different solvers
    # Convert empty disjunctions to 'false'
    fixed_content = re.sub(r'\(or\s*\)', r'false', fixed_content)
    # Convert empty conjunctions to 'true' 
    fixed_content = re.sub(r'\(and\s*\)', r'true', fixed_content)
    
    # Make sure check-sat and get-model are present at the end
    if "(check-sat)" not in fixed_content:
        fixed_content += "\n(check-sat)"
    if "(get-model)" not in fixed_content:
        fixed_content += "\n(get-model)"
    
    # Write the formatted content
    with open(temp_file, 'w') as f:
        f.write(fixed_content)
        
    print(f"Created compatible SMT-LIB2 file at {temp_file} for logic {logic}")
    return temp_file
    
    return temp_file

def run_cvc5(smt_file, logic=None, timeout=300):
    """Run the SMT file using CVC5 executable with specific logic."""
    start_time = time.time()
    
    try:
        # Check if CVC5 exists
        if not os.path.exists(CVC5_PATH):
            return create_error_result("CVC5", smt_file, logic, f"CVC5 not found at {CVC5_PATH}")
        
        # Create a temporary SMT2 file with the proper logic
        temp_file = create_smt2_file_with_logic(smt_file, logic if logic else "QF_LIA")
        
        # Build command with proper arguments
        cmd = [
            CVC5_PATH, 
            "--lang=smt2"  # Specify SMT-LIB v2 format
        ]
        
        # Add file at the end
        cmd.append(temp_file)
        
        cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
        print(f"Executing: {cmd_str}")
        
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass
          # Determine status
        output = result.stdout.lower()
        if "sat" in output and "unsat" not in output:
            status = "sat"
            print("CVC5 found a solution!")
        elif "unsat" in output:
            status = "unsat"
        else:
            status = "unknown"
            
            # Print error messages for debugging
            if result.stderr:
                print(f"CVC5 stderr: {result.stderr}")
                
            # Print more diagnostic information
            print(f"CVC5 stdout: {result.stdout[:500]}..." if len(result.stdout) > 500 else result.stdout)
            print(f"File size: {os.path.getsize(temp_file)} bytes")
            print(f"Command used: {cmd_str}")
        
        # Extract model if satisfiable
        model = None
        if status == "sat":
            # Get everything after "sat" up to the end
            model_text = result.stdout[result.stdout.find("sat") + 3:].strip()
            if model_text:
                model = model_text
        
        solver_name = f"CVC5" if not logic else f"CVC5 ({logic})"
        
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "logic": logic if logic else "default",
            "status": status,
            "time": end_time - start_time,
            "model": model,
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.TimeoutExpired:
        try:
            os.remove(temp_file)
        except:
            pass
        return create_timeout_result("CVC5", smt_file, logic, timeout)
    except Exception as e:
        end_time = time.time()
        try:
            os.remove(temp_file)
        except:
            pass
        return create_error_result("CVC5", smt_file, logic, str(e), end_time - start_time)

def run_yices(smt_file, logic=None, timeout=300):
    """Run the SMT file using Yices executable with specific logic."""
    start_time = time.time()
    
    try:
        # Check if Yices exists
        if not os.path.exists(YICES_PATH):
            return create_error_result("Yices", smt_file, logic, f"Yices not found at {YICES_PATH}")
        
        # Check for yices-smt2 binary (better for SMT-LIB v2)
        yices_smt2_path = YICES_PATH.replace("yices-smt.exe", "yices-smt2.exe")
        if os.path.exists(yices_smt2_path):
            print(f"Using Yices SMT2 binary: {yices_smt2_path}")
            yices_path = yices_smt2_path
        else:
            yices_path = YICES_PATH
        
        # Create a temporary SMT2 file with the proper logic
        temp_file = create_smt2_file_with_logic(smt_file, logic if logic else "ALL")
        
        # Build command
        cmd = [yices_path]
        
        # Add file at the end
        cmd.append(temp_file)
        
        cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
        print(f"Executing: {cmd_str}")
        
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass
          # Determine status
        output = result.stdout.lower() + result.stderr.lower()
        if "sat" in output and "unsat" not in output:
            status = "sat"
            print("Yices found a solution!")
        elif "unsat" in output:
            status = "unsat"
        else:
            status = "unknown"
            
            # Print error for debugging
            if result.stderr:
                print(f"Yices error: {result.stderr}")
                
            # Print more diagnostic information
            print(f"Yices stdout: {result.stdout[:500]}..." if len(result.stdout) > 500 else result.stdout)
            print(f"File size: {os.path.getsize(temp_file)} bytes")
            print(f"Command used: {cmd_str}")
            
            # Check if the error is due to syntax issues
            if "syntax error" in output or "parse error" in output:
                print("Yices reported syntax errors. Attempting basic SMT-LIB2 compatibility fixes...")
        
        # Extract model if satisfiable
        model = None
        if status == "sat":
            model = extract_yices_model(result.stdout)
        
        solver_name = f"Yices" if not logic else f"Yices ({logic})"
        
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "logic": logic if logic else "default",
            "status": status,
            "time": end_time - start_time,
            "model": model,
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.TimeoutExpired:
        try:
            os.remove(temp_file)
        except:
            pass
        return create_timeout_result("Yices", smt_file, logic, timeout)
    except Exception as e:
        end_time = time.time()
        try:
            os.remove(temp_file)
        except:
            pass
        return create_error_result("Yices", smt_file, logic, str(e), end_time - start_time)

def extract_cvc5_model(output):
    """Extract model information from CVC5 output."""
    model_lines = []
    capturing = False
    
    for line in output.splitlines():
        if line.strip() == "":
            continue
        
        if "sat" in line.lower():
            capturing = True
            continue
        
        if capturing:
            model_lines.append(line)
    
    return "\n".join(model_lines) if model_lines else output

def extract_yices_model(output):
    """Extract model information from Yices output."""
    model_lines = []
    for line in output.splitlines():
        if line.strip() == "":
            continue
        
        if line.startswith("(=") or line.startswith("(define"):
            model_lines.append(line)
    
    return "\n".join(model_lines) if model_lines else output

def save_model(solver, file, logic, model):
    """Save model to a file."""
    if model:
        os.makedirs("models", exist_ok=True)
        logic_str = logic.replace("/", "_")
        filename = f"models/{file}_{solver}_{logic_str}_model.txt"
        with open(filename, "w") as f:
            f.write(str(model))
        print(f"Model saved to {filename}")

def create_error_result(solver, smt_file, logic, error, time=0):
    """Create an error result object."""
    solver_name = f"{solver}" if not logic else f"{solver} ({logic})"
    return {
        "solver": solver_name,
        "file": os.path.basename(smt_file),
        "logic": logic if logic else "default",
        "status": "error",
        "time": time,
        "model": None,
        "error": error
    }

def create_timeout_result(solver, smt_file, logic, timeout):
    """Create a timeout result object."""
    solver_name = f"{solver}" if not logic else f"{solver} ({logic})"
    return {
        "solver": solver_name,
        "file": os.path.basename(smt_file),
        "logic": logic if logic else "default",
        "status": "timeout",
        "time": timeout,
        "model": None,
        "error": "Timeout"
    }

def main():
    """Main function to evaluate SMT solvers on files."""
    # Import SMT simplifier for challenging files
    try:
        from smt_simplifier import simplify_smt_file, fix_common_syntax_issues
        HAS_SIMPLIFIER = True
    except ImportError:
        print("SMT simplifier module not found. Some compatibility fixes won't be available.")
        HAS_SIMPLIFIER = False
    
    # Get all SMT files (both .smt and .smt2)
    smt_files = glob.glob("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\*.smt")
    smt_files.extend(glob.glob("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\*.smt2"))
    
    if not smt_files:
        print("No SMT files found!")
        return
    
    print(f"Found {len(smt_files)} SMT files to process")
    os.makedirs("models", exist_ok=True)
    
    # Create directory for simplified versions of problematic files
    os.makedirs("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\simplified", exist_ok=True)
    
    results = []
      # Process each file with all solvers and logics
    for smt_file in smt_files:
        file_basename = os.path.basename(smt_file)
        print(f"\n{'='*60}")
        print(f"Processing {file_basename}...")
        print(f"{'='*60}")
        
        # Run with Z3 API (default logic)
        print("\nRunning with Z3 Python API (default logic)...")
        z3_result = run_z3_api(smt_file)
        results.append(z3_result)
        print(f"Z3 API status: {z3_result['status']} in {z3_result['time']:.2f}s")
        
        if z3_result['status'] == 'sat' and z3_result['model']:
            print(f"Z3 found a solution!")
            save_model("Z3", file_basename, "default", z3_result['model'])
            has_z3_solution = True
        else:
            has_z3_solution = False
        
        # Run with CVC5 (default logic)
        print("\nRunning with CVC5 (default logic)...")
        cvc5_result = run_cvc5(smt_file)
        results.append(cvc5_result)
        print(f"CVC5 status: {cvc5_result['status']} in {cvc5_result['time']:.2f}s")
        
        # Try with simplified file if needed
        if cvc5_result['status'] not in ['sat', 'unsat'] and HAS_SIMPLIFIER and has_z3_solution:
            print("Trying with simplified SMT file for CVC5...")
            simplified_file = os.path.join("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\simplified", 
                                          f"{file_basename}_simplified_cvc5.smt2")
            simplify_smt_file(smt_file, simplified_file)
            
            # Try again with the simplified file
            cvc5_simplify_result = run_cvc5(simplified_file)
            results.append(cvc5_simplify_result)
            print(f"CVC5 (simplified) status: {cvc5_simplify_result['status']} in {cvc5_simplify_result['time']:.2f}s")
            
            if cvc5_simplify_result['status'] == 'sat' and cvc5_simplify_result['model']:
                print(f"CVC5 found a solution with simplified SMT!")
                save_model("CVC5", file_basename, "simplified", cvc5_simplify_result['model'])
        elif cvc5_result['status'] == 'sat' and cvc5_result['model']:
            print(f"CVC5 found a solution!")
            save_model("CVC5", file_basename, "default", cvc5_result['model'])
        
        # Run with Yices (default logic)
        print("\nRunning with Yices (default logic)...")
        yices_result = run_yices(smt_file)
        results.append(yices_result)
        print(f"Yices status: {yices_result['status']} in {yices_result['time']:.2f}s")
        
        # Try with simplified file if needed
        if yices_result['status'] not in ['sat', 'unsat'] and HAS_SIMPLIFIER and has_z3_solution:
            print("Trying with simplified SMT file for Yices...")
            simplified_file = os.path.join("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\simplified", 
                                          f"{file_basename}_simplified_yices.smt2")
            simplified_file = simplify_smt_file(smt_file, simplified_file)
            # Apply additional Yices-specific fixes
            fix_common_syntax_issues(simplified_file, simplified_file)
            
            # Try again with the simplified file
            yices_simplify_result = run_yices(simplified_file)
            results.append(yices_simplify_result)
            print(f"Yices (simplified) status: {yices_simplify_result['status']} in {yices_simplify_result['time']:.2f}s")
            
            if yices_simplify_result['status'] == 'sat' and yices_simplify_result['model']:
                print(f"Yices found a solution with simplified SMT!")
                save_model("Yices", file_basename, "simplified", yices_simplify_result['model'])
        elif yices_result['status'] == 'sat' and yices_result['model']:
            print(f"Yices found a solution!")
            save_model("Yices", file_basename, "default", yices_result['model'])
        
        # Test different logics with each solver
        for logic in SMT_LOGICS:
            print(f"\n{'-'*40}")
            print(f"Testing with logic: {logic}")
            print(f"{'-'*40}")
            
            # Z3 with specific logic
            print(f"Running Z3 with {logic}...")
            z3_logic_result = run_z3_api(smt_file, logic)
            results.append(z3_logic_result)
            print(f"Z3 ({logic}) status: {z3_logic_result['status']} in {z3_logic_result['time']:.2f}s")
            
            if z3_logic_result['status'] == 'sat' and z3_logic_result['model']:
                save_model("Z3", file_basename, logic, z3_logic_result['model'])
                has_z3_solution_with_logic = True
            else:
                has_z3_solution_with_logic = False
            
            # CVC5 with specific logic
            print(f"Running CVC5 with {logic}...")
            cvc5_logic_result = run_cvc5(smt_file, logic)
            results.append(cvc5_logic_result)
            print(f"CVC5 ({logic}) status: {cvc5_logic_result['status']} in {cvc5_logic_result['time']:.2f}s")
            
            # Try with simplified file if needed
            if cvc5_logic_result['status'] not in ['sat', 'unsat'] and HAS_SIMPLIFIER and has_z3_solution_with_logic:
                print(f"Trying with simplified SMT file for CVC5 with {logic}...")
                simplified_file = os.path.join("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\simplified", 
                                              f"{file_basename}_simplified_cvc5_{logic}.smt2")
                simplify_smt_file(smt_file, simplified_file, logic)
                
                # Try again with the simplified file
                cvc5_simplify_logic_result = run_cvc5(simplified_file, logic)
                results.append(cvc5_simplify_logic_result)
                print(f"CVC5 ({logic}, simplified) status: {cvc5_simplify_logic_result['status']} in {cvc5_simplify_logic_result['time']:.2f}s")
                
                if cvc5_simplify_logic_result['status'] == 'sat' and cvc5_simplify_logic_result['model']:
                    print(f"CVC5 found a solution with simplified SMT and logic {logic}!")
                    save_model("CVC5", file_basename, f"{logic}_simplified", cvc5_simplify_logic_result['model'])
            elif cvc5_logic_result['status'] == 'sat' and cvc5_logic_result['model']:
                save_model("CVC5", file_basename, logic, cvc5_logic_result['model'])
            
            # Yices with specific logic
            print(f"Running Yices with {logic}...")
            yices_logic_result = run_yices(smt_file, logic) 
            results.append(yices_logic_result)
            print(f"Yices ({logic}) status: {yices_logic_result['status']} in {yices_logic_result['time']:.2f}s")
            
            # Try with simplified file if needed
            if yices_logic_result['status'] not in ['sat', 'unsat'] and HAS_SIMPLIFIER and has_z3_solution_with_logic:
                print(f"Trying with simplified SMT file for Yices with {logic}...")
                simplified_file = os.path.join("e:\\Programmieren\\Bachelor-Thesis\\SMT_Modules\\simplified", 
                                              f"{file_basename}_simplified_yices_{logic}.smt2")
                simplified_file = simplify_smt_file(smt_file, simplified_file, logic)
                # Apply additional Yices-specific fixes
                fix_common_syntax_issues(simplified_file, simplified_file)
                
                # Try again with the simplified file
                yices_simplify_logic_result = run_yices(simplified_file, logic)
                results.append(yices_simplify_logic_result)
                print(f"Yices ({logic}, simplified) status: {yices_simplify_logic_result['status']} in {yices_simplify_logic_result['time']:.2f}s")
                
                if yices_simplify_logic_result['status'] == 'sat' and yices_simplify_logic_result['model']:
                    print(f"Yices found a solution with simplified SMT and logic {logic}!")
                    save_model("Yices", file_basename, f"{logic}_simplified", yices_simplify_logic_result['model'])
            elif yices_logic_result['status'] == 'sat' and yices_logic_result['model']:
                save_model("Yices", file_basename, logic, yices_logic_result['model'])
    
    # Create summary DataFrame
    df = pd.DataFrame([
        {
            "file": r["file"],
            "solver": r["solver"],
            "logic": r["logic"],
            "status": r["status"],
            "time": r["time"],
            "has_model": "Yes" if r.get("model") else "No"
        }
        for r in results
    ])
    
    # Save results
    df.to_csv("solver_logic_comparison_results.csv", index=False)
    
    # Print summary table
    print("\nSummary:")
    try:
        # Group by file and create pivot table
        summary = df.pivot_table(
            index=["file", "logic"],
            columns="solver",
            values=["status", "time"],
            aggfunc={"status": lambda x: x.iloc[0], "time": "mean"}
        )
        print(summary)
    except Exception as e:
        print(f"Could not create summary table: {e}")
        # Print simplified summary instead
        print(df[["file", "solver", "logic", "status", "time"]].sort_values(["file", "logic", "solver"]))
    
    # Additional analytics
    print("\nSolver performance comparison:")
    solver_stats = df.groupby("solver").agg({
        "status": lambda x: (x == "sat").mean(),  # Percentage of SAT results
        "time": ["mean", "min", "max"],  # Time statistics
        "file": "count"  # Number of runs
    })
    print(solver_stats)
    
    print("\nLogic performance comparison:")
    logic_stats = df.groupby("logic").agg({
        "status": lambda x: (x == "sat").mean(),  # Percentage of SAT results
        "time": ["mean", "min", "max"],  # Time statistics
        "file": "count"  # Number of runs
    })
    print(logic_stats)    
    print("\nResults saved to solver_logic_comparison_results.csv")
    print("Models saved in 'models' directory")
      # Generate simple visualizations
    try:
        import plot_eval
        print("\nGenerating simple visualizations...")
        plot_eval.main()
        print("Visualization complete. Check the 'Plots' directory for results.")
        
        # Print note about HTML-to-PNG conversion (optional)
        print("\nNote: To convert HTML tables to PNG images, you can run:")
        print("  python html_to_image.py Plots/solver_status_summary.html")
    except Exception as e:
        print(f"Failed to generate visualizations: {e}")
        print("You can run visualizations separately with: python simple_plot_solvers.py")


if __name__ == "__main__":
    main()