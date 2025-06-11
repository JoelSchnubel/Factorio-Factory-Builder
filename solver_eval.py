#! .venv\Scripts\python.exe

import os
import time
import subprocess
import tempfile
import re
from z3 import Solver, sat, unsat, unknown, parse_smt2_file
import glob
import json
import csv
import matplotlib.pyplot as plt

CVC5_PATH = "C:\\SMT\\cvc5\\cvc5-Win64-x86_64-static\\bin\\cvc5.exe"
YICES_PATH = "C:\\SMT\\yices\\yices-2.6.5\\bin\\yices-smt2.exe"

# Different logics to test
SMT_LOGICS = [
    None,
    "QF_LIA",   
    "QF_LRA",  
    "QF_UFLIA", 
    "QF_IDL",   
    "QF_RDL",   
    "QF_UF",   
    "QF_BV",   
    "QF_AUFBV",
]

# Define which solvers to use
SOLVERS = ["z3", "cvc5", "yices"]

def create_smt2_file_with_logic(smt_file, logic, solver_name):
    """Create a temporary SMT2 file with the specific logic."""
    # Handle 'None' logic for CVC5 - use QF_ALL as default for CVC5
    display_logic = logic
    if logic is None:
        # CVC5 uses "ALL" or "QF_ALL" as their default logic
        if solver_name == "cvc5":
            logic = "QF_ALL"
        else:
            logic = "ALL"  # Default fallback for other solvers
        print(f"Defaulting to {logic} for {solver_name} since no logic was specified")
    
    temp_file = os.path.join(os.path.dirname(smt_file), f"temp_{display_logic}_{os.path.basename(smt_file)}")
    temp_file = temp_file.replace('.smt', '.smt2')
    
    with open(smt_file, 'r') as f:
        content = f.read()
    
    # Format the content with proper header and footer
    formatted_content = []
    
    # Detect if type declarations are needed based on the content
    needs_int_type = bool(re.search(r'declare-fun\s+\S+\s+\([^)]*\)\s+Int', content))
    needs_real_type = bool(re.search(r'declare-fun\s+\S+\s+\([^)]*\)\s+Real', content))
    needs_bool_type = bool(re.search(r'declare-fun\s+\S+\s+\([^)]*\)\s+Bool', content))
    
    # Add logic declaration
    formatted_content.append(f"(set-logic {logic})")
    
    # Add model production option
    formatted_content.append("(set-option :produce-models true)")
    
    # Logic-specific adaptations
    needs_type_conversion = False
    int_type_substitute = None
    integer_literals_pattern = None
    
    # For CVC5 specific type handling
    if solver_name == "cvc5" or solver_name == "yices":
        # Add necessary type declarations based on the logic
        if logic in ["QF_LRA", "QF_RDL"]:
            # These logics use Real but may need Int to Real conversion
            if needs_int_type:
                # Add mapping from Int to Real
                print(f"Logic {logic} requires Int->Real mapping")
                formatted_content.append("(define-sort Int () Real)")
                needs_type_conversion = True
                int_type_substitute = "Real"
                integer_literals_pattern = r'(\s+)(\d+)(\s+|\))'
        elif logic in ["QF_UF"]:
            # Uninterpreted functions need explicit sort declarations
            if needs_int_type:
                print(f"Logic {logic} requires Int sort declaration")
                # Need to declare a user-defined sort AND convert all integer literals
                formatted_content.append("(declare-sort Int 0)")
                needs_type_conversion = True  # Need to handle integer literals
                integer_literals_pattern = r'(\s+)(\d+)(\s+|\))'  # Pattern to match integer literals
        elif logic in ["QF_BV", "QF_AUFBV"]:
            # For bit-vector logics
            if needs_int_type:
                print(f"Logic {logic} requires Int->BitVec mapping")
                # Use BitVec directly rather than define-sort to avoid nesting issues
                # We'll convert all Int declarations directly in the content
                needs_type_conversion = True
                int_type_substitute = "(_ BitVec 32)"
                # Convert integer literals to bitvector literals
                integer_literals_pattern = r'(\s+)(\d+)(\s+|\))'
    
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
        
        # Skip existing logic and option declarations
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
    
    # Apply logic-specific type conversions if needed
    if needs_type_conversion and integer_literals_pattern and logic in ["QF_BV", "QF_AUFBV"]:
        # For bit-vector logics, convert integer literals to bitvector literals
        fixed_content = re.sub(integer_literals_pattern, r'\1(_ bv\2 32)\3', fixed_content)
    elif needs_type_conversion and logic in ["QF_LRA", "QF_RDL"]:
        # For real-based logics, convert integer literals to real literals
        fixed_content = re.sub(r'(\s+)(\d+)(\s+|\))', r'\1\2.0\3', fixed_content)
        
    # Create a completely new approach for CVC5 based on the logic type
    if solver_name == "cvc5":
        lines = fixed_content.splitlines()
        new_lines = []
        int_constants = {}
        
        # Start with logic declaration and options
        for i, line in enumerate(lines):
            if line.startswith("(set-logic") or line.startswith("(set-option"):
                new_lines.append(line)
                
        # Handle different logics with appropriate type conversions
        if logic in ["QF_BV", "QF_AUFBV"]:
            print(f"Applying special handling for {logic}")
            # For bit-vector logics, convert all Int declarations to BitVec 32
            # Add any necessary integer constants as BitVec constants
            
            # First collect all integer literals used in equations
            integer_literals = set()
            for line in lines:
                if "(assert" in line or "(=" in line:
                    # Find standalone integer literals in assertions
                    for match in re.finditer(r'(?<!\w)(\d+)(?!\w)', line):
                        if match.group(1) not in integer_literals:
                            integer_literals.add(match.group(1))
            
            # Add BitVec constants for these integers
            for literal in integer_literals:
                const_name = f"#bv{literal}"
                new_lines.append(f"(define-fun {const_name} () (_ BitVec 32) (_ bv{literal} 32))")
                int_constants[literal] = const_name
            
            # Now process all the variable declarations and assertions
            for line in lines:
                if line.startswith("(declare-fun"):
                    # Convert Int declarations to BitVec 32
                    if " Int)" in line:
                        line = line.replace(" Int)", " (_ BitVec 32))")
                    new_lines.append(line)
                elif "(assert" in line or "(=" in line:
                    # Replace integer literals with BitVec constants
                    for literal in sorted(integer_literals, key=len, reverse=True):
                        pattern = r'(?<!\w)' + re.escape(literal) + r'(?!\w)'
                        line = re.sub(pattern, int_constants[literal], line)
                    new_lines.append(line)
                elif not line.startswith("(set-logic") and not line.startswith("(set-option"):
                    new_lines.append(line)
                    
        elif logic in ["QF_UF"]:
            print(f"Applying special handling for {logic}")
            # For QF_UF, we need to convert integers to constants of an uninterpreted sort
            
            # First declare the Int sort
            new_lines.append("(declare-sort Int 0)")
            
            # Then declare constants for each integer literal
            integer_literals = set()
            for line in lines:
                if "(assert" in line or "(=" in line:
                    # Find standalone integer literals in assertions
                    for match in re.finditer(r'(?<!\w)(\d+)(?!\w)', line):
                        if match.group(1) not in integer_literals:
                            integer_literals.add(match.group(1))
            
            # Add constants for these integers
            for literal in integer_literals:
                const_name = f"int_const_{literal}"
                new_lines.append(f"(declare-fun {const_name} () Int)")
                int_constants[literal] = const_name
            
            # Now process all the variable declarations and assertions
            for line in lines:
                if line.startswith("(declare-fun") or "(assert" in line or "(=" in line:
                    # Replace integer literals with constants
                    for literal in sorted(integer_literals, key=len, reverse=True):
                        pattern = r'(?<!\w)' + re.escape(literal) + r'(?!\w)'
                        line = re.sub(pattern, int_constants[literal], line)
                    new_lines.append(line)
                elif not line.startswith("(set-logic") and not line.startswith("(set-option"):
                    new_lines.append(line)
                    
        elif logic in ["QF_LRA", "QF_RDL"]:
            print(f"Applying special handling for {logic}")
            # For real-based logics
            
            # First define Int as Real
            new_lines.append("(define-sort Int () Real)")
            
            # Process declarations and assertions, converting literals to reals
            for line in lines:
                if "(assert" in line or "(=" in line:
                    # Convert integer literals to real literals
                    line = re.sub(r'(?<!\w|\.)(\d+)(?!\w|\.)', r'\1.0', line)
                
                if not line.startswith("(set-logic") and not line.startswith("(set-option"):
                    new_lines.append(line)
        else:
            # For other logics, just copy all lines except logic and option declarations
            for line in lines:
                if not line.startswith("(set-logic") and not line.startswith("(set-option"):
                    new_lines.append(line)
        
        fixed_content = "\n".join(new_lines)
    
    # Make sure check-sat and get-model are present at the end
    if "(check-sat)" not in fixed_content:
        fixed_content += "\n(check-sat)"
    if "(get-model)" not in fixed_content:
        fixed_content += "\n(get-model)"
    
    # Write the formatted content
    with open(temp_file, 'w') as f:
        f.write(fixed_content)
        
    print(f"Created compatible SMT-LIB2 file at {temp_file} for logic {display_logic}")
    return temp_file

def adapt_content_for_yices(content, logic):
    """
    Adapt SMT content for Yices
    
    Args:
        content (str): Original SMT file content
        logic (str): Logic to use
        
    Returns:
        str: Adapted content for Yices
    """
    # Add logic declaration if not present and logic is specified
    if logic and not re.search(r'\(set-logic\s+[^\)]+\)', content):
        content = f"(set-logic {logic})\n{content}"
    
    # Make sure produce-models option is set
    if not re.search(r'\(set-option\s+:produce-models\s+true\)', content):
        content = "(set-option :produce-models true)\n" + content
    
    # Logic-specific adaptations
    if logic:
        # For logics that don't support Int type, replace with Real
        if logic in ["QF_LRA", "QF_RDL"]:
            # Replace 'Int' type declarations with 'Real'
            content = re.sub(r'\(\s*declare-fun\s+([^\s]+)\s+\(\s*\)\s+Int\s*\)', 
                             r'(declare-fun \1 () Real)', content)
        
        # For logics without arithmetic, provide appropriate declarations
        if logic in ["QF_UF", "QF_BV", "QF_AUFBV"]:
            # Replace 'Int' type declarations with appropriate types
            if logic == "QF_UF":
                # For QF_UF, declare a sort to use instead of Int
                if not re.search(r'\(declare-sort\s+MyInt\s+0\)', content):
                    content = "(declare-sort MyInt 0)\n" + content
                content = re.sub(r'\(\s*declare-fun\s+([^\s]+)\s+\(\s*\)\s+Int\s*\)', 
                                r'(declare-fun \1 () MyInt)', content)
            elif "BV" in logic:
                # For bit-vector logics, use (_ BitVec 32) or similar
                content = re.sub(r'\(\s*declare-fun\s+([^\s]+)\s+\(\s*\)\s+Int\s*\)', 
                                r'(declare-fun \1 () (_ BitVec 32))', content)
        
        # Remove maximize/minimize commands for logics that don't support optimization
        content = re.sub(r'\(\s*(maximize|minimize)\s+[^\)]+\s*\)', '', content)
    
    # Add check-sat and get-model if needed
    if not re.search(r'\(check-sat\)', content):
        content += "\n(check-sat)"
    
    if not re.search(r'\(get-model\)', content):
        content += "\n(get-model)"
    
    # Check if there's any problematic content (debugging purposes)
    with open('debug_yices_temp.smt2', 'w') as f:
        f.write(content)
    
    return content

def parse_cvc5_model(output):
    """
    Parse model from CVC5 output
    
    Args:
        output (str): Solver output
        
    Returns:
        str: Parsed model
    """
    # Extract model part from output
    model_match = re.search(r'sat\s*((\(model[^$]+)|\(model\))', output, re.DOTALL)
    if model_match:
        return model_match.group(1).strip()
    else:
        return output  # Return full output if model not found

def parse_yices_model(output):
    """
    Parse model from Yices output
    
    Args:
        output (str): Solver output
        
    Returns:
        str: Parsed model
    """
    # Extract model part from output
    model_lines = []
    capture = False
    
    for line in output.splitlines():
        if line.strip() == "sat":
            capture = True
            continue
        if capture:
            model_lines.append(line)
    
    if model_lines:
        return "\n".join(model_lines)
    else:
        return output  # Return full output if model not found

def load_smt_files(directory="SMT_Modules"):
    """
    Load all .smt files from the specified directory
    
    Args:
        directory (str): Directory containing SMT files
    
    Returns:
        dict: Dictionary mapping file names to file content
    """
    smt_files = {}
    
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return smt_files
    
    # Get all .smt files in the directory
    file_paths = glob.glob(os.path.join(directory, "*.smt"))
    
    # Load each file
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                smt_files[file_name] = content
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    print(f"Loaded {len(smt_files)} SMT files from {directory}")
    return smt_files



def run_z3_with_logic(file_path, logic=None):
    """
    Run Z3 with a specific logic on an SMT file
    
    Args:
        file_path (str): Path to the SMT file
        logic (str): Logic to use (None for default)
    
    Returns:
        tuple: (satisfiability, model, execution_time)
    """
    print(f"\nRunning Z3 with logic: {logic or 'Default'}")
    
    start_time = time.time()
    
    # Create Z3 solver with the given logic
    solver = Solver()
    if logic:
        solver.set(logic=logic)
    
    try:
        # Parse the SMT file
        print(f"Parsing SMT file: {os.path.basename(file_path)}")
        formula = parse_smt2_file(file_path)
        solver.add(formula)
        
        # Check satisfiability
        print("Checking satisfiability...")
        result = solver.check()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result == sat:
            print(f"Result: sat (in {execution_time:.2f}s)")
            model = solver.model()
            return "sat", model, execution_time
        elif result == unsat:
            print(f"Result: unsat (in {execution_time:.2f}s)")
            return "unsat", None, execution_time
        else:
            print(f"Result: unknown (in {execution_time:.2f}s)")
            return "unknown", None, execution_time
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Error: {e} (in {execution_time:.2f}s)")
        return "error", str(e), execution_time
    
    
def save_model(model, file_path, logic, solver_name="z3"):
    """
    Save the model to a file
    
    Args:
        model: Solver model (Z3 model object or string for CVC5/Yices)
        file_path (str): Original SMT file path
        logic (str): Logic used
        solver_name (str): Name of the solver used
    """
    if model is None:
        return
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create model file path
    base_name = os.path.basename(file_path)
    logic_str = logic if logic else "default"
    model_file = os.path.join("models", f"{base_name}_{solver_name}_{logic_str}_model.txt")
    
    # Save model
    with open(model_file, 'w') as f:
        f.write(str(model))
    
    print(f"Model saved to {model_file}")
    
    
def run_solver(solver_name, file_path, logic):
    """
    Run the specified solver with the given logic on an SMT file
    
    Args:
        solver_name (str): Name of the solver (z3, cvc5, yices)
        file_path (str): Path to the SMT file
        logic (str): Logic to use (None for default)
    
    Returns:
        tuple: (satisfiability, model, execution_time)
    """
    if solver_name == "z3":
        return run_z3_with_logic(file_path, logic)
    elif solver_name == "cvc5":
        return run_cvc5_with_logic(file_path, logic)
    elif solver_name == "yices":
        return run_yices_with_logic(file_path, logic)
    else:
        return "error", f"Unknown solver: {solver_name}", 0

def evaluate_solvers(directory="SMT_Modules", save_results=True, num_runs=10):
    """
    Evaluate all solvers with different logics on SMT files
    
    Args:
        directory (str): Directory containing SMT files
        save_results (bool): Whether to save results to JSON file
        num_runs (int): Number of times to run each logic to get an average execution time
    
    Returns:
        list: List of result dictionaries
    """
    smt_files = load_smt_files(directory)
    if not smt_files:
        print("No SMT files to evaluate.")
        return []
    
    results = []
    
    for file_name, content in smt_files.items():
        file_path = os.path.join(directory, file_name)
        print(f"\nEvaluating file: {file_name}")
        
        for solver_name in SOLVERS:
            print(f"\nUsing solver: {solver_name}")
            for logic in SMT_LOGICS:
                print(f"Running logic: {logic if logic else 'default'} (10 runs for averaging)")
                
                # Run multiple times and average the results
                total_time = 0
                results_count = {"sat": 0, "unsat": 0, "unknown": 0, "error": 0}
                final_result = None
                final_model = None
                
                for run in range(num_runs):
                    print(f"  Run {run + 1}/{num_runs}...", end="", flush=True)
                    result, model, execution_time = run_solver(solver_name, file_path, logic)
                    print(f" {result} in {execution_time:.2f}s")
                    
                    # Track results
                    results_count[result] += 1
                    total_time += execution_time
                    
                    # Store first non-error result for model saving
                    if final_result is None or (final_result == "error" and result != "error"):
                        final_result = result
                        final_model = model
                
                # Calculate average execution time
                avg_execution_time = total_time / num_runs
                
                # Determine the most common result
                most_common_result = max(results_count.items(), key=lambda x: x[1])[0]
                
                print(f"Average execution time: {avg_execution_time:.2f}s, Most common result: {most_common_result}")
                
                # Create a result dictionary with all information
                result_dict = {
                    "file_name": file_name,
                    "solver": solver_name,
                    "logic": logic if logic else "default",
                    "result": most_common_result,
                    "execution_time": avg_execution_time,
                    "result_counts": results_count
                }
                
                results.append(result_dict)
                
                #if final_result == "sat":
                #    save_model(final_model, file_path, logic, solver_name)
    
    return results

def save_results_to_csv(results, overwrite=True):
    """
    Save evaluation results to a CSV file
    
    Args:
        results (list): List of result dictionaries
        overwrite (bool): Whether to overwrite the existing CSV file
    
    Returns:
        str: Timestamp used for the filename
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV for easier analysis
    csv_file = os.path.join("results", "solver_evaluation.csv" if overwrite else f"solver_evaluation_{timestamp}.csv")
    
    # Extract all unique solvers, logics, and files
    solvers = sorted(set(r["solver"] for r in results))
    logics = sorted(set(r["logic"] for r in results))
    file_names = sorted(set(r["file_name"] for r in results))
    
    # Create one comprehensive CSV file with all information
    with open(csv_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        
        # Write header with solver columns for both time and result
        header = ['File', 'Logic']
        for solver in solvers:
            header.extend([
                f"{solver}_result",
                f"{solver}_time",
                f"{solver}_sat",
                f"{solver}_unsat",
                f"{solver}_unknown", 
                f"{solver}_error"
            ])
        writer.writerow(header)
        
        # Group by file name and logic
        for file_name in file_names:
            for logic in logics:
                row = [file_name, logic]
                
                for solver in solvers:
                    # Find the result for this combination
                    solver_result = next(
                        (r for r in results if r["file_name"] == file_name 
                         and r["logic"] == logic and r["solver"] == solver), 
                        None
                    )
                    
                    if solver_result:
                        # Add result and time
                        row.extend([
                            solver_result["result"],
                            f"{solver_result['execution_time']:.2f}",
                            solver_result["result_counts"]["sat"],
                            solver_result["result_counts"]["unsat"],
                            solver_result["result_counts"]["unknown"],
                            solver_result["result_counts"]["error"]
                        ])
                    else:
                        # No result for this combination
                        row.extend(["N/A", "0.00", "0", "0", "0", "0"])
                
                writer.writerow(row)
    
    print(f"\nResults saved to CSV: {csv_file}")
    
    return timestamp


def run_cvc5_with_logic(file_path, logic=None):
    """
    Run CVC5 with a specific logic on an SMT file
    
    Args:
        file_path (str): Path to the SMT file
        logic (str): Logic to use (None for default)
    
    Returns:
        tuple: (satisfiability, model, execution_time)
    """
    print(f"\nRunning CVC5 with logic: {logic or 'Default'}")
    
    # Check if CVC5 executable exists
    if not os.path.exists(CVC5_PATH):
        print(f"CVC5 executable not found at: {CVC5_PATH}")
        return "error", "CVC5 executable not found", 0
    
    # Create a temporary file with CVC5 specific adaptations
    temp_file_path = create_smt2_file_with_logic(file_path, logic, "cvc5")
    if not temp_file_path:
        return "error", "Failed to create temporary file", 0
    
    # Build command
    cmd = [CVC5_PATH, "--produce-models"]
    # The create_smt2_file_with_logic function already handles the default logic for CVC5
    # So we should use whatever logic it actually used (QF_ALL for None)
    if logic:
        cmd.extend(["--force-logic", logic])
    cmd.append(temp_file_path)
    
    start_time = time.time()
    try:
        # Run CVC5 process
        print(f"Executing CVC5 on: {os.path.basename(file_path)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Keep temporary files for debugging
        debug_file_path = "debug_cvc5_temp.smt2"
        try:
            import shutil
            shutil.copy(temp_file_path, debug_file_path)
            print(f"Copied temp file to {debug_file_path} for debugging")
            
            # Optional: keep temp files for failed runs only
            if "sat" not in stdout.lower() and stderr.strip():
                # Don't remove the file if there was an error
                print(f"Keeping temporary file {temp_file_path} due to error")
            else:
                os.remove(temp_file_path)
        except Exception as e:
            print(f"Warning: Failed to handle temporary file {temp_file_path}: {e}")
        
        # Process output
        if "sat" in stdout.lower():
            print(f"Result: sat (in {execution_time:.2f}s)")
            # Parse the model from CVC5 output
            model = parse_cvc5_model(stdout)
            return "sat", model, execution_time
        elif "unsat" in stdout.lower():
            print(f"Result: unsat (in {execution_time:.2f}s)")
            return "unsat", None, execution_time
        elif "unknown" in stdout.lower():
            print(f"Result: unknown (in {execution_time:.2f}s)")
            return "unknown", None, execution_time
        else:
            # Check for errors
            if stderr.strip():
                print(f"Error: {stderr.strip()} (in {execution_time:.2f}s)")
                return "error", stderr.strip(), execution_time
            else:
                print(f"Unknown result: {stdout.strip()} (in {execution_time:.2f}s)")
                return "unknown", None, execution_time
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Error: {e} (in {execution_time:.2f}s)")
        return "error", str(e), execution_time


def run_yices_with_logic(file_path, logic=None):
    """
    Run Yices with a specific logic on an SMT file
    
    Args:
        file_path (str): Path to the SMT file
        logic (str): Logic to use (None for default)
    
    Returns:
        tuple: (satisfiability, model, execution_time)
    """
    print(f"\nRunning Yices with logic: {logic or 'Default'}")
    
    # Check if Yices executable exists
    if not os.path.exists(YICES_PATH):
        print(f"Yices executable not found at: {YICES_PATH}")
        return "error", "Yices executable not found", 0
    
    # Create a temporary file with Yices specific adaptations
    temp_file_path = create_smt2_file_with_logic(file_path, logic, "yices")
    if not temp_file_path:
        return "error", "Failed to create temporary file", 0
    
    # Build command
    cmd = [YICES_PATH]
    #if logic:
    #    cmd.extend([f"--logic={logic}"])
    cmd.append(temp_file_path)
    
    start_time = time.time()
    try:
        # Run Yices process
        print(f"Executing Yices on: {os.path.basename(file_path)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Clean up temporary file
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"Warning: Failed to remove temporary file {temp_file_path}: {e}")
        
        # Process output
        if "sat" in stdout.lower():
            print(f"Result: sat (in {execution_time:.2f}s)")
            # Parse the model from Yices output
            model = parse_yices_model(stdout)
            return "sat", model, execution_time
        elif "unsat" in stdout.lower():
            print(f"Result: unsat (in {execution_time:.2f}s)")
            return "unsat", None, execution_time
        elif "unknown" in stdout.lower():
            print(f"Result: unknown (in {execution_time:.2f}s)")
            return "unknown", None, execution_time
        else:
            # Check for errors
            if stderr.strip():
                print(f"Error: {stderr.strip()} (in {execution_time:.2f}s)")
                return "error", stderr.strip(), execution_time
            else:
                print(f"Unknown result: {stdout.strip()} (in {execution_time:.2f}s)")
                return "unknown", None, execution_time
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Error: {e} (in {execution_time:.2f}s)")
        return "error", str(e), execution_time



def main():
    print("Starting SMT solver evaluation...")
    print("Available solvers:", SOLVERS)
    print("Available logics:", SMT_LOGICS)

    # Run the evaluation
    results = evaluate_solvers()

    # Print summary
    if results:
        print("\nEvaluation Summary:")
        total_files = len(set(r["file_name"] for r in results))
        print(f"Total files evaluated: {total_files}")
        for solver in SOLVERS:
            solver_results = [r for r in results if r["solver"] == solver]
            if not solver_results:
                continue
            print(f"\n{solver.upper()} Results:")
            sat_count = sum(1 for r in solver_results if r["result"] == "sat")
            unsat_count = sum(1 for r in solver_results if r["result"] == "unsat")
            unknown_count = sum(1 for r in solver_results if r["result"] == "unknown")
            error_count = sum(1 for r in solver_results if r["result"] == "error")
            print(f"  sat: {sat_count}")
            print(f"  unsat: {unsat_count}")
            print(f"  unknown: {unknown_count}")
            print(f"  error: {error_count}")
            avg_time = sum(r["execution_time"] for r in solver_results) / len(solver_results)
            print(f"  Average execution time: {avg_time:.2f}s")

        # Save results to a single CSV (overwrite)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join("results", "solver_evaluation.csv")
        import csv
        solvers = sorted(set(r["solver"] for r in results))
        logics = sorted(set(r["logic"] for r in results))
        file_names = sorted(set(r["file_name"] for r in results))
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['File', 'Logic', 'Solver', 'Result', 'Execution Time (s)', 'SAT', 'UNSAT', 'UNKNOWN', 'ERROR']
            writer.writerow(header)
            for r in results:
                writer.writerow([
                    r["file_name"],
                    r["logic"],
                    r["solver"],
                    r["result"],
                    f"{r['execution_time']:.2f}",
                    r["result_counts"]["sat"],
                    r["result_counts"]["unsat"],
                    r["result_counts"]["unknown"],
                    r["result_counts"]["error"]
                ])
        print(f"\nResults saved to CSV: {csv_file}")

        # Always run the plotting script on the CSV
        print("\nGenerating solver performance plots...")
        import subprocess
        try:
            subprocess.run([
                "python", "plot_solver_results.py", csv_file
            ], check=True)
            print("Plots generated successfully.")
        except Exception as e:
            print(f"Failed to generate plots: {e}")
            print(f"You can manually run: python plot_solver_results.py {csv_file}")

    print("\nEvaluation complete. Results saved to the 'results' directory.")
    print("Visualizations saved to the 'Plots' directory.")

if __name__ == "__main__":
    main()