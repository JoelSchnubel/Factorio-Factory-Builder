#!/usr/bin/env python
"""
Z3 Logic Debugger - A tool to track where logic problems occur when using Z3 API on SMT files.
This script helps identify issues with different logics and tracks exactly where problems occur.
"""

import os
import time
import sys
import glob
import traceback
from collections import defaultdict

def run_z3_api_with_debug(smt_file, logic=None, timeout=300, debug=True):
    """
    Run Z3 API on an SMT file with detailed debugging to find logic-related issues.
    
    Args:
        smt_file: Path to the SMT file
        logic: SMT logic to use (e.g., "QF_LIA", "QF_BV")
        timeout: Timeout in seconds
        debug: Whether to enable verbose debugging
        
    Returns:
        Dictionary with results and debug information
    """    
    start_time = time.time()
    debug_info = defaultdict(list)
    stage = "initialization"
    
    try:
        # Import Z3 with debugging
        try:
            from z3 import Solver, parse_smt2_file, set_param, is_expr
            debug_info["imports"].append("Z3 imports successful")
        except ImportError as e:
            return {
                "solver": "Z3 API",
                "file": os.path.basename(smt_file),
                "status": "error",
                "time": 0,
                "error": f"Z3 import error: {str(e)}",
                "debug_info": dict(debug_info)
            }
        
        stage = "setting_params"
        if debug:
            # Enable Z3 tracing for debugging
            set_param("verbose", 10)
            debug_info[stage].append("Verbose level set to 10")
        
        # Set timeout in milliseconds
        set_param("timeout", timeout * 1000)
        debug_info[stage].append(f"Timeout set to {timeout} seconds")
        
        # Parse the SMT file
        stage = "parsing"
        debug_info[stage].append(f"Attempting to parse {os.path.basename(smt_file)}")
        
        try:
            formula = parse_smt2_file(smt_file)
            debug_info[stage].append("Parsing successful")
            if debug:
                debug_info[stage].append(f"Formula type: {type(formula)}")
                if isinstance(formula, list):
                    debug_info[stage].append(f"Formula contains {len(formula)} assertions")
        except Exception as e:
            debug_info[stage].append(f"Parsing failed: {str(e)}")
            raise
        
        # Create solver with specified logic if provided
        stage = "solver_creation"
        solver = Solver()
        debug_info[stage].append("Solver created")
        
        if logic:
            try:
                solver.set(logic=logic)
                debug_info[stage].append(f"Logic set to {logic}")
            except Exception as e:
                debug_info[stage].append(f"Setting logic '{logic}' failed: {str(e)}")
                raise
        
        # Add formula to solver
        stage = "add_formula"
        try:
            solver.add(formula)
            debug_info[stage].append("Formula added to solver")
        except Exception as e:
            debug_info[stage].append(f"Adding formula failed: {str(e)}")
            raise
        
        # Check for satisfiability
        stage = "check"
        try:
            debug_info[stage].append("Starting solver.check()")
            result = solver.check()
            status = str(result).lower()
            debug_info[stage].append(f"Check result: {status}")
        except Exception as e:
            debug_info[stage].append(f"Check failed: {str(e)}")
            raise
        
        # Get model if satisfiable
        stage = "model"
        model = None
        model_str = None
        
        if status == "sat":
            try:
                model = solver.model()
                model_str = str(model)
                debug_info[stage].append("Model retrieved successfully")
                debug_info[stage].append(f"Model size: {len(model)} declarations")
            except Exception as e:
                debug_info[stage].append(f"Model retrieval failed: {str(e)}")
                # Continue even if model retrieval fails
        
        end_time = time.time()
        
        solver_name = f"Z3 API" if not logic else f"Z3 API ({logic})"
        
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "logic": logic if logic else "default",
            "status": status,
            "time": end_time - start_time,
            "model": model_str,
            "debug_info": dict(debug_info)
        }
        
    except Exception as e:
        end_time = time.time()
        error_tb = traceback.format_exc()
        
        debug_info[stage].append(f"ERROR: {str(e)}")
        debug_info["traceback"] = error_tb.split('\n')
        
        solver_name = f"Z3 API" if not logic else f"Z3 API ({logic})"
        
        return {
            "solver": solver_name,
            "file": os.path.basename(smt_file),
            "logic": logic if logic else "default",
            "status": "error",
            "time": end_time - start_time,
            "model": None,
            "error": str(e),
            "error_stage": stage,
            "debug_info": dict(debug_info)
        }

def analyze_smt_file_structure(smt_file):
    """
    Analyze an SMT file structure to find logic declarations and other key elements.
    
    Args:
        smt_file: Path to the SMT file
        
    Returns:
        Dictionary with file analysis
    """
    analysis = {
        "file": os.path.basename(smt_file),
        "size_bytes": os.path.getsize(smt_file),
        "declared_logic": None,
        "has_check_sat": False,
        "has_get_model": False,
        "declarations": [],
        "assertions": 0,
        "potential_issues": []
    }
    
    try:
        with open(smt_file, 'r') as f:
            content = f.read()
            
        lines = content.splitlines()
        
        # Check for logic declarations
        for line in lines:
            line = line.strip()
            if line.startswith("(set-logic "):
                analysis["declared_logic"] = line[len("(set-logic "):-1].strip()
            
            if "(check-sat)" in line:
                analysis["has_check_sat"] = True
                
            if "(get-model)" in line:
                analysis["has_get_model"] = True
                
            if line.startswith("(declare-"):
                analysis["declarations"].append(line)
                
            if line.startswith("(assert "):
                analysis["assertions"] += 1
        
        # Check for potential issues
        if not analysis["declared_logic"]:
            analysis["potential_issues"].append("No logic explicitly declared")
            
        if not analysis["has_check_sat"]:
            analysis["potential_issues"].append("Missing (check-sat) command")
            
        if not analysis["has_get_model"]:
            analysis["potential_issues"].append("Missing (get-model) command")
            
        # Check for Z3 specific directives
        if "(maximize " in content or "(minimize " in content:
            analysis["potential_issues"].append("Contains Z3-specific optimization directives")
            
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

def test_logic_compatibility(smt_file, logics_to_test=None):
    """
    Test an SMT file with different logics to find compatible ones.
    
    Args:
        smt_file: Path to the SMT file
        logics_to_test: List of logics to test (if None, use standard list)
        
    Returns:
        Dictionary with results for each logic tested
    """
    if logics_to_test is None:
        logics_to_test = [
            "QF_LIA",   # Linear Integer Arithmetic
            "QF_LRA",   # Linear Real Arithmetic
            "QF_UFLIA", # Uninterpreted Functions with Linear Integer Arithmetic
            "QF_IDL",   # Integer Difference Logic
            "QF_RDL",   # Real Difference Logic
            "QF_UF",    # Uninterpreted Functions
            "QF_BV",    # Bit Vectors
            "QF_AUFBV", # Arrays, Uninterpreted Functions, and Bit Vectors
            # Add more as needed
        ]
    
    # First run with default logic
    print(f"Testing {os.path.basename(smt_file)} with default logic...")
    default_result = run_z3_api_with_debug(smt_file)
    
    results = {"default": default_result}
    
    # Test with each specific logic
    for logic in logics_to_test:
        print(f"Testing {os.path.basename(smt_file)} with logic {logic}...")
        logic_result = run_z3_api_with_debug(smt_file, logic)
        results[logic] = logic_result
    
    return results

def save_debug_report(results, output_file):
    """
    Save detailed debug information to a file
    
    Args:
        results: Dictionary with results from testing logics
        output_file: Path to save the report
    """
    with open(output_file, 'w') as f:
        f.write(f"Z3 Logic Debug Report\n")
        f.write(f"===================\n\n")
        
        f.write(f"File: {results['default']['file']}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Logic Compatibility Summary:\n")
        f.write("==========================\n")
        
        for logic, result in results.items():
            status = result["status"]
            runtime = result.get("time", 0)
            error = result.get("error", "")
            
            if error:
                status_str = f"{status} - {error}"
            else:
                status_str = status
                
            f.write(f"{logic: <12}: {status_str: <30} ({runtime:.3f}s)\n")
        
        f.write("\n\nDetailed Debug Information:\n")
        f.write("==========================\n")
        
        for logic, result in results.items():
            f.write(f"\n\nLogic: {logic}\n")
            f.write("-" * (len(logic) + 7) + "\n")
            
            f.write(f"Status: {result['status']}\n")
            f.write(f"Time: {result.get('time', 0):.3f}s\n")
            
            if "error" in result:
                f.write(f"Error: {result['error']}\n")
                
            if "error_stage" in result:
                f.write(f"Error stage: {result['error_stage']}\n")
                
            if "debug_info" in result:
                f.write("\nDebug Trace:\n")
                for stage, info in result["debug_info"].items():
                    f.write(f"  * {stage}:\n")
                    for item in info:
                        f.write(f"      - {item}\n")
            
            if "model" in result and result["model"]:
                f.write("\nModel (truncated):\n")
                model_str = str(result["model"])
                f.write(model_str[:1000] + "..." if len(model_str) > 1000 else model_str)
                f.write("\n")
        
    print(f"Debug report saved to {output_file}")

def main():
    """
    Main function to debug Z3 API on SMT files.
    """
    if len(sys.argv) < 2:
        print("Usage: python z3_logic_debugger.py <smt_file_or_directory> [logic1,logic2,...]")
        print("Example: python z3_logic_debugger.py problem.smt2 QF_LIA,QF_BV")
        print("         python z3_logic_debugger.py smt_files_dir")
        return
    
    path = sys.argv[1]
    
    # Check if specific logics were provided
    custom_logics = None
    if len(sys.argv) > 2:
        custom_logics = sys.argv[2].split(',')
        print(f"Testing with custom logics: {custom_logics}")
    
    # Create output directory
    os.makedirs("z3_debug_reports", exist_ok=True)
    
    if os.path.isdir(path):
        # Process all SMT files in directory
        smt_files = glob.glob(os.path.join(path, "*.smt")) + glob.glob(os.path.join(path, "*.smt2"))
        if not smt_files:
            print(f"No SMT files found in {path}")
            return
            
        print(f"Found {len(smt_files)} SMT files to process")
        
        for smt_file in smt_files:
            print(f"\n\nProcessing {os.path.basename(smt_file)}...")
            
            # Analyze file structure
            analysis = analyze_smt_file_structure(smt_file)
            print(f"File analysis: {os.path.basename(smt_file)}, " 
                  f"size: {analysis['size_bytes']} bytes, "
                  f"declared logic: {analysis['declared_logic'] or 'None'}, "
                  f"assertions: {analysis['assertions']}")
            
            if analysis['potential_issues']:
                print("Potential issues:")
                for issue in analysis['potential_issues']:
                    print(f"  - {issue}")
            
            # Test logic compatibility
            results = test_logic_compatibility(smt_file, custom_logics)
            
            # Save detailed report
            base_name = os.path.basename(smt_file).replace('.', '_')
            report_file = os.path.join("z3_debug_reports", f"{base_name}_debug.txt")
            save_debug_report(results, report_file)
    else:
        # Process a single file
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
            
        # Analyze file structure
        analysis = analyze_smt_file_structure(path)
        print(f"File analysis: {os.path.basename(path)}, " 
              f"size: {analysis['size_bytes']} bytes, "
              f"declared logic: {analysis['declared_logic'] or 'None'}, "
              f"assertions: {analysis['assertions']}")
        
        if analysis['potential_issues']:
            print("Potential issues:")
            for issue in analysis['potential_issues']:
                print(f"  - {issue}")
        
        # Test logic compatibility
        results = test_logic_compatibility(path, custom_logics)
        
        # Save detailed report
        base_name = os.path.basename(path).replace('.', '_')
        report_file = os.path.join("z3_debug_reports", f"{base_name}_debug.txt")
        save_debug_report(results, report_file)

if __name__ == "__main__":
    main()
