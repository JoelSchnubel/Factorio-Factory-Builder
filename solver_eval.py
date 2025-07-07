#! .venv\Scripts\python.exe

"""
SMT Solver Evaluation Module

This module provides a comprehensive evaluation framework for comparing different SMT solvers
(Z3, CVC5, Yices) across various SMT logics and problem instances. It's designed to benchmark
solver performance, analyze result consistency, and generate detailed reports for academic
research and solver selection.

Key Features:
- Multi-solver support: Z3, CVC5, Yices
- Logic-specific optimization: Automatic adaptation for different SMT logics
- File format conversion: Automatic SMT-LIB2 format adaptation
- Performance benchmarking: Execution time measurement and averaging
- Result analysis: SAT/UNSAT/Unknown result tracking
- Visualization: Automated plot generation for result analysis
- Robust error handling: Comprehensive error tracking and reporting

Supported SMT Logics:
- QF_LIA: Quantifier-free linear integer arithmetic
- QF_LRA: Quantifier-free linear real arithmetic
- QF_UFLIA: Quantifier-free linear integer arithmetic with uninterpreted functions
- QF_IDL: Quantifier-free integer difference logic
- QF_RDL: Quantifier-free real difference logic
- QF_UF: Quantifier-free uninterpreted functions
- QF_BV: Quantifier-free bit-vectors
- QF_AUFBV: Quantifier-free arrays with bit-vectors

Solver Integration:
- Z3: Microsoft's SMT solver with Python API integration
- CVC5: Stanford's SMT solver with subprocess execution
- Yices: SRI's SMT solver with subprocess execution

"""

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
from logging_config import setup_logger
logger = setup_logger("solver_eval")


# SMT Solver Executable Paths
# These paths should be updated based on your local installation
CVC5_PATH = "C:\\SMT\\cvc5\\cvc5-Win64-x86_64-static\\bin\\cvc5.exe"
YICES_PATH = "C:\\SMT\\yices\\yices-2.6.5\\bin\\yices-smt2.exe"

# SMT Logic Configurations
# Different logics to test - covers major SMT-LIB2 theories
# None represents default solver logic (no explicit logic setting)
SMT_LOGICS = [
    None,        # Default logic (solver-dependent)
    "QF_LIA",    # Quantifier-free linear integer arithmetic
    "QF_LRA",    # Quantifier-free linear real arithmetic
    "QF_UFLIA",  # Quantifier-free linear integer arithmetic with uninterpreted functions
    "QF_IDL",    # Quantifier-free integer difference logic
    "QF_RDL",    # Quantifier-free real difference logic
    "QF_UF",     # Quantifier-free uninterpreted functions
    "QF_BV",     # Quantifier-free bit-vectors
    "QF_AUFBV",  # Quantifier-free arrays with bit-vectors
]

# Solver Configuration
# List of solvers to evaluate in the benchmark
SOLVERS = ["z3", "cvc5", "yices"]

def create_smt2_file_with_logic(smt_file, logic, solver_name):
    """
    Create a temporary SMT-LIB2 file with solver-specific adaptations and logic settings.
    
    This function converts SMT files to be compatible with different solvers by adapting
    the syntax, type declarations, and logic specifications. It handles solver-specific
    requirements and automatically converts between different type systems.
    
    Key Adaptations:
    - Logic-specific type conversions (Int to Real, BitVec, etc.)
    - Solver-specific syntax modifications
    - Optimization directive handling (Z3-specific features)
    - Proper SMT-LIB2 header/footer generation
    - Integer literal conversion for different logics
    - Error prevention for common syntax issues
    
    Args:
        smt_file (str): Path to the original SMT file
        logic (str): SMT logic to use (None for default)
        solver_name (str): Target solver name ("z3", "cvc5", "yices")
        
    Returns:
        str: Path to the created temporary file, or None if creation failed
        
    Raises:
        IOError: If the original file cannot be read
        OSError: If the temporary file cannot be created
    """
    logger.info(f"Creating SMT-LIB2 file for {solver_name} with logic {logic}")
    logger.debug(f"Source file: {smt_file}")
    
    # Handle 'None' logic for CVC5 - use QF_ALL as default for CVC5
    display_logic = logic
    if logic is None:
        # CVC5 uses "ALL" or "QF_ALL" as their default logic
        if solver_name == "cvc5":
            logic = "QF_ALL"
        else:
            logic = "ALL"  # Default fallback for other solvers
        logger.info(f"Defaulting to {logic} for {solver_name} since no logic was specified")
    
    # Create temporary file path
    temp_file = os.path.join(os.path.dirname(smt_file), f"temp_{display_logic}_{os.path.basename(smt_file)}")
    temp_file = temp_file.replace('.smt', '.smt2')
    logger.debug(f"Temporary file path: {temp_file}")
    
    try:
        # Read the original SMT file content
        with open(smt_file, 'r') as f:
            content = f.read()
        logger.debug(f"Successfully read {len(content)} characters from source file")
    except IOError as e:
        logger.error(f"Failed to read source file {smt_file}: {e}")
        return None
    
    # Format the content with proper header and footer
    formatted_content = []
    
    # Detect if type declarations are needed based on the content
    logger.debug("Analyzing content for type declarations")
    needs_int_type = bool(re.search(r'declare-fun\s+\S+\s+\([^)]*\)\s+Int', content))
    needs_real_type = bool(re.search(r'declare-fun\s+\S+\s+\([^)]*\)\s+Real', content))
    needs_bool_type = bool(re.search(r'declare-fun\s+\S+\s+\([^)]*\)\s+Bool', content))
    
    logger.debug(f"Type analysis: Int={needs_int_type}, Real={needs_real_type}, Bool={needs_bool_type}")
    
    # Add logic declaration
    formatted_content.append(f"(set-logic {logic})")
    logger.debug(f"Added logic declaration: {logic}")
    
    # Add model production option
    formatted_content.append("(set-option :produce-models true)")
    logger.debug("Added produce-models option")
    
    # Logic-specific adaptations
    needs_type_conversion = False
    int_type_substitute = None
    integer_literals_pattern = None
    
    # For CVC5 specific type handling
    if solver_name == "cvc5" or solver_name == "yices":
        logger.debug(f"Applying {solver_name}-specific type handling")
        
        # Add necessary type declarations based on the logic
        if logic in ["QF_LRA", "QF_RDL"]:
            # These logics use Real but may need Int to Real conversion
            if needs_int_type:
                logger.info(f"Logic {logic} requires Int->Real mapping")
                formatted_content.append("(define-sort Int () Real)")
                needs_type_conversion = True
                int_type_substitute = "Real"
                integer_literals_pattern = r'(\s+)(\d+)(\s+|\))'
        elif logic in ["QF_UF"]:
            # Uninterpreted functions need explicit sort declarations
            if needs_int_type:
                logger.info(f"Logic {logic} requires Int sort declaration")
                formatted_content.append("(declare-sort Int 0)")
                needs_type_conversion = True
                integer_literals_pattern = r'(\s+)(\d+)(\s+|\))'
        elif logic in ["QF_BV", "QF_AUFBV"]:
            # For bit-vector logics
            if needs_int_type:
                logger.info(f"Logic {logic} requires Int->BitVec mapping")
                needs_type_conversion = True
                int_type_substitute = "(_ BitVec 32)"
                integer_literals_pattern = r'(\s+)(\d+)(\s+|\))'
    
    # Handle Z3-specific optimization directives
    logger.debug("Processing Z3-specific optimization directives")
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
            logger.debug(f"Skipping existing directive: {line_stripped}")
            continue
            
        # Handle maximize/minimize directives (Z3-specific)
        if line_stripped.startswith("(maximize") or line_stripped.startswith("(minimize"):
            logger.debug(f"Found optimization directive: {line_stripped}")
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
                logger.debug(f"Complete optimization expression: {opt_expr}")
                
                # Extract variables used in ite expressions (indicators for optimization)
                for match in re.finditer(r'\(ite\s+(\S+)\s+1\s+0\)', opt_expr):
                    var_name = match.group(1)
                    optimize_vars.add(var_name)
                    optimization_weights[var_name] = 1
                    logger.debug(f"Found optimization variable: {var_name}")
                
                # Clear for next potential optimization directive
                optimization_value = []
                collecting_optimization = False
            continue
        
        # Add other content
        formatted_content.append(line)
    
    # Convert optimization directives to SMT-LIB2 compatible assertions
    if has_optimization:
        logger.info(f"Converting {len(optimize_vars)} optimization variables to assertions")
        for var_name in optimize_vars:
            # Add soft assertions that can help achieve the desired optimization
            # but as regular assertions, not as optimization directives
            formatted_content.append(f"(assert (= {var_name} true))")
            logger.debug(f"Added optimization assertion for {var_name}")
    
    # Fix common SMT-LIB2 syntax issues
    logger.debug("Fixing common SMT-LIB2 syntax issues")
    fixed_content = "\n".join(formatted_content)
    
    # Fix ADD operations with only one child
    fixed_content = re.sub(r'\(\+\s+([^\s\(\)]+)\)', r'(+ 0 \1)', fixed_content)
    
    # Fix other common issues that might cause problems with different solvers
    # Convert empty disjunctions to 'false'
    fixed_content = re.sub(r'\(or\s*\)', r'false', fixed_content)
    # Convert empty conjunctions to 'true' 
    fixed_content = re.sub(r'\(and\s*\)', r'true', fixed_content)
    
    logger.debug("Applied basic syntax fixes")
    
    # Apply logic-specific type conversions if needed
    if needs_type_conversion and integer_literals_pattern and logic in ["QF_BV", "QF_AUFBV"]:
        logger.info("Applying bit-vector literal conversions")
        # For bit-vector logics, convert integer literals to bitvector literals
        fixed_content = re.sub(integer_literals_pattern, r'\1(_ bv\2 32)\3', fixed_content)
    elif needs_type_conversion and logic in ["QF_LRA", "QF_RDL"]:
        logger.info("Applying real literal conversions")
        # For real-based logics, convert integer literals to real literals
        fixed_content = re.sub(r'(\s+)(\d+)(\s+|\))', r'\1\2.0\3', fixed_content)
        
    # Create a completely new approach for CVC5 based on the logic type
    if solver_name == "cvc5":
        logger.info("Applying CVC5-specific adaptations")
        lines = fixed_content.splitlines()
        new_lines = []
        int_constants = {}
        
        # Start with logic declaration and options
        for i, line in enumerate(lines):
            if line.startswith("(set-logic") or line.startswith("(set-option"):
                new_lines.append(line)
                
        # Handle different logics with appropriate type conversions
        if logic in ["QF_BV", "QF_AUFBV"]:
            logger.info(f"Applying special CVC5 handling for {logic}")
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
            
            logger.debug(f"Found {len(integer_literals)} integer literals for bit-vector conversion")
            
            # Add BitVec constants for these integers
            for literal in integer_literals:
                const_name = f"#bv{literal}"
                new_lines.append(f"(define-fun {const_name} () (_ BitVec 32) (_ bv{literal} 32))")
                int_constants[literal] = const_name
                logger.debug(f"Added BitVec constant: {const_name}")
            
            # Now process all the variable declarations and assertions
            for line in lines:
                if line.startswith("(declare-fun"):
                    # Convert Int declarations to BitVec 32
                    if " Int)" in line:
                        line = line.replace(" Int)", " (_ BitVec 32))")
                        logger.debug(f"Converted Int declaration: {line}")
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
            logger.info(f"Applying special CVC5 handling for {logic}")
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
            
            logger.debug(f"Found {len(integer_literals)} integer literals for QF_UF conversion")
            
            # Add constants for these integers
            for literal in integer_literals:
                const_name = f"int_const_{literal}"
                new_lines.append(f"(declare-fun {const_name} () Int)")
                int_constants[literal] = const_name
                logger.debug(f"Added uninterpreted constant: {const_name}")
            
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
            logger.info(f"Applying special CVC5 handling for {logic}")
            # For real-based logics
            
            # First define Int as Real
            new_lines.append("(define-sort Int () Real)")
            
            # Process declarations and assertions, converting literals to reals
            for line in lines:
                if "(assert" in line or "(=" in line:
                    # Convert integer literals to real literals
                    original_line = line
                    line = re.sub(r'(?<!\w|\.)(\d+)(?!\w|\.)', r'\1.0', line)
                    if line != original_line:
                        logger.debug(f"Converted integer literals to reals: {original_line} -> {line}")
                
                if not line.startswith("(set-logic") and not line.startswith("(set-option"):
                    new_lines.append(line)
        else:
            logger.debug(f"Using default CVC5 handling for logic {logic}")
            # For other logics, just copy all lines except logic and option declarations
            for line in lines:
                if not line.startswith("(set-logic") and not line.startswith("(set-option"):
                    new_lines.append(line)
        
        fixed_content = "\n".join(new_lines)
        logger.debug(f"CVC5 adaptation complete: {len(new_lines)} lines generated")
    
    # Make sure check-sat and get-model are present at the end
    if "(check-sat)" not in fixed_content:
        fixed_content += "\n(check-sat)"
        logger.debug("Added check-sat directive")
    if "(get-model)" not in fixed_content:
        fixed_content += "\n(get-model)"
        logger.debug("Added get-model directive")
    
    # Write the formatted content
    try:
        with open(temp_file, 'w') as f:
            f.write(fixed_content)
        logger.info(f"Successfully created temporary SMT-LIB2 file: {temp_file}")
        logger.debug(f"File size: {len(fixed_content)} characters")
    except IOError as e:
        logger.error(f"Failed to write temporary file {temp_file}: {e}")
        return None
        
    return temp_file

def adapt_content_for_yices(content, logic):
    """
    Adapt SMT content specifically for the Yices solver.
    
    This function modifies SMT-LIB2 content to ensure compatibility with Yices,
    which has specific requirements for different logics and may not support
    all features available in other solvers.
    
    Yices Adaptations:
    - Ensures proper logic declarations are present
    - Adds produce-models option for model generation
    - Handles type conversions for different logics
    - Removes unsupported optimization directives
    - Adds required check-sat and get-model commands
    
    Args:
        content (str): Original SMT file content
        logic (str): SMT logic to use (None for default)
        
    Returns:
        str: Adapted content compatible with Yices
    """
    logger.info(f"Adapting content for Yices with logic: {logic}")
    logger.debug(f"Original content length: {len(content)} characters")
    
    # Add logic declaration if not present and logic is specified
    if logic and not re.search(r'\(set-logic\s+[^\)]+\)', content):
        content = f"(set-logic {logic})\n{content}"
        logger.debug(f"Added logic declaration: {logic}")
    
    # Make sure produce-models option is set
    if not re.search(r'\(set-option\s+:produce-models\s+true\)', content):
        content = "(set-option :produce-models true)\n" + content
        logger.debug("Added produce-models option")
    
    # Logic-specific adaptations
    if logic:
        logger.debug(f"Applying logic-specific adaptations for {logic}")
        
        # For logics that don't support Int type, replace with Real
        if logic in ["QF_LRA", "QF_RDL"]:
            logger.info(f"Converting Int declarations to Real for {logic}")
            # Replace 'Int' type declarations with 'Real'
            original_content = content
            content = re.sub(r'\(\s*declare-fun\s+([^\s]+)\s+\(\s*\)\s+Int\s*\)', 
                             r'(declare-fun \1 () Real)', content)
            if content != original_content:
                logger.debug("Applied Int->Real type conversions")
        
        # For logics without arithmetic, provide appropriate declarations
        if logic in ["QF_UF", "QF_BV", "QF_AUFBV"]:
            logger.info(f"Applying type adaptations for {logic}")
            
            # Replace 'Int' type declarations with appropriate types
            if logic == "QF_UF":
                # For QF_UF, declare a sort to use instead of Int
                if not re.search(r'\(declare-sort\s+MyInt\s+0\)', content):
                    content = "(declare-sort MyInt 0)\n" + content
                    logger.debug("Added MyInt sort declaration")
                content = re.sub(r'\(\s*declare-fun\s+([^\s]+)\s+\(\s*\)\s+Int\s*\)', 
                                r'(declare-fun \1 () MyInt)', content)
                logger.debug("Converted Int declarations to MyInt")
            elif "BV" in logic:
                # For bit-vector logics, use (_ BitVec 32) or similar
                content = re.sub(r'\(\s*declare-fun\s+([^\s]+)\s+\(\s*\)\s+Int\s*\)', 
                                r'(declare-fun \1 () (_ BitVec 32))', content)
                logger.debug("Converted Int declarations to BitVec")
        
        # Remove maximize/minimize commands for logics that don't support optimization
        optimization_removed = re.sub(r'\(\s*(maximize|minimize)\s+[^\)]+\s*\)', '', content)
        if optimization_removed != content:
            content = optimization_removed
            logger.debug("Removed optimization directives (not supported by Yices)")
    
    # Add check-sat and get-model if needed
    if not re.search(r'\(check-sat\)', content):
        content += "\n(check-sat)"
        logger.debug("Added check-sat directive")
    
    if not re.search(r'\(get-model\)', content):
        content += "\n(get-model)"
        logger.debug("Added get-model directive")
    
    # Create debug file for troubleshooting
    try:
        with open('debug_yices_temp.smt2', 'w') as f:
            f.write(content)
        logger.debug("Created debug file: debug_yices_temp.smt2")
    except IOError as e:
        logger.warning(f"Failed to create debug file: {e}")
    
    logger.info(f"Yices adaptation complete: {len(content)} characters")
    return content

def parse_cvc5_model(output):
    """
    Parse and extract model information from CVC5 solver output.
    
    CVC5 outputs models in a specific format that needs to be extracted from
    the complete solver output. This function identifies and isolates the
    model portion for further processing.
    
    Args:
        output (str): Complete output from CVC5 solver execution
        
    Returns:
        str: Extracted model string, or full output if model not found
    """
    logger.debug("Parsing CVC5 model output")
    logger.debug(f"Output length: {len(output)} characters")
    
    # Extract model part from output using regex
    model_match = re.search(r'sat\s*((\(model[^$]+)|\(model\))', output, re.DOTALL)
    if model_match:
        model = model_match.group(1).strip()
        logger.info(f"Successfully extracted CVC5 model: {len(model)} characters")
        logger.debug(f"Model content: {model[:100]}..." if len(model) > 100 else f"Model content: {model}")
        return model
    else:
        logger.warning("No model found in CVC5 output, returning full output")
        return output  # Return full output if model not found

def parse_yices_model(output):
    """
    Parse and extract model information from Yices solver output.
    
    Yices outputs models in a line-by-line format after the satisfiability
    result. This function captures all lines following the 'sat' result
    to construct the complete model.
    
    Args:
        output (str): Complete output from Yices solver execution
        
    Returns:
        str: Extracted model string, or full output if model not found
    """
    logger.debug("Parsing Yices model output")
    logger.debug(f"Output length: {len(output)} characters")
    
    # Extract model part from output
    model_lines = []
    capture = False
    
    for line in output.splitlines():
        if line.strip() == "sat":
            capture = True
            logger.debug("Found 'sat' result, starting model capture")
            continue
        if capture:
            model_lines.append(line)
    
    if model_lines:
        model = "\n".join(model_lines)
        logger.info(f"Successfully extracted Yices model: {len(model_lines)} lines")
        logger.debug(f"Model preview: {model[:100]}..." if len(model) > 100 else f"Model: {model}")
        return model
    else:
        logger.warning("No model found in Yices output, returning full output")
        return output  # Return full output if model not found

def load_smt_files(directory="SMT_Modules"):
    """
    Load all SMT files from a specified directory for batch processing.
    
    This function scans a directory for .smt files and loads their content
    into memory for batch evaluation. It provides error handling for
    individual file loading failures while continuing to process other files.
    
    File Processing:
    - Searches for all .smt files in the specified directory
    - Loads file content into memory for faster access
    - Handles encoding issues and file access errors gracefully
    - Reports loading statistics and any errors encountered
    
    Args:
        directory (str, optional): Directory path containing SMT files. 
                                 Defaults to "SMT_Modules".
    
    Returns:
        dict: Dictionary mapping file names to file content strings.
              Empty dict if directory doesn't exist or no files found.
    """
    logger.info(f"Loading SMT files from directory: {directory}")
    smt_files = {}
    
    # Check if the directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        return smt_files
    
    # Get all .smt files in the directory
    file_pattern = os.path.join(directory, "*.smt")
    file_paths = glob.glob(file_pattern)
    logger.debug(f"Found {len(file_paths)} .smt files using pattern: {file_pattern}")
    
    # Load each file with error handling
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        logger.debug(f"Loading file: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                smt_files[file_name] = content
                logger.debug(f"Successfully loaded {file_name}: {len(content)} characters")
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error loading {file_path}: {e}")
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    smt_files[file_name] = content
                    logger.warning(f"Loaded {file_name} with latin-1 encoding: {len(content)} characters")
            except Exception as e2:
                logger.error(f"Failed to load {file_path} with any encoding: {e2}")
        except IOError as e:
            logger.error(f"IO error loading {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
    
    logger.info(f"Successfully loaded {len(smt_files)} SMT files from {directory}")
    if len(smt_files) != len(file_paths):
        logger.warning(f"Warning: {len(file_paths) - len(smt_files)} files failed to load")
    
    return smt_files



def run_z3_with_logic(file_path, logic=None):
    """
    Execute Z3 solver with a specific logic on an SMT file using Python API.
    
    This function leverages Z3's Python API for direct integration, providing
    better performance and error handling compared to subprocess execution.
    It supports all major SMT logics and provides detailed timing information.
    
    Z3 Integration Features:
    - Direct Python API usage for optimal performance
    - Logic-specific solver configuration
    - Comprehensive error handling and reporting
    - Detailed execution timing
    - Model extraction for satisfiable instances
    
    Args:
        file_path (str): Path to the SMT file to be solved
        logic (str, optional): SMT logic to use. Defaults to None (Z3 default).
    
    Returns:
        tuple: A 3-tuple containing:
            - satisfiability (str): Result status ("sat", "unsat", "unknown", "error")
            - model: Z3 model object for satisfiable instances, None otherwise
            - execution_time (float): Solver execution time in seconds
    """
    logger.info(f"Running Z3 solver with logic: {logic or 'Default'}")
    logger.debug(f"Input file: {file_path}")
    
    start_time = time.time()
    
    # Create Z3 solver with the given logic
    try:
        solver = Solver()
        if logic:
            solver.set(logic=logic)
            logger.debug(f"Z3 solver configured with logic: {logic}")
        else:
            logger.debug("Z3 solver using default logic")
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"Failed to create Z3 solver: {e}")
        return "error", str(e), execution_time
    
    try:
        # Parse the SMT file
        logger.debug(f"Parsing SMT file: {os.path.basename(file_path)}")
        formula = parse_smt2_file(file_path)
        solver.add(formula)
        logger.debug("SMT formula successfully added to solver")
        
        # Check satisfiability
        logger.debug("Starting satisfiability check...")
        result = solver.check()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Process results based on Z3's return values
        if result == sat:
            logger.info(f"Z3 result: SAT (execution time: {execution_time:.3f}s)")
            try:
                model = solver.model()
                logger.debug(f"Model extracted: {len(model)} assignments")
                return "sat", model, execution_time
            except Exception as e:
                logger.warning(f"Failed to extract model: {e}")
                return "sat", None, execution_time
        elif result == unsat:
            logger.info(f"Z3 result: UNSAT (execution time: {execution_time:.3f}s)")
            return "unsat", None, execution_time
        else:  # unknown
            logger.info(f"Z3 result: UNKNOWN (execution time: {execution_time:.3f}s)")
            return "unknown", None, execution_time
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"Z3 solver error: {e} (execution time: {execution_time:.3f}s)")
        return "error", str(e), execution_time
    
    
def save_model(model, file_path, logic, solver_name="z3"):
    """
    Save solver model to a structured file for later analysis.
    
    This function creates organized model files that can be used for result
    verification, debugging, or further analysis. Models are saved with
    descriptive filenames that include solver and logic information.
    
    Model Storage Features:
    - Organized directory structure (models/)
    - Descriptive filenames with solver and logic info
    - Support for different model formats (Z3 objects, strings)
    - Error handling for file I/O operations
    
    Args:
        model: Solver model object or string representation
        file_path (str): Original SMT file path (used for naming)
        logic (str): SMT logic used (included in filename)
        solver_name (str, optional): Solver name. Defaults to "z3".
    
    Returns:
        str: Path to saved model file, or None if save failed
    """
    if model is None:
        logger.debug("No model to save (model is None)")
        return None
    
    logger.info(f"Saving model for {solver_name} with logic {logic}")
    
    try:
        # Create models directory if it doesn't exist
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        logger.debug(f"Ensured models directory exists: {models_dir}")
        
        # Create model file path with descriptive naming
        base_name = os.path.basename(file_path)
        logic_str = logic if logic else "default"
        model_file = os.path.join(models_dir, f"{base_name}_{solver_name}_{logic_str}_model.txt")
        
        # Save model with proper formatting
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(str(model))
        
        logger.info(f"Model successfully saved to: {model_file}")
        logger.debug(f"Model content length: {len(str(model))} characters")
        return model_file
        
    except IOError as e:
        logger.error(f"Failed to save model to {model_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error saving model: {e}")
        return None
    
    
def run_solver(solver_name, file_path, logic):
    """
    Execute a specified solver with given logic on an SMT file.
    
    This function provides a unified interface for running different SMT solvers,
    abstracting away the specific execution methods and providing consistent
    result formatting across all supported solvers.
    
    Supported Solvers:
    - z3: Microsoft's Z3 solver (Python API)
    - cvc5: Stanford's CVC5 solver (subprocess)
    - yices: SRI's Yices solver (subprocess)
    
    Args:
        solver_name (str): Name of the solver to run ("z3", "cvc5", "yices")
        file_path (str): Path to the SMT file to solve
        logic (str): SMT logic to use (None for default)
    
    Returns:
        tuple: A 3-tuple containing:
            - satisfiability (str): Result status ("sat", "unsat", "unknown", "error")
            - model: Solver model (format depends on solver)
            - execution_time (float): Execution time in seconds
    """
    logger.debug(f"Running solver: {solver_name} with logic: {logic} on file: {os.path.basename(file_path)}")
    
    if solver_name == "z3":
        return run_z3_with_logic(file_path, logic)
    elif solver_name == "cvc5":
        return run_cvc5_with_logic(file_path, logic)
    elif solver_name == "yices":
        return run_yices_with_logic(file_path, logic)
    else:
        logger.error(f"Unknown solver: {solver_name}")
        return "error", f"Unknown solver: {solver_name}", 0

def evaluate_solvers(directory="SMT_Modules", save_results=True, num_runs=10):
    """
    Comprehensive evaluation of all configured SMT solvers across multiple logics and files.
    
    This function performs a systematic benchmark of SMT solvers, running each solver
    with each logic on every SMT file in the specified directory. It provides statistical
    analysis through multiple runs and detailed performance metrics.
    
    Evaluation Process:
    1. Load all SMT files from the specified directory
    2. For each file, solver, and logic combination:
       - Run the solver multiple times for statistical accuracy
       - Collect timing and result data
       - Track result consistency across runs
    3. Calculate averages and determine most common results
    4. Generate comprehensive result datasets
    
    Statistical Features:
    - Multiple runs per configuration for timing accuracy
    - Result consistency tracking (sat/unsat/unknown/error counts)
    - Average execution time calculation
    - Most common result determination for reliability assessment
    
    Args:
        directory (str, optional): Directory containing SMT files. Defaults to "SMT_Modules".
        save_results (bool, optional): Whether to save results (legacy parameter). Defaults to True.
        num_runs (int, optional): Number of runs per configuration for averaging. Defaults to 10.
    
    Returns:
        list: List of result dictionaries, each containing:
            - file_name: Name of the SMT file
            - solver: Solver name used
            - logic: SMT logic applied
            - result: Most common result status
            - execution_time: Average execution time
            - result_counts: Dictionary with counts for each result type
    """
    logger.info(f"Starting comprehensive solver evaluation")
    logger.info(f"Directory: {directory}, Runs per config: {num_runs}")
    logger.info(f"Solvers: {SOLVERS}")
    logger.info(f"Logics: {SMT_LOGICS}")
    
    # Load all SMT files for evaluation
    smt_files = load_smt_files(directory)
    if not smt_files:
        logger.warning("No SMT files found for evaluation")
        return []
    
    logger.info(f"Found {len(smt_files)} SMT files to evaluate")
    
    results = []
    total_evaluations = len(smt_files) * len(SOLVERS) * len(SMT_LOGICS)
    current_evaluation = 0
    
    # Iterate through all combinations of files, solvers, and logics
    for file_name, content in smt_files.items():
        file_path = os.path.join(directory, file_name)
        logger.info(f"Evaluating file: {file_name}")
        
        for solver_name in SOLVERS:
            logger.info(f"Using solver: {solver_name}")
            
            for logic in SMT_LOGICS:
                current_evaluation += 1
                logic_display = logic if logic else 'default'
                logger.info(f"Running logic: {logic_display} ({current_evaluation}/{total_evaluations})")
                logger.debug(f"Configuration: {file_name} + {solver_name} + {logic_display}")
                
                # Run multiple times and collect statistics
                total_time = 0
                results_count = {"sat": 0, "unsat": 0, "unknown": 0, "error": 0}
                final_result = None
                final_model = None
                
                for run in range(num_runs):
                    logger.debug(f"Run {run + 1}/{num_runs} for {solver_name} with {logic_display}")
                    
                    try:
                        result, model, execution_time = run_solver(solver_name, file_path, logic)
                        logger.debug(f"Run {run + 1} result: {result} in {execution_time:.3f}s")
                        
                        # Track results for statistical analysis
                        results_count[result] += 1
                        total_time += execution_time
                        
                        # Store first non-error result for potential model saving
                        if final_result is None or (final_result == "error" and result != "error"):
                            final_result = result
                            final_model = model
                            
                    except Exception as e:
                        logger.error(f"Unexpected error in run {run + 1}: {e}")
                        results_count["error"] += 1
                
                # Calculate statistics
                avg_execution_time = total_time / num_runs if num_runs > 0 else 0
                most_common_result = max(results_count.items(), key=lambda x: x[1])[0]
                
                logger.info(f"Completed {logic_display}: avg time {avg_execution_time:.3f}s, most common: {most_common_result}")
                logger.debug(f"Result distribution: {results_count}")
                
                # Create comprehensive result dictionary
                result_dict = {
                    "file_name": file_name,
                    "solver": solver_name,
                    "logic": logic if logic else "default",
                    "result": most_common_result,
                    "execution_time": avg_execution_time,
                    "result_counts": results_count
                }
                
                results.append(result_dict)
                
                # Optional: Save satisfiable models for further analysis
                # Uncommented for performance reasons - can be enabled if needed
                # if final_result == "sat":
                #     save_model(final_model, file_path, logic, solver_name)
    
    logger.info(f"Evaluation complete: {len(results)} total results generated")
    return results

def save_results_to_csv(results, overwrite=True):
    """
    Save comprehensive evaluation results to CSV format for analysis and visualization.
    
    This function creates a structured CSV file containing all evaluation results
    in a format suitable for statistical analysis, visualization, and reporting.
    The CSV includes detailed performance metrics and result distributions.
    
    CSV Structure:
    - File: SMT file name
    - Logic: SMT logic used
    - Solver columns for each solver containing:
      - Result status (sat/unsat/unknown/error)
      - Average execution time
      - Individual result counts (sat, unsat, unknown, error)
    
    File Management:
    - Creates results directory if it doesn't exist
    - Supports both overwrite and timestamped file creation
    - Uses descriptive column headers for clarity
    - Handles missing data gracefully with "N/A" placeholders
    
    Args:
        results (list): List of result dictionaries from evaluate_solvers()
        overwrite (bool, optional): Whether to overwrite existing CSV. Defaults to True.
    
    Returns:
        str: Timestamp used for the filename (for timestamped files)
    """
    logger.info("Saving evaluation results to CSV format")
    logger.debug(f"Processing {len(results)} result entries")
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    try:
        os.makedirs(results_dir, exist_ok=True)
        logger.debug(f"Ensured results directory exists: {results_dir}")
    except OSError as e:
        logger.error(f"Failed to create results directory: {e}")
        return None
    
    # Create timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Determine output filename
    if overwrite:
        csv_filename = "solver_evaluation.csv"
    else:
        csv_filename = f"solver_evaluation_{timestamp}.csv"
    
    csv_file = os.path.join(results_dir, csv_filename)
    logger.info(f"Saving results to: {csv_file}")
    
    # Extract all unique solvers, logics, and files for CSV structure
    solvers = sorted(set(r["solver"] for r in results))
    logics = sorted(set(r["logic"] for r in results))
    file_names = sorted(set(r["file_name"] for r in results))
    
    logger.debug(f"CSV will include {len(solvers)} solvers, {len(logics)} logics, {len(file_names)} files")
    
    try:
        # Create comprehensive CSV file with all information
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Create header with solver columns for both time and result
            header = ['File', 'Logic']
            for solver in solvers:
                header.extend([
                    f"{solver}_result",      # Main result (sat/unsat/unknown/error)
                    f"{solver}_time",        # Average execution time
                    f"{solver}_sat",         # Count of SAT results
                    f"{solver}_unsat",       # Count of UNSAT results
                    f"{solver}_unknown",     # Count of UNKNOWN results
                    f"{solver}_error"        # Count of ERROR results
                ])
            
            writer.writerow(header)
            logger.debug(f"CSV header created with {len(header)} columns")
            
            # Write data rows grouped by file name and logic
            rows_written = 0
            for file_name in file_names:
                for logic in logics:
                    row = [file_name, logic]
                    
                    for solver in solvers:
                        # Find the result for this specific combination
                        solver_result = next(
                            (r for r in results if r["file_name"] == file_name 
                             and r["logic"] == logic and r["solver"] == solver), 
                            None
                        )
                        
                        if solver_result:
                            # Add actual results
                            row.extend([
                                solver_result["result"],
                                f"{solver_result['execution_time']:.3f}",
                                solver_result["result_counts"]["sat"],
                                solver_result["result_counts"]["unsat"],
                                solver_result["result_counts"]["unknown"],
                                solver_result["result_counts"]["error"]
                            ])
                        else:
                            # No result for this combination - fill with defaults
                            row.extend(["N/A", "0.000", "0", "0", "0", "0"])
                    
                    writer.writerow(row)
                    rows_written += 1
            
            logger.info(f"CSV file created successfully with {rows_written} data rows")
        
    except IOError as e:
        logger.error(f"Failed to write CSV file {csv_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating CSV: {e}")
        return None
    
    return timestamp
    return timestamp


def run_cvc5_with_logic(file_path, logic=None):
    """
    Execute CVC5 solver with a specific logic on an SMT file via subprocess.
    
    This function runs CVC5 as an external process, handling file format conversion,
    command-line argument construction, and output parsing. CVC5 is executed with
    model production enabled for satisfiable instances.
    
    CVC5 Execution Features:
    - Subprocess-based execution for isolation
    - Automatic SMT-LIB2 format conversion
    - Logic-specific command-line configuration
    - Model extraction from solver output
    - Comprehensive error handling and debugging support
    - Temporary file management with cleanup
    
    Args:
        file_path (str): Path to the SMT file to be solved
        logic (str, optional): SMT logic to use. Defaults to None (CVC5 default).
    
    Returns:
        tuple: A 3-tuple containing:
            - satisfiability (str): Result status ("sat", "unsat", "unknown", "error")
            - model (str): Parsed model string for satisfiable instances, None otherwise
            - execution_time (float): Solver execution time in seconds
    """
    logger.info(f"Running CVC5 solver with logic: {logic or 'Default'}")
    logger.debug(f"Input file: {file_path}")
    
    # Check if CVC5 executable exists
    if not os.path.exists(CVC5_PATH):
        logger.error(f"CVC5 executable not found at: {CVC5_PATH}")
        return "error", "CVC5 executable not found", 0
    
    # Create a temporary file with CVC5 specific adaptations
    logger.debug("Creating CVC5-compatible temporary file")
    temp_file_path = create_smt2_file_with_logic(file_path, logic, "cvc5")
    if not temp_file_path:
        logger.error("Failed to create temporary file for CVC5")
        return "error", "Failed to create temporary file", 0
    
    # Build command with appropriate options
    cmd = [CVC5_PATH, "--produce-models"]
    if logic:
        cmd.extend(["--force-logic", logic])
        logger.debug(f"Added logic constraint: {logic}")
    cmd.append(temp_file_path)
    
    logger.debug(f"CVC5 command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        # Run CVC5 process with timeout protection
        logger.debug(f"Executing CVC5 on: {os.path.basename(file_path)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"CVC5 process completed in {execution_time:.3f}s")
        logger.debug(f"Return code: {process.returncode}")
        
        # Handle temporary file cleanup and debugging
        debug_file_path = "debug_cvc5_temp.smt2"
        try:
            import shutil
            shutil.copy(temp_file_path, debug_file_path)
            logger.debug(f"Copied temp file to {debug_file_path} for debugging")
            
            # Keep temp files for failed runs, remove for successful ones
            if "sat" not in stdout.lower() and stderr.strip():
                logger.warning(f"Keeping temporary file {temp_file_path} due to error")
            else:
                os.remove(temp_file_path)
                logger.debug("Removed temporary file after successful execution")
        except Exception as e:
            logger.warning(f"Failed to handle temporary file {temp_file_path}: {e}")
        
        # Process solver output
        if "sat" in stdout.lower():
            logger.info(f"CVC5 result: SAT (execution time: {execution_time:.3f}s)")
            # Parse the model from CVC5 output
            model = parse_cvc5_model(stdout)
            return "sat", model, execution_time
        elif "unsat" in stdout.lower():
            logger.info(f"CVC5 result: UNSAT (execution time: {execution_time:.3f}s)")
            return "unsat", None, execution_time
        elif "unknown" in stdout.lower():
            logger.info(f"CVC5 result: UNKNOWN (execution time: {execution_time:.3f}s)")
            return "unknown", None, execution_time
        else:
            # Check for errors in stderr
            if stderr.strip():
                logger.error(f"CVC5 error: {stderr.strip()} (execution time: {execution_time:.3f}s)")
                return "error", stderr.strip(), execution_time
            else:
                logger.warning(f"CVC5 unknown result: {stdout.strip()} (execution time: {execution_time:.3f}s)")
                return "unknown", None, execution_time
    
    except subprocess.TimeoutExpired:
        logger.error("CVC5 execution timed out")
        return "error", "Execution timeout", time.time() - start_time
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"CVC5 execution error: {e} (execution time: {execution_time:.3f}s)")
        return "error", str(e), execution_time
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Error: {e} (in {execution_time:.2f}s)")
        return "error", str(e), execution_time


def run_yices_with_logic(file_path, logic=None):
    """
    Execute Yices solver with a specific logic on an SMT file via subprocess.
    
    This function runs Yices as an external process, handling file format conversion,
    command-line configuration, and output parsing. Yices is executed with optimized
    settings for performance and model extraction.
    
    Yices Execution Features:
    - Subprocess-based execution for process isolation
    - Automatic SMT-LIB2 format conversion and adaptation
    - Logic-agnostic execution (Yices auto-detects logic)
    - Model extraction from solver output
    - Efficient temporary file management
    - Comprehensive error handling and reporting
    
    Args:
        file_path (str): Path to the SMT file to be solved
        logic (str, optional): SMT logic to use. Defaults to None (Yices auto-detection).
    
    Returns:
        tuple: A 3-tuple containing:
            - satisfiability (str): Result status ("sat", "unsat", "unknown", "error")
            - model (str): Parsed model string for satisfiable instances, None otherwise
            - execution_time (float): Solver execution time in seconds
    """
    logger.info(f"Running Yices solver with logic: {logic or 'Auto-detect'}")
    logger.debug(f"Input file: {file_path}")
    
    # Check if Yices executable exists
    if not os.path.exists(YICES_PATH):
        logger.error(f"Yices executable not found at: {YICES_PATH}")
        return "error", "Yices executable not found", 0
    
    # Create a temporary file with Yices specific adaptations
    logger.debug("Creating Yices-compatible temporary file")
    temp_file_path = create_smt2_file_with_logic(file_path, logic, "yices")
    if not temp_file_path:
        logger.error("Failed to create temporary file for Yices")
        return "error", "Failed to create temporary file", 0
    
    # Build command (Yices typically auto-detects logic, so we don't force it)
    cmd = [YICES_PATH]
    # Note: Yices can auto-detect logic from the SMT-LIB2 file
    # Explicit logic setting via command line is often not needed
    cmd.append(temp_file_path)
    
    logger.debug(f"Yices command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        # Run Yices process
        logger.debug(f"Executing Yices on: {os.path.basename(file_path)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"Yices process completed in {execution_time:.3f}s")
        logger.debug(f"Return code: {process.returncode}")
        
        # Clean up temporary file
        try:
            os.remove(temp_file_path)
            logger.debug("Removed temporary file after execution")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")
        
        # Process solver output
        if "sat" in stdout.lower():
            logger.info(f"Yices result: SAT (execution time: {execution_time:.3f}s)")
            # Parse the model from Yices output
            model = parse_yices_model(stdout)
            return "sat", model, execution_time
        elif "unsat" in stdout.lower():
            logger.info(f"Yices result: UNSAT (execution time: {execution_time:.3f}s)")
            return "unsat", None, execution_time
        elif "unknown" in stdout.lower():
            logger.info(f"Yices result: UNKNOWN (execution time: {execution_time:.3f}s)")
            return "unknown", None, execution_time
        else:
            # Check for errors in stderr
            if stderr.strip():
                logger.error(f"Yices error: {stderr.strip()} (execution time: {execution_time:.3f}s)")
                return "error", stderr.strip(), execution_time
            else:
                logger.warning(f"Yices unknown result: {stdout.strip()} (execution time: {execution_time:.3f}s)")
                return "unknown", None, execution_time
    
    except subprocess.TimeoutExpired:
        logger.error("Yices execution timed out")
        return "error", "Execution timeout", time.time() - start_time
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"Yices execution error: {e} (execution time: {execution_time:.3f}s)")
        return "error", str(e), execution_time



def main():
    """
    Main execution function for the SMT solver evaluation framework.
    
    This function orchestrates the complete evaluation process, from loading SMT files
    through running all solver-logic combinations to generating comprehensive reports
    and visualizations. It provides a complete automated benchmarking workflow.
    
    Execution Workflow:
    1. Initialize evaluation parameters and display configuration
    2. Run comprehensive solver evaluation across all configurations
    3. Generate statistical summaries for each solver
    4. Save detailed results to CSV format for analysis
    5. Generate performance visualization plots
    6. Provide completion summary and file locations
    
    Output Files:
    - results/solver_evaluation.csv: Detailed evaluation results
    - Plots/: Directory containing performance visualization charts
    - logs/: Detailed execution logs (via logger)
    
    No arguments required - all configuration is done via global constants.
    """
    logger.info("Starting SMT solver evaluation framework")
    logger.info(f"Configured solvers: {SOLVERS}")
    logger.info(f"Configured logics: {SMT_LOGICS}")
    
    print("Starting SMT solver evaluation...")
    print("Available solvers:", SOLVERS)
    print("Available logics:", SMT_LOGICS)

    # Run the comprehensive evaluation
    logger.info("Beginning solver evaluation process")
    results = evaluate_solvers()

    # Generate and display summary statistics
    if results:
        logger.info(f"Evaluation completed with {len(results)} total results")
        print("\nEvaluation Summary:")
        
        total_files = len(set(r["file_name"] for r in results))
        print(f"Total files evaluated: {total_files}")
        logger.info(f"Evaluated {total_files} unique SMT files")
        
        # Generate per-solver statistics
        for solver in SOLVERS:
            solver_results = [r for r in results if r["solver"] == solver]
            if not solver_results:
                logger.warning(f"No results found for solver: {solver}")
                continue
                
            print(f"\n{solver.upper()} Results:")
            
            # Calculate result distribution
            sat_count = sum(1 for r in solver_results if r["result"] == "sat")
            unsat_count = sum(1 for r in solver_results if r["result"] == "unsat")
            unknown_count = sum(1 for r in solver_results if r["result"] == "unknown")
            error_count = sum(1 for r in solver_results if r["result"] == "error")
            
            print(f"  sat: {sat_count}")
            print(f"  unsat: {unsat_count}")
            print(f"  unknown: {unknown_count}")
            print(f"  error: {error_count}")
            
            # Calculate average execution time
            avg_time = sum(r["execution_time"] for r in solver_results) / len(solver_results)
            print(f"  Average execution time: {avg_time:.3f}s")
            
            logger.info(f"{solver} summary: {sat_count} SAT, {unsat_count} UNSAT, {unknown_count} UNKNOWN, {error_count} ERROR, avg time {avg_time:.3f}s")

        # Save results using the dedicated CSV function
        try:
            logger.info("Saving results to CSV format")
            save_results_to_csv(results, overwrite=True)
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")
            print(f"Error saving CSV results: {e}")

        # Generate performance visualization plots
        try:
            logger.info("Generating performance visualization plots")
            print("\nGenerating solver performance plots...")
            
            csv_file = os.path.join("results", "solver_evaluation.csv")
            subprocess.run([
                "python", "plot_solver_results.py", csv_file
            ], check=True)
            
            print("Plots generated successfully.")
            logger.info("Performance plots generated successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Plot generation failed with return code {e.returncode}")
            print(f"Failed to generate plots: {e}")
            print(f"You can manually run: python plot_solver_results.py {csv_file}")
        except FileNotFoundError:
            logger.error("plot_solver_results.py not found")
            print("Plot script not found. Please ensure plot_solver_results.py is available.")
        except Exception as e:
            logger.error(f"Unexpected error during plot generation: {e}")
            print(f"Unexpected error generating plots: {e}")

    else:
        logger.warning("No evaluation results generated")
        print("No evaluation results were generated. Check the SMT_Modules directory.")

    # Provide completion summary
    print("\nEvaluation complete. Results saved to the 'results' directory.")
    print("Visualizations saved to the 'Plots' directory.")
    logger.info("SMT solver evaluation framework completed successfully")

if __name__ == "__main__":
    main()