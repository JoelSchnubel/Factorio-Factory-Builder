#! .venv\Scripts\python.exe

import os
import re
import shutil

def simplify_smt_file(input_file, output_file=None, logic="QF_LIA"):
    if output_file is None:
        output_file = input_file + ".simplified.smt2"

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    header = [
        f"(set-logic {logic})",
        "(set-option :produce-models true)"
    ]
   
    declarations = []
    for line in content.splitlines():
        if line.strip().startswith("(declare-"):
            declarations.append(line.strip())
    
    assertions = []
    skip_line = False
    inside_optimization = False
    for line in content.splitlines():
        line_stripped = line.strip()
        
        if line_stripped.startswith("(maximize") or line_stripped.startswith("(minimize"):
            inside_optimization = True
            skip_line = True
            continue
        
        if inside_optimization:
        
            open_count = line_stripped.count("(")
            close_count = line_stripped.count(")")
            
            if close_count > open_count:  
                inside_optimization = False
            
            skip_line = True
            continue
        
        # Include assertions but not optimizations
        if line_stripped.startswith("(assert") and not skip_line:
            assertions.append(line_stripped)
    
    # Add check-sat and get-model
    footer = [
        "(check-sat)",
        "(get-model)"
    ]
    
    # Write the simplified file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(header))
        f.write("\n")
        f.write("\n".join(declarations))
        f.write("\n")
        f.write("\n".join(assertions))
        f.write("\n")
        f.write("\n".join(footer))
    
    print(f"Created simplified SMT file at {output_file}")
    return output_file

def fix_common_syntax_issues(input_file, output_file=None):
    """Fix common syntax issues in SMT-LIB2 files that cause problems with some solvers."""
    if output_file is None:
        output_file = input_file + ".fixed.smt2"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix various syntax issues
    # 1. Fix empty operators
    content = re.sub(r'\(or\s*\)', r'false', content)
    content = re.sub(r'\(and\s*\)', r'true', content)
    
    # 2. Fix ADD operations with only one child
    content = re.sub(r'\(\+\s+([^\s\(\)]+)\)', r'(+ 0 \1)', content)
    
    # 3. Replace Z3-specific functions like minimize/maximize with comments
    content = re.sub(r'\(maximize.*?\)', r'; Z3-specific optimization removed', content, flags=re.DOTALL)
    content = re.sub(r'\(minimize.*?\)', r'; Z3-specific optimization removed', content, flags=re.DOTALL)
    
    # 4. Fix variable names with characters not compatible with all solvers
    # Use a more careful regex that avoids replacing | inside string literals
    varcounter = 0
    def replace_var(match):
        nonlocal varcounter
        varcounter += 1
        # Keep common characters, replace problematic ones
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', match.group(1))
        return f"var_{safe_name}_{varcounter}"
    
    content = re.sub(r'\|([^|]+)\|', replace_var, content)
    
    # Write the fixed file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed SMT file syntax at {output_file}")
    return output_file

def create_solver_specific_versions(smt_file, logics=None):
    if logics is None:
        logics = ["QF_LIA"]
    
    base_dir = os.path.dirname(smt_file)
    base_name = os.path.basename(smt_file).replace(".smt", "").replace(".smt2", "")
    
    # Create solver-specific directory
    solver_dir = os.path.join(base_dir, "solver_specific")
    os.makedirs(solver_dir, exist_ok=True)
    
    result_files = {}
    
    # Create specific versions for each solver and logic
    for logic in logics:
        # Make a CVC5 version
        cvc5_file = os.path.join(solver_dir, f"{base_name}_cvc5_{logic}.smt2")
        simplified = simplify_smt_file(smt_file, cvc5_file, logic)
        result_files[f"cvc5_{logic}"] = cvc5_file
        
        # Make a Yices version 
        yices_file = os.path.join(solver_dir, f"{base_name}_yices_{logic}.smt2")
        simplified = simplify_smt_file(smt_file, yices_file, logic)
        # Apply additional Yices-specific fixes
        fix_common_syntax_issues(simplified, yices_file)
        result_files[f"yices_{logic}"] = yices_file
    
    return result_files

if __name__ == "__main__":
    # Simple command line interface
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python smt_simplifier.py input_file.smt2 [output_file.smt2]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    simplified = simplify_smt_file(input_file, output_file)
    print(f"Simplified SMT file created: {simplified}")
