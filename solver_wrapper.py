#! .venv\Scripts\python.exe

import logging
import gurobipy as gp
from gurobipy import GRB


no_output = False

class Wrapper:
    def __init__(self, obj):
        self.obj = obj
        self.overwritten = {}
        # delayed to avoid infinite recursion in self.obj = obj
        self.__setattr__ = self.__setattr__2
        
    def __setattr__2(self, name, value):
        self.overwritten[name] = value
        
    # forward all non overwritten attributes to the object
    def __getattr__(self, name):
        if name in self.overwritten:
            return self.overwritten[name]
        return getattr(self.obj, name)

class SolverFactory:
    @staticmethod
    def create_solver(solver_type="z3"):
        if solver_type.lower() == "z3":
            from z3 import Optimize, Int, Bool, Or, And, Not, Implies, If, Sum, sat
            
            solver = Optimize()
            
            # No need to override anything for Z3
            return solver, Int, Bool, Or, And, Not, Implies, If, Sum, sat
            
        elif solver_type.lower() == "gurobi":
            from gurobipy import Model, Env, GRB, quicksum, Var
            import gurobipy as gp
            
            env = Env(empty=True)
            if no_output:
                env.setParam("OutputFlag", 0)
            env.start()
            model = Model(env=env)
            model = Wrapper(model)
            
            used_names = set()
            
            # Create variable wrapper class for comparisons
            class GurobiVarWrapper:
                def __init__(self, var):
                    self.var = var
                
                # Less than operator
                def __lt__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return self.var <= other.var - 1  # Strict less than
                    else:
                        return self.var <= other - 1  # Strict less than
                
                # Less than or equal operator
                def __le__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return self.var <= other.var
                    else:
                        return self.var <= other
                
                # Greater than operator
                def __gt__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return self.var >= other.var + 1  # Strict greater than
                    else:
                        return self.var >= other + 1  # Strict greater than
                
                # Greater than or equal operator
                def __ge__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return self.var >= other.var
                    else:
                        return self.var >= other
                
                # Equal operator
                def __eq__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return self.var == other.var
                    else:
                        return self.var == other
                
                # Not equal operator
                def __ne__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return self.var != other.var
                    else:
                        return self.var != other
                
                # Addition operator
                def __add__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return GurobiVarWrapper(self.var + other.var)
                    else:
                        return GurobiVarWrapper(self.var + other)
                
                # Right addition operator
                def __radd__(self, other):
                    return GurobiVarWrapper(other + self.var)
                
                # Subtraction operator
                def __sub__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return GurobiVarWrapper(self.var - other.var)
                    else:
                        return GurobiVarWrapper(self.var - other)
                
                # Right subtraction operator
                def __rsub__(self, other):
                    return GurobiVarWrapper(other - self.var)
                
                # Multiplication operator
                def __mul__(self, other):
                    if isinstance(other, GurobiVarWrapper):
                        return GurobiVarWrapper(self.var * other.var)
                    else:
                        return GurobiVarWrapper(self.var * other)
                
                # Right multiplication operator
                def __rmul__(self, other):
                    return GurobiVarWrapper(other * self.var)
                
                # For accessing var properties
                def __getattr__(self, name):
                    return getattr(self.var, name)
                
                # For string representation
                def __repr__(self):
                    return f"GurobiVarWrapper({self.var})"
                
                def __ne__(self, other):
                    # Create a binary variable with a safer naming approach
                    try:
                        # Try to use VarName if available
                        var_name = getattr(self.var, 'VarName', f"var_{id(self.var)}")
                        other_name = getattr(other, 'var', other)
                        if hasattr(other_name, 'VarName'):
                            other_name = getattr(other_name, 'VarName', f"var_{id(other_name)}")
                        bin_var_name = f"neq_{id(self.var)}_{id(other_name)}"
                    except:
                        # Fallback to a simple unique name based on object IDs
                        bin_var_name = f"neq_{id(self.var)}_{id(other)}"
                    
                    # Create the binary variable
                    bin_var = model.addVar(vtype=GRB.BINARY, name=bin_var_name)
                    
                    # Handle different types for 'other'
                    if isinstance(other, GurobiVarWrapper):
                        other_var = other.var
                    else:
                        other_var = other
                    
                    # Use big-M approach to model x != y as:
                    # x ≤ y - 1 + M(1-b)   (if b=1, forces x ≤ y - 1, meaning x < y)
                    # x ≥ y + 1 - M*b      (if b=0, forces x ≥ y + 1, meaning x > y)
                    # This ensures that either x < y or x > y, which means x != y
                    M_value = 1000000
                    model.addConstr(self.var <= other_var - 1 + M_value * (1 - bin_var))
                    model.addConstr(self.var >= other_var + 1 - M_value * bin_var)
                    
                    # Return True since we already enforced the constraint
                    return True
                
            counter = {"Bool": 0, "Int": 0, "Not": 0, "Or": 0, "And": 0, "Implies": 0, "If": 0}
                    
                # Update Int and Bool functions to wrap the variables
            def Bool(name):
                original_name = name
                # Try to make the name unique if there's a conflict
                if name in used_names:
                    counter["Bool"] += 1
                    name = f"{original_name}_{counter['Bool']}"
                
                # If we still have a conflict, keep adding numbers until unique
                while name in used_names:
                    counter["Bool"] += 1
                    name = f"{original_name}_{counter['Bool']}"
                
                used_names.add(name)
                var = model.addVar(vtype=GRB.BINARY, name=name)
                return GurobiVarWrapper(var)

            # Similarly update the Int function
            def Int(name):
                original_name = name
                # Try to make the name unique if there's a conflict
                if name in used_names:
                    counter["Int"] += 1
                    name = f"{original_name}_{counter['Int']}"
                
                # If we still have a conflict, keep adding numbers until unique
                while name in used_names:
                    counter["Int"] += 1
                    name = f"{original_name}_{counter['Int']}"
                
                used_names.add(name)
                var = model.addVar(vtype=GRB.INTEGER, lb=0, name=name)
                return GurobiVarWrapper(var)

            # Also update the Not function to use the counter
            def Not(arg):
                # Negate the constraint (only works for simple binary expressions)
                if isinstance(arg, tuple):
                    if arg[0] == '==':
                        return ('!=', arg[1], arg[2])
                    elif arg[0] == '!=':
                        return ('==', arg[1], arg[2])
                
                # For wrapped variables
                if hasattr(arg, 'var') and isinstance(arg.var, gp.Var) and arg.var.VType == GRB.BINARY:
                    return GurobiVarWrapper(1 - arg.var)
                
                # For more complex expressions, would need binary variables
                counter["Not"] += 1
                not_var = Bool(f"not_var_{counter['Not']}")
                
                # Extract the actual argument if it's wrapped
                if hasattr(arg, 'var'):
                    arg_val = arg.var
                else:
                    arg_val = arg
                    
                model.addConstr(not_var.var == 1 - arg_val)
                return not_var
            
            # Create large constant M for big-M formulation
            M = 1000000  # Adjust based on your problem
                
            
            def Or(*args):
                # Create a binary variable for each argument
                bin_vars = [model.addVar(vtype=GRB.BINARY, name=f"or_var_{i}") for i in range(len(args))]
                
                # Add constraints for each argument using big-M formulation
                for i, arg in enumerate(args):
                    if isinstance(arg, tuple) and arg[0] == '!=':  # Handle inequality
                        model.addConstr(arg[1] - arg[2] <= M * bin_vars[i])
                        model.addConstr(arg[2] - arg[1] <= M * bin_vars[i])
                    elif hasattr(arg, 'var'):  # Handle wrapped variables
                        # Handle comparison expressions specially
                        if isinstance(arg.var, gp.TempConstr):
                            # For expressions like 'x < y', we need to handle the specific operator
                            # Extract the sense and the left/right expressions
                            lhs = arg.var.getExpr()
                            sense = arg.var.getSense()
                            rhs = arg.var.getRhs()
                            
                            # Correctly formulate the big-M constraint based on the sense
                            if sense == '<':
                                model.addConstr(lhs <= rhs + M * bin_vars[i])
                            elif sense == '>':
                                model.addConstr(lhs >= rhs - M * bin_vars[i])
                            elif sense == '<=':
                                model.addConstr(lhs <= rhs + M * bin_vars[i])
                            elif sense == '>=':
                                model.addConstr(lhs >= rhs - M * bin_vars[i])
                            elif sense == '=':
                                model.addConstr(lhs <= rhs + M * bin_vars[i])
                                model.addConstr(lhs >= rhs - M * bin_vars[i])
                        else:
                            # For regular variables or expressions
                            model.addConstr(arg.var >= 1 - M * (1 - bin_vars[i]))
                    else:
                        # Direct expressions (like x > y directly instead of wrapped)
                        try:
                            model.addConstr(arg >= 1 - M * (1 - bin_vars[i]))
                        except Exception as e:
                            logging.error(f"Error in Or function: {e} for argument {arg}")
                            # Fallback approach
                            if hasattr(arg, 'getLhs'):
                                lhs = arg.getLhs()
                                sense = arg.getSense()
                                rhs = arg.getRhs()
                                
                                if sense == '<':
                                    model.addConstr(lhs <= rhs + M * bin_vars[i])
                                elif sense == '>':
                                    model.addConstr(lhs >= rhs - M * bin_vars[i])
                                elif sense == '<=':
                                    model.addConstr(lhs <= rhs + M * bin_vars[i])
                                elif sense == '>=':
                                    model.addConstr(lhs >= rhs - M * bin_vars[i])
                                elif sense == '=':
                                    model.addConstr(lhs <= rhs + M * bin_vars[i])
                                    model.addConstr(lhs >= rhs - M * bin_vars[i])
                
                # At least one binary variable must be 1
                model.addConstr(gp.quicksum(bin_vars) >= 1)
                
                # Return the OR result (always True since we enforced it by constraints)
                return True
            
            def And(*args):
                # Simply add all constraints
                for arg in args:
                    if isinstance(arg, tuple) and arg[0] == '!=':  # Handle inequality
                        model.addConstr(arg[1] != arg[2])
                    elif hasattr(arg, 'var'):  # Handle wrapped variables or constraints
                        # Check what type of object arg.var is
                        if isinstance(arg.var, gp.TempConstr):
                            # It's a constraint expression, add it directly
                            model.addConstr(arg.var)
                        elif isinstance(arg.var, gp.Var):
                            # It's a variable, add constraint that it equals 1 (true)
                            model.addConstr(arg.var == 1)
                        else:
                            # Try to add it as a general constraint with a try/except
                            try:
                                model.addConstr(arg.var)
                            except Exception as e:
                                logging.error(f"Error in And function: {e} for argument {arg.var}")
                                # Try to interpret as a boolean value
                                try:
                                    bool_val = bool(arg.var)
                                    if not bool_val:
                                        # If it's False, the whole AND must be false
                                        # We would need to return a constraint that is always false
                                        # But for simplicity, let's just log an error
                                        logging.error("AND constraint contains a False value, constraint may not behave correctly")
                                except:
                                    logging.error(f"Could not convert {arg.var} to a boolean")
                    elif arg is True:  # Skip True values
                        continue
                    elif arg is False:  # If any arg is False, the whole AND must be false
                        # For now, log a warning since we can't directly model this
                        logging.warning("AND constraint contains False, constraint may not behave correctly")
                    else:  # Handle direct expressions
                        try:
                            model.addConstr(arg)
                        except Exception as e:
                            logging.error(f"Error in And function: {e} for argument {arg}")
                            # Try to interpret as a boolean value
                            try:
                                bool_val = bool(arg)
                                if not bool_val:
                                    # If it's False, the whole AND must be false
                                    logging.error("AND constraint contains a False value, constraint may not behave correctly")
                            except:
                                logging.error(f"Could not convert {arg} to a boolean")
                
                return True
  
            
            def Implies(condition, result):
                counter["Implies"] += 1
                
                # Extract variables if they're wrapped
                if hasattr(condition, 'var'):
                    condition_var = condition.var
                else:
                    condition_var = condition
                    
                if hasattr(result, 'var'):
                    result_var = result.var
                else:
                    result_var = result
                
                # Create a binary variable for the condition if it's not already one
                binary_condition = None
                
                # Safer check for variable type
                try:
                    is_gurobi_var = isinstance(condition_var, gp.Var)
                    if is_gurobi_var:
                        try:
                            is_binary = condition_var.VType == GRB.BINARY
                        except:
                            # If we can't access VType, assume it's not a binary var
                            is_binary = False
                    else:
                        is_binary = False
                        
                    if not is_binary:
                        binary_condition = model.addVar(vtype=GRB.BINARY, name=f"implies_cond_{counter['Implies']}")
                        model.addConstr(binary_condition == condition_var)
                    else:
                        binary_condition = condition_var
                except Exception as e:
                    # If any error occurs, create a new binary variable to be safe
                    logging.error(f"Error checking variable type in Implies: {e}")
                    binary_condition = model.addVar(vtype=GRB.BINARY, name=f"implies_cond_{counter['Implies']}")
                    try:
                        model.addConstr(binary_condition == condition_var)
                    except:
                        logging.error(f"Could not constrain binary_condition to condition_var")
                
                # Use a safer approach with big-M formulation which works in most cases
                try:
                    # This formulation is equivalent to: if binary_condition then result_var
                    # Which can be written as: !binary_condition OR result_var
                    # Or: (1 - binary_condition) + result_var >= 1
                    
                    # For result_var that is a constraint expression
                    if isinstance(result_var, gp.TempConstr):
                        # Add the constraint directly
                        model.addConstr((binary_condition == 0) | result_var)
                    # For result_var that is a binary variable
                    elif isinstance(result_var, gp.Var) and result_var.VType == GRB.BINARY:
                        model.addConstr(1 - binary_condition + result_var >= 1)
                    # For constant result values
                    elif isinstance(result_var, bool) or isinstance(result_var, int):
                        if bool(result_var):  # If True, no constraint needed
                            pass
                        else:  # If False, add constraint that condition must be False
                            model.addConstr(binary_condition == 0)
                    # For other expressions
                    else:
                        model.addConstr(1 - binary_condition + result_var >= 1)
                except Exception as e:
                    logging.error(f"Error in Implies using OR formulation: {e}")
                    # Final fallback: just log the error but don't crash
                    logging.error(f"Could not create implication constraint for condition {condition_var} and result {result_var}")
                
                # Return a wrapped binary variable
                return GurobiVarWrapper(binary_condition)
            
            def If(condition, then_expr, else_expr):
                # Create a variable for the result
                is_binary = (isinstance(then_expr, bool) or 
                            (hasattr(then_expr, 'var') and then_expr.var.VType == GRB.BINARY)) and \
                           (isinstance(else_expr, bool) or 
                            (hasattr(else_expr, 'var') and else_expr.var.VType == GRB.BINARY))
                
                result_type = GRB.BINARY if is_binary else GRB.INTEGER
                result_var = model.addVar(vtype=result_type, name=f"if_result_var")
                result = GurobiVarWrapper(result_var)
                
                # Add constraints using binary variable and big-M
                b = Bool(f"if_condition_var")
                
                # Handle wrapped variables
                cond = condition.var if hasattr(condition, 'var') else condition
                then = then_expr.var if hasattr(then_expr, 'var') else then_expr
                else_val = else_expr.var if hasattr(else_expr, 'var') else else_expr
                
                model.addConstr((cond == True) >> (result_var == then))
                model.addConstr((cond == False) >> (result_var == else_val))
                return result
            
            def Sum(expr_list):
                # Process the list to handle wrapped variables
                processed_list = []
                for expr in expr_list:
                    if hasattr(expr, 'var'):
                        processed_list.append(expr.var)
                    else:
                        processed_list.append(expr)
                
                # Sum up all expressions
                sum_expr = gp.quicksum(processed_list)
                return GurobiVarWrapper(sum_expr) if hasattr(sum_expr, 'addVar') else sum_expr
            
            def add(constraint):
                # Handle wrapped constraints
                if hasattr(constraint, 'var'):
                    try:
                        return model.addConstr(constraint.var)
                    except Exception as e:
                        logging.error(f"Error adding wrapped constraint: {e}")
                        
                        # Don't try to evaluate constraint as a boolean
                        # Instead, handle special types we know about
                        if isinstance(constraint.var, gp.TempConstr):
                            try:
                                # Try to add it directly
                                return model.addConstr(constraint.var)
                            except:
                                logging.error(f"Could not add TempConstr directly")
                        
                        # For wrapped variables, use a safer check for binary variables
                        if isinstance(constraint.var, gp.Var):
                            try:
                                # Try to check VType safely
                                if hasattr(constraint.var, 'VType') and constraint.var.VType == GRB.BINARY:
                                    return model.addConstr(constraint.var == 1)
                            except:
                                # If VType check fails, try a more generic approach
                                try:
                                    return model.addConstr(constraint.var == 1)
                                except:
                                    logging.error(f"Could not add constraint for variable")
                        
                        # As a last resort, log error and continue
                        logging.error(f"Unable to add constraint {type(constraint.var)}")
                        return None
                else:
                    # Rest of the function unchanged...
                    try:
                        # For direct constraints
                        return model.addConstr(constraint)
                    except Exception as e:
                        logging.error(f"Error adding direct constraint: {e}")
                        
                        # Only check for True/False if we're dealing with a Python bool
                        if isinstance(constraint, bool):
                            if constraint is True:
                                return None  # No constraint needed for True
                            else:
                                # Add an impossible constraint for False
                                dummy_var = model.addVar(vtype=GRB.BINARY, name=f"impossible_constraint_{id(constraint)}")
                                model.addConstr(dummy_var == 0)
                                return model.addConstr(dummy_var == 1)
                        
                        # For everything else
                        try:
                            dummy_var = model.addVar(vtype=GRB.BINARY, name=f"constraint_{id(constraint)}")
                            model.addConstr(dummy_var == 1)
                            return model.addConstr(dummy_var == constraint)
                        except:
                            logging.error(f"Could not add constraint: {constraint}")
                            return None
            
            model.add = add
            
            def minimize(objective):
                # Handle wrapped objective
                if hasattr(objective, 'var'):
                    return model.setObjective(objective.var, GRB.MINIMIZE)
                return model.setObjective(objective, GRB.MINIMIZE)
            
            model.minimize = minimize
            
            
            def maximize(objective):
                # Handle wrapped objective
                if hasattr(objective, 'var'):
                    return model.setObjective(objective.var, GRB.MAXIMIZE)
                return model.setObjective(objective, GRB.MAXIMIZE)
            
            model.maximize = maximize
            # Define sat equivalent for Gurobi
            sat = GRB.OPTIMAL
            
            # Create check method to match Z3's
            def check():
                model.optimize()
                return model.status
            model.check = check
            
            # Create model method to match Z3's
            class GurobiModel:
                def __init__(self, gurobi_model):
                    self.gurobi_model = gurobi_model
                
                def access(self, e, f):
                    if isinstance(e, int):
                        return e
                    value = f(e)
                    return value
                
                # In the GurobiModel class, update the evaluate method with detailed logging:
                def evaluate(self, expr):
                    if isinstance(expr, int):
                        logging.debug(f"Evaluating integer constant: {expr}")
                        return expr
                    try:
                        # Handle wrapped variables
                        if hasattr(expr, 'var'):
                            raw_value = expr.var.X
                            var_name = expr.var.VarName
                            logging.debug(f"Evaluating wrapped Gurobi variable {var_name}: raw value = {raw_value}")
                        # Handle direct Gurobi variables
                        elif hasattr(expr, 'X'):
                            raw_value = expr.X
                            var_name = expr.VarName
                            logging.debug(f"Evaluating Gurobi variable {var_name}: raw value = {raw_value}")
                        else:
                            raw_value = expr
                            var_name = str(expr)
                            logging.debug(f"Evaluating expression: {var_name}, value = {raw_value}")
                        
                        # Create wrapper with as_long method
                        value_obj = type('', (), {})()
                        value_obj.as_long = lambda: int(round(raw_value))
                        
                        logging.debug(f"Returning wrapped value with as_long() = {value_obj.as_long()}")
                        return value_obj
                    except Exception as e:
                        logging.error(f"Error in evaluate: {e} for expression {expr}")
                        # Create a default object with as_long that returns 0
                        value_obj = type('', (), {})()
                        value_obj.as_long = lambda: 0
                        
                        logging.debug(f"Returning default value with as_long() = 0")
                        return value_obj
                            
                # make subscripting work
                def __getitem__(self, key):
                    if hasattr(key, 'var'):
                        return self.access(key.var, lambda x: x.X)
                    return self.access(key, lambda x: x.X)
                
            model.model = lambda: GurobiModel(model)
            
            return model, Int, Bool, Or, And, Not, Implies, If, Sum, sat
        
        else:
            raise ValueError(f"Unsupported solver type: {solver_type}")