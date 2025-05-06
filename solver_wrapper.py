#! .venv\Scripts\python.exe

from logging_config import setup_logger
logger = setup_logger("Pathfinder")

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

class GurobiVarWrapper:
    def __init__(self, var):
        self.var = var
    
    def __lt__(self, other):
        # Create a Gurobi constraint directly
        if hasattr(other, 'var'):
            return self.var <= other.var - 1  # x < y is equivalent to x <= y - 1 for integers
        else:
            return self.var <= other - 1
    
    def __le__(self, other):
        if hasattr(other, 'var'):
            return self.var <= other.var
        else:
            return self.var <= other
    
    def __gt__(self, other):
        if hasattr(other, 'var'):
            return self.var >= other.var + 1  # x > y is equivalent to x >= y + 1 for integers
        else:
            return self.var >= other + 1
    
    def __ge__(self, other):
        if hasattr(other, 'var'):
            return self.var >= other.var
        else:
            return self.var >= other
    
    def __eq__(self, other):
        if hasattr(other, 'var'):
            return self.var == other.var
        else:
            return self.var == other
        
    def __add__(self, other):
        if hasattr(other, 'var'):
            return self.var + other.var
        return self.var + other
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if hasattr(other, 'var'):
            return self.var - other.var
        return self.var - other
    
    def __rsub__(self, other):
        if hasattr(other, 'var'):
            return other.var - self.var
        return other - self.var
    
    def __mul__(self, other):
        if hasattr(other, 'var'):
            return self.var * other.var
        return self.var * other
    
    def __rmul__(self, other):
        return self.__mul__(other)


# Factory for creating solvers with common interface
class SolverFactory:
    @staticmethod
    def create_solver(solver_type="z3"):
        if solver_type.lower() == "z3":
            from z3 import Optimize, Int, Bool, Or, And, Not, Implies, If, Sum, sat
            solver = Optimize()
            return solver, Int, Bool, Or, And, Not, Implies, If, Sum, sat
            
        elif solver_type.lower() == "gurobi":
            
            from gurobipy import Model, Env, GRB, quicksum, Var
            env = Env(empty=True)
            
            if no_output:
                env.setParam("OutputFlag", 0)
            env.start()
            model = Model(env=env)
            model = Wrapper(model)

            used_names = set()
            
            def Int(name):
                assert name not in used_names
                used_names.add(name)
                var = model.addVar(vtype=GRB.INTEGER, name=name)
                return GurobiVarWrapper(var)

            def Bool(name):
                assert name not in used_names
                used_names.add(name)
                var = model.addVar(vtype=GRB.BINARY, name=name)
                return GurobiVarWrapper(var)
            
            def add_constraint(expr):
                # Try to add the constraint directly
                try:
                    return model.addConstr(expr)
                except Exception as e:
                    # If that fails, try to handle different types
                    if hasattr(expr, 'var'):
                        return model.addConstr(expr.var)
                    elif isinstance(expr, bool):
                        if not expr:
                            # Create an impossible constraint for False
                            dummy = model.addVar(vtype=GRB.BINARY, name=f"false_const")
                            model.addConstr(dummy == 0)
                            model.addConstr(dummy == 1)
                        # True constraints don't need to be added
                        return None
                    else:
                        # Last resort - try to convert to a boolean constraint
                        return model.addConstr(expr >= 1)
                
            def minimize(obj):
                model.setObjective(obj, GRB.MINIMIZE)
              
            def maximize(obj):
                model.setObjective(obj, GRB.MAXIMIZE)
                
            def check():
                model.optimize()
                print("Status:", model.status)
                return model.status

            def And(*args):
                # For AND, all conditions must be true
                # In Gurobi, we simply add each constraint individually
                for arg in args:
                    if isinstance(arg, bool):
                        # If any constant False, the whole AND is false
                        if not arg:
                            # Create an impossible constraint (0 == 1)
                            dummy = model.addVar(vtype=GRB.BINARY, name=f"false_{id(arg)}")
                            model.addConstr(dummy == 0)
                            model.addConstr(dummy == 1)  # Makes the model infeasible
                            return False
                        # For True, nothing to do
                    elif isinstance(arg, GurobiVarWrapper):
                        # For wrapped variables, extract the underlying var
                        if hasattr(arg.var, 'VType') and arg.var.VType == GRB.BINARY:
                            model.addConstr(arg.var == 1)
                        else:
                            model.addConstr(arg)
                    elif hasattr(arg, 'var'):
                        # For other objects with var attribute
                        model.addConstr(arg.var)
                    else:
                        try:
                            # Direct constraint expressions
                            model.addConstr(arg)
                        except Exception as e:
                            # Handle expressions like x + y < 100
                            if '+' in str(arg) and '<' in str(arg):
                                parts = str(arg).split('<')
                                left_expr = eval(parts[0].strip())
                                right_val = int(parts[1].strip())
                                model.addConstr(left_expr <= right_val - 1)
                            elif '+' in str(arg) and '<=' in str(arg):
                                parts = str(arg).split('<=')
                                left_expr = eval(parts[0].strip())
                                right_val = int(parts[1].strip())
                                model.addConstr(left_expr <= right_val)
                            else:
                                # Try our best with other expressions
                                print(f"Warning: Could not add constraint {arg} directly: {e}")
                                return False
                
                # All constraints were added successfully
                return True
            
            def Or(*args):
                # For OR, at least one condition must be true
                if not args:
                    return True  # Empty OR is vacuously true
                    
                # Handle simple boolean cases
                if all(isinstance(arg, bool) for arg in args):
                    return any(args)
                    
                # For a single argument, just return it
                if len(args) == 1:
                    return args[0]
                
                # For binary variables, we can use a sum constraint:
                # sum(vars) >= 1  means at least one variable must be true
                binary_vars = []
                for arg in args:
                    if isinstance(arg, bool):
                        if arg:  # If any is True, the whole OR is true
                            return True
                        # Skip False values for summation
                    elif isinstance(arg, GurobiVarWrapper) and hasattr(arg.var, 'VType') and arg.var.VType == GRB.BINARY:
                        binary_vars.append(arg.var)
                    else:
                        # For non-binary expressions, we need to handle them differently
                        result_var = model.addVar(vtype=GRB.BINARY, name=f"or_expr_{id(arg)}")
                        # Add constraints to link the expression with the result var
                        try:
                            # Try to add a direct constraint
                            model.addGenConstrIndicator(result_var, True, arg)
                        except:
                            # Fallback: Add a constraint that arg implies result_var
                            M = 1e6  # A large value for big-M method
                            if isinstance(arg, GurobiVarWrapper):
                                model.addConstr(arg.var <= M * result_var)
                            else:
                                # Last resort
                                print(f"Warning: Complex expression in Or: {arg}")
                                return False
                        binary_vars.append(result_var)
                
                # Create constraint: sum of binary vars >= 1
                if binary_vars:
                    if len(binary_vars) == 1:
                        # If only one binary variable, just return it == 1
                        return binary_vars[0] == 1
                    else:
                        result_var = model.addVar(vtype=GRB.BINARY, name=f"or_result_{id(args)}")
                        model.addConstr(sum(binary_vars) >= result_var)
                        model.addConstr(sum(binary_vars) <= len(binary_vars) * result_var)
                        return result_var == 1
                
                # If we get here, all args were False
                return False
            
            def Not(arg):
                # For NOT, we need to negate the condition
                if isinstance(arg, bool):
                    return not arg
                elif isinstance(arg, GurobiVarWrapper):
                    # For binary variables, we need to set them to 0 (false)
                    if hasattr(arg.var, 'VType') and arg.var.VType == GRB.BINARY:
                        result_var = model.addVar(vtype=GRB.BINARY, name=f"not_{id(arg)}")
                        model.addConstr(result_var + arg.var == 1)  # result = 1-arg
                        return GurobiVarWrapper(result_var)
                    else:
                        # For other expressions, create a binary result variable
                        print("Warning: Not with non-binary variable might not work as expected")
                        return GurobiVarWrapper(model.addVar(vtype=GRB.BINARY, name=f"not_complex_{id(arg)}"))
                elif hasattr(arg, 'var'):
                    # For other objects with var attribute
                    if hasattr(arg.var, 'VType') and arg.var.VType == GRB.BINARY:
                        result_var = model.addVar(vtype=GRB.BINARY, name=f"not_attr_{id(arg)}")
                        model.addConstr(result_var + arg.var == 1)  # result = 1-arg
                        return GurobiVarWrapper(result_var)
                    else:
                        print("Warning: Not with complex expression might not work as expected")
                        return GurobiVarWrapper(model.addVar(vtype=GRB.BINARY, name=f"not_complex_attr_{id(arg)}"))
                else:
                    # Direct expressions - this is tricky and might not always work
                    print("Warning: Not with direct expression might not work as expected")
                    return not arg  # Try the Python 'not' operator
                    
            def Implies(a, b):
                # For implication a → b, equivalent to (¬a ∨ b)
                if isinstance(a, bool) and isinstance(b, bool):
                    return not a or b
                
                # Create a binary result variable
                result_var = model.addVar(vtype=GRB.BINARY, name=f"implies_{id((a,b))}")
                
                # Handle different cases
                if isinstance(a, GurobiVarWrapper) and hasattr(a.var, 'VType') and a.var.VType == GRB.BINARY:
                    if isinstance(b, GurobiVarWrapper) and hasattr(b.var, 'VType') and b.var.VType == GRB.BINARY:
                        # Both a and b are binary variables
                        model.addConstr(b.var >= a.var * result_var)
                        model.addConstr(b.var >= a.var - (1 - result_var))
                    else:
                        # a is binary, b is an expression
                        # Use indicator constraint
                        try:
                            model.addGenConstrIndicator(a.var, True, b)
                            return result_var == 1
                        except:
                            print("Warning: Complex implication with non-binary RHS")
                            return False
                else:
                    # a is not a binary variable, more complex case
                    try:
                        # Try Or(Not(a), b)
                        not_a = Not(a)
                        return Or(not_a, b)
                    except:
                        print("Warning: Complex implication with non-binary LHS")
                        return False
                
                return result_var == 1
            
            def If(cond, true_expr, false_expr):
                # For If, we need to create a conditional constraint
                if isinstance(cond, bool):
                    return true_expr if cond else false_expr
                
                # Create result variable based on types
                if isinstance(true_expr, (int, float)) and isinstance(false_expr, (int, float)):
                    # Numeric literals
                    result_var = model.addVar(vtype=GRB.INTEGER, name=f"if_result_{id((cond,true_expr,false_expr))}")
                elif isinstance(true_expr, GurobiVarWrapper) and isinstance(false_expr, GurobiVarWrapper):
                    # Both expressions are variables
                    if true_expr.var.VType == GRB.BINARY and false_expr.var.VType == GRB.BINARY:
                        result_var = model.addVar(vtype=GRB.BINARY, name=f"if_result_{id((cond,true_expr,false_expr))}")
                    else:
                        result_var = model.addVar(vtype=GRB.INTEGER, name=f"if_result_{id((cond,true_expr,false_expr))}")
                else:
                    # Default to INTEGER
                    result_var = model.addVar(vtype=GRB.INTEGER, name=f"if_result_{id((cond,true_expr,false_expr))}")
                
                # Handle the condition
                if isinstance(cond, GurobiVarWrapper) and hasattr(cond.var, 'VType') and cond.var.VType == GRB.BINARY:
                    # Binary condition
                    if isinstance(true_expr, (int, float)) and isinstance(false_expr, (int, float)):
                        # Simple numeric literals
                        model.addConstr(result_var == cond.var * true_expr + (1 - cond.var) * false_expr)
                    elif isinstance(true_expr, GurobiVarWrapper) and isinstance(false_expr, GurobiVarWrapper):
                        # Both are variables
                        M = 1e6  # Big-M constant
                        model.addConstr(result_var <= true_expr.var + M * (1 - cond.var))
                        model.addConstr(result_var >= true_expr.var - M * (1 - cond.var))
                        model.addConstr(result_var <= false_expr.var + M * cond.var)
                        model.addConstr(result_var >= false_expr.var - M * cond.var)
                    else:
                        # Mixed types or expressions
                        try:
                            model.addGenConstrIndicator(cond.var, True, result_var == true_expr)
                            model.addGenConstrIndicator(cond.var, False, result_var == false_expr)
                        except:
                            print("Warning: Complex If expression with mixed types")
                            return true_expr  # Fallback to true branch
                else:
                    # Complex condition
                    print("Warning: If with non-binary condition might not work as expected")
                    return true_expr  # Fallback to true branch
                
                return GurobiVarWrapper(result_var)

            def model_func():
                # Return the model object
                return model
            
            def getitem(var):
                if hasattr(var, 'var'):
                    # If var is a wrapped variable, get its value
                    return var.var.X
                return None
                        
            # Assign methods to the model
            model.get = getitem
            model.Int = Int
            model.Bool = Bool
            model.add = add_constraint
            model.minimize = minimize
            model.maximize = maximize
            model.check = check
            model.model = model_func
            sat = [2, 9]  # 2 = optimal, 9 = suboptimal
            
            return model, Int, Bool, Or, And, Not, Implies, If, quicksum, sat

# Test code for the wrapper
if __name__ == "__main__":


    print("\nTesting with Gurobi...")
    solver, Int, Bool, Or, And, Not, Implies, If, Sum, sat = SolverFactory.create_solver("gurobi")
    
    x = Int("x")
    y = Int("y")
    b = Bool("b")
    
    print(f"Created Gurobi variables: x = {x}, y = {y}, b = {b}")
    # Same constraints as Z3
    solver.add(x >= 1)
    solver.add(x < 10)
    solver.add(y >= 0)
    solver.add(y < 5)
    solver.add(x + y < 100)
    
    
    solver.add(Or(x + y < 10, b == 1))
    solver.add(And(x + y < 100, b == 1))
    # Add individual constraints instead of using And
    
    solver.add(b == 1)
    
    status = solver.check()
    print("Gurobi Status:", status)
    m = solver.model()
    
    print(f"Gurobi Solution: x = {m.get(x)}, y = {m.get(y)}, b = {m.get(b)}")