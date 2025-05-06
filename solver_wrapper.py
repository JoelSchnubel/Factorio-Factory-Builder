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
            
            from gurobipy import  Model, Env ,GRB, quicksum, Var
            env = Env(empty=True)
            
            if no_output:
                env.setParam("OutputFlag",0)
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
                            model.addConstr(arg.var)
                    elif hasattr(arg, 'var'):
                        # For other objects with var attribute
                        model.addConstr(arg.var)
                    else:
                        # Direct constraint expressions
                        model.addConstr(arg)
                
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
                        # For non-binary expressions, we need indicator constraints
                        # This is more complex - may need a different approach
                        raise NotImplementedError("OR with non-binary variables not yet supported")
                
                # Create constraint: sum of binary vars >= 1
                if binary_vars:
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
                elif hasattr(arg, 'var'):
                    # For binary variables, we need to set them to 0 (false)
                    if hasattr(arg.var, 'VType') and arg.var.VType == GRB.BINARY:
                        model.addConstr(arg.var == 0)
                    else:
                        # Other variable expressions (like comparisons)
                        model.addConstr(arg.var == 0)
                else:
                    # Direct constraint expressions
                    model.addConstr(arg == 0)
                    
            def Implies(a, b):
                # For implication, we need to ensure that if a is true, b must also be true
                # In Gurobi, we can use a big-M constraint or simply add the constraints directly
                if isinstance(a, bool) and isinstance(b, bool):
                    return not a or b
                elif hasattr(a, 'var') and hasattr(b, 'var'):
                    # For binary variables, we need to set them accordingly
                    if hasattr(a.var, 'VType') and a.var.VType == GRB.BINARY:
                        model.addConstr(Or(Not(a.var), b.var))
                    else:
                        # Other variable expressions (like comparisons)
                        model.addConstr(Implies(a.var, b.var))
                else:
                    # Direct constraint expressions
                    model.addConstr(Implies(a, b))
            
            def If(cond, true_expr, false_expr):
                # For If, we need to create a conditional constraint
                if isinstance(cond, bool):
                    return true_expr if cond else false_expr
                elif hasattr(cond, 'var'):
                    # For binary variables, we need to set them accordingly
                    if hasattr(cond.var, 'VType') and cond.var.VType == GRB.BINARY:
                        model.addConstr(If(cond.var == 1, true_expr, false_expr))
                    else:
                        # Other variable expressions (like comparisons)
                        model.addConstr(If(cond.var, true_expr, false_expr))
                else:
                    # Direct constraint expressions
                    model.addConstr(If(cond, true_expr, false_expr))

            
            def model_func():
                # Return the model object
                return model
            
            def getitem(var):
                if hasattr(var, 'var'):
                    # If var is a wrapped variable, get its value
                    return var.var.X
                return None
                        
            

            model.get = getitem
            model.Int = Int
            model.Bool = Bool
            model.add = add_constraint
            model.minimize = minimize
            model.maximize = maximize
            model.check = check
            model.model = model_func
            sat = [2,9] # 2 = optimal, 9 = suboptimal
            
            return model, Int, Bool, Or, And, Not, Implies, If, quicksum, sat

# Test code for the wrapper
if __name__ == "__main__":
    import gurobipy as gp 
    from gurobipy import GRB

    # Create a new model
    m = gp.Model()

    # Create variables
    x1 = m.addVar(vtype=GRB.INTEGER, name="x")
    y1 = m.addVar(vtype=GRB.INTEGER, name="y")
    z1 = m.addVar(vtype=GRB.INTEGER, name="z")
    m.addConstr(x1+y1 >= 10)
    

    # Now test with Gurobi
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
    
    solver.add(And(x + y < 100, b == 1))
    
    status = solver.check()
    print("Gurobi Status:", status)
    m = solver.model()
    
    print(f"Gurobi Solution: x = {m.get(x)}, y = {m.get(y)}, b = {m.get(b)}")
    
    #if solver.check() == sat:
    #    m = solver.model()
    #    print(f"Gurobi Solution: x = {m[x]}, y = {m[y]}, b = {m[b]}")
    #else:
    #    print("Gurobi found no solution.")
