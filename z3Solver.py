#! .venv\Scripts\python.exe

from z3 import *

class Z3Solver:
    def __init__(self, grid_width, grid_height, production_data):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.production_data = production_data

        # Initialize the Z3 grid
        self.assemblers = [[Bool(f"A_{x}_{y}") for y in range(self.grid_height)] for x in range(self.grid_width)]
        self.inserters = [[Bool(f"I_{x}_{y}") for y in range(self.grid_height)] for x in range(self.grid_width)]
        self.belt_start = [[Bool(f"B_start_{x}_{y}") for y in range(self.grid_height)] for x in range(self.grid_width)]
        self.belt_end = [[Bool(f"B_end_{x}_{y}") for y in range(self.grid_height)] for x in range(self.grid_width)]
        self.item_ids = [[None for y in range(self.grid_height)] for x in range(self.grid_width)]  # To store item IDs
        
        # Initialize Z3 solver
        self.solver = Solver()
        self.belt_pairs = []  # To store belt start and end positions

    def process_input(self):
        """
        Process the input data and assign assemblers, inserters, and belts based on production rates.
        """
        assembler_counter = 0
        for item_id, item_data in self.production_data.items():
            assemblers = item_data.get('assemblers', 0)
            inserters = item_data.get('inserters', 0)
            belts = item_data.get('belts', 0)
            
            # For now, assign assemblers in a linear fashion. This can be optimized.
            for _ in range(assemblers):
                
                placed = self.place_assembler(assembler_counter, item_id)
                if not placed:
                    print(f"Error placing assembler for {item_id}")
                assembler_counter += 1
                
    def place_output_belt(self, x, y, item_id):
        """
        Place an output belt from the assembler producing electronic-circuit.
        """
        if x < self.grid_width:
            self.solver.add(self.belt_start[x][y] == True)
            print(f"Output belt for {item_id} placed at ({x},{y})")
            self.belt_pairs.append((self.belt_start[x][y], None))

    def place_input_belt(self, x, y, item_id):
        """
        Place input belts for copper-plate and iron-plate.
        """
        if x >= 0:
            self.solver.add(self.belt_end[x][y] == True)
            print(f"Input belt for {item_id} placed at ({x},{y})")
            self.belt_pairs.append((None, self.belt_end[x][y]))
            
    def place_assembler(self, assembler_id, item_id):
        """
        Place an assembler on the grid and assign it an item ID.
        """
        for x in range(self.grid_width - 2):
            for y in range(self.grid_height - 2):
                assembler_block = [self.assemblers[x + i][y + j] for i in range(3) for j in range(3)]
                if all(self.item_ids[x + i][y + j] is None for i in range(3) for j in range(3)):
                    # Reserve this block for the assembler
                    for i in range(3):
                        for j in range(3):
                            self.item_ids[x + i][y + j] = item_id
                    return True
        return False

    def add_assembler_constraints(self):
        # Constraints for assembler placement (3x3 block)
        for x in range(self.grid_width - 2):
            for y in range(self.grid_height - 2):
                assembler_block = [self.assemblers[x + i][y + j] for i in range(3) for j in range(3)]
                self.solver.add(Or(assembler_block))  # Add assembler constraint for 3x3 blocks

    def add_inserter_constraints(self):
        # Inserters should be adjacent to assemblers
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                adjacent_cells = []
                if x > 0:
                    adjacent_cells.append(self.assemblers[x - 1][y])
                if x < self.grid_width - 1:
                    adjacent_cells.append(self.assemblers[x + 1][y])
                if y > 0:
                    adjacent_cells.append(self.assemblers[x][y - 1])
                if y < self.grid_height - 1:
                    adjacent_cells.append(self.assemblers[x][y + 1])
                
                # Inserters must be adjacent to an assembler
                self.solver.add(Implies(self.inserters[x][y], Or(adjacent_cells)))

    def add_belt_constraints(self):
        # Belt start from assembler output and end at assembler input
        for x in range(self.grid_width - 2):
            for y in range(self.grid_height - 2):
                # Belt starts at right of the assembler
                if x < self.grid_width - 3:
                    self.solver.add(Implies(self.assemblers[x][y], self.belt_start[x + 3][y + 1]))
                    self.belt_pairs.append((self.belt_start[x + 3][y + 1], None))
                
                # Belt ends at left of the next assembler
                if x > 0:
                    self.solver.add(Implies(self.assemblers[x][y], self.belt_end[x - 1][y + 1]))
                    self.belt_pairs[-1] = (self.belt_pairs[-1][0], self.belt_end[x - 1][y + 1])

    def add_non_overlap_constraints(self):
        # Ensure no overlap between assemblers, inserters, belts
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                self.solver.add(Or(Not(self.assemblers[x][y]), Not(self.inserters[x][y])))
                self.solver.add(Or(Not(self.belt_start[x][y]), Not(self.assemblers[x][y])))
                self.solver.add(Or(Not(self.belt_end[x][y]), Not(self.assemblers[x][y])))

    def solve(self):
        # Solve the constraints
        if self.solver.check() == sat:
            return self.solver.model()
        else:
            return None

    def display_solution(self, model):
        # Display the grid with assemblers, inserters, and belt start/end
        if model:
            print("Assembler and Inserter Placement:")
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    if model.evaluate(self.assemblers[x][y]):
                        print(f"Assembler (Item ID: {self.item_ids[x][y]}) at ({x},{y})")
                    if model.evaluate(self.inserters[x][y]):
                        print(f"Inserter at ({x},{y})")
                    if model.evaluate(self.belt_start[x][y]):
                        print(f"Belt start at ({x},{y})")
                    if model.evaluate(self.belt_end[x][y]):
                        print(f"Belt end at ({x},{y})")
        else:
            print("No solution found")

# Example usage
def main():
    # Define grid dimensions
    grid_width = 10
    grid_height = 10

    # Production data input (as given)
    production_data = {
        'electronic-circuit': {'amount_per_minute': 50, 'assemblers': 1, 'inserters': 1, 'belts': 1},
        'copper-cable': {'amount_per_minute': 150.0, 'assemblers': 2, 'inserters': 2, 'belts': 1},
        'copper-plate': {'amount_per_minute': 75.0},
        'iron-plate': {'amount_per_minute': 50.0}
    }

    # Initialize solver with grid size and production data
    z3_solver = Z3Solver(grid_width, grid_height, production_data)

    # Process the input to place assemblers
    z3_solver.process_input()

    # Add constraints for assemblers, inserters, belts, and non-overlapping
    z3_solver.add_assembler_constraints()
    z3_solver.add_inserter_constraints()
    z3_solver.add_belt_constraints()
    z3_solver.add_non_overlap_constraints()

    # Solve the constraints
    model = z3_solver.solve()

    # Display the result
    z3_solver.display_solution(model)
    
    

if __name__ == "__main__":
    main()
