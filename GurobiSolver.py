#! .venv\Scripts\python.exe

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from logging_config import setup_logger
logger = setup_logger("GurobiSolver")

# Constants
ITEMS_PER_SECOND = 7.5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CELL_SIZE = 40

INSERTER_COLOR_MAP = {
    'input': (255, 165, 0),  # Orange 
    'output': (255, 20, 147)  # Deep Pink
}

BELT_COLOR_MAP = {
    'start': (0, 128, 0),  # Green 
    'end': (255, 0, 0)     # Red
}

class Assembler:
    def __init__(self, id, inserters, item=None, capacity=0, model=None):
        self.id = id
        self.x = model.addVar(vtype=GRB.INTEGER, name=f'{id}_x')
        self.y = model.addVar(vtype=GRB.INTEGER, name=f'{id}_y')
        self.inserters = inserters
        self.item = item
        self.capacity = capacity
     
    def __str__(self) -> str:
        inserters_str = ', '.join([str(inserter) for inserter in self.inserters])
        return (f"Assembler(id={self.id}, position=({self.x}, {self.y}), "
                f"capacity={self.capacity}, inserters=[{inserters_str}])")
        
class Inserter:
    def __init__(self, id, type, belt, item=None, model=None):
        self.id = id
        self.x = model.addVar(vtype=GRB.INTEGER, name=f'{id}_x')
        self.y = model.addVar(vtype=GRB.INTEGER, name=f'{id}_y')
        self.type = type  # 'input' or 'output'
        self.belt = belt
        self.item = item
        self.direct_interaction = False

    def __str__(self) -> str:
        return (f"Inserter(id={self.id}, position=({self.x}, {self.y}), type={self.type}), item={self.item}"
                f",Belt={str(self.belt)}")

class Belt:
    def __init__(self, id, type, item=None, int_x=0, int_y=0, model=None):
        self.id = id
        self.x = model.addVar(vtype=GRB.INTEGER, name=f'{id}_x')
        self.y = model.addVar(vtype=GRB.INTEGER, name=f'{id}_y')
        self.type = type  # 'start' or 'end'
        self.item = item
        self.is_used = model.addVar(vtype=GRB.BINARY, name=f"{id}_used")
        self.int_x = int_x
        self.int_y = int_y
        
    def __str__(self) -> str:
        return f"Belt(id={self.id}, position=({self.x}, {self.y}), type={self.type}), item={self.item}"

class GurobiSolver:
    def __init__(self, width, height, production_data):
        self.width = width
        self.height = height
        self.production_data = production_data
        
        # Create Gurobi environment and model
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)  # Suppress Gurobi output
        self.env.start()
        self.model = gp.Model("FactoryOptimization", env=self.env)
        
        # Big M constant for indicator constraints
        self.M = 1000000
        
        # Create variables for assemblers, input inserters, output inserters, and belts
        self.assemblers = []
        self.placed_assembler = []
        self.global_input_belts = []
        self.global_output_belts = [] 
        self.additional_constraints = []
        
        self.input_information = None
        self.output_information = None
        self.obstacle_maps = []
    
    def build_constraints(self):
        """Build all factory layout constraints"""
        self.create_assemblers()
        
        self.add_global_belt_overlap_assembler_constraint()
        self.add_global_belt_overlap_inserter_constraint()
        self.add_global_belt_overlap_belt_constraint()
        
        self.add_bound_constraints_assembler()
        self.add_bound_constraints_belts_and_inserters()
        
        self.add_assembler_overlap_assembler_constraint()
        self.add_assembler_overlap_inserter_constraint()
        self.add_assembler_overlap_belt_constraint()
        
        self.add_inserter_overlap_inserter_constraint()
        self.add_inserter_overlap_belt_constraint()
        
        self.add_belt_overlap_belt_constraint()
        
        self.add_inserter_adjacent_to_assembler()
        
        self.add_space_for_output_of_assembler()
        
        self.add_input_inserter_merge_assembler_constraint()
        
        self.add_minimize_belts()

    def create_assemblers(self):
        """Create assemblers based on production data"""
        logger.info("Starting to create assemblers based on production data.")
        assembler_count = 0
        
        for item_id, item_info in self.production_data.items():
            logger.debug(f"Processing item '{item_id}' with production data: {item_info}")

            if 'assemblers' in item_info and item_info['assemblers'] > 0:
                for i in range(item_info['assemblers']):
                    logger.info(f"Creating assembler for item '{item_id}', instance {i}")
                    
                    # Create input inserters for the assembler
                    input_inserters = []
                    
                    for inserter_info in item_info['input_inserters']:
                        for j in range(inserter_info['inserters']):
                            input_inserter_id = f"{inserter_info['id']}_in_{assembler_count}_{i}_{j}"
                            belt_id = f"{inserter_info['id']}_end_{assembler_count}_{i}_{j}"
                            
                            logger.info(f"Creating input inserter for {inserter_info['id']} at {input_inserter_id}")

                            # Create the input inserter with a unique ID and associated belt
                            input_inserters.append(
                                Inserter(
                                    id=input_inserter_id,
                                    type='input',
                                    item=inserter_info['id'],
                                    model=self.model,
                                    belt=Belt(
                                        id=belt_id,
                                        type="end",  
                                        item=inserter_info['id'],
                                        model=self.model
                                    )
                                )
                            )
                            
                    # create the assembler with the input inserters and unique ID
                    assembler = Assembler(
                        id=f"{item_id}_{assembler_count}_{i}",
                        inserters=input_inserters,
                        item=item_id,
                        capacity=item_info['capacity'],
                        model=self.model
                    )
                    
                    self.assemblers.append(assembler)
                    logger.debug(f"Created assembler with ID: {assembler.id} and input inserters: {[inserter.id for inserter in input_inserters]}")
                    logger.debug(f"Created assembler with ID: {assembler.id} with item = {assembler.item}")
                    logger.debug(f"Created assembler with ID: {assembler.id} with capacity = {assembler.capacity}")
                assembler_count += 1

        logger.info(f"Created {len(self.assemblers)} assemblers in total.")
    
    def add_manuel_IO_constraints(self, input_information, output_information):
        """Add manual input/output constraints based on user-defined information"""
        logger.info("Starting to add manual I/O constraints.")
        
        self.input_information = input_information
        self.output_information = output_information

        for item, data in input_information.items():
            logger.debug(f"Processing input item '{item}' with data: {data}")
            
            # Create and position the input belts (start point)
            belt = Belt(id=f"{item}_{data['input']}_input", type='input', item=item, model=self.model)
            logger.info(f"Adding input belt for item '{item}' at position {data['input']}")
            self.model.addConstr(belt.x == data["input"][1])
            self.model.addConstr(belt.y == data["input"][0])
            self.global_input_belts.append(belt)
            
            # Create and position the output belts (end point)
            belt = Belt(id=f"{item}_{data['output']}_input", type='input', item=item, model=self.model)
            logger.info(f"Adding output belt for item '{item}' at position {data['output']}")
            self.model.addConstr(belt.x == data["output"][1])
            self.model.addConstr(belt.y == data["output"][0])
            self.global_input_belts.append(belt)

            # Process path data directly if it exists
            if data['paths'] is not None and item in data['paths']:
                logger.debug(f"Processing path data for input item '{item}'")
                
                for path_data in data['paths'][item]:
                    path = path_data['path']
                    
                    # Add all points in the path as belts (excluding start and end points)
                    for pos in path[1:-1]:  # Skip first and last points as they're already added as input/output
                        x, y = pos
                        logger.info(f"Adding belt path for '{item}' at position ({x}, {y})")
                        belt = Belt(id=f"{item}_({x}, {y})_input", type='input', item=item, 
                                    int_x=x, int_y=y, model=self.model)
                        self.model.addConstr(belt.x == x)
                        self.model.addConstr(belt.y == y)
                        self.global_input_belts.append(belt)

        for item, data in output_information.items():
            logger.debug(f"Processing output item '{item}' with data: {data}")

            # Create and position the input belts for output item (start point)
            belt = Belt(id=f"{item}_{data['input']}_output", type='output', item=item, model=self.model)
            logger.info(f"Adding input belt for output item '{item}' at position {data['input']}")
            self.model.addConstr(belt.x == data["input"][1])
            self.model.addConstr(belt.y == data["input"][0])
            self.global_output_belts.append(belt)
            
            # Create and position the output belts (end point)
            belt = Belt(id=f"{item}_{data['output']}_output", type='output', item=item, model=self.model)
            logger.info(f"Adding output belt for output item '{item}' at position {data['output']}")
            self.model.addConstr(belt.x == data["output"][1])
            self.model.addConstr(belt.y == data["output"][0])
            self.global_output_belts.append(belt)

            # Process path data directly if it exists
            if data['paths'] is not None and item in data['paths']:
                logger.debug(f"Processing path data for output item '{item}'")
                
                for path_data in data['paths'][item]:
                    path = path_data['path']
                    
                    # Add all points in the path as belts (excluding start and end points)
                    for pos in path[1:-1]:  # Skip first and last points as they're already added as input/output
                        x, y = pos
                        logger.info(f"Adding belt path for '{item}' at position ({x}, {y})")
                        belt = Belt(id=f"{item}_({x}, {y})_output", type='output', item=item, 
                                    int_x=x, int_y=y, model=self.model)
                        self.model.addConstr(belt.x == x)
                        self.model.addConstr(belt.y == y)
                        self.global_output_belts.append(belt)

        logger.info("Finished adding manual I/O constraints. Calling solver.")
        self.solve()

    def add_bound_constraints_assembler(self):
        """Add boundary constraints for assemblers (3x3 grid)"""
        logger.info("Adding boundary constraints for assemblers.")
        for assembler in self.assemblers:
            logger.debug(f"Setting boundary constraints for assembler ID {assembler.id}")
            self.model.addConstr(assembler.x >= 0)
            self.model.addConstr(assembler.x <= self.width - 3)
            self.model.addConstr(assembler.y >= 0)
            self.model.addConstr(assembler.y <= self.height - 3)

    def add_bound_constraints_belts_and_inserters(self):
        """Add boundary constraints for belts and inserters"""
        logger.info("Adding boundary constraints for belts and inserters.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                logger.debug(f"Setting boundary constraints for inserter ID {inserter.id} and belt ID {belt.id}")
                self.model.addConstr(inserter.x >= 0)
                self.model.addConstr(inserter.x <= self.width - 1)
                self.model.addConstr(inserter.y >= 0)
                self.model.addConstr(inserter.y <= self.height - 1)
                self.model.addConstr(belt.x >= 0)
                self.model.addConstr(belt.x <= self.width - 1)
                self.model.addConstr(belt.y >= 0)
                self.model.addConstr(belt.y <= self.height - 1)

    def add_global_belt_overlap_assembler_constraint(self):
        """Prevent global belts from overlapping with assemblers"""
        logger.info("Adding assembler overlap constraints to avoid global belts.")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for belt in belts:
                # Add an OR constraint using binary variables and big-M
                overlap_constr = self.model.addVar(vtype=GRB.BINARY, name=f"overlap_{assembler.id}_{belt.id}")
                
                # Belt is to the left of assembler: belt.x < assembler.x
                self.model.addConstr(belt.x <= assembler.x - 1 + self.M * (1 - overlap_constr))
                
                # Belt is to the right of assembler: belt.x > assembler.x + 2
                right_var = self.model.addVar(vtype=GRB.BINARY, name=f"right_{assembler.id}_{belt.id}")
                self.model.addConstr(belt.x >= assembler.x + 3 - self.M * (1 - right_var))
                
                # Belt is above assembler: belt.y < assembler.y
                above_var = self.model.addVar(vtype=GRB.BINARY, name=f"above_{assembler.id}_{belt.id}")
                self.model.addConstr(belt.y <= assembler.y - 1 + self.M * (1 - above_var))
                
                # Belt is below assembler: belt.y > assembler.y + 2
                below_var = self.model.addVar(vtype=GRB.BINARY, name=f"below_{assembler.id}_{belt.id}")
                self.model.addConstr(belt.y >= assembler.y + 3 - self.M * (1 - below_var))
                
                # At least one of these conditions must be true
                self.model.addConstr(overlap_constr + right_var + above_var + below_var >= 1)

    def add_global_belt_overlap_inserter_constraint(self):
        """Prevent global belts from overlapping with inserters"""
        logger.info("Adding inserter overlap constraints to avoid global belts.")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                for belt in belts:
                    # An inserter and belt can't be at the same position
                    # Instead of (inserter.x != belt.x) | (inserter.y != belt.y)
                    # Use binary variables and big-M method
                    
                    # Create binary variables for x and y dimensions
                    x_diff = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_{inserter.id}_{belt.id}")
                    y_diff = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_{inserter.id}_{belt.id}")
                    
                    # If x_diff = 1, then inserter.x is different from belt.x
                    # x_diff = 1 -> inserter.x + 1 <= belt.x OR belt.x + 1 <= inserter.x
                    x_diff_left = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_left_{inserter.id}_{belt.id}")
                    x_diff_right = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_right_{inserter.id}_{belt.id}")
                    
                    self.model.addConstr(inserter.x + 1 <= belt.x + self.M * (1 - x_diff_left))
                    self.model.addConstr(belt.x + 1 <= inserter.x + self.M * (1 - x_diff_right))
                    self.model.addConstr(x_diff == x_diff_left + x_diff_right)
                    self.model.addConstr(x_diff <= 1)  # At most one direction can be true
                    
                    # Similar for y dimension
                    y_diff_above = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_above_{inserter.id}_{belt.id}")
                    y_diff_below = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_below_{inserter.id}_{belt.id}")
                    
                    self.model.addConstr(inserter.y + 1 <= belt.y + self.M * (1 - y_diff_above))
                    self.model.addConstr(belt.y + 1 <= inserter.y + self.M * (1 - y_diff_below))
                    self.model.addConstr(y_diff == y_diff_above + y_diff_below)
                    self.model.addConstr(y_diff <= 1)  # At most one direction can be true
                    
                    # Either x or y must be different
                    self.model.addConstr(x_diff + y_diff >= 1, name=f"no_overlap_{inserter.id}_{belt.id}")
                    
                    

    def add_global_belt_overlap_belt_constraint(self):
        """Prevent belts with different items from overlapping"""
        logger.info("Adding belt overlap constraints to avoid conflicts with global belts (different items only).")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                inserter_belt = inserter.belt
                for belt in belts:
                    if belt.item != inserter_belt.item:
                        # Belts with different items can't occupy the same position
                        # Replace (inserter_belt.x != belt.x) | (inserter_belt.y != belt.y)
                        
                        # Create binary variables for x and y dimensions
                        x_diff = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_belt_{inserter_belt.id}_{belt.id}")
                        y_diff = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_belt_{inserter_belt.id}_{belt.id}")
                        
                        # Similar to above method
                        x_diff_left = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_left_belt_{inserter_belt.id}_{belt.id}")
                        x_diff_right = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_right_belt_{inserter_belt.id}_{belt.id}")
                        
                        self.model.addConstr(inserter_belt.x + 1 <= belt.x + self.M * (1 - x_diff_left))
                        self.model.addConstr(belt.x + 1 <= inserter_belt.x + self.M * (1 - x_diff_right))
                        self.model.addConstr(x_diff == x_diff_left + x_diff_right)
                        self.model.addConstr(x_diff <= 1)
                        
                        y_diff_above = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_above_belt_{inserter_belt.id}_{belt.id}")
                        y_diff_below = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_below_belt_{inserter_belt.id}_{belt.id}")
                        
                        self.model.addConstr(inserter_belt.y + 1 <= belt.y + self.M * (1 - y_diff_above))
                        self.model.addConstr(belt.y + 1 <= inserter_belt.y + self.M * (1 - y_diff_below))
                        self.model.addConstr(y_diff == y_diff_above + y_diff_below)
                        self.model.addConstr(y_diff <= 1)
                        
                        self.model.addConstr(x_diff + y_diff >= 1, name=f"diff_item_belt_{inserter_belt.id}_{belt.id}")

    def add_assembler_overlap_assembler_constraint(self):
        """Prevent assemblers from overlapping with each other"""
        logger.info("Adding assembler overlap constraints to prevent assembler-assembler overlap.")
        for i, assembler in enumerate(self.assemblers):
            for j, other_assembler in enumerate(self.assemblers):
                if i < j:  # Only process unique pairs
                    # Create binary variables for each constraint
                    left_var = self.model.addVar(vtype=GRB.BINARY, name=f"left_{assembler.id}_{other_assembler.id}")
                    right_var = self.model.addVar(vtype=GRB.BINARY, name=f"right_{assembler.id}_{other_assembler.id}")
                    above_var = self.model.addVar(vtype=GRB.BINARY, name=f"above_{assembler.id}_{other_assembler.id}")
                    below_var = self.model.addVar(vtype=GRB.BINARY, name=f"below_{assembler.id}_{other_assembler.id}")
                    
                    # a1 is to the left of a2: a1.x + 3 <= a2.x
                    self.model.addConstr(assembler.x + 2 <= other_assembler.x - 1 + self.M * (1 - left_var))
                    
                    # a1 is to the right of a2: a1.x >= a2.x + 3
                    self.model.addConstr(assembler.x >= other_assembler.x + 3 - self.M * (1 - right_var))
                    
                    # a1 is above a2: a1.y + 3 <= a2.y
                    self.model.addConstr(assembler.y + 2 <= other_assembler.y - 1 + self.M * (1 - above_var))
                    
                    # a1 is below a2: a1.y >= a2.y + 3
                    self.model.addConstr(assembler.y >= other_assembler.y + 3 - self.M * (1 - below_var))
                    
                    # At least one of these constraints must be satisfied
                    self.model.addConstr(left_var + right_var + above_var + below_var >= 1)

    def add_assembler_overlap_inserter_constraint(self):
        """Prevent assemblers from overlapping with inserters"""
        logger.info("Adding assembler-inserter overlap constraints.")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    for inserter in other_assembler.inserters:
                        # Create binary variables for each constraint
                        left_var = self.model.addVar(vtype=GRB.BINARY, name=f"left_{assembler.id}_{inserter.id}")
                        right_var = self.model.addVar(vtype=GRB.BINARY, name=f"right_{assembler.id}_{inserter.id}")
                        above_var = self.model.addVar(vtype=GRB.BINARY, name=f"above_{assembler.id}_{inserter.id}")
                        below_var = self.model.addVar(vtype=GRB.BINARY, name=f"below_{assembler.id}_{inserter.id}")
                        
                        # Inserter is to the left of assembler
                        self.model.addConstr(inserter.x <= assembler.x - 1 + self.M * (1 - left_var))
                        
                        # Inserter is to the right of assembler
                        self.model.addConstr(inserter.x >= assembler.x + 3 - self.M * (1 - right_var))
                        
                        # Inserter is above assembler
                        self.model.addConstr(inserter.y <= assembler.y - 1 + self.M * (1 - above_var))
                        
                        # Inserter is below assembler
                        self.model.addConstr(inserter.y >= assembler.y + 3 - self.M * (1 - below_var))
                        
                        # At least one constraint must be satisfied
                        self.model.addConstr(left_var + right_var + above_var + below_var >= 1)

    def add_assembler_overlap_belt_constraint(self):
        """Prevent assemblers from overlapping with belts of different items"""
        logger.info("Adding assembler-belt overlap constraints (excluding same-item belts).")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    for inserter in other_assembler.inserters:
                        belt = inserter.belt
                        if belt.item != assembler.item:
                            # Create binary variables for each constraint
                            left_var = self.model.addVar(vtype=GRB.BINARY, name=f"left_{assembler.id}_{belt.id}")
                            right_var = self.model.addVar(vtype=GRB.BINARY, name=f"right_{assembler.id}_{belt.id}")
                            above_var = self.model.addVar(vtype=GRB.BINARY, name=f"above_{assembler.id}_{belt.id}")
                            below_var = self.model.addVar(vtype=GRB.BINARY, name=f"below_{assembler.id}_{belt.id}")
                            
                            # Belt is to the left of assembler
                            self.model.addConstr(belt.x <= assembler.x - 1 + self.M * (1 - left_var))
                            
                            # Belt is to the right of assembler
                            self.model.addConstr(belt.x >= assembler.x + 3 - self.M * (1 - right_var))
                            
                            # Belt is above assembler
                            self.model.addConstr(belt.y <= assembler.y - 1 + self.M * (1 - above_var))
                            
                            # Belt is below assembler
                            self.model.addConstr(belt.y >= assembler.y + 3 - self.M * (1 - below_var))
                            
                            # At least one constraint must be satisfied
                            self.model.addConstr(left_var + right_var + above_var + below_var >= 1)

    def add_inserter_overlap_inserter_constraint(self):
        """Prevent inserters from overlapping with each other"""
        logger.info("Adding inserter overlap constraints to prevent inserter-inserter overlap.")
        # Check inserters within the same assembler
        for assembler in self.assemblers:
            for i, inserter in enumerate(assembler.inserters):
                for j, other_inserter in enumerate(assembler.inserters):
                    if i < j:  # Only process unique pairs
                        # Create binary variables for x and y dimensions
                        x_diff = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_ins_{inserter.id}_{other_inserter.id}")
                        y_diff = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_ins_{inserter.id}_{other_inserter.id}")
                        
                        # Configure the x_diff variable
                        x_diff_left = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_left_ins_{inserter.id}_{other_inserter.id}")
                        x_diff_right = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_right_ins_{inserter.id}_{other_inserter.id}")
                        
                        self.model.addConstr(inserter.x + 1 <= other_inserter.x + self.M * (1 - x_diff_left))
                        self.model.addConstr(other_inserter.x + 1 <= inserter.x + self.M * (1 - x_diff_right))
                        self.model.addConstr(x_diff == x_diff_left + x_diff_right)
                        self.model.addConstr(x_diff <= 1)
                        
                        # Configure the y_diff variable
                        y_diff_above = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_above_ins_{inserter.id}_{other_inserter.id}")
                        y_diff_below = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_below_ins_{inserter.id}_{other_inserter.id}")
                        
                        self.model.addConstr(inserter.y + 1 <= other_inserter.y + self.M * (1 - y_diff_above))
                        self.model.addConstr(other_inserter.y + 1 <= inserter.y + self.M * (1 - y_diff_below))
                        self.model.addConstr(y_diff == y_diff_above + y_diff_below)
                        self.model.addConstr(y_diff <= 1)
                        
                        # Either x or y must be different
                        self.model.addConstr(x_diff + y_diff >= 1, name=f"no_overlap_ins_{inserter.id}_{other_inserter.id}")
        
        # Check inserters between different assemblers
        for i, assembler in enumerate(self.assemblers):
            for j, other_assembler in enumerate(self.assemblers):
                if i < j:  # Only process unique pairs
                    for inserter in assembler.inserters:
                        for other_inserter in other_assembler.inserters:
                            # Create binary variables for x and y dimensions
                            x_diff = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_ins_diff_{inserter.id}_{other_inserter.id}")
                            y_diff = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_ins_diff_{inserter.id}_{other_inserter.id}")
                            
                            # Configure the x_diff variable
                            x_diff_left = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_left_ins_diff_{inserter.id}_{other_inserter.id}")
                            x_diff_right = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_right_ins_diff_{inserter.id}_{other_inserter.id}")
                            
                            self.model.addConstr(inserter.x + 1 <= other_inserter.x + self.M * (1 - x_diff_left))
                            self.model.addConstr(other_inserter.x + 1 <= inserter.x + self.M * (1 - x_diff_right))
                            self.model.addConstr(x_diff == x_diff_left + x_diff_right)
                            self.model.addConstr(x_diff <= 1)
                            
                            # Configure the y_diff variable
                            y_diff_above = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_above_ins_diff_{inserter.id}_{other_inserter.id}")
                            y_diff_below = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_below_ins_diff_{inserter.id}_{other_inserter.id}")
                            
                            self.model.addConstr(inserter.y + 1 <= other_inserter.y + self.M * (1 - y_diff_above))
                            self.model.addConstr(other_inserter.y + 1 <= inserter.y + self.M * (1 - y_diff_below))
                            self.model.addConstr(y_diff == y_diff_above + y_diff_below)
                            self.model.addConstr(y_diff <= 1)
                            
                            # Either x or y must be different
                            self.model.addConstr(x_diff + y_diff >= 1, name=f"no_overlap_ins_diff_{inserter.id}_{other_inserter.id}")


    def add_inserter_overlap_belt_constraint(self):
        """Prevent inserters from overlapping with belts"""
        logger.info("Adding inserter-belt overlap constraints to prevent inserters from overlapping with belts.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        for other_inserter in other_assembler.inserters:
                            other_belt = other_inserter.belt
                            # Similar pattern as above
                            x_diff = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_ins_belt_{inserter.id}_{other_belt.id}")
                            y_diff = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_ins_belt_{inserter.id}_{other_belt.id}")
                            
                            x_diff_left = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_left_ins_belt_{inserter.id}_{other_belt.id}")
                            x_diff_right = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_right_ins_belt_{inserter.id}_{other_belt.id}")
                            
                            self.model.addConstr(inserter.x + 1 <= other_belt.x + self.M * (1 - x_diff_left))
                            self.model.addConstr(other_belt.x + 1 <= inserter.x + self.M * (1 - x_diff_right))
                            self.model.addConstr(x_diff == x_diff_left + x_diff_right)
                            self.model.addConstr(x_diff <= 1)
                            
                            y_diff_above = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_above_ins_belt_{inserter.id}_{other_belt.id}")
                            y_diff_below = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_below_ins_belt_{inserter.id}_{other_belt.id}")
                            
                            self.model.addConstr(inserter.y + 1 <= other_belt.y + self.M * (1 - y_diff_above))
                            self.model.addConstr(other_belt.y + 1 <= inserter.y + self.M * (1 - y_diff_below))
                            self.model.addConstr(y_diff == y_diff_above + y_diff_below)
                            self.model.addConstr(y_diff <= 1)
                            
                            self.model.addConstr(x_diff + y_diff >= 1, name=f"no_overlap_{inserter.id}_{other_belt.id}")

    def add_belt_overlap_belt_constraint(self):
        """Prevent belts with different items from overlapping"""
        logger.info("Adding belt overlap constraints to prevent belt-belt overlap (except same-item belts).")
        for i, assembler in enumerate(self.assemblers):
            for inserter in assembler.inserters:
                belt = inserter.belt
                for j, other_assembler in enumerate(self.assemblers):
                    if i < j:  # Only process unique pairs
                        for other_inserter in other_assembler.inserters:
                            other_belt = other_inserter.belt
                            if belt.item != other_belt.item:
                                # Apply the same pattern
                                x_diff = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_belts_{belt.id}_{other_belt.id}")
                                y_diff = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_belts_{belt.id}_{other_belt.id}")
                                
                                x_diff_left = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_left_belts_{belt.id}_{other_belt.id}")
                                x_diff_right = self.model.addVar(vtype=GRB.BINARY, name=f"x_diff_right_belts_{belt.id}_{other_belt.id}")
                                
                                self.model.addConstr(belt.x + 1 <= other_belt.x + self.M * (1 - x_diff_left))
                                self.model.addConstr(other_belt.x + 1 <= belt.x + self.M * (1 - x_diff_right))
                                self.model.addConstr(x_diff == x_diff_left + x_diff_right)
                                self.model.addConstr(x_diff <= 1)
                                
                                y_diff_above = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_above_belts_{belt.id}_{other_belt.id}")
                                y_diff_below = self.model.addVar(vtype=GRB.BINARY, name=f"y_diff_below_belts_{belt.id}_{other_belt.id}")
                                
                                self.model.addConstr(belt.y + 1 <= other_belt.y + self.M * (1 - y_diff_above))
                                self.model.addConstr(other_belt.y + 1 <= belt.y + self.M * (1 - y_diff_below))
                                self.model.addConstr(y_diff == y_diff_above + y_diff_below)
                                self.model.addConstr(y_diff <= 1)
                                
                                self.model.addConstr(x_diff + y_diff >= 1, name=f"no_overlap_{belt.id}_{other_belt.id}")

    def add_inserter_adjacent_to_assembler(self):
        """Ensure inserters are adjacent to their assemblers and belts are positioned correctly"""
        logger.info("Adding inserter adjacency constraints.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                # Define all possible positions adjacent to the assembler
                input_positions = [
                    (assembler.x, assembler.y - 1), (assembler.x + 1, assembler.y - 1), (assembler.x + 2, assembler.y - 1),  # Top
                    (assembler.x - 1, assembler.y), (assembler.x - 1, assembler.y + 1), (assembler.x - 1, assembler.y + 2),  # Left
                    (assembler.x + 3, assembler.y), (assembler.x + 3, assembler.y + 1), (assembler.x + 3, assembler.y + 2),  # Right
                    (assembler.x, assembler.y + 3), (assembler.x + 1, assembler.y + 3), (assembler.x + 2, assembler.y + 3)   # Bottom
                ]
                
                # Create binary variables for each position
                position_vars = []
                for i, pos in enumerate(input_positions):
                    pos_var = self.model.addVar(vtype=GRB.BINARY, name=f"pos_{inserter.id}_{i}")
                    position_vars.append(pos_var)
                    
                    # If this position is chosen, set inserter coordinates
                    self.model.addConstr(inserter.x >= pos[0] - self.M * (1 - pos_var))
                    self.model.addConstr(inserter.x <= pos[0] + self.M * (1 - pos_var))
                    self.model.addConstr(inserter.y >= pos[1] - self.M * (1 - pos_var))
                    self.model.addConstr(inserter.y <= pos[1] + self.M * (1 - pos_var))
                
                # Exactly one position must be chosen
                self.model.addConstr(quicksum(position_vars) == 1)
                
                # Position the belt based on inserter position
                belt = inserter.belt
                
                # Top positions (0, 1, 2)
                for i in range(3):
                    # If inserter is in top position, belt is one position above
                    self.model.addConstr(belt.x >= inserter.x - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.x <= inserter.x + self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y >= inserter.y - 1 - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y <= inserter.y - 1 + self.M * (1 - position_vars[i]))
                
                # Left positions (3, 4, 5)
                for i in range(3, 6):
                    # If inserter is in left position, belt is one position to the left
                    self.model.addConstr(belt.x >= inserter.x - 1 - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.x <= inserter.x - 1 + self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y >= inserter.y - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y <= inserter.y + self.M * (1 - position_vars[i]))
                
                # Right positions (6, 7, 8)
                for i in range(6, 9):
                    # If inserter is in right position, belt is one position to the right
                    self.model.addConstr(belt.x >= inserter.x + 1 - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.x <= inserter.x + 1 + self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y >= inserter.y - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y <= inserter.y + self.M * (1 - position_vars[i]))
                
                # Bottom positions (9, 10, 11)
                for i in range(9, 12):
                    # If inserter is in bottom position, belt is one position below
                    self.model.addConstr(belt.x >= inserter.x - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.x <= inserter.x + self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y >= inserter.y + 1 - self.M * (1 - position_vars[i]))
                    self.model.addConstr(belt.y <= inserter.y + 1 + self.M * (1 - position_vars[i]))

    def add_space_for_output_of_assembler(self):
        """Ensure there is space for output of each assembler"""
        logger.info("Adding output space constraints.")
        for assembler in self.assemblers:
            # Define potential positions for output
            output_positions = [
                # Each entry is a pair of positions [(x1, y1), (x2, y2)]
                [(assembler.x, assembler.y - 1), (assembler.x, assembler.y - 2)],       # Top left
                [(assembler.x + 1, assembler.y - 1), (assembler.x + 1, assembler.y - 2)], # Top middle
                [(assembler.x + 2, assembler.y - 1), (assembler.x + 2, assembler.y - 2)], # Top right
                
                [(assembler.x, assembler.y + 3), (assembler.x, assembler.y + 4)],       # Bottom left
                [(assembler.x + 1, assembler.y + 3), (assembler.x + 1, assembler.y + 4)], # Bottom middle
                [(assembler.x + 2, assembler.y + 3), (assembler.x + 2, assembler.y + 4)], # Bottom right
                
                [(assembler.x - 1, assembler.y), (assembler.x - 2, assembler.y)],       # Left top
                [(assembler.x - 1, assembler.y + 1), (assembler.x - 2, assembler.y + 1)], # Left middle
                [(assembler.x - 1, assembler.y + 2), (assembler.x - 2, assembler.y + 2)], # Left bottom
                
                [(assembler.x + 3, assembler.y), (assembler.x + 4, assembler.y)],       # Right top
                [(assembler.x + 3, assembler.y + 1), (assembler.x + 4, assembler.y + 1)], # Right middle
                [(assembler.x + 3, assembler.y + 2), (assembler.x + 4, assembler.y + 2)]  # Right bottom
            ]
            
            # Create binary variables for each pair of positions
            pair_vars = []
            for i, (pos1, pos2) in enumerate(output_positions):
                pair_var = self.model.addVar(vtype=GRB.BINARY, name=f"output_pair_{assembler.id}_{i}")
                pair_vars.append(pair_var)
                
                # Create binary variables to check if positions are in bounds
                pos1_in_bounds = self.model.addVar(vtype=GRB.BINARY, name=f"pos1_in_bounds_{assembler.id}_{i}")
                pos2_in_bounds = self.model.addVar(vtype=GRB.BINARY, name=f"pos2_in_bounds_{assembler.id}_{i}")
                
                # Check if position 1 is within bounds using binary variables and big-M
                # For x: 0 <= pos1[0] < self.width
                pos1_x_lower = self.model.addVar(vtype=GRB.BINARY, name=f"pos1_x_lower_{assembler.id}_{i}")
                pos1_x_upper = self.model.addVar(vtype=GRB.BINARY, name=f"pos1_x_upper_{assembler.id}_{i}")
                
                self.model.addConstr(pos1[0] >= 0 - self.M * (1 - pos1_x_lower))
                self.model.addConstr(pos1[0] <= self.width - 1 + self.M * (1 - pos1_x_upper))
                
                # For y: 0 <= pos1[1] < self.height
                pos1_y_lower = self.model.addVar(vtype=GRB.BINARY, name=f"pos1_y_lower_{assembler.id}_{i}")
                pos1_y_upper = self.model.addVar(vtype=GRB.BINARY, name=f"pos1_y_upper_{assembler.id}_{i}")
                
                self.model.addConstr(pos1[1] >= 0 - self.M * (1 - pos1_y_lower))
                self.model.addConstr(pos1[1] <= self.height - 1 + self.M * (1 - pos1_y_upper))
                
                # pos1 is in bounds if all conditions are satisfied
                self.model.addConstr(pos1_in_bounds <= pos1_x_lower)
                self.model.addConstr(pos1_in_bounds <= pos1_x_upper)
                self.model.addConstr(pos1_in_bounds <= pos1_y_lower)
                self.model.addConstr(pos1_in_bounds <= pos1_y_upper)
                self.model.addConstr(pos1_in_bounds >= pos1_x_lower + pos1_x_upper + pos1_y_lower + pos1_y_upper - 3)
                
                # Similar for position 2
                pos2_x_lower = self.model.addVar(vtype=GRB.BINARY, name=f"pos2_x_lower_{assembler.id}_{i}")
                pos2_x_upper = self.model.addVar(vtype=GRB.BINARY, name=f"pos2_x_upper_{assembler.id}_{i}")
                
                self.model.addConstr(pos2[0] >= 0 - self.M * (1 - pos2_x_lower))
                self.model.addConstr(pos2[0] <= self.width - 1 + self.M * (1 - pos2_x_upper))
                
                pos2_y_lower = self.model.addVar(vtype=GRB.BINARY, name=f"pos2_y_lower_{assembler.id}_{i}")
                pos2_y_upper = self.model.addVar(vtype=GRB.BINARY, name=f"pos2_y_upper_{assembler.id}_{i}")
                
                self.model.addConstr(pos2[1] >= 0 - self.M * (1 - pos2_y_lower))
                self.model.addConstr(pos2[1] <= self.height - 1 + self.M * (1 - pos2_y_upper))
                
                self.model.addConstr(pos2_in_bounds <= pos2_x_lower)
                self.model.addConstr(pos2_in_bounds <= pos2_x_upper)
                self.model.addConstr(pos2_in_bounds <= pos2_y_lower)
                self.model.addConstr(pos2_in_bounds <= pos2_y_upper)
                self.model.addConstr(pos2_in_bounds >= pos2_x_lower + pos2_x_upper + pos2_y_lower + pos2_y_upper - 3)
                
                # If positions are in bounds, check if they are empty
                pos1_empty = self.model.addVar(vtype=GRB.BINARY, name=f"pos1_empty_{assembler.id}_{i}")
                pos2_empty = self.model.addVar(vtype=GRB.BINARY, name=f"pos2_empty_{assembler.id}_{i}")
                
                # Only check if position is occupied if it's in bounds
                # For position 1: if in bounds, check if empty; if not in bounds, consider occupied
                occupied_var1 = self.model.addVar(vtype=GRB.BINARY, name=f"occupied_var1_{assembler.id}_{i}")
                self.model.addConstr(occupied_var1 <= 1 - pos1_in_bounds + 
                                    self.is_position_occupied_var(pos1[0], pos1[1]) * pos1_in_bounds)
                self.model.addConstr(pos1_empty == 1 - occupied_var1)
                
                # For position 2: similar logic
                occupied_var2 = self.model.addVar(vtype=GRB.BINARY, name=f"occupied_var2_{assembler.id}_{i}")
                self.model.addConstr(occupied_var2 <= 1 - pos2_in_bounds + 
                                    self.is_position_occupied_var(pos2[0], pos2[1]) * pos2_in_bounds)
                self.model.addConstr(pos2_empty == 1 - occupied_var2)
                
                # For a pair to be valid, both positions must be in bounds and empty
                self.model.addConstr(pair_var <= pos1_in_bounds)
                self.model.addConstr(pair_var <= pos2_in_bounds)
                self.model.addConstr(pair_var <= pos1_empty)
                self.model.addConstr(pair_var <= pos2_empty)
            
            # At least one pair must be valid for output
            if pair_vars:
                self.model.addConstr(quicksum(pair_vars) >= 1)

    def is_position_occupied_var(self, x_expr, y_expr):
        """Create a variable indicating if a position represented by expressions might be occupied"""
        # Create a unique name for the variable based on expression IDs
        var_name = f"occupied_{id(x_expr)}_{id(y_expr)}"
        occupied_var = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        # A position is occupied if any assembler, inserter, or belt is at that position
        occupation_indicators = []
        
        # Check assemblers (3x3 grid)
        for assembler in self.assemblers:
            assembler_occupies = self.model.addVar(vtype=GRB.BINARY, name=f"assembler_at_{var_name}_{assembler.id}")
            
            # Check if x_expr is within assembler's x range
            x_in_range = self.model.addVar(vtype=GRB.BINARY, name=f"x_in_{var_name}_{assembler.id}")
            self.model.addConstr(x_expr >= assembler.x - self.M * (1 - x_in_range))
            self.model.addConstr(x_expr <= assembler.x + 2 + self.M * (1 - x_in_range))
            
            # Check if y_expr is within assembler's y range
            y_in_range = self.model.addVar(vtype=GRB.BINARY, name=f"y_in_{var_name}_{assembler.id}")
            self.model.addConstr(y_expr >= assembler.y - self.M * (1 - y_in_range))
            self.model.addConstr(y_expr <= assembler.y + 2 + self.M * (1 - y_in_range))
            
            # Position is occupied by assembler if it's in both ranges
            self.model.addConstr(assembler_occupies <= x_in_range)
            self.model.addConstr(assembler_occupies <= y_in_range)
            self.model.addConstr(assembler_occupies >= x_in_range + y_in_range - 1)
            
            occupation_indicators.append(assembler_occupies)
            
            # Check inserters and belts for this assembler
            for inserter in assembler.inserters:
                # Check if inserter is at the position
                inserter_occupies = self.model.addVar(vtype=GRB.BINARY, name=f"inserter_at_{var_name}_{inserter.id}")
                
                # x_expr == inserter.x && y_expr == inserter.y
                x_match = self.model.addVar(vtype=GRB.BINARY, name=f"x_match_ins_{var_name}_{inserter.id}")
                self.model.addConstr(x_expr <= inserter.x + self.M * (1 - x_match))
                self.model.addConstr(x_expr >= inserter.x - self.M * (1 - x_match))
                
                y_match = self.model.addVar(vtype=GRB.BINARY, name=f"y_match_ins_{var_name}_{inserter.id}")
                self.model.addConstr(y_expr <= inserter.y + self.M * (1 - y_match))
                self.model.addConstr(y_expr >= inserter.y - self.M * (1 - y_match))
                
                self.model.addConstr(inserter_occupies <= x_match)
                self.model.addConstr(inserter_occupies <= y_match)
                self.model.addConstr(inserter_occupies >= x_match + y_match - 1)
                
                occupation_indicators.append(inserter_occupies)
                
                # Check if belt is at the position
                belt = inserter.belt
                belt_occupies = self.model.addVar(vtype=GRB.BINARY, name=f"belt_at_{var_name}_{belt.id}")
                
                # x_expr == belt.x && y_expr == belt.y
                x_match_belt = self.model.addVar(vtype=GRB.BINARY, name=f"x_match_belt_{var_name}_{belt.id}")
                self.model.addConstr(x_expr <= belt.x + self.M * (1 - x_match_belt))
                self.model.addConstr(x_expr >= belt.x - self.M * (1 - x_match_belt))
                
                y_match_belt = self.model.addVar(vtype=GRB.BINARY, name=f"y_match_belt_{var_name}_{belt.id}")
                self.model.addConstr(y_expr <= belt.y + self.M * (1 - y_match_belt))
                self.model.addConstr(y_expr >= belt.y - self.M * (1 - y_match_belt))
                
                self.model.addConstr(belt_occupies <= x_match_belt)
                self.model.addConstr(belt_occupies <= y_match_belt)
                self.model.addConstr(belt_occupies >= x_match_belt + y_match_belt - 1)
                
                occupation_indicators.append(belt_occupies)
        
        # Check global belts
        for belt in self.global_input_belts + self.global_output_belts:
            global_belt_occupies = self.model.addVar(vtype=GRB.BINARY, name=f"global_belt_at_{var_name}_{belt.id}")
            
            # x_expr == belt.x && y_expr == belt.y
            x_match_global = self.model.addVar(vtype=GRB.BINARY, name=f"x_match_global_{var_name}_{belt.id}")
            self.model.addConstr(x_expr <= belt.x + self.M * (1 - x_match_global))
            self.model.addConstr(x_expr >= belt.x - self.M * (1 - x_match_global))
            
            y_match_global = self.model.addVar(vtype=GRB.BINARY, name=f"y_match_global_{var_name}_{belt.id}")
            self.model.addConstr(y_expr <= belt.y + self.M * (1 - y_match_global))
            self.model.addConstr(y_expr >= belt.y - self.M * (1 - y_match_global))
            
            self.model.addConstr(global_belt_occupies <= x_match_global)
            self.model.addConstr(global_belt_occupies <= y_match_global)
            self.model.addConstr(global_belt_occupies >= x_match_global + y_match_global - 1)
            
            occupation_indicators.append(global_belt_occupies)
        
        # Position is occupied if any of the indicators is true
        if occupation_indicators:
            self.model.addConstr(occupied_var >= quicksum(occupation_indicators) / len(occupation_indicators))
            self.model.addConstr(occupied_var <= quicksum(occupation_indicators))
        else:
            self.model.addConstr(occupied_var == 0)  # Nothing to check = not occupied
        
        return occupied_var

    def add_input_inserter_merge_assembler_constraint(self):
        """Ensure input inserters can merge with appropriate assembler edges"""
        logger.info("Adding input inserter merge constraints.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                if not inserter.direct_interaction:
                    for other_assembler in self.assemblers:
                        if assembler.id != other_assembler.id and inserter.item == other_assembler.item and other_assembler.capacity > 0:
                            # Define valid edge positions around the other assembler
                            edge_positions = [
                                (other_assembler.x, other_assembler.y),       # Top left
                                (other_assembler.x + 1, other_assembler.y),   # Top middle
                                (other_assembler.x + 2, other_assembler.y),   # Top right
                                (other_assembler.x, other_assembler.y + 1),   # Middle left
                                (other_assembler.x + 2, other_assembler.y + 1), # Middle right
                                (other_assembler.x, other_assembler.y + 2),   # Bottom left
                                (other_assembler.x + 1, other_assembler.y + 2), # Bottom middle
                                (other_assembler.x + 2, other_assembler.y + 2)  # Bottom right
                            ]
                            
                            # Create binary variables for each position
                            merge_vars = []
                            for i, pos in enumerate(edge_positions):
                                merge_var = self.model.addVar(vtype=GRB.BINARY, name=f"merge_{inserter.id}_{other_assembler.id}_{i}")
                                merge_vars.append(merge_var)
                                
                                # If this position is chosen for merging
                                self.model.addConstr(inserter.belt.x >= pos[0] - self.M * (1 - merge_var))
                                self.model.addConstr(inserter.belt.x <= pos[0] + self.M * (1 - merge_var))
                                self.model.addConstr(inserter.belt.y >= pos[1] - self.M * (1 - merge_var))
                                self.model.addConstr(inserter.belt.y <= pos[1] + self.M * (1 - merge_var))
                            
                            # If any merge position is used, mark as direct interaction
                            merge_sum = self.model.addVar(vtype=GRB.BINARY, name=f"merge_sum_{inserter.id}_{other_assembler.id}")
                            self.model.addConstr(merge_sum == quicksum(merge_vars))
                            
                            # When merge occurs, reduce other assembler's capacity
                            # This would need dynamic constraint updates, handled separately

    def add_minimize_belts(self):
        """Maximize belt overlaps to minimize total belt count"""
        logger.info("Adding belt overlap optimization objective.")
        
        # Create variables to track overlaps
        overlap_vars = []
        
        # Track inserter belt overlaps with global belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                
                # Check overlaps with global input/output belts
                for global_belt in self.global_input_belts + self.global_output_belts:
                    if belt.item == global_belt.item:
                        overlap_var = self.model.addVar(vtype=GRB.BINARY, name=f"overlap_{belt.id}_{global_belt.id}")
                        
                        # Belt overlaps if positions are the same
                        self.model.addConstr(belt.x - global_belt.x <= self.M * (1 - overlap_var))
                        self.model.addConstr(global_belt.x - belt.x <= self.M * (1 - overlap_var))
                        self.model.addConstr(belt.y - global_belt.y <= self.M * (1 - overlap_var))
                        self.model.addConstr(global_belt.y - belt.y <= self.M * (1 - overlap_var))
                        
                        overlap_vars.append(overlap_var)
                
                # Check overlaps with assembler edges
                for other_assembler in self.assemblers:
                    if inserter.item == other_assembler.item and assembler.id != other_assembler.id:
                        edge_positions = [
                            (other_assembler.x, other_assembler.y),       # Top left
                            (other_assembler.x + 1, other_assembler.y),   # Top middle
                            (other_assembler.x + 2, other_assembler.y),   # Top right
                            (other_assembler.x, other_assembler.y + 1),   # Middle left
                            (other_assembler.x + 2, other_assembler.y + 1), # Middle right
                            (other_assembler.x, other_assembler.y + 2),   # Bottom left
                            (other_assembler.x + 1, other_assembler.y + 2), # Bottom middle
                            (other_assembler.x + 2, other_assembler.y + 2)  # Bottom right
                        ]
                        
                        for i, pos in enumerate(edge_positions):
                            edge_overlap_var = self.model.addVar(vtype=GRB.BINARY, name=f"edge_overlap_{belt.id}_{other_assembler.id}_{i}")
                            
                            # Belt overlaps with edge position
                            self.model.addConstr(belt.x - pos[0] <= self.M * (1 - edge_overlap_var))
                            self.model.addConstr(pos[0] - belt.x <= self.M * (1 - edge_overlap_var))
                            self.model.addConstr(belt.y - pos[1] <= self.M * (1 - edge_overlap_var))
                            self.model.addConstr(pos[1] - belt.y <= self.M * (1 - edge_overlap_var))
                            
                            overlap_vars.append(edge_overlap_var)
        
        # Set the objective to maximize belt overlaps
        if overlap_vars:
            self.model.setObjective(quicksum(overlap_vars), GRB.MAXIMIZE)

    def find_non_overlapping_inserters(self):
        """Identify inserters that don't overlap with global belts but have a matching item"""
        logger.info("Identifying non-overlapping inserters.")
        non_overlapping_inserters = []

        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                same_item_belts = [gb for gb in self.global_input_belts + self.global_output_belts if gb.item == belt.item]

                # Check if the belt overlaps with any global belt of the same type
                belt_x = belt.x.x  # Get Gurobi variable value
                belt_y = belt.y.x
                
                is_overlapping = any(gb.x.x == belt_x and gb.y.x == belt_y for gb in same_item_belts)

                if not is_overlapping and same_item_belts:
                    logger.info(f"Inserter {inserter.id} does not overlap but has matching item.")
                    non_overlapping_inserters.append((inserter, same_item_belts))
                else:
                    logger.info(f"Inserter {inserter.id} is locked (either overlaps or has no matching global belt).")

        return non_overlapping_inserters

    def lock_initial_solution(self, non_overlapping_inserters):
        """Lock positions of everything except non-overlapping inserters"""
        logger.info("Locking initial solution.")
        
        non_overlapping_set = {inserter for inserter, _ in non_overlapping_inserters}

        # Keep assembler positions unchanged
        for assembler in self.assemblers:
            assembler_x = assembler.x.x  # Get current value from Gurobi
            assembler_y = assembler.y.x
            
            logger.info(f"Locking assembler {assembler.id} at ({assembler_x}, {assembler_y}).")
            self.model.addConstr(assembler.x == assembler_x)
            self.model.addConstr(assembler.y == assembler_y)

            for inserter in assembler.inserters:
                if inserter in non_overlapping_set:
                    logger.info(f"Inserter {inserter.id} is free to move.")
                    continue

                inserter_x = inserter.x.x
                inserter_y = inserter.y.x
                logger.info(f"Locking inserter {inserter.id} at ({inserter_x}, {inserter_y}).")
                
                self.model.addConstr(inserter.x == inserter_x)
                self.model.addConstr(inserter.y == inserter_y)

                # Keep belt positions fixed
                belt = inserter.belt
                belt_x = belt.x.x
                belt_y = belt.y.x
                
                self.model.addConstr(belt.x == belt_x)
                self.model.addConstr(belt.y == belt_y)

    def minimize_non_overlapping_inserters(self, non_overlapping_inserters):
        """Minimize distance from non-overlapping inserters to closest global belt"""
        logger.info("Minimizing distance for non-overlapping inserters.")

        if not non_overlapping_inserters:
            return

        distance_vars = []

        for inserter, same_item_belts in non_overlapping_inserters:
            belt = inserter.belt
            
            # For each possible global belt, create a distance variable
            for gb_idx, gb in enumerate(same_item_belts):
                dist_var = self.model.addVar(name=f"dist_{belt.id}_to_{gb.id}")
                
                # Calculate Manhattan distance: |x1-x2| + |y1-y2|
                x_diff_pos = self.model.addVar(name=f"x_diff_pos_{belt.id}_to_{gb.id}")
                x_diff_neg = self.model.addVar(name=f"x_diff_neg_{belt.id}_to_{gb.id}")
                y_diff_pos = self.model.addVar(name=f"y_diff_pos_{belt.id}_to_{gb.id}")
                y_diff_neg = self.model.addVar(name=f"y_diff_neg_{belt.id}_to_{gb.id}")
                
                # x_diff = belt.x - gb.x
                self.model.addConstr(belt.x - gb.x == x_diff_pos - x_diff_neg)
                self.model.addConstr(x_diff_pos >= 0)
                self.model.addConstr(x_diff_neg >= 0)
                
                # y_diff = belt.y - gb.y
                self.model.addConstr(belt.y - gb.y == y_diff_pos - y_diff_neg)
                self.model.addConstr(y_diff_pos >= 0)
                self.model.addConstr(y_diff_neg >= 0)
                
                # dist = |x_diff| + |y_diff| = x_diff_pos + x_diff_neg + y_diff_pos + y_diff_neg
                self.model.addConstr(dist_var == x_diff_pos + x_diff_neg + y_diff_pos + y_diff_neg)
                
                # Is this the selected belt to be close to?
                selected_var = self.model.addVar(vtype=GRB.BINARY, name=f"selected_{belt.id}_to_{gb.id}")
                
                # If selected, contribute to objective
                dist_selected = self.model.addVar(name=f"dist_selected_{belt.id}_to_{gb.id}")
                self.model.addConstr(dist_selected >= dist_var - self.M * (1 - selected_var))
                self.model.addConstr(dist_selected <= dist_var + self.M * (1 - selected_var))
                self.model.addConstr(dist_selected <= self.M * selected_var)
                
                distance_vars.append(dist_selected)
            
            # Ensure exactly one global belt is selected for each inserter
            self.model.addConstr(quicksum(self.model.getVarByName(f"selected_{belt.id}_to_{gb.id}") 
                                        for gb_idx, gb in enumerate(same_item_belts)) == 1)

        # Set objective to minimize total distance
        self.model.setObjective(quicksum(distance_vars), GRB.MINIMIZE)

    def solve(self):
        """Solve the optimization problem"""
        logger.info("Starting to solve the optimization model.")
        
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            logger.info(f"Solver status: {self.model.status} - Solution found")
            
            # Identify non-overlapping inserters
            non_overlapping_inserters = self.find_non_overlapping_inserters()
            
            if non_overlapping_inserters:
                # Lock the initial solution
                self.lock_initial_solution(non_overlapping_inserters)
                
                # Add distance minimization for non-overlapping inserters
                self.minimize_non_overlapping_inserters(non_overlapping_inserters)
                
                # Solve again with the distance minimization objective
                self.model.optimize()
                logger.info(f"Final solver status: {self.model.status}")
            
            # Build the map representation
            self.build_map()
            return True
        else:
            logger.warning(f"Solver failed to find a solution. Status: {self.model.status}")
            return False

    def build_map(self):
        """Build a map representation of the factory layout"""
        # Initialize with Python lists instead of NumPy array
        obstacle_map = [[0 for _ in range(self.width)] for _ in range(self.height)]

        belt_point_information = []
        assembler_information = []
        inserter_information = []
        
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            # Add input and output belts
            for belt in self.global_input_belts:
                x = int(belt.x.x)
                y = int(belt.y.x)
                
                obstacle_map[y][x] = 1
                
            for belt in self.global_output_belts:
                x = int(belt.x.x)
                y = int(belt.y.x)
                obstacle_map[y][x] = 22
            
            # Mark assemblers in the obstacle map
            for assembler in self.assemblers:
                x = int(assembler.x.x)
                y = int(assembler.y.x)
                
                assembler_information.append([assembler.item, x, y])
                
                # Mark 3x3 area around the assembler as occupied
                for dx in range(3):
                    for dy in range(3):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            obstacle_map[y + dy][x + dx] = 33

                # Mark inserters in the obstacle map
                for inserter in assembler.inserters:
                    ix = int(inserter.x.x)
                    iy = int(inserter.y.x)
                    
                    direction = "north"
                    
                    # Check if inserter is above the assembler
                    if iy == y - 1 and ix >= x and ix <= x + 2:
                        direction = "south"  # Facing down toward the assembler
                    # Check if inserter is below the assembler
                    elif iy == y + 3 and ix >= x and ix <= x + 2:
                        direction = "north"  # Facing up toward the assembler
                    # Check if inserter is to the left of the assembler
                    elif ix == x - 1 and iy >= y and iy <= y + 2:
                        direction = "east"   # Facing right toward the assembler
                    # Check if inserter is to the right of the assembler
                    elif ix == x + 3 and iy >= y and iy <= y + 2:
                        direction = "west"   # Facing left toward the assembler
                    
                    inserter_information.append([inserter.item, ix, iy, direction])
                    
                    # Mark inserter position as occupied
                    if inserter.type == "input" and 0 <= ix < self.width and 0 <= iy < self.height:
                        obstacle_map[iy][ix] = 44
                            
                    belt = inserter.belt
                    if belt is not None:
                        bx = int(belt.x.x)
                        by = int(belt.y.x)

                        if belt.type == "start" and 0 <= bx < self.width and 0 <= by < self.height:
                            obstacle_map[by][bx] = 88
                        elif belt.type == "end" and 0 <= bx < self.width and 0 <= by < self.height:
                            obstacle_map[by][bx] = 99
                            
                        if bx > -1 and by > -1:
                            start_id = belt.id.split('_')[0]
                            belt_point_information.append([start_id, bx, by, belt.type])
        
        return obstacle_map, belt_point_information, assembler_information, inserter_information

    def get_init_map(self):
        """Get initial map with just global belts"""
        obstacle_map = np.zeros((self.height, self.width), dtype=int)
        
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            for belt in self.global_input_belts:
                x = int(belt.x.x)
                y = int(belt.y.x)
                obstacle_map[y][x] = 1
                    
            for belt in self.global_output_belts:
                x = int(belt.x.x)
                y = int(belt.y.x)
                obstacle_map[y][x] = 22
                
        return obstacle_map

    def restrict_current_setup(self):
        """Add constraints to forbid the current solution"""
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            # Collect current positions of all assemblers and inserters
            position_constraints = []
            
            for assembler in self.assemblers:
                current_x = int(assembler.x.x)
                current_y = int(assembler.y.x)
                
                # Create a binary variable for this assembler's current position
                pos_var = self.model.addVar(vtype=GRB.BINARY, name=f"curr_pos_{assembler.id}")
                
                # This variable is 1 if the assembler is at its current position
                self.model.addConstr(current_x - assembler.x <= self.M * (1 - pos_var))
                self.model.addConstr(assembler.x - current_x <= self.M * (1 - pos_var))
                self.model.addConstr(current_y - assembler.y <= self.M * (1 - pos_var))
                self.model.addConstr(assembler.y - current_y <= self.M * (1 - pos_var))
                
                position_constraints.append(pos_var)
                
                # For each inserter of this assembler
                for inserter in assembler.inserters:
                    inserter_x = int(inserter.x.x)
                    inserter_y = int(inserter.y.x)
                    
                    ins_pos_var = self.model.addVar(vtype=GRB.BINARY, name=f"curr_pos_{inserter.id}")
                    
                    self.model.addConstr(inserter_x - inserter.x <= self.M * (1 - ins_pos_var))
                    self.model.addConstr(inserter.x - inserter_x <= self.M * (1 - ins_pos_var))
                    self.model.addConstr(inserter_y - inserter.y <= self.M * (1 - ins_pos_var))
                    self.model.addConstr(inserter.y - inserter_y <= self.M * (1 - ins_pos_var))
                    
                    position_constraints.append(ins_pos_var)
            
            # Add a constraint that says "at least one of these positions must be different"
            if position_constraints:
                self.model.addConstr(quicksum(position_constraints) <= len(position_constraints) - 1)
                logger.info("Added constraint to forbid the current solution.")
        else:
            logger.warning("Cannot restrict current setup - no valid solution exists.")