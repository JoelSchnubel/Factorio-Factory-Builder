#! .venv\Scripts\python.exe

from z3 import *
from z3 import Optimize , Context 
import numpy as np
from solver_wrapper import SolverFactory
import json
from logging_config import setup_logger
import os

set_param('parallel.enable',True)
set_param('smt.threads', 8)

logger = setup_logger("SMT_Solver")

# can also be other machine like chamical plant, oil refinery, etc.
class Assembler:
    def __init__(self,id,inserters,item=None,capacity=0, solver=None,width=3,height=3):
        self.id = id
        self.x = solver.Int(f'{id}_x')
        self.y = solver.Int(f'{id}_y')
        

        self.width = width
        self.height = height
        
        self.inserters = inserters
        self.item = item
        
        #self.capacity = Int(f'{id}_capacity')
        self.capacity = capacity
        
        # for chemical plant, oil refinery, etc.
        self.fluid_orientation = solver.Int(f'{id}_fluid_orientation')  # 0-3 for different orientations
     
    def __str__(self) -> str:
            inserters_str = ', '.join([str(inserter) for inserter in self.inserters])
            return (f"Assembler(id={self.id}, position=({self.x}, {self.y}), "
                    f"capacity={self.capacity}, inserters=[{inserters_str}])")
   
        
        
class Inserter:
    def __init__(self,id,type,belt,item=None, solver=None, is_fluid=False):
        self.id = id
        self.x = solver.Int(f'{id}_x')
        self.y = solver.Int(f'{id}_y')
        self.type = type # 'input' or 'output'
        
        self.belt = belt
        self.item = item
        
        self.direct_interaction =False

        self.is_fluid = is_fluid  # Falg to indicate if the inserter is a pipe 
   
        
    def __str__(self) -> str:
        
            return (f"Inserter(id={self.id}, position=({self.x}, {self.y}), type={self.type}), item={self.item}"
                    f",Belt={str(self.belt)}")
class Belt:
    def __init__(self,id,type,item=None,int_x=0,int_y=0, solver=None):
        self.id = id
        self.x = solver.Int(f'{id}_x')
        self.y = solver.Int(f'{id}_y')
        self.type = type # 'start' or 'end'
        self.item = item
        
        self.is_used = solver.Bool(f"{id}_used")  # Z3 Boolean for belt usage
        
        self.int_x =int_x
        self.int_y =int_y
        
    def __str__(self) -> str:
        return f"Belt(id={self.id}, position=({self.x}, {self.y}), type={self.type}), item={self.item}"



class PowerPole:
    def __init__(self, id, solver=None):
        self.id = id
        self.x = solver.Int(f'{id}_x')
        self.y = solver.Int(f'{id}_y')
        self.is_used = solver.Bool(f'{id}_used') 
        
    def __str__(self) -> str:
        return f"PowerPole(id={self.id}, position=({self.x}, {self.y}), used={self.is_used})"


class SMTSolver:
    def __init__(self, width, height, production_data,solver_type="z3"):
        self.width = width
        self.height = height
        self.production_data = production_data
        self.machines_data = self.load_json("machine_data.json")
        self.config = self.load_json("config.json")
        
        if self.config is None:
            logger.warning("No config.json file found. Using default settings.")
            self.config = {
                "power": {
                    "default_type": "medium-electric-pole"
                }
            }
        
        #self.solver = Optimize()
        self.solver_ops = SolverFactory.create_solver(solver_type)
        self.solver, self.Int, self.Bool, self.Or, self.And, self.Not, self.Implies, self.If, self.Sum, self.sat = self.solver_ops
        # Set solver-specific constants
        if solver_type.lower() == "gurobi":
            self.M = 1000000
        
        # Create variables for assemblers, input inserters, output inserters, and belts
        self.assemblers = []
        self.placed_assembler = []
        self.global_input_belts = []
        self.global_output_belts = [] 

        self.additional_constraints = []
        
        self.input_information = None
        self.output_information = None

        self.obstacle_maps= []
        
        self.model = None

        self.power_poles = [] 
        self.power_pole_type = None  
     
             
    def load_json(self,recipe_file):
        logger.debug(f"Loading JSON data from {recipe_file}")
        with open(recipe_file, "r") as file:
                data = json.load(file)
                return data
            
    def build_constraints(self):
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

    
    
    
    
    def setup_power_poles(self, max_poles=None):
        logger.info("Setting up power poles")
        
        # Get power pole settings from config
        if "power_poles" not in self.machines_data:
            logger.warning("No power pole data found in machine_data.json")
            return
            
        # Determine power pole type from config
        power_config = self.config.get("power", {})
        
        if not power_config.get("place_power_poles", True): 
            logger.warning("Power poles are disabled in config")
            return
        
        pole_type = power_config.get("default_type", "medium-electric-pole")
        
        if pole_type not in self.machines_data["power_poles"]:
            logger.warning(f"Power pole type {pole_type} not found in machine_data.json, using medium-electric-pole")
            pole_type = "medium-electric-pole"
        
        pole_data = self.machines_data["power_poles"][pole_type]
        self.power_pole_type = pole_type
        self.power_pole_radius = pole_data["supply_area_radius"]
        self.power_pole_width = pole_data["dimensions"]["width"]
        self.power_pole_height = pole_data["dimensions"]["height"]
          # Estimate number of poles needed based on grid size and pole coverage
        if max_poles is None:            
            grid_area = self.width * self.height
            # Area covered by each pole is a square with sides of 2*radius+1
            # Square coverage: side length is 2*radius+1 (to account for the radius on both sides plus the center)
            pole_coverage = (2 * self.power_pole_radius + 1) * (2 * self.power_pole_radius + 1)
            max_poles = min(30, max(1, int(grid_area / pole_coverage * 1.5)))  # Add 50% extra for safety
        
        logger.info(f"Creating {max_poles} power poles of type {pole_type}")
        
        # Create power poles
        for i in range(max_poles):
            pole = PowerPole(id=f"power_pole_{i}", solver=self)
            self.power_poles.append(pole)
            
        # Add constraints for power poles
        self.add_power_pole_constraints()

    
    def add_power_pole_constraints(self):

        logger.info("Adding power pole constraints")
        
        # Log current assembler positions and dimensions
        for assembler in self.assemblers:
            logger.debug(f"Assembler {assembler.id}: width={assembler.width}, height={assembler.height}")
            logger.debug(f"Inserters for assembler {assembler.id}: {[ins.id for ins in assembler.inserters]}")
        
        logger.debug(f"Power pole type: {self.power_pole_type}, radius: {self.power_pole_radius}")
        
        # 1. Power poles must be within grid bounds if used
        for pole in self.power_poles:
            self.solver.add(self.Implies(
                pole.is_used,
                self.And(
                    pole.x >= 0, 
                    pole.x <= self.width - self.power_pole_width,
                    pole.y >= 0,
                    pole.y <= self.height - self.power_pole_height
                )
            ))
            logger.debug(f"Added grid bounds constraint for pole {pole.id}")
        
        # 2. Power poles must not overlap with assemblers
        for pole in self.power_poles:
            for assembler in self.assemblers:
                self.solver.add(self.Implies(
                    pole.is_used,
                    self.Or(
                        pole.x + self.power_pole_width <= assembler.x,
                        pole.x >= assembler.x + assembler.width,
                        pole.y + self.power_pole_height <= assembler.y,
                        pole.y >= assembler.y + assembler.height
                    )
                ))
            logger.debug(f"Added assembler overlap constraint for pole {pole.id}")
          # 3. Power poles must not overlap with each other
        for i, pole1 in enumerate(self.power_poles):
            for j, pole2 in enumerate(self.power_poles[i+1:], i+1):  # Fixed: Only compare with later poles
                self.solver.add(self.Implies(
                    self.And(pole1.is_used, pole2.is_used),
                    self.Or(
                        pole1.x + self.power_pole_width <= pole2.x,
                        pole1.x >= pole2.x + self.power_pole_width,
                        pole1.y + self.power_pole_height <= pole2.y,
                        pole1.y >= pole2.y + self.power_pole_height
                    )
                ))
            logger.debug(f"Added pole-pole overlap constraint for pole {pole1.id}")
            
        # 3b. Power poles must not overlap with inserters
        for pole in self.power_poles:
            for assembler in self.assemblers:
                for inserter in assembler.inserters:
                    self.solver.add(self.Implies(
                        pole.is_used,
                        self.Or(
                            pole.x + self.power_pole_width <= inserter.x,
                            pole.x > inserter.x,
                            pole.y + self.power_pole_height <= inserter.y,
                            pole.y > inserter.y
                        )
                    ))
            logger.debug(f"Added inserter overlap constraint for pole {pole.id}")
            
        # 3c. Power poles must not overlap with belts
        for pole in self.power_poles:
            for assembler in self.assemblers:
                for inserter in assembler.inserters:
                    belt = inserter.belt
                    if belt:
                        self.solver.add(self.Implies(
                            pole.is_used,
                            self.Or(
                                pole.x + self.power_pole_width <= belt.x,
                                pole.x > belt.x,
                                pole.y + self.power_pole_height <= belt.y,
                                pole.y > belt.y
                            )
                        ))
            logger.debug(f"Added belt overlap constraint for pole {pole.id}")
            
        # 3d. Power poles must not overlap with global belts
        for pole in self.power_poles:
            for belt in self.global_input_belts + self.global_output_belts:
                self.solver.add(self.Implies(
                    pole.is_used,
                    self.Or(
                        pole.x + self.power_pole_width <= belt.x,
                        pole.x > belt.x,
                        pole.y + self.power_pole_height <= belt.y,
                        pole.y > belt.y
                    )
                ))
            logger.debug(f"Added global belt overlap constraint for pole {pole.id}")
        
        logger.debug("Adding coverage variables for assemblers and inserters")
        # 4. Create coverage variables for each entity (assembler and inserter)
        # Add a covered attribute to each entity
        for assembler in self.assemblers:
            assembler.is_covered = self.Bool(f"assembler_{assembler.id}_covered")
            
            for inserter in assembler.inserters:
                inserter.is_covered = self.Bool(f"inserter_{inserter.id}_covered")
        
        # 5. Define coverage conditions for each entity using relaxed distance calculations
        logger.debug("Defining coverage conditions for assemblers")
        for assembler in self.assemblers:
            # For assemblers, check if ANY of its tiles is covered by ANY power pole
            assembler_coverage_conditions = []
            
            # Generate all tile positions for the assembler
            for dx in range(assembler.width):
                for dy in range(assembler.height):
                    assembler_tile_x = assembler.x + dx
                    assembler_tile_y = assembler.y + dy
                    
                    # For each pole, check if it could cover this tile
                    for pole in self.power_poles:                       
                        pole_center_x = pole.x + self.power_pole_width // 2
                        pole_center_y = pole.y + self.power_pole_height // 2
                        # Square coverage area: check if tile is within the supply area in both x and y directions
                        x_distance_constraint = Abs(assembler_tile_x - pole_center_x) <= self.power_pole_radius
                        y_distance_constraint = Abs(assembler_tile_y - pole_center_y) <= self.power_pole_radius
                        distance_constraint = self.And(x_distance_constraint, y_distance_constraint)
                      
                        # Add coverage condition (pole is used AND distance is within radius)
                        assembler_coverage_conditions.append(self.And(pole.is_used, distance_constraint))
            
            if not assembler_coverage_conditions:
                logger.warning(f"No coverage conditions for assembler {assembler.id}!")
                # Create at least one condition to avoid empty OR
                assembler_coverage_conditions.append(self.Bool(f"dummy_coverage_{assembler.id}"))
                self.solver.add(assembler_coverage_conditions[0] == False)
            
            # Assembler is covered if ANY of its tiles is covered by ANY power pole
            self.solver.add(assembler.is_covered == self.Or(assembler_coverage_conditions))
            logger.debug(f"Added coverage conditions for assembler {assembler.id}: {len(assembler_coverage_conditions)} conditions")
            
        logger.debug("Defining coverage conditions for inserters")    
        # For inserters, check if they're covered by ANY power pole
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                inserter_coverage_conditions = []
                
                for pole in self.power_poles:                    
                    pole_center_x = pole.x + self.power_pole_width // 2
                    pole_center_y = pole.y + self.power_pole_height // 2
                    # Square coverage area for inserters as well
                    x_distance_constraint = Abs(inserter.x - pole_center_x) <= self.power_pole_radius
                    y_distance_constraint = Abs(inserter.y - pole_center_y) <= self.power_pole_radius
                    distance_constraint = self.And(x_distance_constraint, y_distance_constraint)
                    
                    inserter_coverage_conditions.append(self.And(pole.is_used, distance_constraint))
                
                if not inserter_coverage_conditions:
                    logger.warning(f"No coverage conditions for inserter {inserter.id}!")
                    # Create at least one condition to avoid empty OR
                    inserter_coverage_conditions.append(self.Bool(f"dummy_coverage_{inserter.id}"))
                    self.solver.add(inserter_coverage_conditions[0] == False)
                    
                self.solver.add(inserter.is_covered == self.Or(inserter_coverage_conditions))
                logger.debug(f"Added coverage conditions for inserter {inserter.id}: {len(inserter_coverage_conditions)} conditions")
        
        # 6. HARD CONSTRAINT: All entities MUST be covered
        logger.debug("Adding hard coverage constraints - all entities must be covered")
        for assembler in self.assemblers:
            self.solver.add(assembler.is_covered)
            logger.debug(f"Added hard constraint: Assembler {assembler.id} must be covered")
            
            for inserter in assembler.inserters:
                self.solver.add(inserter.is_covered)
                logger.debug(f"Added hard constraint: Inserter {inserter.id} must be covered")
        
        # 7. Ensure at least one pole is used if we have entities
        if self.assemblers:
            self.solver.add(self.Or([pole.is_used for pole in self.power_poles]))
            logger.debug("Added constraint to ensure at least one pole is used")
        
    def minimize_power_poles(self):
        """
        Minimize the number of power poles used while maintaining coverage.
        This should be called after the initial solution is found.
        """
        logger.info("=== Minimizing Number of Power Poles ===")
        
        # First solve the problem with all constraints
        result = self.solver.check()
        if result == self.sat:
            self.model = self.solver.model()
            logger.info("Found initial solution before minimizing power poles")
            
            # Lock the positions of all entities except power poles
            self.lock_non_power_pole_positions()
            
            # Create optimization objective: minimize sum of is_used variables
            total_poles = self.Int("total_power_poles")
            self.solver.add(total_poles == self.Sum([self.If(pole.is_used, 1, 0) for pole in self.power_poles]))
            self.solver.minimize(total_poles)
            
            # Solve again with the minimization objective
            final_result = self.solver.check()
            if final_result == self.sat:
                self.model = self.solver.model()
                used_poles = sum(1 for pole in self.power_poles 
                            if self.model.evaluate(pole.is_used))
                logger.info(f"Successfully minimized power poles: using {used_poles} poles")
                
                return True
            else:
                logger.warning(f"Failed to minimize power poles: {final_result}")
                return False
        else:
            logger.warning("No initial solution found, cannot minimize power poles")
            return False

    def lock_non_power_pole_positions(self):
        """
        Lock the positions of all entities except power poles.
        This allows only power poles to be repositioned during minimization.
        """
        logger.info("Locking positions of all entities except power poles")
        
        # Lock assembler positions
        for assembler in self.assemblers:
            self.solver.add(assembler.x == self.model.evaluate(assembler.x))
            self.solver.add(assembler.y == self.model.evaluate(assembler.y))
            
            # Lock inserter positions
            for inserter in assembler.inserters:
                self.solver.add(inserter.x == self.model.evaluate(inserter.x))
                self.solver.add(inserter.y == self.model.evaluate(inserter.y))
                
                # Lock belt positions
                if inserter.belt:
                    self.solver.add(inserter.belt.x == self.model.evaluate(inserter.belt.x))
                    self.solver.add(inserter.belt.y == self.model.evaluate(inserter.belt.y))
        
    
    def add_manuel_IO_constraints(self, input_information, output_information):
        logger.info("Starting to add manual I/O constraints.")
        
        self.input_information = input_information
        self.output_information = output_information

        for item, data in input_information.items():
            logger.debug(f"Processing input item '{item}' with data: {data}")
            
            # Create and position the input belts (start point)
            belt = Belt(id=f"{item}_{data['input']}_input", type='input', item=item,solver=self)
            logger.info(f"Adding input belt for item '{item}' at position {data['input']}")
            self.solver.add(self.And(belt.x == data["input"][1], belt.y == data["input"][0]))
            self.global_input_belts.append(belt)
            
            # Create and position the output belts (end point)
            belt = Belt(id=f"{item}_{data['output']}_input", type='input', item=item,solver=self)
            logger.info(f"Adding output belt for item '{item}' at position {data['output']}")
            self.solver.add(self.And(belt.x == data["output"][1], belt.y == data["output"][0]))
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
                        belt = Belt(id=f"{item}_({x}, {y})_input", type='input', item=item, int_x=x, int_y=y,solver=self)
                        self.solver.add(self.And(belt.x == x, belt.y == y))
                        self.global_input_belts.append(belt)

        for item, data in output_information.items():
            logger.debug(f"Processing output item '{item}' with data: {data}")

            # Create and position the input belts for output item (start point)
            belt = Belt(id=f"{item}_{data['input']}_output", type='output', item=item,solver=self)
            logger.info(f"Adding input belt for output item '{item}' at position {data['input']}")
            self.solver.add(self.And(belt.x == data["input"][1], belt.y == data["input"][0]))
            self.global_output_belts.append(belt)
            
            # Create and position the output belts (end point)
            belt = Belt(id=f"{item}_{data['output']}_output", type='output', item=item,solver=self)
            logger.info(f"Adding output belt for output item '{item}' at position {data['output']}")
            self.solver.add(self.And(belt.x == data["output"][1], belt.y == data["output"][0]))
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
                        belt = Belt(id=f"{item}_({x}, {y})_output", type='output', item=item, int_x=x, int_y=y,solver=self)
                        self.solver.add(self.And(belt.x == x, belt.y == y))
                        self.global_output_belts.append(belt)

        logger.info("Finished adding manual I/O constraints. Calling solver.")
        self.solve()

    def add_constraint(self, constraint):
        logger.info("Adding a new constraint.")
        self.additional_constraints.append(constraint)
        self.solver.add(constraint)
        logger.debug(f"Added constraint: {constraint}")
        
    def get_machine_type_for_recipe(self, recipe_id):
        """Determine the machine type to use for a given recipe."""
        if "recipe_machine_mapping" in self.machines_data:
            recipes = self.machines_data["recipe_machine_mapping"].get("recipes", {})
            if recipe_id in recipes:
                return recipes[recipe_id]
        
        # Default to assembling-machine-2 
        return self.machines_data["recipe_machine_mapping"].get("default", "assembling-machine-2")
    
    def create_assemblers(self):
        logger.info("Starting to create assemblers based on production data.")
        assembler_count = 0
        
         # Dictionary to map item IDs to their types from recipes data
        item_types = {}
        
        # Get recipes data to check for fluid items
        try:
            with open("recipes.json", "r") as f:
                recipes_data = json.load(f)
                for item in recipes_data:
                    if isinstance(item, dict) and "id" in item and "type" in item:
                        item_types[item["id"]] = item["type"]
        except Exception as e:
            logger.error(f"Failed to load recipes.json: {e}")
        
        
        
        for item_id, item_info in self.production_data.items():
            logger.debug(f"Processing item '{item_id}' with production data: {item_info}")

            if 'assemblers' in item_info and item_info['assemblers'] > 0:
                
                machine_type = self.get_machine_type_for_recipe(item_id)
                
                default_assembler = self.config.get("machines", {}).get("default_assembler", "assembling-machine-1")
            
            
                # Get machine dimensions
                machine_info = self.machines_data["assemblers"].get(machine_type, self.machines_data["assemblers"][default_assembler])
            
                width = machine_info["dimensions"]["width"]
                height = machine_info["dimensions"]["height"]
                
                fixed_inputs = machine_info.get("fixed_inputs", [])
                fixed_outputs = machine_info.get("fixed_outputs", [])
                
                logger.info(f"Using {machine_type} for {item_id} with dimensions {width}x{height}")
            
                
                for i in range(item_info['assemblers']):
                    logger.info(f"Creating assembler for item '{item_id}', instance {i}")
                    
                    # Create input inserters for the assembler
                    input_inserters = []
                    handled_fluid_items = set()
                    
                    for inserter_info in item_info['input_inserters']:
                        
                        input_item = inserter_info['id']
                        
                        # Check if item is a fluid by looking up its type
                        is_fluid = item_types.get(input_item, "") == "Liquid"
                        logger.info(f"Item '{input_item}' is {'fluid' if is_fluid else 'solid'}")
                    
                        if is_fluid:
                             if input_item not in handled_fluid_items:
                                handled_fluid_items.add(input_item)
                                # Create exactly one fluid inserter
                                input_inserter_id = f"{input_item}_in_{assembler_count}_{i}_0"
                                belt_id = f"{input_item}_end_{assembler_count}_{i}_0"
                                
                                logger.info(f"Creating one fluid inserter for {input_item}")
                                input_inserters.append(
                                    Inserter(
                                        id=input_inserter_id,
                                        type='input',
                                        item=input_item,
                                        solver=self,
                                        is_fluid=True,
                                        belt=Belt(
                                            id=belt_id,
                                            type="end",
                                            item=input_item,
                                            solver=self
                                        )
                                    )
                                )
                        
                        else:
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
                                        solver = self,
                                        is_fluid=is_fluid,
                                        belt=Belt(
                                            id=belt_id,
                                            type="end",  
                                            item=inserter_info['id'],
                                            solver = self
                                        
                                        )
                                    )
                                )
                            
                    # create the assembler with the input inserters and unique ID
                    assembler = Assembler(
                        id=f"{item_id}_{assembler_count}_{i}",
                        inserters=input_inserters,
                        item=item_id,
                        capacity=item_info['capacity'],
                        solver = self,
                        width=width,
                        height=height
                    )
                    
                    # Add fixed input/output points
                    assembler.fixed_inputs = fixed_inputs
                    assembler.fixed_outputs = fixed_outputs
                    
                    self.assemblers.append(assembler)
                    logger.debug(f"Created assembler with ID: {assembler.id} and input inserters: {[inserter.id for inserter in input_inserters]}")
                    logger.debug(f"Created assembler with ID: {assembler.id} with item = {assembler.item}")
                    logger.debug(f"Created assembler with ID: {assembler.id} with capacity = {assembler.capacity}")
                assembler_count += 1

        logger.info(f"Created {len(self.assemblers)} assemblers in total.")
        
    # assembler is 3x3 and not allowed to get out of bounds 
    def add_bound_constraints_assembler(self):
        logger.info("Adding boundary constraints for assemblers.")
        for assembler in self.assemblers:
            logger.debug(f"Setting boundary constraints for assembler ID {assembler.id}")
            self.solver.add(self.And(assembler.x >= 0, assembler.x <= self.width - assembler.width))
            self.solver.add(self.And(assembler.y >= 0, assembler.y <= self.height - assembler.height))

    # belts and inserter bound constraints
    def add_bound_constraints_belts_and_inserters(self):
        logger.info("Adding boundary constraints for belts and inserters.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                logger.debug(f"Setting boundary constraints for inserter ID {inserter.id} and belt ID {belt.id}")
                self.solver.add(self.And(inserter.x >= 0, inserter.x < self.width))
                self.solver.add(self.And(inserter.y >= 0, inserter.y < self.height))
                self.solver.add(self.And(belt.x >= 0, belt.x < self.width))
                self.solver.add(self.And(belt.y >= 0, belt.y < self.height))

    # user input/putput belts are not allowed to overlap with assembler
    def add_global_belt_overlap_assembler_constraint(self):
        logger.info("Adding assembler overlap constraints to avoid global belts.")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for belt in belts:
                #logger.debug(f"Preventing overlap between assembler ID {assembler.id} and belt ID {belt.id}")
                self.solver.add(self.Or(
                    self.Or(belt.x < assembler.x, belt.x > assembler.x + assembler.width-1),
                    self.Or(belt.y < assembler.y, belt.y > assembler.y + assembler.height-1)
                ))
                
    # user input/putput belts are not allowed to overlap with inserter
    def add_global_belt_overlap_inserter_constraint(self):
        logger.info("Adding inserter overlap constraints to avoid global belts.")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                for belt in belts:
                    #logger.debug(f"Preventing overlap between inserter ID {inserter.id} and belt ID {belt.id}")
                    self.solver.add(self.Or(
                            belt.x != inserter.x,
                            belt.y != inserter.y
                    ))
                       
    # user input/putput belts are allowed to overlap with belts that have the same item
    def add_global_belt_overlap_belt_constraint(self):
        logger.info("Adding belt overlap constraints to prevent conflicts with global belts (different items only).")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                inserter_belt = inserter.belt
                for belt in belts:
                    if belt.item != inserter_belt.item:
                        #logger.debug(f"Preventing overlap between belt ID {belt.id} and inserter belt ID {inserter_belt.id}")
                        self.solver.add(self.Or(
                            belt.x != inserter_belt.x,
                            belt.y != inserter_belt.y
                        ))
                                                
    # assembler are not allowed to overlap with other assemblers
    def add_assembler_overlap_assembler_constraint(self):
        logger.info("Adding assembler overlap constraints to prevent assembler-assembler overlap.")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    #logger.debug(f"Preventing overlap between assembler ID {assembler.id} and assembler ID {other_assembler.id}")
                    self.solver.add(self.Or(
                        assembler.x + assembler.width <= other_assembler.x, 
                        assembler.x >= other_assembler.x + other_assembler.width,
                        assembler.y + assembler.height <= other_assembler.y, 
                        assembler.y >= other_assembler.y + other_assembler.height
                    ))
                    
    # assemblers are not allowed to overlap with inserters
    def add_assembler_overlap_inserter_constraint(self):
        logger.info("Adding assembler-inserter overlap constraints.")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    for inserter in other_assembler.inserters:
                        #logger.debug(f"Preventing overlap between assembler ID {assembler.id} and inserter ID {inserter.id}")
                        self.solver.add(self.Or(
                            self.Or(inserter.x < assembler.x, inserter.x > assembler.x + assembler.width-1),
                            self.Or(inserter.y < assembler.y, inserter.y > assembler.y + assembler.height-1)
                        ))
    # assembler and belts are only allowed to overlap if they have the same item 
    def add_assembler_overlap_belt_constraint(self):
        logger.info("Adding assembler-belt overlap constraints (excluding same-item belts).")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    for inserter in other_assembler.inserters:
                        belt = inserter.belt
                        if belt.item != assembler.item:
                            #logger.debug(f"Preventing overlap between assembler ID {assembler.id} and belt ID {belt.id}")
                            self.solver.add(self.Or(
                                self.Or(belt.x < assembler.x, belt.x > assembler.x + assembler.width-1),
                                self.Or(belt.y < assembler.y, belt.y > assembler.y + assembler.height-1)
                            ))
                            
    # inserters are not allowed to overlap each other 
    # also for same assembler
    def add_inserter_overlap_inserter_constraint(self):
        logger.info("Adding inserter overlap constraints to prevent inserter-inserter overlap.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                
                for other_inserter in assembler.inserters:
                    if inserter.id != other_inserter.id:
                        self.solver.add(self.Or(
                            inserter.x != other_inserter.x,
                            inserter.y != other_inserter.y
                        ))
                        
                      
                
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        for other_inserter in other_assembler.inserters:
                            #logger.debug(f"Preventing overlap between inserter ID {inserter.id} and inserter ID {other_inserter.id}")
                            self.solver.add(self.Or(
                                inserter.x != other_inserter.x,
                                inserter.y != other_inserter.y
                            ))
                            
    # inserter and belts ar not allowed to overlap
    def add_inserter_overlap_belt_constraint(self):
        logger.info("Adding inserter-belt overlap constraints to prevent inserters from overlapping with belts.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        for other_inserter in other_assembler.inserters:
                            other_belt = other_inserter.belt
                            #logger.debug(f"Preventing overlap between inserter ID {inserter.id} and belt ID {other_belt.id}")
                            self.solver.add(self.Or(
                                inserter.x != other_belt.x,
                                inserter.y != other_belt.y
                            ))
                        
                            
    # belts are allowed to overlap if they have the same item
    def add_belt_overlap_belt_constraint(self):
        logger.info("Adding belt overlap constraints to prevent belt-belt overlap (except same-item belts).")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        for other_inserter in other_assembler.inserters:
                            other_belt = other_inserter.belt
                            if belt.item != other_belt.item:
                                #logger.debug(f"Preventing overlap between belt ID {belt.id} and other belt ID {other_belt.id}")
                                
                                self.solver.add(self.Or(
                                    belt.x != other_belt.x,
                                    belt.y != other_belt.y
                                ))
                            
        
    # inserter needs to be adjacent to assembler
    def add_inserter_adjacent_to_assembler(self):

        for assembler in self.assemblers:
            
            default_assembler = self.config.get("machines", {}).get("default_assembler", "assembling-machine-1")
        
            machine_type = self.get_machine_type_for_recipe(assembler.item)
            # Use default_assembler from config instead of "default" literal
            machine_info = self.machines_data["assemblers"].get(machine_type, self.machines_data["assemblers"][default_assembler])
            
            # For solid inserters - calculate positions around the machine based on dimensions
            width = assembler.width
            height = assembler.height
            
            solid_positions = []
            fluid_positions = []
             
            # Top Edge
            for dx in range(width):
                solid_positions.append((assembler.x + dx, assembler.y - 1))
            # Bottom edge
            for dx in range(width):
                solid_positions.append((assembler.x + dx, assembler.y + height))
            # Left edge
            for dy in range(height):
                solid_positions.append((assembler.x - 1, assembler.y + dy))
            # Right edge
            for dy in range(height):
                solid_positions.append((assembler.x + width, assembler.y + dy))
            
            
            # Check if this machine has fluid connection pairs
            fluid_connection_pairs = machine_info.get("fluid_connection_pairs", [])
            
            if fluid_connection_pairs:
                # Constrain orientation to valid values (0 to num_orientations-1)
               
                for orientation_idx, connection_pair in enumerate(fluid_connection_pairs):
                    for input_pos in connection_pair["inputs"]:
                        pos_x = assembler.x + input_pos["dx"]
                        pos_y = assembler.y + input_pos["dy"]
                        fluid_positions.append((pos_x, pos_y))
        
     
               
            
            for inserter in assembler.inserters:
                # solids
                if not inserter.is_fluid:
                    # For solids, use standard positions around the machine
                    logger.info(f"Allowing solid inserter {inserter.id} to be placed at any valid edge position")
                    self.solver.add(self.Or([self.And(inserter.x == pos[0], inserter.y == pos[1]) for pos in solid_positions]))
                    
                    # For solids, the belt should be positioned properly based on inserter location
                    belt = inserter.belt
                    if belt is not None:
                        belt_constraints = []
                        
                        # Top edge
                        for dx in range(width):
                            belt_constraints.append(self.And(
                                inserter.x == assembler.x + dx, 
                                inserter.y == assembler.y - 1,
                                belt.x == inserter.x,
                                belt.y == inserter.y - 1
                            ))
                        
                        # Bottom edge
                        for dx in range(width):
                            belt_constraints.append(self.And(
                                inserter.x == assembler.x + dx, 
                                inserter.y == assembler.y + height,
                                belt.x == inserter.x,
                                belt.y == inserter.y + 1
                            ))
                        
                        # Left edge
                        for dy in range(height):
                            belt_constraints.append(self.And(
                                inserter.x == assembler.x - 1, 
                                inserter.y == assembler.y + dy,
                                belt.x == inserter.x - 1,
                                belt.y == inserter.y
                            ))
                        
                        # Right edge
                        for dy in range(height):
                            belt_constraints.append(self.And(
                                inserter.x == assembler.x + width, 
                                inserter.y == assembler.y + dy,
                                belt.x == inserter.x + 1,
                                belt.y == inserter.y
                            ))
                        
                        self.solver.add(self.Or(belt_constraints))
                # fluids        
                else:
                    logger.info(f"Allowing fluid inserter {inserter.id} to be placed at any valid edge position")
                    self.solver.add(self.Or([self.And(inserter.x == pos[0], inserter.y == pos[1]) for pos in fluid_positions]))
                    
                    # belt should be positioned at the same position as the inserter
                    belt = inserter.belt
                    self.solver.add(self.And(
                        belt.x == inserter.x,
                        belt.y == inserter.y
                    ))
    
    # ensure at least one space 1 and 2 tiles away from the assembler is free to ensure a possible output
    def add_space_for_output_of_assembler(self):
        for assembler in self.assemblers:
            output_positions = [[(assembler.x, assembler.y - 1),(assembler.x, assembler.y - 2)], # upper left 
                                [(assembler.x + 1, assembler.y - 1),(assembler.x + 1, assembler.y - 2)], # upper middle
                                [(assembler.x + 2, assembler.y - 1),(assembler.x + 2, assembler.y - 2)], # upper right
                                
                                [(assembler.x, assembler.y + 3),(assembler.x, assembler.y + 4)], # bottom left       
                                [(assembler.x + 1, assembler.y + 3),(assembler.x + 1, assembler.y + 4)], # bottom middle                                                                                               
                                [(assembler.x + 2,assembler.y + 3),(assembler.x + 2, assembler.y + 4)], # bottom right       
                                
                                [(assembler.x-1, assembler.y),(assembler.x-2, assembler.y)], # left up
                                [(assembler.x-1, assembler.y+1),(assembler.x-2, assembler.y+1)], # left middle
                                [(assembler.x-1, assembler.y+2),(assembler.x-2, assembler.y+2)], # left bottom
                                
                                [(assembler.x+4, assembler.y),(assembler.x+5, assembler.y)], # right up
                                [(assembler.x+4, assembler.y+1),(assembler.x+5, assembler.y+1)], # right middle
                                [(assembler.x+4, assembler.y+2),(assembler.x+5, assembler.y+2)], # right bottom
            ]
            
            pair_constraints = []
            for pos1, pos2 in output_positions:
                # Ensure both positions in the pair are empty
                pair_constraints.append(self.And(
                    self.Not(self.is_position_occupied(pos1[0], pos1[1])),
                    self.Not(self.is_position_occupied(pos2[0], pos2[1]))
                ))

            if pair_constraints:
                # Add a constraint that at least one pair is empty
                logger.info(f"Adding output space constraints for assembler {assembler.id}")
                self.solver.add(self.Or(pair_constraints))
            
    
    def is_position_occupied(self,x,y):
        
        occupied_conditions = []
        for assembler in self.assemblers:
            # Assembler occupies a wxh area
            occupied_conditions.append(
                self.And(x >= assembler.x, x <= assembler.x + assembler.width - 1,
                    y >= assembler.y, y <= assembler.y + assembler.height - 1)
            )
            for inserter in assembler.inserters:
                
                # Check if any inserter occupies the position
                occupied_conditions.append(
                    self.And(x == inserter.x, y == inserter.y)
                )
                
                # Check if any belt occupies the position
                belt = inserter.belt
                occupied_conditions.append(
                    self.And(x == belt.x, y == belt.y)
                )
        
        # Check if any belt occupies the position
        for belt in self.global_input_belts + self.global_output_belts:
            occupied_conditions.append(
                self.And(x == belt.x, y == belt.y)
            )
        

        return self.Or(occupied_conditions)


    # input inserter should be next to other assembler or belt that transports/produces the same item
    def add_input_inserter_merge_user_belt_constraint(self):
        belts = self.global_input_belts + self.global_output_belts
        merge_constraints = []
        
        logger.debug("Adding constraints to merge input inserters with belts of matching items.")
        
        for belt in belts:
            for assembler in self.assemblers:
                for inserter in assembler.inserters:
                    inserter_belt = inserter.belt
        
                    if belt.item == inserter_belt.item:
                        logger.debug(  f"Matching item found: Belt (ID: {belt.id}, Item: {belt.item}) "
                                        f"and Inserter Belt (ID: {inserter_belt.id}, Item: {inserter_belt.item})")
                        merge_constraints.append(self.And(inserter_belt.x == belt.x ,inserter_belt.y == belt.y))
                    
        
        if merge_constraints:
            logger.debug("Adding merge constraints for input inserters to the solver.")
            self.solver.add(self.Or(merge_constraints))
        else:
            logger.debug("No merge constraints were added as no matching items were found.")
                
        
        
    # we can force the belt to overlap with either an belt defined by the user or force merge it with an assemblers outline.

    def add_input_inserter_merge_assembler_constraint(self):
        
        merge_constraints = []
        
        logger.debug("Adding constraints to merge input inserters with assembler edges for items with matching types.")
         
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
            
                # Create a list to hold the constraints for each valid assembler
                assembler_constraints = []
        
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        
                        #logger.debug(f"Assembler {assembler.id} and other assembler {other_assembler.id}")
                        #logger.debug(f"Inserter item {inserter.item} and other assembler item {other_assembler.item}")
                        
                        # if my input inserter and the other assembler produces/transports same item, set the belt on one of the edge positions of the assembler -> all but the middle
                        if inserter.item == other_assembler.item and other_assembler.capacity > 0 and not inserter.direct_interaction:
                            
                            logger.debug(f"------------------------")
                            logger.debug(  f"Assembler {assembler.id} and other assembler {other_assembler.id} "
                                        f"have matching items: {inserter.item}")
                            
                            
                            
                            merge_positions = [ (other_assembler.x,other_assembler.y), # upper left
                                                (other_assembler.x+1,other_assembler.y  ), # upper middle
                                                (other_assembler.x+2,other_assembler.y  ), # upper right
                                                (other_assembler.x,other_assembler.y+1  ), # middle left
                                                # leave out middle
                                                (other_assembler.x+2,other_assembler.y+1), # middle right
                                                (other_assembler.x,other_assembler.y+2  ), # lower left
                                                (other_assembler.x+1,other_assembler.y+2), # lower middle
                                                (other_assembler.x+2,other_assembler.y+2), # lower right
                                            ]
                            
                            
                            
      
                            constraints = [
                                self.And(inserter.belt.x == pos[0], inserter.belt.y == pos[1])
                                for pos in merge_positions
                            ]
                            
                            if constraints:
                                
                                logger.debug(f"Adding constraints to position belt for inserter {inserter.id} and assembler {other_assembler.id} with capcity: {other_assembler.capacity}.")
                                logger.debug(f"Adding 'Or' between valid positions for inserter {inserter.id} and reducing capacity of {other_assembler.id} to {other_assembler.capacity}")
                                logger.debug(f"{inserter.id} merged is set to {inserter.direct_interaction}")

                                other_assembler.capacity -= 1
                                inserter.direct_interaction = True
                                assembler_constraints.append(self.Or(constraints))
                                merge_constraints.append(self.Or(constraints))
                                self.solver.add(merge_constraints)
                            else:
                                logger.debug(f"no valid constraints can be build")
                                
                        else:
                                
                                merge_positions = [(other_assembler.x,other_assembler.y), # upper left
                                                (other_assembler.x+1,other_assembler.y  ), # upper middle
                                                (other_assembler.x+2,other_assembler.y  ), # upper right
                                                (other_assembler.x,other_assembler.y+1  ), # middle left
                                                # leave out middle
                                                (other_assembler.x+2,other_assembler.y+1), # middle right
                                                (other_assembler.x+2,other_assembler.y+2), # lower left
                                                (other_assembler.x+1,other_assembler.y+2), # lower middle
                                                (other_assembler.x+2,other_assembler.y+2), # lower right
                                            ]

                               

                                constraints = [
                                self.Not(self.Or(inserter.belt.x == pos[0], inserter.belt.y == pos[1]))
                                for pos in merge_positions
                                ]
                                #self.solver.add(constraints)
                                assembler_constraints.append(self.Or(constraints))
                                # add constraint to diallow the positionong of the assembler next to the inserter

                                logger.debug(f"dissallow merge for inserter {inserter.id} and assembler {other_assembler.id} with capcity: {other_assembler.capacity}.")
                                
           
                        
    
    
    # minimizes the number of non merged belts 
    def add_minimize_belts(self):
        logger.info("===== Adding constraints to maximize belt overlaps =====")
        
        overlap_variables = []  # To track overlaps for optimization
        inserter_belts = []

        # Collect all inserter belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                inserter_belts.append(inserter.belt)
                
         # Check overlaps with global input/output belts
        global_belts = self.global_input_belts + self.global_output_belts
            
        
            
        for inserter_belt in inserter_belts:
            for global_belt in global_belts:
                if inserter_belt.item == global_belt.item:
                    # Define overlap condition
                    is_overlapping = self.And(
                        inserter_belt.x == global_belt.x,
                        inserter_belt.y == global_belt.y
                    )
                    # Add to overlap tracking
                    overlap_var = self.Bool(f"overlap_{inserter_belt.id}_{global_belt.id}")
                    self.solver.add(self.Implies(overlap_var, is_overlapping))
                    overlap_variables.append(overlap_var)
                    
 
 
        # Check overlaps with other assemblers
        for assembler in self.assemblers:
            
            assembler_edges = []
            
            # Top edge
            for dx in range(assembler.width):
                assembler_edges.append((assembler.x + dx, assembler.y))
            
            # Bottom edge
            for dx in range(assembler.width):
                assembler_edges.append((assembler.x + dx, assembler.y + assembler.height - 1))
            
            # Left edge (excluding corners which we've already added)
            for dy in range(1, assembler.height - 1):
                assembler_edges.append((assembler.x, assembler.y + dy))
            
            # Right edge (excluding corners which we've already added)
            for dy in range(1, assembler.height - 1):
                assembler_edges.append((assembler.x + assembler.width - 1, assembler.y + dy))
    
            
            for edge in assembler_edges:
                is_overlapping = self.And(
                    inserter_belt.x == edge[0],
                    inserter_belt.y == edge[1]
                )
                # Add to overlap tracking
                overlap_var = self.Bool(f"overlap_{inserter_belt.id}_assembler_{assembler.id}")
                self.solver.add(self.Implies(overlap_var, is_overlapping))
                overlap_variables.append(overlap_var)
        
        if overlap_variables: 
            # Maximize the sum of all overlap variables
            logger.info("Adding optimization goal to maximize belt overlaps.")
            self.solver.maximize(self.Sum([self.If(var, 1, 0) for var in overlap_variables]))
        
        
    def find_non_overlapping_inserters(self):
        """Finds inserters that do NOT overlap with a global belt but have a matching item."""
        logger.info("=== Identifying Non-Overlapping Inserters ===")

        non_overlapping_inserters = []

        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                same_item_belts = [gb for gb in self.global_input_belts + self.global_output_belts if gb.item == belt.item]

                is_overlapping = any(
                    self.model.evaluate(self.And(belt.x == gb.x, belt.y == gb.y)) for gb in same_item_belts
                )

                if not is_overlapping and same_item_belts:
                    logger.info(f"Inserter {inserter.id} does not overlap but has matching item.")
                    
                    non_overlapping_inserters.append((inserter, same_item_belts))
                else:
                    logger.info(f"Inserter {inserter.id} is locked (either overlaps or has no matching global belt).")

        return non_overlapping_inserters


    def lock_initial_solution(self, non_overlapping_inserters):
        """Locks all positions except non-overlapping inserters."""
        logger.info("=== Locking Initial Solution ===")
       

        non_overlapping_set = {inserter for inserter, _ in non_overlapping_inserters}

        # Keep assembler positions unchanged
        for assembler in self.assemblers:
            logger.info(f"Locking assembler {assembler.id} at ({assembler.x}, {assembler.y}).")
            self.solver.add(assembler.x == self.model.evaluate(assembler.x))
            self.solver.add(assembler.y == self.model.evaluate(assembler.y))

            for inserter in assembler.inserters:
                if inserter in non_overlapping_set:
                    logger.info(f"Inserter {inserter.id} is free to move.")
                   
                    continue

                logger.info(f"Locking inserter {inserter.id} at ({inserter.x}, {inserter.y}).")
               
                self.solver.add(inserter.x == self.model.evaluate(inserter.x))
                self.solver.add(inserter.y == self.model.evaluate(inserter.y))

                # Keep belt positions fixed
                belt = inserter.belt
                self.solver.add(belt.x == self.model.evaluate(belt.x))
                self.solver.add(belt.y == self.model.evaluate(belt.y))
                
                
    def minimize_non_overlapping_inserters(self, non_overlapping_inserters):
       
        """Minimizes distance for non-overlapping inserters to the closest source. either global belt or assembler"""
        
        logger.info("=== Minimizing Distance for Non-Overlapping Inserters ===")
        

        total_distance = self.Int("total_distance")
        distance_constraints = []

        for inserter, same_item_belts in non_overlapping_inserters:
            belt = inserter.belt
            min_distance = self.Int(f"min_dist_{belt.id}")
            
            # Compute Manhattan distances to global belts
            belt_distances = [Abs(belt.x - gb.x) + Abs(belt.y - gb.y) for gb in same_item_belts]
            
            # Find assemblers that produce the needed item
            producing_assemblers = [
                asm for asm in self.assemblers 
                if asm.item == inserter.item and asm.id != inserter.id.split('_in_')[0]
            ]
        

                
            logger.info(f"Found {len(producing_assemblers)} producing assemblers for item {inserter.item}.")
            # Compute Manhattan distances to producing assemblers (using their edges)
            assembler_distances = []
            for asm in producing_assemblers:
                # Get all edge positions of the assembler
                edge_positions = []
                
                # Top edge
                for dx in range(asm.width):
                    edge_positions.append((asm.x + dx, asm.y))
                
                # Bottom edge
                for dx in range(asm.width):
                    edge_positions.append((asm.x + dx, asm.y + asm.height - 1))
                
                # Left edge (excluding corners)
                for dy in range(1, asm.height - 1):
                    edge_positions.append((asm.x, asm.y + dy))
                
                # Right edge (excluding corners)
                for dy in range(1, asm.height - 1):
                    edge_positions.append((asm.x + asm.width - 1, asm.y + dy))
    
                # Find minimum distance to any edge of the assembler
                for edge_pos in edge_positions:
                    assembler_distances.append(Abs(belt.x - edge_pos[0]) + Abs(belt.y - edge_pos[1]))
        

            
            all_distances = belt_distances + assembler_distances
           

            if all_distances:
                logger.info(f"Inserter {inserter.id} has {len(belt_distances)} distances to matching global belts " +
                        f"and {len(assembler_distances)} distances to assemblers producing {inserter.item}.")
                
                # Minimize the distance to the closest source (belt or assembler edge)
                self.solver.add(self.Or([min_distance == d for d in all_distances]))
                self.solver.add(min_distance >= 0)  # Ensure non-negative distance
                distance_constraints.append(min_distance)
            else:
                logger.warning(f"Inserter {inserter.id} has no matching global belts or assemblers producing {inserter.item}.")

        if distance_constraints:
            self.solver.add(total_distance == self.Sum(distance_constraints))
            self.solver.minimize(total_distance)
            logger.info("Added minimization constraint for total distance.")
        else:
            logger.warning("No distance constraints were added - no inserters to optimize")
            
    
    def solve(self):

        
        output_item = next(iter(self.production_data.keys()))
        output_amount = 0
        if output_item in self.production_data:
            output_amount = self.production_data[output_item].get("amount_per_minute", 0)       
        # Otherwise, use the original solving approach
        os.makedirs("SMT_Modules", exist_ok=True)  # Create the directory if it doesn't exist
        model_file = os.path.join("SMT_Modules", f"solver_model_{output_item}_{output_amount}.smt")
        self.write_model_to_file(file_path=model_file)
        
        result = self.solver.check()
        logger.info(f"Solver check result: {result}")

        if result == self.sat:
            self.model = self.solver.model() 

         
            #self.debug_print_model_values()
            
            # Identify non-overlapping inserters
            non_overlapping_inserters = self.find_non_overlapping_inserters()
            # Add constraints to preserve the solved setup
            self.lock_initial_solution(non_overlapping_inserters)
            
            self.minimize_non_overlapping_inserters(non_overlapping_inserters)
            
            
            distance_result = self.solver.check()
            if distance_result == self.sat:
                self.model = self.solver.model()
                logger.info("Successfully minimized inserter distances")
                
                # Now minimize power poles if enabled
                # if hasattr(self, 'power_poles') and self.power_poles:
                #     self.minimize_power_poles()
                # else:
                #     logger.warning("Power poles are not enabled, skipping minimization.")
            
            self.model = self.solver.model() 
            logger.info(f"Final solver check result successful")
            self.build_map()

            
    
           
    
            return 
            
        else:
            logger.warning("Solver failed to find a solution.")
            return result
            
    def get_init_map(self):
        obstacle_map = np.zeros((self.height, self.width), dtype=int)
        for belt in self.global_input_belts:
                x = self.model.evaluate(belt.x).as_long()
                y = self.model.evaluate(belt.y).as_long()
                obstacle_map[y][x] = 1
                
        for belt in self.global_output_belts:
                x = self.model.evaluate(belt.x).as_long()
                y = self.model.evaluate(belt.y).as_long()
                obstacle_map[y][x] = 22
        return obstacle_map
    def build_map(self):
        
        
        obstacle_map = [[0 for _ in range(self.width)] for _ in range(self.height)]

        belt_point_information = []
        assembler_information = []
        inserter_information = []
        fluid_connection_info = []
        power_pole_information = []
        
        if self.solver.check() == self.sat:
            
            # Process power poles
            if hasattr(self, 'power_poles') and self.power_poles:
                for pole in self.power_poles:
                    if self.model.evaluate(pole.is_used):
                        x = self.model.evaluate(pole.x).as_long()
                        y = self.model.evaluate(pole.y).as_long()

                        obstacle_map[y][x] = 55  
                        
                        # Add to power pole information
                        power_pole_information.append([
                            self.power_pole_type, x, y
                        ])
            
            
            # Add input and output belts
            for belt in self.global_input_belts:
                x = self.model.evaluate(belt.x).as_long()
                y = self.model.evaluate(belt.y).as_long()
                
                obstacle_map[y][x] = 1
                
            for belt in self.global_output_belts:
                x = self.model.evaluate(belt.x).as_long()
                y = self.model.evaluate(belt.y).as_long()
                obstacle_map[y][x] = 22
            
            # Mark assemblers in the obstacle map
            for assembler in self.assemblers:
                x = self.model.evaluate(assembler.x).as_long()
                y = self.model.evaluate(assembler.y).as_long()
                width = assembler.width
                height = assembler.height
            
                machine_type = self.get_machine_type_for_recipe(assembler.item)
                default_assembler = self.config.get("machines", {}).get("default_assembler", "assembling-machine-1")
                machine_info = self.machines_data["assemblers"].get(machine_type, self.machines_data["assemblers"][default_assembler])
                
                # Get selected orientation for fluid connections
                orientation_idx = 0
     
                if hasattr(assembler, 'fluid_orientation'):
                    # First check if we have any fluid inserters
                    fluid_inserters = [ins for ins in assembler.inserters if ins.is_fluid]
                    
                    if fluid_inserters and machine_info.get("fluid_connection_pairs"):
                        fluid_connection_pairs = machine_info.get("fluid_connection_pairs", [])
                        
                        # Get positions of fluid inserters
                        fluid_positions = []
                        for inserter in fluid_inserters:
                            ix = self.model.evaluate(inserter.x).as_long()
                            iy = self.model.evaluate(inserter.y).as_long()
                            fluid_positions.append((ix, iy, inserter.item))
                        
                        logger.info(f"Fluid inserters for {assembler.id} are at: {fluid_positions}")
                        
                        # Find the orientation that matches these positions
                        best_match_orientation = None
                        
                        for idx, connection_pair in enumerate(fluid_connection_pairs):
                            valid_positions = []
                            # Get all valid input positions for this orientation
                            for input_pos in connection_pair["inputs"]:
                                abs_x = x + input_pos["dx"]
                                abs_y = y + input_pos["dy"]
                                valid_positions.append((abs_x, abs_y))
                            
                            # Check if all fluid inserters match valid positions in this orientation
                            matches = True
                            for fluid_pos in fluid_positions:
                                fluid_x, fluid_y, _ = fluid_pos
                                if (fluid_x, fluid_y) not in valid_positions:
                                    matches = False
                                    break
                            
                            if matches:
                                best_match_orientation = idx
                                logger.info(f"Found matching orientation {idx} for assembler {assembler.id}")
                                break
                        
                        if best_match_orientation is not None:
                            orientation_idx = best_match_orientation
                        else:
                            # If no exact match, try to find the best fit orientation
                            max_matches = 0
                            for idx, connection_pair in enumerate(fluid_connection_pairs):
                                matches_count = 0
                                valid_positions = []
                                
                                for input_pos in connection_pair["inputs"]:
                                    abs_x = x + input_pos["dx"]
                                    abs_y = y + input_pos["dy"]
                                    valid_positions.append((abs_x, abs_y))
                                
                                for fluid_pos in fluid_positions:
                                    fluid_x, fluid_y, _ = fluid_pos
                                    if (fluid_x, fluid_y) in valid_positions:
                                        matches_count += 1
                                
                                if matches_count > max_matches:
                                    max_matches = matches_count
                                    orientation_idx = idx
                            
                            logger.info(f"Best fit orientation for {assembler.id} is {orientation_idx} with {max_matches} matching inputs")
                    else:
                        try:
                            # Try the standard Z3 approach
                            orientation_val = self.model.evaluate(assembler.fluid_orientation)
                            if hasattr(orientation_val, 'as_long'):
                                orientation_idx = orientation_val.as_long()
                            else:
                                # For other solvers, try to extract the numerical value differently
                                orientation_idx = int(str(orientation_val))
                        except Exception as e:
                            logger.warning(f"Could not get orientation for {assembler.id}: {e}")
                            orientation_idx = 0  # Default to first orientation
              
                        
                # Add assembler with dimensions and orientation to info
                assembler_information.append([
                    assembler.item, x, y, width, height, machine_type, orientation_idx
                ])
                
                
                # Mark 3x3 area around the assembler as occupied
                for dx in range(width):
                    for dy in range(height):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            obstacle_map[y + dy][x + dx] = 33

                
                
                
                
                # Mark fluid connection points if this machine has them
                fluid_connection_pairs = machine_info.get("fluid_connection_pairs", [])
                if fluid_connection_pairs and orientation_idx < len(fluid_connection_pairs):
                    connection_pair = fluid_connection_pairs[orientation_idx]
                    
                    # Mark all input points
                    for input_pos in connection_pair["inputs"]:
                        conn_x = x + input_pos["dx"]
                        conn_y = y + input_pos["dy"]
                        
                        if 0 <= conn_x < self.width and 0 <= conn_y < self.height:
                            obstacle_map[conn_y][conn_x] = 66  # Mark as fluid input
                            
                            fluid_connection_info.append({
                                "assembler_id": assembler.id,
                                "position": (conn_x, conn_y),
                                "relative_pos": (input_pos["dx"], input_pos["dy"]),
                                "type": "input",
                                "machine_type": machine_type,
                                "orientation": orientation_idx
                            })
                    
                    # Mark all output points
                    for output_pos in connection_pair["outputs"]:
                        conn_x = x + output_pos["dx"]
                        conn_y = y + output_pos["dy"]
                        
                        if 0 <= conn_x < self.width and 0 <= conn_y < self.height:
                            obstacle_map[conn_y][conn_x] = 77  # Mark as fluid output
                            
                            fluid_connection_info.append({
                                "assembler_id": assembler.id,
                                "position": (conn_x, conn_y),
                                "relative_pos": (output_pos["dx"], output_pos["dy"]),
                                "type": "output",
                                "machine_type": machine_type,
                                "orientation": orientation_idx
                            })
                
                
                # Mark inserters in the obstacle map
                for inserter in assembler.inserters:
                    ix = self.model.evaluate(inserter.x).as_long()
                    iy = self.model.evaluate(inserter.y).as_long()
                    
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
                    elif ix == x + 3 and iy >= y and iy + 2:
                        direction = "west"   # Facing left toward the assembler
                    
                    
                    inserter_information.append([inserter.item, ix, iy, direction])
                    
                    # Add special logging for iron-plate inserters at position (9, 5)
                    if ix == 9 and iy == 5:
                        logger.info(f"Critical inserter added: {inserter.item} at position (9, 5) with direction {direction}")
                    
                    # Mark inserter position as occupied
                    if inserter.type == "input":
                        if 0 <= ix < self.width and 0 <= iy < self.height:
                            obstacle_map[iy][ix] = 44
                            
                    belt = inserter.belt
                    if belt is not None:
                        bx = self.model.evaluate(belt.x).as_long()
                        by = self.model.evaluate(belt.y).as_long()

                        if belt.type == "start":
                            if 0 <= bx < self.width and 0 <= by < self.height:
                                obstacle_map[by][bx] = 88
                                    
                        else:
                            if 0 <= bx < self.width and 0 <= by < self.height:
                                obstacle_map[by][bx] = 99
                            
                        if bx > -1 and by > -1:
                            start_id = belt.id.split('_')[0]
                            belt_point_information.append([start_id, bx, by, belt.type])
                            
                    logger.debug(f"Added inserter {inserter.id} at ({ix}, {iy}) with direction {direction} and item {inserter.item}")   
                    logger.debug(f"Added belt {belt.id} at ({bx}, {by}) with type {belt.type} and item {belt.item}")
                        

    
        else:
            logger.info('not sat')
            logger.info("=== Finished building map ===")
            
        logger.debug(f"Obstacle map size: {len(obstacle_map)}x{len(obstacle_map[0])}")
        logger.debug(f"Belt point information: {belt_point_information}")
        logger.debug(f"Assembler information: {assembler_information}")
        logger.debug(f"Inserter information: {inserter_information}")
        logger.debug(f"Power pole information: {power_pole_information}")
       
        # Store all information as instance variables for later use
        self.obstacle_map = obstacle_map
        self.belt_point_information = belt_point_information
        self.assembler_information = assembler_information
        self.inserter_information = inserter_information
        self.fluid_connection_info = fluid_connection_info
        self.power_pole_information = power_pole_information
        
        return obstacle_map, belt_point_information, assembler_information, inserter_information, fluid_connection_info, power_pole_information

    
        
    def restrict_current_setup(self):
        """
        Restricts the current setup of assemblers and inserters by adding a negated constraint
        for the current configuration to the solver.
        """
        if self.solver.check() == sat:
            model = self.solver.model()
            constraints = []
            # Capture assembler positions
            for assembler in self.assemblers:
                x = model.evaluate(assembler.x).as_long()
                y = model.evaluate(assembler.y).as_long()
                assembler_constraint = And(assembler.x == x, assembler.y == y)
                constraints.append(assembler_constraint)
                # Capture inserter positions associated with the assembler
                for inserter in assembler.inserters:
                    ix = model.evaluate(inserter.x).as_long()
                    iy = model.evaluate(inserter.y).as_long()
                    inserter_constraint = self.And(inserter.x == ix, inserter.y == iy)
                    constraints.append(inserter_constraint)
            # Add the negated constraint to forbid this setup
            if constraints:
                forbidden_constraint = self.Not(self.And(*constraints))
                self.solver.add(forbidden_constraint)
                logger.info("Added a constraint to forbid the current setup of assemblers and inserters.")
        else:
            logger.info("No valid configuration to restrict (solver state is not SAT).")
            
    
    def write_model_to_file(self, file_path, logic="QF_LIA"):
        """Write the current constraint system to a file in SMT-LIB2 format
        
        Args:
            file_path: Path to save the SMT file
            logic: SMT logic to use (default: QF_LIA)
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
              # Change extension to .smt2 if it's not already
            if not file_path.lower().endswith('.smt2'):
                file_path = file_path.replace('.smt', '.smt2')
            
            # Get the base SMT-LIB2 expression from solver
            smt_content = self.solver.sexpr()
            
            # Ensure the file has proper SMT-LIB2 format elements
            formatted_content = []
            
            # Add the logic declaration at the beginning
            formatted_content.append(f"(set-logic {logic})")
            
            # Add model-production option
            formatted_content.append("(set-option :produce-models true)")
            
            # Add the original content, skipping any existing logic declarations
            for line in smt_content.split('\n'):
                if not line.strip().startswith("(set-logic") and not line.strip().startswith("(set-option"):
                    formatted_content.append(line)
            
            # Add check-sat and get-model commands at the end
            if "(check-sat)" not in smt_content:
                formatted_content.append("(check-sat)")
            
            if "(get-model)" not in smt_content:
                formatted_content.append("(get-model)")
            
            # Write the enhanced SMT-LIB2 content
            with open(file_path, 'w') as f:
                f.write('\n'.join(formatted_content))
            
            logger.info(f"Successfully wrote SMT-LIB2 model to {file_path} with logic {logic}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to write SMT-LIB2 model to file: {e}")
            return False
   