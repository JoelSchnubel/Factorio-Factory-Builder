#! .venv\Scripts\python.exe

from z3 import *
import pygame
import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for detailed information, change to INFO for less verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("factory_log.log", mode='w'),  # Specify the log file name
    ]
)


ITEMS_PER_SECOND = 7.5

# Define constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Define the size of each grid cell
CELL_SIZE = 40

INSERTER_COLOR_MAP = {
    'input': (255, 165, 0),  # Orange 
    'output': (255, 20, 147) # Deep Pink
}

BELT_COLOR_MAP = {
    'start': (0, 128, 0),  # Green 
    'end': (255, 0, 0)     # Red
}



class Assembler:
    def __init__(self,id,inserters,item=None,capacity=0):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        
        self.inserters = inserters
        self.item = item
        
        #self.capacity = Int(f'{id}_capacity')
        self.capacity = capacity
     
    def __str__(self) -> str:
            inserters_str = ', '.join([str(inserter) for inserter in self.inserters])
            return (f"Assembler(id={self.id}, position=({self.x}, {self.y}), "
                    f"capacity={self.capacity}, inserters=[{inserters_str}])")
   
        
        
class Inserter:
    def __init__(self,id,type,belt,item=None):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        self.type = type # 'input' or 'output'
        
        self.belt = belt
        self.item = item
        
        self.direct_interaction = Bool(f"{id}_direct")

        self.is_merged = False
    
        
    def __str__(self) -> str:
        
            return (f"Inserter(id={self.id}, position=({self.x}, {self.y}), type={self.type}), item={self.item}"
                    f",Belt={str(self.belt)}")
class Belt:
    def __init__(self,id,type,item=None,int_x=0,int_y=0):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        self.type = type # 'start' or 'end'
        self.item = item
        
        self.is_used = Bool(f"{id}_used")  # Z3 Boolean for belt usage
        
        self.int_x =int_x
        self.int_y =int_y
        
    def __str__(self) -> str:
        return f"Belt(id={self.id}, position=({self.x}, {self.y}), type={self.type}), item={self.item}"


class Z3Solver:
    def __init__(self, width, height, production_data):
        self.width = width
        self.height = height
        self.production_data = production_data
        self.solver = Optimize()

        # Create variables for assemblers, input inserters, output inserters, and belts
        self.assemblers = []
        self.placed_assembler = []
        self.global_input_belts = []
        self.global_output_belts = [] 

        self.additional_constraints = []
        
        self.input_information = None
        self.output_information = None

        self.obstacle_maps= []
     
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
        
        self.add_input_inserter_merge_user_belt_constraint()
        self.add_input_inserter_merge_assembler_constraint()
        
        
        #self.add_minimize_belts()

    def add_manuel_IO_constraints(self, input_information, output_information):
        logging.info("Starting to add manual I/O constraints.")
        
        self.input_information = input_information
        self.output_information = output_information

        for item, data in input_information.items():
            logging.debug(f"Processing input item '{item}' with data: {data}")
            
            # Create and position the input belts
            belt = Belt(id=f"{item}_{data['input']}_input", type='input', item=item)
            logging.info(f"Adding input belt for item '{item}' at position {data['input']}")
            self.solver.add(And(belt.x == data["input"][1], belt.y == data["input"][0]))
            self.global_input_belts.append(belt)
            
            belt = Belt(id=f"{item}_{data['output']}_input", type='input', item=item)
            logging.info(f"Adding output belt for item '{item}' at position {data['output']}")
            self.solver.add(And(belt.x == data["output"][1], belt.y == data["output"][0]))
            self.global_input_belts.append(belt)

            grid = data['grid']
            if grid is not None:
                logging.debug(f"Processing grid path for input item '{item}'")
                for row in range(self.height):
                    for col in range(self.width):
                        if grid[col][row] == 2:  # Belt path marked with '2'
                            logging.info(f"Adding belt path for '{item}' at grid position ({col}, {row})")
                            belt = Belt(id=f"{item}_({col}, {row})_input", type='input', item=item, int_x=col, int_y=row)
                            self.solver.add(And(belt.x == col, belt.y == row))
                            self.global_input_belts.append(belt)

        for item, data in output_information.items():
            logging.debug(f"Processing output item '{item}' with data: {data}")

            # Create and position the output belts
            belt = Belt(id=f"{item}_{data['input']}_output", type='output', item=item)
            logging.info(f"Adding input belt for output item '{item}' at position {data['input']}")
            self.solver.add(And(belt.x == data["input"][1], belt.y == data["input"][0]))
            self.global_output_belts.append(belt)
            
            belt = Belt(id=f"{item}_{data['output']}_output", type='output', item=item)
            logging.info(f"Adding output belt for output item '{item}' at position {data['output']}")
            self.solver.add(And(belt.x == data["output"][1], belt.y == data["output"][0]))
            self.global_output_belts.append(belt)

            grid = data['grid']
            if grid is not None:
                logging.debug(f"Processing grid path for output item '{item}'")
                for row in range(self.height):
                    for col in range(self.width):
                        if grid[col][row] == 2:  # Belt path marked with '2'
                            logging.info(f"Adding belt path for '{item}' at grid position ({col}, {row})")
                            belt = Belt(id=f"{item}_({col}, {row})_output", type='output', item=item, int_x=col, int_y=row)
                            self.solver.add(And(belt.x == col, belt.y == row))
                            self.global_output_belts.append(belt)

        logging.info("Finished adding manual I/O constraints. Calling solver.")
        self.solve()

    def add_constraint(self, constraint):
        logging.info("Adding a new constraint.")
        self.additional_constraints.append(constraint)
        self.solver.add(constraint)
        logging.debug(f"Added constraint: {constraint}")

    def create_assemblers(self):
        logging.info("Starting to create assemblers based on production data.")
        assembler_count = 0
        
        for item_id, item_info in self.production_data.items():
            logging.debug(f"Processing item '{item_id}' with production data: {item_info}")

            if 'assemblers' in item_info and item_info['assemblers'] > 0:
                for i in range(item_info['assemblers']):
                    logging.info(f"Creating assembler for item '{item_id}', instance {i}")
                    
                    # Create input inserters for the assembler
                    input_inserters = []
                    
                    for inserter_info in item_info['input_inserters']:
                        for j in range(inserter_info['amount']):
                            input_inserter_id = f"{inserter_info['id']}_in_{assembler_count}_{i}_{j}"
                            belt_id = f"{inserter_info['id']}_end_{assembler_count}_{i}_{j}"
                            
                            logging.info(f"Creating input inserter for {inserter_info['id']} at {input_inserter_id}")

                            # Create the input inserter with a unique ID and associated belt
                            input_inserters.append(
                                Inserter(
                                    id=input_inserter_id,
                                    type='input',
                                    item=inserter_info['id'],
                                    belt=Belt(
                                        id=belt_id,
                                        type="end",  # or "start" depending on your logic
                                        item=inserter_info['id']
                                    )
                                )
                            )
                            
                    
                    assembler = Assembler(
                        id=f"{item_id}_{assembler_count}_{i}",
                        inserters=input_inserters,
                        item=item_id,
                        capacity=item_info['capacity']
                    )
                    
                    
                    self.assemblers.append(assembler)
                    logging.debug(f"Created assembler with ID: {assembler.id} and input inserters: {[inserter.id for inserter in input_inserters]}")
                    logging.debug(f"Created assembler with ID: {assembler.id} with item = {assembler.item}")
                    logging.debug(f"Created assembler with ID: {assembler.id} with capacity = {assembler.capacity}")
                assembler_count += 1

        logging.info(f"Created {len(self.assemblers)} assemblers in total.")
        
    # assembler is 3x3 and not allowed to get out of bounds 
    def add_bound_constraints_assembler(self):
        logging.info("Adding boundary constraints for assemblers.")
        for assembler in self.assemblers:
            logging.debug(f"Setting boundary constraints for assembler ID {assembler.id}")
            self.solver.add(And(assembler.x >= 0, assembler.x <= self.width - 3))
            self.solver.add(And(assembler.y >= 0, assembler.y <= self.height - 3))

    # belts and inserter bound constraints
    def add_bound_constraints_belts_and_inserters(self):
        logging.info("Adding boundary constraints for belts and inserters.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                logging.debug(f"Setting boundary constraints for inserter ID {inserter.id} and belt ID {belt.id}")
                self.solver.add(And(inserter.x >= 0, inserter.x < self.width))
                self.solver.add(And(inserter.y >= 0, inserter.y < self.height))
                self.solver.add(And(belt.x >= 0, belt.x < self.width))
                self.solver.add(And(belt.y >= 0, belt.y < self.height))

    # user input/putput belts are not allowed to overlap with assembler
    def add_global_belt_overlap_assembler_constraint(self):
        logging.info("Adding assembler overlap constraints to avoid global belts.")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for belt in belts:
                #logging.debug(f"Preventing overlap between assembler ID {assembler.id} and belt ID {belt.id}")
                self.solver.add(Or(
                    Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                    Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                ))
    # user input/putput belts are not allowed to overlap with inserter
    def add_global_belt_overlap_inserter_constraint(self):
        logging.info("Adding inserter overlap constraints to avoid global belts.")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                for belt in belts:
                    #logging.debug(f"Preventing overlap between inserter ID {inserter.id} and belt ID {belt.id}")
                    self.solver.add(Or(
                        Or(belt.x < inserter.x, belt.x > inserter.x + 2),
                        Or(belt.y < inserter.y, belt.y > inserter.y + 2)
                    ))
    # user input/putput belts are allowed to overlap with belts that have the same item
    def add_global_belt_overlap_belt_constraint(self):
        logging.info("Adding belt overlap constraints to avoid conflicts with global belts (different items only).")
        belts = self.global_input_belts + self.global_output_belts
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                inserter_belt = inserter.belt
                for belt in belts:
                    if belt.item != inserter_belt.item:
                        #logging.debug(f"Preventing overlap between belt ID {belt.id} and inserter belt ID {inserter_belt.id}")
                        self.solver.add(Or(
                            Or(belt.x < inserter_belt.x, belt.x > inserter_belt.x + 2),
                            Or(belt.y < inserter_belt.y, belt.y > inserter_belt.y + 2)
                        ))
                        
    # assembler are not allowed to overlap with other assemblers
    def add_assembler_overlap_assembler_constraint(self):
        logging.info("Adding assembler overlap constraints to prevent assembler-assembler overlap.")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    #logging.debug(f"Preventing overlap between assembler ID {assembler.id} and assembler ID {other_assembler.id}")
                    self.solver.add(Or(assembler.x + 2 < other_assembler.x, assembler.x > other_assembler.x + 2,
                                    assembler.y + 2 < other_assembler.y, assembler.y > other_assembler.y + 2))
                    
    # assemblers are not allowed to overlap with inserters
    def add_assembler_overlap_inserter_constraint(self):
        logging.info("Adding assembler-inserter overlap constraints.")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    for inserter in other_assembler.inserters:
                        #logging.debug(f"Preventing overlap between assembler ID {assembler.id} and inserter ID {inserter.id}")
                        self.solver.add(Or(
                            Or(inserter.x < assembler.x, inserter.x > assembler.x + 2),
                            Or(inserter.y < assembler.y, inserter.y > assembler.y + 2)
                        ))
    # assembler and belts are only allowed to overlap if they have the same item 
    def add_assembler_overlap_belt_constraint(self):
        logging.info("Adding assembler-belt overlap constraints (excluding same-item belts).")
        for assembler in self.assemblers:
            for other_assembler in self.assemblers:
                if assembler.id != other_assembler.id:
                    for inserter in other_assembler.inserters:
                        belt = inserter.belt
                        if belt.item != assembler.item:
                            #logging.debug(f"Preventing overlap between assembler ID {assembler.id} and belt ID {belt.id}")
                            self.solver.add(Or(
                                Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                                Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                            ))
                            
    # inserters are not allowed to overlap each other 
    # also for same assembler
    def add_inserter_overlap_inserter_constraint(self):
        logging.info("Adding inserter overlap constraints to prevent inserter-inserter overlap.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                
                for other_inserter in assembler.inserters:
                    if inserter.id != other_inserter.id:
                        self.solver.add(Or(
                                Or(inserter.x < other_inserter.x, inserter.x > other_inserter.x),
                                Or(inserter.y < other_inserter.y, inserter.y > other_inserter.y)
                            ))
                
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        for other_inserter in other_assembler.inserters:
                            #logging.debug(f"Preventing overlap between inserter ID {inserter.id} and inserter ID {other_inserter.id}")
                            self.solver.add(Or(
                                Or(inserter.x < other_inserter.x, inserter.x > other_inserter.x),
                                Or(inserter.y < other_inserter.y, inserter.y > other_inserter.y)
                            ))
                            
    # inserter and belts ar not allowed to overlap
    def add_inserter_overlap_belt_constraint(self):
        logging.info("Adding inserter-belt overlap constraints to prevent inserters from overlapping with belts.")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        for other_inserter in other_assembler.inserters:
                            other_belt = other_inserter.belt
                            #logging.debug(f"Preventing overlap between inserter ID {inserter.id} and belt ID {other_belt.id}")
                            self.solver.add(Or(
                                Or(inserter.x < other_belt.x, inserter.x > other_belt.x),
                                Or(inserter.y < other_belt.y, inserter.y > other_belt.y)
                            ))
                            
    # belts are allowed to overlap if they have the same item
    def add_belt_overlap_belt_constraint(self):
        logging.info("Adding belt overlap constraints to prevent belt-belt overlap (except same-item belts).")
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                belt = inserter.belt
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        for other_inserter in other_assembler.inserters:
                            other_belt = other_inserter.belt
                            if belt.item != other_belt.item:
                                #logging.debug(f"Preventing overlap between belt ID {belt.id} and other belt ID {other_belt.id}")
                                self.solver.add(Or(
                                    Or(belt.x < other_belt.x, belt.x > other_belt.x),
                                    Or(belt.y < other_belt.y, belt.y > other_belt.y)
                                ))
                
        
    # inserter needs to be adjacent to assembler
    def add_inserter_adjacent_to_assembler(self):

        for assembler in self.assemblers:
            
            input_positions = [
                    (assembler.x, assembler.y - 1), (assembler.x + 1, assembler.y - 1), (assembler.x + 2, assembler.y - 1),  # Top row
                    (assembler.x, assembler.y + 3), (assembler.x + 1, assembler.y + 3), (assembler.x + 2, assembler.y + 3),  # Bottom row
                    (assembler.x - 1, assembler.y), (assembler.x - 1, assembler.y + 1), (assembler.x - 1, assembler.y + 2),  # Left column
                    (assembler.x + 3, assembler.y), (assembler.x + 3, assembler.y + 1), (assembler.x + 3, assembler.y + 2)   # Right column
                ]
            
            
            
            for inserter in assembler.inserters:
                # Ensure each inserter is adjacent to its assembler
                logging.info(f"Ensuring inserter {inserter.id} is adjacent to assembler {assembler.id}")
                self.solver.add(Or([And(inserter.x == pos[0], inserter.y == pos[1]) for pos in input_positions]))

                belt = inserter.belt
                if belt is not None:
                    logging.info(f"Adding belt position constraint for inserter {inserter.id} with belt {belt.id}")
                    # Ensure that the belt corresponding to the inserter is at the opposite side of the assembler
                    self.solver.add(
                    Or(
                                # Inserter to the left of the assembler, belt is to the left of the inserter
                                And(inserter.x == assembler.x - 1, belt.x == inserter.x - 1, belt.y == inserter.y),
                                # Inserter to the right of the assembler, belt is to the right of the inserter
                                And(inserter.x == assembler.x + 3, belt.x == inserter.x + 1, belt.y == inserter.y),
                                # Inserter above the assembler, belt is above the inserter
                                And(inserter.y == assembler.y - 1, belt.x == inserter.x, belt.y == inserter.y - 1),
                                # Inserter below the assembler, belt is below the inserter
                                And(inserter.y == assembler.y + 3, belt.x == inserter.x, belt.y == inserter.y + 1)
                        ))
    
    # ensure at least one space 1 and 2 tiles away from the assembler is free to ensure a possible output
    def add_space_for_output_of_assembler(self):
        for assembler in self.assemblers:
            output_positions = [[(assembler.x, assembler.y - 1),(assembler.x, assembler.y - 2)], # upper left 
                                [(assembler.x + 1, assembler.y - 1),(assembler.x + 1, assembler.y - 2)], # upper middle
                                [(assembler.x + 2, assembler.y - 1),(assembler.x + 2, assembler.y - 2)], # upper right
                                
                                [(assembler.x, assembler.y + 3),(assembler.x, assembler.y + 4)], # bottom left       
                                [(assembler.x + 1, assembler.y + 3),(assembler.x + 1, assembler.y + 4)], # bottom middle                                                                                               
                                [(assembler.x + 2, assembler.y + 3),(assembler.x + 2, assembler.y + 4)], # bottom right       
                                
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
                pair_constraints.append(And(
                    Not(self.is_position_occupied(pos1[0], pos1[1])),
                    Not(self.is_position_occupied(pos2[0], pos2[1]))
                ))

            if pair_constraints:
                # Add a constraint that at least one pair is empty
                logging.info(f"Adding output space constraints for assembler {assembler.id}")
                self.solver.add(Or(pair_constraints))
            
    
    def is_position_occupied(self,x,y):
        
        occupied_conditions = []
        for assembler in self.assemblers:
            # Assembler occupies a 3x3 area
            occupied_conditions.append(
                And(x >= assembler.x, x <= assembler.x + 2,
                    y >= assembler.y, y <= assembler.y + 2)
            )
            for inserter in assembler.inserters:
                
                # Check if any inserter occupies the position
                occupied_conditions.append(
                    And(x == inserter.x, y == inserter.y)
                )
                
                # Check if any belt occupies the position
                belt = inserter.belt
                occupied_conditions.append(
                    And(x == belt.x, y == belt.y)
                )
        
        # Check if any belt occupies the position
        for belt in self.global_input_belts + self.global_output_belts:
            occupied_conditions.append(
                And(x == belt.x, y == belt.y)
            )
        
        # Return True if any of the occupied conditions hold, otherwise False
        #logging.debug(f"Position ({x},{y}) occupied check: {occupied_conditions}")
        return Or(occupied_conditions)


    # input inserter should be next to other assembler or belt that transports/produces the same item
    def add_input_inserter_merge_user_belt_constraint(self):
        belts = self.global_input_belts + self.global_output_belts
        merge_constraints = []
        
        logging.debug("Adding constraints to merge input inserters with belts of matching items.")
        
        for belt in belts:
            for assembler in self.assemblers:
                for inserter in assembler.inserters:
                    inserter_belt = inserter.belt
        
                    if belt.item == inserter_belt.item:
                        logging.debug(  f"Matching item found: Belt (ID: {belt.id}, Item: {belt.item}) "
                                        f"and Inserter Belt (ID: {inserter_belt.id}, Item: {inserter_belt.item})")
                        merge_constraints.append(And(inserter_belt.x == belt.x ,inserter_belt.y == belt.y))
                    
        
        if merge_constraints:
            logging.debug("Adding merge constraints for input inserters to the solver.")
            self.solver.add(Or(merge_constraints))
        else:
            logging.debug("No merge constraints were added as no matching items were found.")
                
        
        
    # we can force the belt to overlap with either an belt defined by the user or force merge it with an assemblers outline. assembler (x,y) = its upper left corner
    
    # merge assemblers belt with other assembler -> eg. assembler = 
    
    # additionally we need to look at the capacity of an assembler:
    # the assembler we wan
    def add_input_inserter_merge_assembler_constraint(self):
        
        merge_constraints = []
        
        logging.debug("Adding constraints to merge input inserters with assembler edges for items with matching types.")
         
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                
                # Create a list to hold the constraints for each valid assembler
                assembler_constraints = []
        
                for other_assembler in self.assemblers:
                    if assembler.id != other_assembler.id:
                        
                        #logging.debug(f"Assembler {assembler.id} and other assembler {other_assembler.id}")
                        #logging.debug(f"Inserter item {inserter.item} and other assembler item {other_assembler.item}")
                        
                        # if my input inserter and the other assembler produces/transports same item, set the belt on one of the edge positions of the assembler -> all but the middle
                        if inserter.item == other_assembler.item: #and assembler.capacity > 0:
                            
                            
                            logging.debug(  f"Assembler {assembler.id} and other assembler {other_assembler.id} "
                                        f"have matching items: {inserter.item}")
                            
                            logging.debug(f"Assembler {assembler.id} and other assembler {other_assembler.id} have matching items: {inserter.item}")
                            
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
                            
                            
                            belt = inserter.belt 
      
                            constraints = [
                                And(inserter.belt.x == pos[0], inserter.belt.y == pos[1])
                                for pos in merge_positions
                            ]
                            
                            if constraints:
                                logging.debug(f"Adding constraints to position belt for inserter {inserter.id} "
                                            f"at positions: {merge_positions}")
                                
                                assembler_constraints.append(Or(constraints))
                                #merge_constraints.append(Or(constraints))
                                #self.solver.add(merge_constraints)
                                
                            else:
                
                                logging.debug(f"No valid merge positions found for inserter {inserter.id}.")
                                
                # If there are valid constraints for this inserter, combine them with 'Or' between different assemblers
                if assembler_constraints:
                    # reduce capacity by 1 
                    assembler.capacity -= 1
                    
                    
                    logging.debug(f"Adding 'Or' between valid positions for inserter {inserter.id}.")
                    merge_constraints.append(Or(assembler_constraints))  # This Or ensures multiple assemblers are considered
                    self.solver.add(merge_constraints)
                
                else:
                    logging.debug(f"No valid merge positions found for inserter {inserter.id}.")
                    
        #if merge_constraints:
        #    self.solver.add(Or(merge_constraints))  # This 'Or' handles multiple inserters' constraints
                        
    
    
    # minimizes the number of non merged belts 
    def add_minimize_belts(self):
        logging.info("Adding constraints to maximize overlaps of inserter belts with global belts and assemblers.")

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
                    is_overlapping = And(
                        inserter_belt.x == global_belt.x,
                        inserter_belt.y == global_belt.y
                    )
                    # Add to overlap tracking
                    overlap_var = Bool(f"overlap_{inserter_belt.id}_{global_belt.id}")
                    self.solver.add(Implies(overlap_var, is_overlapping))
                    overlap_variables.append(overlap_var)
                    
        
        # Check overlaps with other assemblers
        for assembler in self.assemblers:
            assembler_edges = [
                (assembler.x, assembler.y),  # Upper-left
                (assembler.x + 1, assembler.y),  # Upper-middle
                (assembler.x + 2, assembler.y),  # Upper-right
                (assembler.x, assembler.y + 1),  # Middle-left
                (assembler.x + 2, assembler.y + 1),  # Middle-right
                (assembler.x, assembler.y + 2),  # Lower-left
                (assembler.x + 1, assembler.y + 2),  # Lower-middle
                (assembler.x + 2, assembler.y + 2),  # Lower-right
            ]
            
            for edge in assembler_edges:
                is_overlapping = And(
                    inserter_belt.x == edge[0],
                    inserter_belt.y == edge[1]
                )
                # Add to overlap tracking
                overlap_var = Bool(f"overlap_{inserter_belt.id}_assembler_{assembler.id}")
                self.solver.add(Implies(overlap_var, is_overlapping))
                overlap_variables.append(overlap_var)
        
            
        # Maximize the sum of all overlap variables
        logging.info("Adding optimization goal to maximize belt overlaps.")
        self.solver.maximize(Sum([If(var, 1, 0) for var in overlap_variables]))
                    
    
    def solve(self):
            result = self.solver.check()
            print("model is:",result)
            
            logging.info(f"Solver check result: {result}")
            self.model = self.solver.model() 
            
            self.build_map()
            
            return 

        
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
        obstacle_map = np.zeros((self.height, self.width), dtype=int)

        belt_point_information = []
        assembler_information = []
        inserter_information = []
        
        if self.solver.check() == sat:
            # add input and output belts:
            for belt in self.global_input_belts:
                x = self.model.evaluate(belt.x).as_long()
                y = self.model.evaluate(belt.y).as_long()
                
                obstacle_map[y][x] = 1
                
            for belt in self.global_output_belts:
                x = self.model.evaluate(belt.x).as_long()
                y = self.model.evaluate(belt.y).as_long()
                obstacle_map[y][x] = 22
            
            #Mark assemblers in the obstacle map
            for assembler in self.assemblers:
                x = self.model.evaluate(assembler.x).as_long()
                y = self.model.evaluate(assembler.y).as_long()
                
                #print(str(assembler.id) + " x: "+str(x)+ "y: "+str(y))
                
                assembler_information.append([assembler.item,x,y])
                
                # Mark 3x3 area around the assembler as occupied
                for dx in range(3):
                    for dy in range(3):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            obstacle_map[y + dy][x + dx] = 33

                #Mark inserters in the obstacle map
                for inserter in assembler.inserters:
                    ix = self.model.evaluate(inserter.x).as_long()
                    iy = self.model.evaluate(inserter.y).as_long()
                    
                    inserter_information.append([inserter.item,ix,iy])
                    
                    #print(str(inserter.id) + " x: "+str(ix)+ "y: "+str(iy))
                    
                    # Mark inserter position as occupied
                    if inserter.type == "input":
                        if 0 <= ix < self.width and 0 <= iy < self.height:
                            obstacle_map[iy][ix] = 44
                            
                    belt = inserter.belt
                    if belt is not None:
                        
                        #print(f"inserter_belt {belt}")
                        
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
                
        else:
            print('not sat')
            
            
        return obstacle_map,belt_point_information,assembler_information,inserter_information

        
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
                    inserter_constraint = And(inserter.x == ix, inserter.y == iy)
                    constraints.append(inserter_constraint)
            # Add the negated constraint to forbid this setup
            if constraints:
                forbidden_constraint = Not(And(*constraints))
                self.solver.add(forbidden_constraint)
                print("Added a constraint to forbid the current setup of assemblers and inserters.")
        else:
            print("No valid configuration to restrict (solver state is not SAT).")
            
            
        

        
# Example usage
def main():
    pass

if __name__ == "__main__":
    main()
