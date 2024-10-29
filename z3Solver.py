#! .venv\Scripts\python.exe

from z3 import *
import pygame
import numpy as np
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
    def __init__(self,id,inserters,item=None):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        
        self.inserters = inserters
        self.item = None
        
    def __str__(self) -> str:
            inserters_str = ', '.join([str(inserter) for inserter in self.inserters])

            return (f"Assembler(id={self.id}, position=({self.x}, {self.y}),"
                    f"inserters=[{inserters_str}]")
        
class Inserter:
    def __init__(self,id,type,belt,item=None):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        self.type = type # 'input' or 'output'
        
        self.belt = belt
        self.item = item
        
        self.direct_interaction = Bool(f"{id}_direct")
        
    
        
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
        self.output_information =None

        self.obstacle_maps= []
     
    def build_sequential(self):
        self.create_assemblers()
        print(f"assemblers, {self.assemblers}")
        init_map =self.get_init_map()
        print(f"init_map{init_map}")
        self.obstacle_maps.append(init_map)
        
        for assembler in self.assemblers[::-1]:
            
            
            self.add_single_assembler_constraint(assembler)

            self.add_single_inserter_constraint(assembler)
            
            self.add_input_belt_merging(assembler)
            self.add_inserter_merging(assembler)
            
            
            #self.add_single_belt_constraint(assembler)
     
            self.add_single_overlapping_constraints(assembler)
            #self.add_overlapping_constraints()
            self.add_bound_constraints()
            self.solve()
            obstacle_map,belt_point_information = self.build_map()
            print(obstacle_map)
            self.obstacle_maps.append(obstacle_map)
            
            self.fix_position(assembler)
            
        return self.placed_assembler, self.model
           
      
    def create_constraints(self):
        self.create_assemblers()
        
            
        self.add_assembler_constraints()
        
        self.add_bound_constraints()
        self.add_overlapping_constraints()
        self.add_inserter_constraints()
        self.add_belt_penalty()
    
        
        
    def fix_position(self,assembler):
        # model is solved at that point, we can eval the positions using the solver
        ax = self.model.evaluate(assembler.x).as_long()
        ay = self.model.evaluate(assembler.y).as_long()
        
        self.solver.add(assembler.x == ax)
        self.solver.add(assembler.y == ay)
        
        for inserter in assembler.inserters:
            
            ix = self.model.evaluate(inserter.x).as_long()
            iy = self.model.evaluate(inserter.y).as_long()

            # Add constraints to "lock" the inserter's position
            self.solver.add(inserter.x == ix)
            self.solver.add(inserter.y == iy)

            # Lock the position of the associated belt
            belt = inserter.belt
            
            if belt is not None:
                bx = self.model.evaluate(belt.x).as_long()
                by = self.model.evaluate(belt.y).as_long()

                # Add constraints to "lock" the belt's position
                self.solver.add(belt.x == bx)
                self.solver.add(belt.y == by)
        
    # add the positional constraints inputted by the user 
    # we have item with all belt info and !belt positions
    def add_manuel_IO_constraints(self,input_information,output_information):
        
        self.input_information  = input_information
        self.output_information = output_information
        
        for item, data in input_information.items():
            belt = Belt(id=item+"_"+str(data["input"])+"_"+"input" , type='input',item=item)
            # define the position of the belt
            self.solver.add(And(belt.x == data["input"][1] ,belt.y == data["input"][0] ))
            self.global_input_belts.append(belt)
            
            belt = Belt(id=item+"_"+str(data["output"])+"_"+"input" , type='input',item=item)
            # define the position of the belt
            self.solver.add(And(belt.x == data["output"][1] ,belt.y == data["output"][0] ))
            self.global_input_belts.append(belt)
            grid = data['grid']
            if grid is not None:
                for row in range(self.height):
                    for col in range(self.width):
                        if grid[col][row] == 2:  # Belt path marked with '2'
                            belt = Belt(id=item+"_("+str(col)+", "+str(row)+")_"+"input" , type='input',item=item,int_x=col,int_y=row)
                            # define the position of the belt
                            self.solver.add(And(belt.x == col ,belt.y == row ))
                            self.global_input_belts.append(belt)
                            
        for item, data in output_information.items():
            belt = Belt(id=item+"_"+str(data["input"])+"_"+"output" , type='output',item=item)
            # define the position of the belt
            self.solver.add(And(belt.x == data["input"][1] ,belt.y == data["input"][0] ))
            self.global_output_belts.append(belt)
            
            belt = Belt(id=item+"_"+str(data["output"])+"_"+"output" , type='output',item=item)
            # define the position of the belt
            self.solver.add(And(belt.x == data["output"][1] ,belt.y == data["output"][0] ))
            self.global_output_belts.append(belt)
            grid = data['grid']
            if grid is not None:
                for row in range(self.height):
                    for col in range(self.width):
                        if grid[col][row] == 2:  # Belt path marked with '2'
                            belt = Belt(id=item+"_()"+str(col)+", "+str(row)+")_"+"output" , type='output',item=item,int_x=col,int_y=row)
                            # define the position of the belt
                            self.solver.add(And(belt.x == col ,belt.y == row ))
                            self.global_output_belts.append(belt)
        self.solve()             
                            
    def add_constraint(self,constraint):
        self.additional_constraints.append(constraint)    
        
        self.solver.add(constraint)
        
    def create_global_belts(self):
        # Create input belts
        for item_id, item_info in self.production_data.items():
            if len( item_info) == 1:  # This identifies input items
                #self.global_input_belts.append({'id': item_id, 'amount': math.ceil(item_info['amount_per_minute']/(ITEMS_PER_SECOND*60))})
                for i in range( math.ceil(item_info['amount_per_minute']/(ITEMS_PER_SECOND*60))):
                    self.global_input_belts.append(Belt(id=item_id+"_"+str(i)+"_"+"start" , type='start'))
                
            # Create output belt for the final product
        for item_id, item_info in self.production_data.items():
            if 'assemblers' in item_info and item_info['assemblers'] > 0:
                self.global_output_belt = Belt(id=item_id+"_"+str(i)+"_"+"end" , type='end')
                break
            
        # Print out the created belts for verification
        print("Global Input Belts:")
        for belt in self.global_input_belts:
            print(belt)

        print("Global Output Belt:")
        if self.global_output_belt:
            print(self.global_output_belt)

    def add_global_belt_constraints(self):
        # Ensure global input belts are placed on the edges and within bounds
        for belt in self.global_input_belts:
            self.solver.add(And(
                Or(
                    belt.x == 0,  # Left edge
                    belt.x == self.width - 1  # Right edge
                ),
                Or(
                    belt.y == 0,  # Top edge
                    belt.y == self.height - 1   # Bottom edge
                ),
                belt.x >= 0, belt.y >= 0  # Non-negative constraints
            ))

        # Ensure global output belt is placed on the edge and within bounds
        if self.global_output_belt:
            self.solver.add(And(
                Or(
                    self.global_output_belt.x == 0,  # Left edge
                    self.global_output_belt.x == self.width - 1  # Right edge
                ),
                Or(
                    self.global_output_belt.y == 0,  # Top edge
                    self.global_output_belt.y == self.height - 1  # Bottom edge
                ),
                self.global_output_belt.x >= 0, self.global_output_belt.y >= 0  # Non-negative constraints
            ))

        # Add non-overlapping constraints for all belts
        all_belts = self.global_input_belts + [self.global_output_belt] if self.global_output_belt else self.global_input_belts
        for i, belt1 in enumerate(all_belts):
            for belt2 in all_belts[i+1:]:
                # Ensure that no two belts occupy the same position
                self.solver.add(Or(
                    belt1.x != belt2.x,
                    belt1.y != belt2.y
                ))
        print("Global belt constraints added.")

    
    def create_assemblers(self):
        # Loop through the production data to create assemblers and their inserters and belts
        ac = 0 # assembler counter
        for item_id, item_info in self.production_data.items():
            if 'assemblers' in item_info and item_info['assemblers'] > 0:
                for i in range(item_info['assemblers']): # i = intern counter 
                    # Create inserters for the assembler
                    input_inserters = [
                    Inserter(
                        id=f"{input_info['id']}_in_{ac}_{i}",
                        type='input',
                        item=input_info['id'],
                        belt=Belt(
                            id=f"{input_info['id']}_end_{ac}_{i}",
                            type="end",
                            item=input_info['id']
                        )
                    ) for input_info in item_info['input_inserters']
                    ]
                    output_inserters = [
                    Inserter(
                        id=f"{item_id}_out_{ac}_{i}",
                        type='output',
                        item=item_id,
                        belt=Belt(
                            id=f"{item_id}_start_{ac}_{i}",
                            type="start",
                            item=item_id
                        )
                    ) for _ in range(item_info.get('output_inserters', 1))
                    ]
                    # Create the assembler with the inserters and belts
                    assembler = Assembler(
                    id=f"{item_id}_{ac}_{i}",
                    inserters=input_inserters + output_inserters,
                    item = item_id
                    )
            
                    
                    self.assemblers.append(assembler)
                ac += 1

        print(f"created { len(self.assemblers) } assemblers")
        
    # defines 3x3 space for assemblers and ensures that no other assembler or inserter or belt is at that space
    
    def add_single_assembler_constraint(self,assembler):
    
        if assembler in self.assemblers:
            # bound constraints for assembler
            self.solver.add(And(assembler.x >= 0, assembler.x <= self.width - 3))
            self.solver.add(And(assembler.y >= 0, assembler.y <= self.height - 3))
            
            # make sure the assembler did not overlap with existing global belts
            belts = self.global_input_belts + self.global_output_belts
            for belt in belts:
                    
                self.solver.add(Or(
                                    Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                                    Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                                ))
                
            # assembler does not overlap with other assemblers or inserters
            for other in self.placed_assembler:
                if assembler.id != other.id:
                    # Non-overlapping assembler regions
                    self.solver.add(Or(assembler.x + 2 < other.x, assembler.x > other.x + 2,
                                    assembler.y + 2 < other.y, assembler.y > other.y + 2))
                    
                    for inserter in other.inserters:
                        # Ensure the inserter is not inside the others assembler's 3x3 space
                        self.solver.add(Or(
                            Or(inserter.x < assembler.x, inserter.x > assembler.x + 2),
                            Or(inserter.y < assembler.y, inserter.y > assembler.y + 2)
                        ))
                        
                        # they can overlap, if they have the same item -> see merging of inserters and direct interaction
                        belt = inserter.belt
                        if belt.item != assembler.item:
                            self.solver.add(Or(
                                Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                                Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                            ))

            self.placed_assembler.append(assembler)
        else:
            raise Exception ("assembler not created")
            
        
    # inserter needs to be adjacent to assembler
    def add_single_inserter_constraint(self,assembler):
        input_positions = [
                (assembler.x, assembler.y - 1), (assembler.x + 1, assembler.y - 1), (assembler.x + 2, assembler.y - 1),  # Top row
                (assembler.x, assembler.y + 3), (assembler.x + 1, assembler.y + 3), (assembler.x + 2, assembler.y + 3),  # Bottom row
                (assembler.x - 1, assembler.y), (assembler.x - 1, assembler.y + 1), (assembler.x - 1, assembler.y + 2),  # Left column
                (assembler.x + 3, assembler.y), (assembler.x + 3, assembler.y + 1), (assembler.x + 3, assembler.y + 2)   # Right column
            ]
        
        for inserter in assembler.inserters:
                # Ensure each inserter is adjacent to its assembler
                self.solver.add(Or([And(inserter.x == pos[0], inserter.y == pos[1]) for pos in input_positions]))
                
                belt = inserter.belt
                # Ensure that the belt corresponding to the inserter is at the opposite side of the assembler
                self.solver.add(Or(
                # Option 1: Direct interaction (no belt)
                inserter.direct_interaction,
                # Option 2: Standard case with a belt
                Or(
                    # Inserter to the left of the assembler, belt is to the left of the inserter
                    And(inserter.x == assembler.x - 1, belt.x == inserter.x - 1, belt.y == inserter.y),
                    # Inserter to the right of the assembler, belt is to the right of the inserter
                    And(inserter.x == assembler.x + 3, belt.x == inserter.x + 1, belt.y == inserter.y),
                    # Inserter above the assembler, belt is above the inserter
                    And(inserter.y == assembler.y - 1, belt.x == inserter.x, belt.y == inserter.y - 1),
                    # Inserter below the assembler, belt is below the inserter
                    And(inserter.y == assembler.y + 3, belt.x == inserter.x, belt.y == inserter.y + 1)
                )
                ))
        
    # @Preprocessing
    # Returns a dictionary with each belt as the key and the following structure as the value:
    # {
    #     belt: {
    #         'left': {
    #             'inserter_pos': (x, y),  # The position of the inserter next to the belt
    #             'assembler_pos': [(x1, y1), (x2, y2), ...]  # List of valid positions for the upper-left corner of 3x3 structures
    #         },
    #         'right': {
    #             'inserter_pos': (x, y),
    #             'assembler_pos': [(x1, y1), ...]
    #         },
    #         'up': {
    #             'inserter_pos': (x, y),
    #             'assembler_pos': [(x1, y1), ...]
    #         },
    #         'down': {
    #             'inserter_pos': (x, y),
    #             'assembler_pos': [(x1, y1), ...]
    #         }
    #     }
    # }
    # This method helps determine where an assembler can be placed next to a global input or output belt
    # based on the availability of adjacent free spaces.
    def check_space_around_path(self, belt_list, item_type):
        obstacle_map = self.obstacle_maps[-1]
        
        
        height = len(obstacle_map)  # Number of rows
        width = len(obstacle_map[0]) if height > 0 else 0  # Number of columns

        available_positions = {}  # To store results with belt as key

        for belt in belt_list:
            if belt.item == item_type:
                belt_x = belt.int_x
                belt_y = belt.int_y

                # Store possible positions
                directions = {}

                # Helper function to check if a 3x3 area is free
                def is_3x3_space_clear(start_x, start_y):
                    if not (0 <= start_x < width and 0 <= start_y < height):
                        return False  # Out of bounds
                    for dx in range(3):
                        for dy in range(3):
                            x, y = start_x + dx, start_y + dy
                            if x >= width or y >= height or obstacle_map[y][x] != 0:
                                return False
                    return True

                # Check left - three possible 3x3 positions aligned vertically
                if belt_x > 0 and obstacle_map[belt_y][belt_x - 1] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        left_x = belt_x - 4  # Place structure left of the belt
                        left_y = belt_y + offset
                        if is_3x3_space_clear(left_x, left_y):
                            inserter_pos = (belt_x - 1, belt_y)
                            directions['left'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((left_x, left_y))

                # Check right - three possible 3x3 positions aligned vertically
                if belt_x < width - 1 and obstacle_map[belt_y][belt_x + 1] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        right_x = belt_x + 2
                        right_y = belt_y + offset
                        if is_3x3_space_clear(right_x, right_y):
                            inserter_pos = (belt_x + 1, belt_y)
                            directions['right'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((right_x, right_y))

                # Check up - three possible 3x3 positions aligned horizontally
                if belt_y > 0 and obstacle_map[belt_y - 1][belt_x] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        up_x = belt_x + offset
                        up_y = belt_y - 4
                        if is_3x3_space_clear(up_x, up_y):
                            inserter_pos = (belt_x, belt_y - 1)
                            directions['up'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((up_x, up_y))

                # Check down - three possible 3x3 positions aligned horizontally
                if belt_y < height - 1 and obstacle_map[belt_y + 1][belt_x] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        down_x = belt_x + offset
                        down_y = belt_y + 2
                        if is_3x3_space_clear(down_x, down_y):
                            inserter_pos = (belt_x, belt_y + 1)
                            directions['down'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((down_x, down_y))

                # Store the directions in the available positions dictionary
                if directions:
                    available_positions[belt] = directions

        return available_positions
                    
                    
    
    # check if we can place an assembler with inserter next to the input belt. if this is possible, 
    def add_input_belt_merging(self,assembler):
        for inserter in assembler.inserters:

            if inserter.type == "input":
                
                
                # these positions are all avaible for the belt
                available_positions  = self.check_space_around_path(self.global_input_belts,inserter.item)
                #print(f"Available positions for {inserter.item}:", available_positions)
                
                if len(available_positions) == 0:
                    print('no space available for input belt merging')
                    return False
                
                inserter_belt = inserter.belt
                
                position_constraints = []
                
                for belt, directions in available_positions.items():
                    
                    position_constraints_per_belt = []
                    
                    for direction, details in directions.items():
                        inserter_pos = details['inserter_pos']
                        assembler_pos_list = details['assembler_pos']
                        
                        belt_constraint= And(inserter_belt.x == belt.int_x ,inserter_belt.y == belt.int_y)
                        inserter_constraint = And(inserter.x == inserter_pos[0] ,inserter.y == inserter_pos[1])
                        
                        assembler_constraints = []
                        
                        for assembler_pos in assembler_pos_list:
                            
                            assembler_constraint = And(assembler.x == assembler_pos[0] ,assembler.y == assembler_pos[1])
                            assembler_constraints.append(assembler_constraint)
                        
                        position_constraints_per_belt.append(belt_constraint)
                        position_constraints_per_belt.append(inserter_constraint)
                        position_constraints_per_belt.append(Or(assembler_constraints))
                    
                    position_constraints.append(And(*position_constraints_per_belt))

            self.solver.add(Or(position_constraints))

    # give the position of an assembler and its inserter, determine the direction the inserter is relative to the assembler
    def determine_orientation(self,assembler_x, assembler_y, inserter_x, inserter_y):
      

        assembler_boundary = {
        "left": [(assembler_x - 1, assembler_y + i) for i in range(3)],
        "right": [(assembler_x + 3, assembler_y + i) for i in range(3)],
        "up": [(assembler_x + i, assembler_y - 1) for i in range(3)],
        "down": [(assembler_x + i, assembler_y + 3) for i in range(3)]
        }

        # Check which side the inserter is adjacent to
        for direction, positions in assembler_boundary.items():
            if (inserter_x, inserter_y) in positions:
                return direction
        return None  # Inserter is not adjacent
        
        
    def check_space_around_assembler(self,assembler,inserter):
        
        obstacle_map = self.obstacle_maps[-1]
        
        
        height = len(obstacle_map)  # Number of rows
        width = len(obstacle_map[0]) if height > 0 else 0  # Number of columns
        
        # Helper function to check if a 3x3 area is free
        def is_3x3_space_clear(start_x, start_y):
                    if not (0 <= start_x < width and 0 <= start_y < height):
                        return False  # Out of bounds
                    for dx in range(3):
                        for dy in range(3):
                            x, y = start_x + dx, start_y + dy
                            if x >= width or y >= height or (obstacle_map[y][x] != 0 and obstacle_map[y][x] != 88): # 88 can be ignored , only markings for belt
                                return False
                    return True

        assembler_x = self.model.evaluate(assembler.x).as_long()
        assembler_y = self.model.evaluate(assembler.y).as_long()
        
        inserter_x = self.model.evaluate(inserter.x).as_long()
        inserter_y = self.model.evaluate(inserter.y).as_long()
        
        
        
        orientation = self.determine_orientation(assembler_x, assembler_y, inserter_x, inserter_y)
        
        print(f"Orientation: {orientation}")
        print(f"Assembler position: ({assembler_x}, {assembler_y}), Inserter position: ({inserter_x}, {inserter_y})")

        
        # returns all possible positions for the assembler
        assembler_positions = []
        # Check left - three possible 3x3 positions aligned vertically
        if orientation == "left":
            for offset in range(-2, 1):
                left_x = inserter_x - 3
                left_y = inserter_y + offset
                if is_3x3_space_clear(left_x, left_y):
                    print(f"Available position found on left at ({left_x}, {left_y})")
                    assembler_positions.append((left_x, left_y))
        
        elif orientation == "right":
            for offset in range(-2, 1):
                right_x = inserter_x + 1
                right_y = inserter_y + offset
                if is_3x3_space_clear(right_x, right_y):
                    print(f"Available position found on right at ({right_x}, {right_y})")
                    assembler_positions.append((right_x, right_y))
        
        elif orientation == "up":
            for offset in range(-2, 1):
                up_x = inserter_x + offset
                up_y = inserter_y - 3
                if is_3x3_space_clear(up_x, up_y):
                    print(f"Available position found above at ({up_x}, {up_y})")
                    assembler_positions.append((up_x, up_y))
        
        elif orientation == "down":
            for offset in range(-2, 1):
                down_x = inserter_x + offset
                down_y = inserter_y + 1
                if is_3x3_space_clear(down_x, down_y):
                    print(f"Available position found below at ({down_x}, {down_y})")
                    assembler_positions.append((down_x, down_y))

        # Final debugging print for found positions
        if not assembler_positions:
            print("No available positions found for assembler.")
        else:
            print(f"Possible assembler positions: {assembler_positions}")
            
        return assembler_positions
    
    # @Helper 
    # define possible positions for inserter merging
    def add_inserter_merging(self,assembler):
        
        end_constraints = []
        
        for other_assembler in self.placed_assembler:
            # should always be True 
            if other_assembler.id != assembler.id:
                
                
                for inserter in assembler.inserters:
                    
                    # get own input inserter
                    if inserter.type == "input":
                        
                        for other_inserter in other_assembler.inserters:
                            
                            # search for all other insert that output the item we need 
                            if other_inserter.type == "output" and inserter.item == other_inserter.item:

                                possible_positions = self.check_space_around_assembler(other_assembler,other_inserter)
                                
                                # if space is avaible, set the position of our inserter to the output insert of the placed assembler
                                if not possible_positions:
                                    inserter.direct_interaction = False
                                    continue  # Skip to the next inserter

                                # If space is available, enable direct interaction
                                print(f'set direct interaction for {assembler.id}')
                                #inserter.direct_interaction = True
                                
                                # merge the 2 inserter
                                inserter_pos_constraint = And(
                                inserter.x == self.model.evaluate(other_inserter.x).as_long(),
                                inserter.y == self.model.evaluate(other_inserter.y).as_long()
                                )
                                
                                # destroy belt of input and output inserter
                                inserter.belt = None
                                other_inserter.belt = None
                                
                                #assembler_constraint = And(assembler.x == possible_positions[0][0], assembler.y == possible_positions[0][1])
                                
                                #self.solver.add(inserter_pos_constraint)
                                #self.solver.add(assembler_constraint)
                                
                                position_constraints = [inserter_pos_constraint]
                                for pos in possible_positions:
                                    assembler_constraint = And(assembler.x == pos[0], assembler.y == pos[1])
                                    position_constraints.append(assembler_constraint)
                                    
                                end_constraints.append(Or(*position_constraints))
        if len(end_constraints)>0:
            self.solver.add(Or(*end_constraints))
        
        
                                
    
    def add_single_belt_penalty(self,assembler):
        belt_penalty = Bool(f"{assembler.id}_belt_penalty")  # Penalty for belt usage
        self.solver.add(belt_penalty >= 0)  # Penalty is non-negative
        
        total_belt_penalty = 0  # Accumulate belt penalties
        
        for inserter in assembler.inserters:
            direct_interaction = Bool(f"{inserter.id}_direct")
            belt_used = Not(direct_interaction)  # Belt is used when direct interaction is false
            total_belt_penalty += If(belt_used, 1, 0)  # Add 1 to penalty if belt is used
            
            
            merged = Bool(f"{inserter.belt.id}_merged")
            
        self.solver.minimize(total_belt_penalty)    
    
    def add_assembler_constraints(self):
        # assembler has pos x,y and 
        for assembler in self.assemblers:
            # Ensure assembler fits within the grid boundaries (W-3 x H-3 grid)
            self.solver.add(And(assembler.x >= 0, assembler.x <= self.width - 3))
            self.solver.add(And(assembler.y >= 0, assembler.y <= self.height - 3))
            
            
            #belts = self.global_input_belts
            #belts.extend(self.global_output_belts)
            
            belts = self.global_input_belts + self.global_output_belts
            # Allow for some flexibility in overlap
            for belt in belts:
                
                self.solver.add(Or(
                                    Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                                    Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                                ))
                                
            
            #Ensure no other assembler the assembler's 3x3 space
            for other in self.assemblers:
                if assembler.id != other.id:
                    # Non-overlapping assembler regions
                    self.solver.add(Or(assembler.x + 2 < other.x, assembler.x > other.x + 2,
                                    assembler.y + 2 < other.y, assembler.y > other.y + 2))
                    
                    for inserter in other.inserters:
                        
                        # Ensure the inserter is not inside the current assembler's 3x3 space
                        self.solver.add(Or(
                            Or(inserter.x < assembler.x, inserter.x > assembler.x + 2),
                            Or(inserter.y < assembler.y, inserter.y > assembler.y + 2)
                        ))

                        # Ensure the belt is not inside the current assembler's 3x3 space
                        #belt = inserter.belt
                        #if belt is not None:
                        #    self.solver.add(Or(
                        #        Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                        #        Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                        #    ))
                                
        print('Assembler constraints added')
    
    # adds constraint that inserts need to be around its assembler as well as its belt

    def add_inserter_constraints(self):
        for assembler in self.assemblers:
            # Define the possible positions of inserters around an assembler (adjacent cells)
            input_positions = [
                (assembler.x, assembler.y - 1), (assembler.x + 1, assembler.y - 1), (assembler.x + 2, assembler.y - 1),  # Top row
                (assembler.x, assembler.y + 3), (assembler.x + 1, assembler.y + 3), (assembler.x + 2, assembler.y + 3),  # Bottom row
                (assembler.x - 1, assembler.y), (assembler.x - 1, assembler.y + 1), (assembler.x - 1, assembler.y + 2),  # Left column
                (assembler.x + 3, assembler.y), (assembler.x + 3, assembler.y + 1), (assembler.x + 3, assembler.y + 2)  # Right column
            ]

            for inserter in assembler.inserters:
                # Ensure each inserter is adjacent to its assembler
                self.solver.add(Or([And(inserter.x == pos[0], inserter.y == pos[1]) for pos in input_positions]))
                
                
                direct_interaction = Bool(f"{inserter.id}_direct")
                direct_preferred = Bool(f"{inserter.id}_direct_preferred")  # flag preference direct insertion over belt usage
                
                belt = inserter.belt
                # Ensure that the belt corresponding to the inserter is at the opposite side of the assembler
                self.solver.add(Or(
                # Option 1: Direct interaction (no belt)
                direct_interaction,
                # Option 2: Standard case with a belt
                Or(
                    # Inserter to the left of the assembler, belt is to the left of the inserter
                    And(inserter.x == assembler.x - 1, belt.x == inserter.x - 1, belt.y == inserter.y),
                    # Inserter to the right of the assembler, belt is to the right of the inserter
                    And(inserter.x == assembler.x + 3, belt.x == inserter.x + 1, belt.y == inserter.y),
                    # Inserter above the assembler, belt is above the inserter
                    And(inserter.y == assembler.y - 1, belt.x == inserter.x, belt.y == inserter.y - 1),
                    # Inserter below the assembler, belt is below the inserter
                    And(inserter.y == assembler.y + 3, belt.x == inserter.x, belt.y == inserter.y + 1)
                )
                ))

                # If direct interaction is chosen, ensure no belt is involved (set belt positions to -1, -1)
                self.solver.add(Implies(direct_interaction, And(belt.x == -1, belt.y == -1)))

                # Enforce direct interaction if direct_preferred is True
                self.solver.add(Implies(direct_preferred, direct_interaction))

                # Ensure correct item flow for direct interactions
                if inserter.type == 'output':
                    #self.solver.add(inserter.item == assembler.item)

                    # Check for merging with input inserter of adjacent assembler
                    for other_assembler in self.assemblers:
                        if assembler.id != other_assembler.id:
                            other_input_positions = [
                                (other_assembler.x, other_assembler.y - 1), (other_assembler.x + 1, other_assembler.y - 1), (other_assembler.x + 2, other_assembler.y - 1),
                                (other_assembler.x, other_assembler.y + 3), (other_assembler.x + 1, other_assembler.y + 3), (other_assembler.x + 2, other_assembler.y + 3),
                                (other_assembler.x - 1, other_assembler.y), (other_assembler.x - 1, other_assembler.y + 1), (other_assembler.x - 1, other_assembler.y + 2),
                                (other_assembler.x + 3, other_assembler.y), (other_assembler.x + 3, other_assembler.y + 1), (other_assembler.x + 3, other_assembler.y + 2)
                            ]

                            for other_inserter in other_assembler.inserters:
                                if other_inserter.type == 'input':
                                    # Check if inserters are adjacent and handle the same item
                                    self.solver.add(Implies(And(direct_interaction,
                                                                inserter.item == other_inserter.item,
                                                                Or([And(inserter.x == other_inserter.x, inserter.y == other_inserter.y)])),
                                                            True))

                                    # Merge input and output inserters
                                    self.solver.add(Implies(And(direct_interaction,
                                                                inserter.item == other_inserter.item),
                                                            And(inserter.x == other_inserter.x, inserter.y == other_inserter.y)))                    
    def add_belt_penalty(self):
        belt_penalty = Int('belt_penalty')  # Penalty for belt usage
        self.solver.add(belt_penalty >= 0)  # Penalty is non-negative
        
        total_belt_penalty = 0  # Accumulate belt penalties
        
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                # Add penalty if belt is used instead of direct interaction
                direct_interaction = Bool(f"{inserter.id}_direct")
                belt_used = Not(direct_interaction)  # Belt is used when direct interaction is false
                total_belt_penalty += If(belt_used, 1, 0)  # Add 1 to penalty if belt is used
        
        # Minimize belt usage by minimizing the belt penalty
        self.solver.minimize(total_belt_penalty)                         
                
    def add_bound_constraints(self):

        # Ensure all inserters stay within the grid boundaries
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                
                #direct_interaction = Bool(f"{inserter.id}_direct")
                belt = inserter.belt
                
                self.solver.add(And(inserter.x >= 0, inserter.x < self.width))
                self.solver.add(And(inserter.y >= 0, inserter.y < self.height))
                
                if belt is not None:
                # If direct interaction is not used, belt must stay within the grid boundaries
                    self.solver.add(And(belt.x >= 0, belt.x < self.width, belt.y >= 0, belt.y < self.height))

                    # If direct interaction is used, the belt's position must be invalid (-1, -1)
                    #self.solver.add(Implies(direct_interaction, And(belt.x == -1, belt.y == -1)))

                
    # what is done: assembler can't overlap other assembler, belt or inserter
    def add_single_overlapping_constraints(self,assembler):
        # belt and inserter of assembler cannot overlap with existing belts
        
        non_overlapping_constraints = []
        
        # Loop through each inserter and belt of the current assembler
        for inserter in assembler.inserters:
            belt = inserter.belt
            
            
            # Check against global belts to prevent overlap
            for global_belt in self.global_input_belts + self.global_output_belts:
                if belt is not None and belt.item != global_belt.item:
                    belt_no_overlap_with_global = Or(
                            belt.x != global_belt.x,
                            belt.y != global_belt.y
                        )
                    non_overlapping_constraints.append(belt_no_overlap_with_global)
                
                # Prevent inserter from overlapping with global belts
                inserter_no_overlap_with_global = Or(
                    inserter.x != global_belt.x,
                    inserter.y != global_belt.y
                )
                
                non_overlapping_constraints.append(inserter_no_overlap_with_global)
                
            
            for other_assembler in self.placed_assembler:
                for other_inserter in other_assembler.inserters:
                    other_belt = other_inserter.belt
                    
                    
                    if inserter.item != other_inserter.item:
                        inserter_no_overlap = Or(
                            inserter.x != other_inserter.x,
                            inserter.y != other_inserter.y
                        )
                        non_overlapping_constraints.append(inserter_no_overlap)
                        
                    # Prevent belt overlap with other belts (unless IDs match)
                    if belt is not None and other_belt is not None and belt.item != other_belt.item:
                        belt_no_overlap = Or(
                            belt.x != other_belt.x,
                            belt.y != other_belt.y
                        )
                        non_overlapping_constraints.append(belt_no_overlap)    
        # Add all non-overlapping constraints to the solver
        if non_overlapping_constraints:
            self.solver.add(*non_overlapping_constraints)            
            
                
    def add_overlapping_constraints(self):
        
        # we made sure that assemblers dont overlap with anything so we only need to look at the inserters and belts
        
        inserters = [inserter for assembler in self.assemblers for inserter in assembler.inserters]
        belts = [inserter.belt for assembler in self.assemblers for inserter in assembler.inserters] + self.global_input_belts +self.global_output_belts
        
        
        #belts += self.global_input_belts
        #belts.append(self.global_output_belt)
        
        #Ensure no inserters overlap with each other except they transport the same item
        for i, inserter1 in enumerate(inserters):
            for inserter2 in inserters[i+1:]:
                if(inserter1.id != inserter2.id) or (inserter1.item != inserter2.item):
                    self.solver.add(Or(
                        inserter1.x != inserter2.x,
                        inserter1.y != inserter2.y
                    ))

        #Ensure no belts overlap with each other, except they have the same item 
        for i, belt1 in enumerate(belts):
            for belt2 in belts[i+1:]:
                if(belt1.id != belt2.id) or  (belt1.item != belt2.item):
                    self.solver.add(Or(
                        belt1.x != belt2.x,
                        belt1.y != belt2.y
                    ))

        #Ensure no inserter overlaps with any belt
        for inserter in inserters:
            for belt in belts:
                self.solver.add(Or(
                    inserter.x != belt.x,
                    inserter.y != belt.y
                ))

    def solve(self):
            result = self.solver.check()
            print("model is:",result)
            
           
            self.model = self.solver.model() 

            return 
            # Print assembler positions
            for assembler in self.assemblers:
                x_pos = self.model.evaluate(assembler.x)  # Get evaluated assembler x
                y_pos = self.model.evaluate(assembler.y)  # Get evaluated assembler y
                print(f"Assembler {assembler.id} -> Position: ({x_pos}, {y_pos})")

                # Print inserter positions for this assembler
                for inserter in assembler.inserters:
                    x_inserter = self.model.evaluate(inserter.x)
                    y_inserter = self.model.evaluate(inserter.y)
                    print(f"  Inserter {inserter.id} -> Position: ({x_inserter}, {y_inserter})")

                    # Print belt position corresponding to the inserter
                    belt = inserter.belt
                    x_belt = self.model.evaluate(belt.x)
                    y_belt = self.model.evaluate(belt.y)
                    print(f"    Belt {belt.id} -> Position: ({x_belt}, {y_belt})")
        
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
            for assembler in self.placed_assembler:
                x = self.model.evaluate(assembler.x).as_long()
                y = self.model.evaluate(assembler.y).as_long()
                
                print(str(assembler.id) + " x: "+str(x)+ "y: "+str(y))
                
                # Mark 3x3 area around the assembler as occupied
                for dx in range(3):
                    for dy in range(3):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            obstacle_map[y + dy][x + dx] = 33

                #Mark inserters in the obstacle map
                for inserter in assembler.inserters:
                    ix = self.model.evaluate(inserter.x).as_long()
                    iy = self.model.evaluate(inserter.y).as_long()
                    # Mark inserter position as occupied
                    if 0 <= ix < self.width and 0 <= iy < self.height:
                        obstacle_map[iy][ix] = 44

                    if belt is not None:
                        
                        try:
                            # Mark the corresponding belt if not directly feeding
                            belt = inserter.belt
                            bx = self.model.evaluate(belt.x).as_long()
                            by = self.model.evaluate(belt.y).as_long()

                            if 0 <= bx < self.width and 0 <= by < self.height:
                                obstacle_map[by][bx] = 88
                            
                            if bx > -1 and by > -1:
                                start_id = belt.id.split('_')[0]
                                belt_point_information.append([start_id, bx, by, belt.type])
                        except:
                            print("der scheißt einfach aufs if")
        else:
            print('not sat')
            
            
        return obstacle_map,belt_point_information
    
        
    def visualize_factory(self):
        
        
        
        # Initialize Pygame
        pygame.init()
        
        # Set up the window size (based on grid dimensions)
        window_width = self.width * CELL_SIZE 
        window_height = self.height * CELL_SIZE
        window = pygame.display.set_mode((window_width, window_height))
        
        pygame.display.set_caption('Factory Layout Visualization')
        
        # Set up the clock for controlling the frame rate
        clock = pygame.time.Clock()
        
        # Run the game loop
        running = True
        while running:
            # Handle events (like quitting the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Fill the background with white
            window.fill(WHITE)
            
            # Draw the grid
            for x in range(0, window_width, CELL_SIZE):
                for y in range(0, window_height, CELL_SIZE):
                    pygame.draw.rect(window, BLACK, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), 1)
            
            for belt in self.global_input_belts:
                belt_x = self.model.evaluate(belt.x).as_long()
                belt_y = self.model.evaluate(belt.y).as_long()
                belt_rect = pygame.Rect(belt_x * CELL_SIZE, belt_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                belt_color = BELT_COLOR_MAP.get(belt.type, GREEN)
                
                pygame.draw.rect(window, belt_color, belt_rect)
                
                font = pygame.font.Font(None, 20)
                text_surface = font.render(belt.id, True, BLACK)
                window.blit(text_surface, (belt_rect.x+5 , belt_rect.y+5 ))
                
            
            
            out_belt_x = self.model.evaluate(self.global_output_belt.x).as_long()
            out_belt_y = self.model.evaluate(self.global_output_belt.y).as_long()
            belt_rect = pygame.Rect(out_belt_x * CELL_SIZE, out_belt_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            belt_color = BELT_COLOR_MAP.get("end", GREEN)
                
            pygame.draw.rect(window, belt_color, belt_rect)
                
            font = pygame.font.Font(None, 20)
            text_surface = font.render(self.global_output_belt.id, True, BLACK)
            window.blit(text_surface, (belt_rect.x+5 , belt_rect.y+5 ))
                
            # Draw the assemblers, inserters, and belts
            for assembler in self.assemblers:
                
                # Draw assembler (blue 3x3 grid)
                assembler_x = self.model.evaluate(assembler.x).as_long()
                assembler_y = self.model.evaluate(assembler.y).as_long()
                assembler_rect = pygame.Rect(assembler_x * CELL_SIZE, assembler_y * CELL_SIZE, 3 * CELL_SIZE, 3 * CELL_SIZE)
                pygame.draw.rect(window, BLUE, assembler_rect)
                
                # Draw assembler name
                font = pygame.font.Font(None, 24)
                text_surface = font.render(assembler.id, True, BLACK)
                window.blit(text_surface, (assembler_rect.x + 5, assembler_rect.y + 5))
                
                # Draw inserters (red 1x1 grid)
                for inserter in assembler.inserters:
                    inserter_x = self.model.evaluate(inserter.x).as_long()
                    inserter_y = self.model.evaluate(inserter.y).as_long()
                    inserter_rect = pygame.Rect(inserter_x * CELL_SIZE, inserter_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    # Change inserter color based on type
                    inserter_color = INSERTER_COLOR_MAP.get(inserter.type, RED)
                    pygame.draw.rect(window, inserter_color, inserter_rect)
                    
                    # Draw the associated belt (green 1x1 grid)
                    belt_x = self.model.evaluate(inserter.belt.x).as_long()
                    belt_y = self.model.evaluate(inserter.belt.y).as_long()
                    belt_rect = pygame.Rect(belt_x * CELL_SIZE, belt_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    belt_color = BELT_COLOR_MAP.get(inserter.belt.type, GREEN)
                    pygame.draw.rect(window, belt_color, belt_rect)
            
            # Update the display
            pygame.display.flip()
            
            # Limit the frame rate to 30 FPS
            clock.tick(30)
        
        # Quit Pygame properly when the loop ends
        pygame.quit()
        
        
        
# Example usage
def main():
    # Define grid dimensions
    grid_width = 20
    grid_height = 20
    
    production_data= {'electronic-circuit': {'output_per_minute': 10, 'assemblers': 1, 'output_inserters': 1, 'input_inserters': [{'id': 'copper-cable', 'amount': 3}, {'id': 'iron-plate', 'amount': 1}], 'belts': 1}, 'copper-cable': {'output_per_minute': 30.0, 'assemblers': 1, 'output_inserters': 1, 'input_inserters': [{'id': 'copper-plate', 'amount': 1}], 'belts': 1}, 'copper-plate': {'amount_per_minute': 15.0}, 'iron-plate': {'amount_per_minute': 10.0}}

    z3_solver = Z3Solver(grid_width, grid_height, production_data)
    z3_solver.solve()
    z3_solver.visualize_factory()
    
    obstacle_map,pairs = z3_solver.build_map()
    print(obstacle_map)
    print(pairs)

if __name__ == "__main__":
    main()
