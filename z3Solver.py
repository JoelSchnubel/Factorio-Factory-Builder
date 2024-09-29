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
    def __init__(self,id,inserters):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        
        self.inserters = inserters
        
        
    def __str__(self) -> str:
            inserters_str = ', '.join([str(inserter) for inserter in self.inserters])

            return (f"Assembler(id={self.id}, position=({self.x}, {self.y}),"
                    f"inserters=[{inserters_str}]")
        
class Inserter:
    def __init__(self,id,type,belt):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        self.type = type # 'input' or 'output'
        
        self.belt = belt
        
    def __str__(self) -> str:
        
            return (f"Inserter(id={self.id}, position=({self.x}, {self.y}), type={self.type})"
                    f",Belt={str(self.belt)}")
class Belt:
    def __init__(self,id,type):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        self.type = type # 'start' or 'end'
        
    def __str__(self) -> str:
        return f"Belt(id={self.id}, position=({self.x}, {self.y}), type={self.type})"


class Z3Solver:
    def __init__(self, width, height, production_data):
        self.width = width
        self.height = height
        self.production_data = production_data
        self.solver = Solver()

        # Create variables for assemblers, input inserters, output inserters, and belts
        self.assemblers = []
        
        self.global_input_belts = []
        self.global_output_belt = None

        self.additional_constraints = []
        
        self.create_global_belts()
        self.create_assemblers()
        
        self.add_global_belt_constraints()
        self.add_assembler_constraints()
        self.add_bound_constraints()
        self.add_overlapping_constraints()
        self.add_inserter_constraints()
        
        
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

    # make sure they are only at the edges
    

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
                    input_inserters = [Inserter(f"{input_info['id']}_in_{ac}_{i}", 'input',belt = Belt(f"{input_info['id']}_end_{ac}_{i}","end")) for input_info in item_info['input_inserters']]
                    output_inserters = [Inserter(f"{item_id}_out_{ac}_{i}", 'output',belt = Belt(f"{item_id}_start_{ac}_{i}","start")) for i in range(item_info.get('output_inserters', 1))]
                    
                    # Create the assembler with the inserters and belts
                    assembler = Assembler(f"{item_id}_{ac}_{i}", input_inserters + output_inserters)
            
                    
                    self.assemblers.append(assembler)
                ac += 1
            
    # defines 3x3 space for assemblers and ensures that no other assembler or inserter or belt is at that space
    def add_assembler_constraints(self):
        # assembler has pos x,y and 
        for assembler in self.assemblers:
            # Ensure assembler fits within the grid boundaries (W-3 x H-3 grid)
            self.solver.add(And(assembler.x >= 0, assembler.x <= self.width - 3))
            self.solver.add(And(assembler.y >= 0, assembler.y <= self.height - 3))
            
            # make sure assembler does not 
            
            belts = self.global_input_belts
            belts.append(self.global_output_belt)
            for belt in belts:
                self.solver.add(And(
                            Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                            Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                        ))
            
            # Ensure no other assembler the assembler's 3x3 space
            for other in self.assemblers:
                if assembler.id != other.id:
                    # Non-overlapping assembler regions
                    self.solver.add(Or(assembler.x + 2 < other.x, assembler.x > other.x + 2,
                                       assembler.y + 2 < other.y, assembler.y > other.y + 2))
                    

                    for inserter in other.inserters:
                        
                        # Ensure the inserter is not inside the current assembler's 3x3 space
                        self.solver.add(And(
                            Or(inserter.x < assembler.x, inserter.x > assembler.x + 2),
                            Or(inserter.y < assembler.y, inserter.y > assembler.y + 2)
                        ))

                        # Ensure the belt is not inside the current assembler's 3x3 space
                        belt = inserter.belt
                        self.solver.add(And(
                            Or(belt.x < assembler.x, belt.x > assembler.x + 2),
                            Or(belt.y < assembler.y, belt.y > assembler.y + 2)
                        ))
                                
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
                
                belt = inserter.belt
                # Ensure that the belt corresponding to the inserter is at the opposite side of the assembler
                self.solver.add(Or(
                        # If inserter is to the left of the assembler, belt is to the left of the inserter
                        And(inserter.x == assembler.x - 1, belt.x == inserter.x - 1, belt.y == inserter.y),
                        # If inserter is to the right of the assembler, belt is to the right of the inserter
                        And(inserter.x == assembler.x + 3, belt.x == inserter.x + 1, belt.y == inserter.y),
                        # If inserter is above the assembler, belt is above the inserter
                        And(inserter.y == assembler.y - 1, belt.x == inserter.x, belt.y == inserter.y - 1),
                        # If inserter is below the assembler, belt is below the inserter
                        And(inserter.y == assembler.y + 3, belt.x == inserter.x, belt.y == inserter.y + 1)
                    ))
                
    def add_bound_constraints(self):

        # Ensure all inserters stay within the grid boundaries
        for assembler in self.assemblers:
            for inserter in assembler.inserters:
                
                belt = inserter.belt
                
                self.solver.add(And(inserter.x >= 0, inserter.x < self.width))
                self.solver.add(And(inserter.y >= 0, inserter.y < self.height))
                
                # Ensure all belts stay within the grid boundaries
                self.solver.add(And(belt.x >= 0, belt.x < self.width))
                self.solver.add(And(belt.y >= 0, belt.y < self.height))

                

    print('Bound constraints for belts and inserters added.')
                
    def add_overlapping_constraints(self):
        
        # we made sure that assemblers dont overlap with anythign so we only need to look at the inserters and belts
        
        inserters = [inserter for assembler in self.assemblers for inserter in assembler.inserters]
        belts = [inserter.belt for assembler in self.assemblers for inserter in assembler.inserters]
        
        
        #belts += self.global_input_belts
        #belts.append(self.global_output_belt)
        
        #Ensure no inserters overlap with each other
        for i, inserter1 in enumerate(inserters):
            for inserter2 in inserters[i+1:]:
                if(inserter1.id != inserter2.id):
                    self.solver.add(Or(
                        inserter1.x != inserter2.x,
                        inserter1.y != inserter2.y
                    ))

        #Ensure no belts overlap with each other
        for i, belt1 in enumerate(belts):
            for belt2 in belts[i+1:]:
                if(belt1.id != belt2.id):
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
        
        
    def build_map(self):
        start_points = []
        end_points = {}
        obstacle_map = np.zeros((self.height, self.width), dtype=int)
        
        belts = self.global_input_belts
        belts.append(self.global_output_belt)
        #Mark assemblers in the obstacle map
        for assembler in self.assemblers:
            x = self.model.evaluate(assembler.x).as_long()
            y = self.model.evaluate(assembler.y).as_long()
            # Mark 3x3 area around the assembler as occupied
            for dx in range(3):
                for dy in range(3):
                    if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                        obstacle_map[y + dy][x + dx] = 1
                        
            # Mark inserters in the obstacle map
            for inserter in assembler.inserters:
                x = self.model.evaluate(inserter.x).as_long()
                y = self.model.evaluate(inserter.y).as_long()
                # Mark inserter position as occupied
                if 0 <= x < self.width and 0 <= y < self.height:
                    obstacle_map[y][x] = 1

                belts.append(inserter.belt)
        
        
        # Mark belts in the obstacle map
        for belt in belts:
            x = self.model.evaluate(belt.x).as_long()
            y = self.model.evaluate(belt.y).as_long()
            # Mark belt position as occupied
            if 0 <= x < self.width and 0 <= y < self.height:
                obstacle_map[y][x] = 1
                
            if belt.type == 'start':
                # Store start point with the ID without the suffix
                start_id = belt.id.split('_')[0]  # Get the part before "_"
                start_points.append((start_id, (x, y)))
                
            elif belt.type == 'end':
                end_id = belt.id.split('_')[0]  # Get the part before "_"
                if end_id not in end_points:
                    end_points[end_id] = []
                end_points[end_id].append((x, y))
                
            # Create pairs of start and end points
            pairs = []
            for start_id, start_position in start_points:
                if start_id in end_points:
                    for end_position in end_points[start_id]:
                        pairs.append((start_position, end_position))


        return obstacle_map,pairs
        
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
