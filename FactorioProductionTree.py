#! .venv\Scripts\python.exe

import json
import logging
import pygame
import math
import sys ,os
from z3 import And , Or
from z3Solver  import Z3Solver
from AStarPathFinderold import AStarPathFinderold

from AStarPathFinder import AStarPathFinder

# Define constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
LIGHT_GREY = (211,211,211)

# Define the size of each grid cell
CELL_SIZE = 50

INSERTER_COLOR_MAP = {
    'input': (255, 165, 0),  # Orange 
    'output': (255, 20, 147) # Deep Pink
}

BELT_COLOR_MAP = {
    'start': (0, 128, 0),  # Green 
    'end': (255, 0, 0)     # Red
}


class FactorioProductionTree:
    def __init__(self,grid_width,grid_height) -> None:
        # Load the data from JSON
        self.setup_logging("FactorioProductionTree")
        items_data = self.load_json("recipes.json")
        
        self.machines_data = self.load_json("machine_data.json")  # Machine speeds and capacities
        
        self.grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Create a lookup dictionary for items by their ID
        self.item_lookup = {item["id"]: item for item in items_data}

        # info for inout items with belts
        self.input_items=[]
        self.input_information = None
        
        self.output_item = None
        self.output_information = None
        
        # init after calulation of production data
        self.z3_solver = None
        self.AStar = None
        
        self.obstacle_map = None
    
        
        
    def load_json(self,recipe_file):
        with open(recipe_file, "r") as file:
                recipes = json.load(file)
                return recipes
            
            
    def setup_logging(self, file_name):
        # Configure the logging

        # Clear the log file at the start
        open(file_name, "w").close()  # Truncate the file

        logging.basicConfig(
            filename=file_name,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    # Recursively calculate the production requirements, including assemblers, inserters, and belts.
    def calculate_production(self, item_id, items_per_minute):

        item = self.item_lookup.get(item_id)
        
        # If the item doesn't exist or there's no recipe, return the required amount as it is.
        if not item or "recipe" not in item:
            return {item_id: {"amount_per_minute": items_per_minute}}

        recipe = item["recipe"]
        time_per_unit = recipe.get("time")
        yield_per_recipe = recipe.get("yield")

        # If the recipe has no ingredients (e.g "iron-plate"), just return the target amount. -> ground insert inputs
        # If you want to set an item as possible Input for the system -> set Ingredients to []
        if time_per_unit is None or yield_per_recipe is None or not recipe["ingredients"]:
            self.input_items.append(item_id)
            return {item_id: {"amount_per_minute": items_per_minute}}

        # Calculate how many recipe runs are needed per minute
        recipe_runs_needed_per_minute = items_per_minute / yield_per_recipe

        
        # Store total required amounts, including assemblers, inserters, and belts
        total_requirements = {
            item_id: {
                "output_per_minute": items_per_minute,
                "assemblers": math.ceil(self._calculate_assemblers(time_per_unit, recipe_runs_needed_per_minute)),
                "output_inserters": math.ceil(self._calculate_inserters(recipe_runs_needed_per_minute)),
                "input_inserters": recipe["ingredients"],
                "belts": 0  
            }
        }

        # Process each ingredient recursively and calculate belt requirements
        for ingredient in recipe["ingredients"]:
            ingredient_id = ingredient["id"]
            ingredient_amount = ingredient["amount"]
            total_ingredient_needed_per_minute = ingredient_amount * recipe_runs_needed_per_minute
            
            sub_requirements = self.calculate_production(ingredient_id, total_ingredient_needed_per_minute)

            # Merge sub-requirements into total requirements
            for sub_item, sub_data in sub_requirements.items():
                if sub_item in total_requirements:
                    total_requirements[sub_item]["amount_per_minute"] += sub_data["amount_per_minute"]
                else:
                    total_requirements[sub_item] = sub_data
            
            # Calculate belts needed to transfer ingredients to the assemblers
            total_requirements[item_id]["belts"] += self._calculate_belts(total_ingredient_needed_per_minute)

        total_requirements[item_id]["belts"] = math.ceil(total_requirements[item_id]["belts"])  # Round up belts
        return total_requirements

    # Calculate how many assemblers are needed to produce the required amount per minute.
    def _calculate_assemblers(self, time_per_unit, recipe_runs_needed_per_minute):
        crafting_speed = self.machines_data["assemblers"]["crafting_speed"]
        items_per_second_per_assembler = crafting_speed / time_per_unit
        items_per_minute_per_assembler = items_per_second_per_assembler * 60
        return recipe_runs_needed_per_minute / items_per_minute_per_assembler

    # Calculate how many inserters are needed to move the required amount of items per minute.
    def _calculate_inserters(self, recipe_runs_needed_per_minute):
       
        items_per_second_per_inserter = self.machines_data["inserters"]["ItemsPerSecond"]
        items_per_minute_per_inserter = items_per_second_per_inserter * 60
        return recipe_runs_needed_per_minute / items_per_minute_per_inserter
    # Calculate how many belts are needed to move the required amount of items per minute.
    def _calculate_belts(self, total_items_needed_per_minute):
      
        items_per_second_per_belt = self.machines_data["belts"]["ItemsPerSecond"]
        items_per_minute_per_belt = items_per_second_per_belt * 60
        return total_items_needed_per_minute / items_per_minute_per_belt
    

    # TODO rework
    def calculate_minimal_grid_size(self,production_output):

        total_assemblers = 0
        total_inserters = 0
        total_belts = 0
        
        for item, requirements in production_output.items():
            total_assemblers += requirements.get('assemblers', 0)
            total_inserters += requirements.get('inserters', 0)
            total_belts += requirements.get('belts', 0)
        
        # Each assembler takes a 3x3 space
        assembler_height = total_assemblers * 3
        # Space for inserters and belts: assume inserters are adjacent and belts run vertically
        inserter_belt_height = total_inserters + total_belts
        
        # Minimal grid width is the total number of assemblers + inserter space
        width = max(10, total_assemblers * 4)
        # Minimal grid height is the height for assemblers + inserters/belts
        height = assembler_height + inserter_belt_height
        
        self.grid_width = width
        self.grid_height = height
        
        return width, height

    
    
    def eval(self):
        return self.grid_height * self.grid_width + sum(row.count(2) for row in self.grid)

    
    def manual_Output(self,output_item):
        side_panel_width = 300
        CELL_SIZE = 50  # Assuming a default cell size
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)  # Green for input
        RED = (255, 0, 0)  # Red for output
        
        direction_angle_map = {
            'right': 90,
            'down': 180,
            'left': 270,
            'up': 0
        }
        
        
        output_image = pygame.image.load(f"assets/{output_item}.png")
        output_image = pygame.transform.scale(output_image, (CELL_SIZE, CELL_SIZE))
        
        input_information = self.input_information
        input_items = self.input_items
        # Load images for the items
        item_images = {item: pygame.image.load(f"assets/{item}.png") for item in input_items}
        
        conveyor_image = pygame.image.load("assets/conveyor.png")
        # Resize images to fit in the grid cells
        item_images = {item: pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE)) for item, image in item_images.items()}
        conveyor_image = pygame.transform.scale(conveyor_image, (CELL_SIZE, CELL_SIZE))
    

        output_information = {output_item:{'input': None, 'output': None,'grid': None, 'direction_grid': None}}
        setting_input = True
        # build obstacle map
        grid_astar =[[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
                                        
        for item, data in input_information.items():
            grid = data['grid']
            if grid is not None:
                for row in range(self.grid_height):
                    for col in range(self.grid_width):
                        if grid[col][row] == 2:  # Belt path marked with '2'
                            grid_astar[col][row] = 1

                # make sure to not cross existing belt start and end points
                if data['input']:
                    grid_astar[data['input'][1]][data['input'][0]]=1

                if data['output']:
                    grid_astar[data['output'][1]][data['output'][0]]=1
    

    
        pygame.init()

        # Set up the window size
        window_width = self.grid_width * CELL_SIZE + side_panel_width
        window_height = self.grid_height * CELL_SIZE
        window = pygame.display.set_mode((window_width, window_height))

        pygame.display.set_caption('IO Items')

        # Set up clock
        clock = pygame.time.Clock()
        running = True
        # Main loop
        while running:
            items = list(output_information.keys()) 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    #pygame.quit()
                    running = False

                # Handle mouse click to place or remove input/output
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    # Make sure the click is inside the grid (not in the side panel)
                    if mouse_x > side_panel_width:
                        col = (mouse_x - side_panel_width) // CELL_SIZE
                        row = mouse_y // CELL_SIZE
                        current_item = items[0]

                        if 0 <= col < self.grid_width and 0 <= row < self.grid_height:
                            # Left click to place input/output
                            if event.button == 1:
                                if setting_input and output_information[current_item]['input'] is None:
                                    output_information[current_item]['input'] = (row, col)
                                    setting_input = False  # Switch to setting output next
                                elif not setting_input and output_information[current_item]['output'] is None:
                                    output_information[current_item]['output'] = (row, col)
                                    setting_input = True  # After setting output, switch to input for next item

                                    # Once both input and output are set, find the path and connect belts
                                    if output_information[current_item]['input'] and output_information[current_item]['output']:
                                            
                                        # Create an AStarPathfinder object for pathfinding
                                        pairs = [(output_information[current_item]['input'], output_information[current_item]['output'])]
                                        astar = AStarPathFinderold(grid_astar, pairs)
                                        connected = astar.connect_belts()  # This updates the grid with belt paths

                                        
                                        if connected:
                                            output_information[current_item]['grid'] = astar.grid
                                            output_information[current_item]['direction_grid'] = astar.direction_grid
                                            
                                        else:
                                            print(f"No valid path found for {current_item}")

                            # Right click to undo last placement (input or output)
                            if event.button == 3:
                                if not setting_input and output_information[current_item]['input'] is not None:
                                    # Undo input if input was set last
                                    output_information[current_item]['input'] = None
                                    setting_input = True  # Go back to setting input
                                    
                                    output_information[current_item]['grid'] = None
                                    output_information[current_item]['direction_grid'] = None
                                    
                                elif output_information[current_item]['output'] is not None:
                                    # Undo output if output was set last
                                    output_information[current_item]['output'] = None
                                    setting_input = False  # Go back to setting output
                                    
                                    output_information[current_item]['grid'] = None
                                    output_information[current_item]['direction_grid'] = None

                # Handle key press to change selected item
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:  # Right arrow to move to the next item
                        current_item_index = (current_item_index + 1) % len(output_information)
                        setting_input = True  # Reset to input when changing item
                    elif event.key == pygame.K_LEFT:  # Left arrow to move to the previous item
                        current_item_index = (current_item_index - 1) % len(output_information)
                        setting_input = True  # Reset to input when changing item

            # Fill the screen with black
            window.fill(BLACK)
            # Draw the grid and place images with background colors
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    rect = pygame.Rect(side_panel_width + col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(window, WHITE, rect, 1)  # Draw grid lines

                    # Check if this cell is assigned as an input or output for any item
                    for item_index, item in enumerate(input_information):
                        if input_information[item]['input'] == (row, col):
                            # Draw green background for input
                            pygame.draw.rect(window, GREEN, rect)
                            window.blit(item_images[item], rect)
                        elif input_information[item]['output'] == (row, col):
                            # Draw red background for output
                            pygame.draw.rect(window, RED, rect)
                            window.blit(item_images[item], rect)
                            
                    for item_index, item in enumerate(output_information):
                        if output_information[item]['input'] == (row, col):
                            # Draw green background for input
                            pygame.draw.rect(window, GREEN, rect)
                            window.blit(output_image, rect)
                        elif output_information[item]['output'] == (row, col):
                            # Draw red background for output
                            pygame.draw.rect(window, RED, rect)
                            window.blit(output_image, rect)
                    
            # draw input items
            for item, data in input_information.items():
                grid = data['grid']
                direction_grid = data['direction_grid']
                if grid is not None:
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            if grid[col][row] == 2:  # Belt path marked with '2'
                                direction = direction_grid[col][row]
                                if direction in direction_angle_map:
                                    # Rotate the belt image based on direction
                                    rotated_belt_image = pygame.transform.rotate(conveyor_image, direction_angle_map[direction]+90)
                                            
                                    # Draw the belt image at the correct position
                                    window.blit(rotated_belt_image, (side_panel_width + col * CELL_SIZE, row * CELL_SIZE))
            # draw output items                      
            for item, data in output_information.items():
                grid = data['grid']
                direction_grid = data['direction_grid']
                if grid is not None:
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            if grid[col][row] == 2:  # Belt path marked with '2'
                                direction = direction_grid[col][row]
                                if direction in direction_angle_map:
                                    # Rotate the belt image based on direction
                                    rotated_belt_image = pygame.transform.rotate(conveyor_image, direction_angle_map[direction]+90)
                                            
                                    # Draw the belt image at the correct position
                                    window.blit(rotated_belt_image, (side_panel_width + col * CELL_SIZE, row * CELL_SIZE))
                
            
            # Draw the side panel
            pygame.draw.rect(window, BLACK, (0, 0, side_panel_width, window_height))  # Side panel background
            font = pygame.font.Font(None, 36)
            item_text = f"Setting: {output_item}"
            setting_text = "Input" if setting_input else "Output"
            text_surface_item = font.render(item_text, True, WHITE)
            text_surface_setting = font.render(setting_text, True, WHITE)
            window.blit(text_surface_item, (10, 50))  # Show the current item being set
            window.blit(text_surface_setting, (10, 100))  # Show whether we're setting input or output

            # Update display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(30)
            
        self.output_information = output_information
        pygame.quit()
    
    def manual_Input(self):
        
        side_panel_width = 300
        CELL_SIZE = 50  # Assuming a default cell size
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)  # Green for input
        RED = (255, 0, 0)  # Red for output

        # Get all input items
        #input_items = ['copper-plate', 'iron-plate']
        
        input_items = self.input_items
        
        direction_angle_map = {
            'right': 90,
            'down': 180,
            'left': 270,
            'up': 0
        }
        # Load images for the items
        item_images = {item: pygame.image.load(f"assets/{item}.png") for item in input_items}
        conveyor_image = pygame.image.load("assets/conveyor.png")
        # Resize images to fit in the grid cells
        item_images = {item: pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE)) for item, image in item_images.items()}
        conveyor_image = pygame.transform.scale(conveyor_image, (CELL_SIZE, CELL_SIZE))

        # Track positions for inputs and outputs (item -> {'input': (row, col), 'output': (row, col)})
        input_information = {item: {'input': None, 'output': None,'grid': None, 'direction_grid': None} for item in input_items}
        current_item_index = 0
        setting_input = True  # True when setting input position, False for output
            
        pygame.init()

        # Set up the window size
        window_width = self.grid_width * CELL_SIZE + side_panel_width
        window_height = self.grid_height * CELL_SIZE
        window = pygame.display.set_mode((window_width, window_height))

        pygame.display.set_caption('IO Items')

        # Set up clock
        clock = pygame.time.Clock()

        running = True
        # Main loop
        while running:
            items = list(input_information.keys()) 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    #pygame.quit()
                    #sys.exit()
                    running = False

                # Handle mouse click to place or remove input/output
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    # Make sure the click is inside the grid (not in the side panel)
                    if mouse_x > side_panel_width:
                        col = (mouse_x - side_panel_width) // CELL_SIZE
                        row = mouse_y // CELL_SIZE
                        current_item = items[current_item_index]

                        if 0 <= col < self.grid_width and 0 <= row < self.grid_height:
                            # Left click to place input/output
                            if event.button == 1:
                                if setting_input and input_information[current_item]['input'] is None:
                                    input_information[current_item]['input'] = (row, col)
                                    setting_input = False  # Switch to setting output next
                                elif not setting_input and input_information[current_item]['output'] is None:
                                    input_information[current_item]['output'] = (row, col)
                                    setting_input = True  # After setting output, switch to input for next item

                                    # Once both input and output are set, find the path and connect belts
                                    if input_information[current_item]['input'] and input_information[current_item]['output']:
                                        
                                        grid_astar =[[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
                                        
                                        for item, data in input_information.items():
                                            
                                            grid = data['grid']
                                            if grid is not None:
                                                for row in range(self.grid_height):
                                                    for col in range(self.grid_width):
                                                        if grid[col][row] == 2:  # Belt path marked with '2'
                                                            grid_astar[col][row] = 1
                                                            
                                            # make sure to not cross existing belt start and end points
                                            if data['input']:
                                                grid_astar[data['input'][1]][data['input'][0]]=1
                                                
                                            if data['output']:
                                                grid_astar[data['output'][1]][data['output'][0]]=1
                                            
                                        # Create an AStarPathfinder object for pathfinding
                                        pairs = [(input_information[current_item]['input'], input_information[current_item]['output'])]
                                        astar = AStarPathFinderold(grid_astar, pairs)
                                        connected = astar.connect_belts()  # This updates the grid with belt paths
                                        
                                        
                                        if connected:
                                            input_information[current_item]['grid'] = astar.grid
                                            input_information[current_item]['direction_grid'] = astar.direction_grid
                                            
                                        else:
                                            print(f"No valid path found for {current_item}")

                            # Right click to undo last placement (input or output)
                            if event.button == 3:
                                if not setting_input and input_information[current_item]['input'] is not None:
                                    # Undo input if input was set last
                                    input_information[current_item]['input'] = None
                                    setting_input = True  # Go back to setting input
                                    
                                    input_information[current_item]['grid'] = None
                                    input_information[current_item]['direction_grid'] = None
                                    
                                elif input_information[current_item]['output'] is not None:
                                    # Undo output if output was set last
                                    input_information[current_item]['output'] = None
                                    setting_input = False  # Go back to setting output
                                    
                                    input_information[current_item]['grid'] = None
                                    input_information[current_item]['direction_grid'] = None

                # Handle key press to change selected item
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:  # Right arrow to move to the next item
                        current_item_index = (current_item_index + 1) % len(input_information)
                        setting_input = True  # Reset to input when changing item
                    elif event.key == pygame.K_LEFT:  # Left arrow to move to the previous item
                        current_item_index = (current_item_index - 1) % len(input_information)
                        setting_input = True  # Reset to input when changing item
            
            # Fill the screen with black
            window.fill(BLACK)
            # Draw the grid and place images with background colors
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    rect = pygame.Rect(side_panel_width + col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(window, WHITE, rect, 1)  # Draw grid lines

                    # Check if this cell is assigned as an input or output for any item
                    for item_index, item in enumerate(input_information):
                        if input_information[item]['input'] == (row, col):
                            # Draw green background for input
                            pygame.draw.rect(window, GREEN, rect)
                            window.blit(item_images[item], rect)
                        elif input_information[item]['output'] == (row, col):
                            # Draw red background for output
                            pygame.draw.rect(window, RED, rect)
                            window.blit(item_images[item], rect)

                    
            for item, data in input_information.items():
                grid = data['grid']
                direction_grid = data['direction_grid']
                if grid is not None:
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            if grid[col][row] == 2:  # Belt path marked with '2'
                                direction = direction_grid[col][row]
                                if direction in direction_angle_map:
                                    # Rotate the belt image based on direction
                                    rotated_belt_image = pygame.transform.rotate(conveyor_image, direction_angle_map[direction]+90)
                                            
                                    # Draw the belt image at the correct position
                                    window.blit(rotated_belt_image, (side_panel_width + col * CELL_SIZE, row * CELL_SIZE))

                
            
            # Draw the side panel
            pygame.draw.rect(window, BLACK, (0, 0, side_panel_width, window_height))  # Side panel background
            font = pygame.font.Font(None, 36)
            item_text = f"Setting: {items[current_item_index]}"
            setting_text = "Input" if setting_input else "Output"
            text_surface_item = font.render(item_text, True, WHITE)
            text_surface_setting = font.render(setting_text, True, WHITE)
            window.blit(text_surface_item, (10, 50))  # Show the current item being set
            window.blit(text_surface_setting, (10, 100))  # Show whether we're setting input or output

            # Update display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(30)
            
            # store input posiitons
        pygame.quit()
        self.input_information = input_information

    def solve(self,production_data,sequential):
        
        # Initialize solver with grid size and production data
        if self.z3_solver is None:
            self.z3_solver = Z3Solver(self.grid_width,self.grid_height, production_data)
        # Process the input to place assemblers
        
        if sequential:
            self.placed_assembler, self.model = self.z3_solver.build_sequential()
            
            
            
        else:
            self.z3_solver.build_constraints()
            self.z3_solver.solve()

        
        
    def add_manual_IO_constraints(self,production_data,sequential):
        if self.z3_solver is None:
            self.z3_solver = Z3Solver(self.grid_width,self.grid_height, production_data)
        
        
        
        self.z3_solver.add_manuel_IO_constraints(self.input_information,self.output_information)
    
    
    def detect_belt_overlap(self,belt):
        belt_coords = (belt[1], belt[2]) 
        # Step 2: Extract input/output belt coordinates
        input_coords = set()
        output_coords = set()

        # Gather all coordinates in input/output grids where a belt (value 2) is present
        for material, data in self.input_information.items():
            input_grid = data['grid']
            for row in range(len(input_grid)):
                for col in range(len(input_grid[0])):
                    if input_grid[row][col] == 2:
                        input_coords.add((row, col))

        for material, data in self.output_information.items():
            output_grid = data['grid']
            for row in range(len(output_grid)):
                for col in range(len(output_grid[0])):
                    if output_grid[row][col] == 2:
                        output_coords.add((row, col))
        
        return belt_coords in input_coords or belt_coords in output_coords

    def detect_assembler_overlap(self,belt,assembler_information):
        belt_coords = (belt[1], belt[2]) 
        
        for assembler in assembler_information:
            assembler_item, assembler_x, assembler_y = assembler
            
            # Check if the belt's coordinates overlap with the assembler's 3x3 area
            for dx in range(3):  # Assembler width: 3
                for dy in range(3):  # Assembler height: 3
                    assembler_coords = (assembler_x + dx, assembler_y + dy)
                    
                    # If the belt's coordinates overlap with any of the assembler's coordinates
                    if belt_coords == assembler_coords:
                        return True  # There is an overlap
        return False  # No overlap detected
        

    def is_position_available(self, pos):
        x, y = pos
        # Check if position is within bounds of the obstacle_map and if it's free (0)
        if 0 <= x < len(self.obstacle_map) and 0 <= y < len(self.obstacle_map[0]):
            return self.obstacle_map[y][x] == 0
        return False  # Out-of-bounds positions are considered unavailable

    def get_retrieval_points(self, belt_point_information, assembler_information):
        retrieval_points = {}  # Dictionary to store all retrieval points for each item

        # Step 1: Check direct input points from `input_information`
        for item, x, y, _ in belt_point_information:
            retrieval_points[item] = {
            'destination': [(x, y)],  # This is the destination point from belt_point_information
            'start_points': []  # List to store relevant start points
            }

            # Step 2: Check if item is in `input_information`
            if item in self.input_information:
                # Get all positions with the matching item from `input_information`
                input_point = self.input_information[item]['input']
                output_point = self.input_information[item]['output']
                input_point = (input_point[1], input_point[0])  # Swap the coordinates
                output_point = (output_point[1], output_point[0])  # Swap the coordinates
                
                # Add these points to the start_points list
                retrieval_points[item]['start_points'].append(input_point)
                retrieval_points[item]['start_points'].append(output_point)
                
                # Retrieve the grid for the item and locate all positions with '2' (indicating belt presence)
                item_grid = self.input_information[item]['grid']
                for row in range(len(item_grid)):
                    for col in range(len(item_grid[0])):
                        if item_grid[row][col] == 2:  # Belt presence for this item
                            retrieval_points[item]["start_points"].append((row, col))
                continue  # Move to the next belt item after checking input information

            # Step 2: Check assembler output points from `assembler_information`
            for asm_item, assembler_x, assembler_y in assembler_information:
                if asm_item == item:
        
                    # Define assembler output positions based on `output_positions` template
                    output_positions = [
                        [(assembler_x, assembler_y - 1), (assembler_x, assembler_y - 2)],  # Upper left
                        [(assembler_x + 1, assembler_y - 1), (assembler_x + 1, assembler_y - 2)],  # Upper middle
                        [(assembler_x + 2, assembler_y - 1), (assembler_x + 2, assembler_y - 2)],  # Upper right
                        [(assembler_x, assembler_y + 3), (assembler_x, assembler_y + 4)],  # Bottom left
                        [(assembler_x + 1, assembler_y + 3), (assembler_x + 1, assembler_y + 4)],  # Bottom middle
                        [(assembler_x + 2, assembler_y + 3), (assembler_x + 2, assembler_y + 4)],  # Bottom right
                        [(assembler_x - 1, assembler_y), (assembler_x - 2, assembler_y)],  # Left up
                        [(assembler_x - 1, assembler_y + 1), (assembler_x - 2, assembler_y + 1)],  # Left middle
                        [(assembler_x - 1, assembler_y + 2), (assembler_x - 2, assembler_y + 2)],  # Left bottom
                        [(assembler_x + 4, assembler_y), (assembler_x + 5, assembler_y)],  # Right up
                        [(assembler_x + 4, assembler_y + 1), (assembler_x + 5, assembler_y + 1)],  # Right middle
                        [(assembler_x + 4, assembler_y + 2), (assembler_x + 5, assembler_y + 2)]  # Right bottom
                    ]

                    # Filter output positions to ensure no overlap with any existing structures
                    for position_pair in output_positions:
                        # Filter output positions to ensure no overlap with any existing structures
                            if self.is_position_available(position_pair[0]) and self.is_position_available(position_pair[1]):
                                retrieval_points[item]["start_points"].append(position_pair[1])

        return retrieval_points

    
    def add_out_point_information(self,output_item,assembler_information):
        retrieval_points = {}
        retrieval_points[output_item] = {
            'destination': [],  # This is the destination point from belt_point_information
            'start_points': []  # List to store relevant start points
        }
        
        output_coords = []
        for material, data in self.output_information.items():
            output_grid = data['grid']
            for row in range(len(output_grid)):
                for col in range(len(output_grid[0])):
                    if output_grid[row][col] == 2:
                        output_coords.append((row, col))
                        
        input_point = self.output_information[output_item]['input']
        output_point = self.output_information[output_item]['output']         
                        
        input_point = (input_point[1], input_point[0])  # Swap the coordinates
        output_point = (output_point[1], output_point[0])  # Swap the coordinates
        
        output_coords.append(input_point)
        output_coords.append(output_point)
        retrieval_points[output_item]["destination"] = output_coords
        
        for asm_item, assembler_x, assembler_y in assembler_information:
                if asm_item == output_item:
        
                    # Define assembler output positions based on `output_positions` template
                    output_positions = [
                        [(assembler_x, assembler_y - 1), (assembler_x, assembler_y - 2)],  # Upper left
                        [(assembler_x + 1, assembler_y - 1), (assembler_x + 1, assembler_y - 2)],  # Upper middle
                        [(assembler_x + 2, assembler_y - 1), (assembler_x + 2, assembler_y - 2)],  # Upper right
                        [(assembler_x, assembler_y + 3), (assembler_x, assembler_y + 4)],  # Bottom left
                        [(assembler_x + 1, assembler_y + 3), (assembler_x + 1, assembler_y + 4)],  # Bottom middle
                        [(assembler_x + 2, assembler_y + 3), (assembler_x + 2, assembler_y + 4)],  # Bottom right
                        [(assembler_x - 1, assembler_y), (assembler_x - 2, assembler_y)],  # Left up
                        [(assembler_x - 1, assembler_y + 1), (assembler_x - 2, assembler_y + 1)],  # Left middle
                        [(assembler_x - 1, assembler_y + 2), (assembler_x - 2, assembler_y + 2)],  # Left bottom
                        [(assembler_x + 4, assembler_y), (assembler_x + 5, assembler_y)],  # Right up
                        [(assembler_x + 4, assembler_y + 1), (assembler_x + 5, assembler_y + 1)],  # Right middle
                        [(assembler_x + 4, assembler_y + 2), (assembler_x + 5, assembler_y + 2)]  # Right bottom
                    ]

                    # Filter output positions to ensure no overlap with any existing structures
                    for position_pair in output_positions:
                            if self.is_position_available(position_pair[0]) and self.is_position_available(position_pair[1]):
                                retrieval_points[output_item]["start_points"].append(position_pair[1])
        return retrieval_points
    # need to solve once before you can execute this
    def build_belts(self,output_item,max_tries):
        
        
        for i in range(max_tries):
            
            print(f'Try Number: {i}')
            
            
            self.obstacle_map,belt_point_information,assembler_information = self.z3_solver.build_map()
            
            print(self.obstacle_map)
            
            print(belt_point_information)
            
            print(f"assembler_information {assembler_information}")
            
            # get rid of belts that are already connected -> overlap with other input and output belts set by user or overlap with assembler -> direct insertion
            belt_point_information = [belt for belt in belt_point_information if not self.detect_belt_overlap(belt)]
            belt_point_information = [belt for belt in belt_point_information if not self.detect_assembler_overlap(belt,assembler_information)]
            
            print(f"belt_point_information {belt_point_information}")
            
            retrieval_points = self.get_retrieval_points(belt_point_information,assembler_information)
            print(f" retrieval points :{retrieval_points}")
            
        
            # add output belt if needed to form all possible to all possible
            retrieval_points.update(self.add_out_point_information(output_item,assembler_information))
            print(f"finished retrieval points :{retrieval_points}")
            
            
            astar_pathfinder = AStarPathFinder(self.obstacle_map)
            paths = astar_pathfinder.find_path_for_item(retrieval_points)

            print(paths)
            
            return True
                
        return False            
            
        
        
    def visualize_factory(self,paths):
        side_panel_width = 300
        
        direction_angle_map = {
            'right': 90,
            'down': 180,
            'left': 270,
            'up': 0
        }

        pygame.init()
        
        # Set up the window size
        window_width = self.grid_width * CELL_SIZE + side_panel_width
        window_height = self.grid_width * CELL_SIZE
        window = pygame.display.set_mode((window_width, window_height))
        
        pygame.display.set_caption('Factory Layout Visualization')

        
        # Set up clock
        clock = pygame.time.Clock()
    
        # assets
        assembler_image = pygame.image.load('assets/assembler.png')
        inserter_image = pygame.image.load('assets/inserter.png')
        conveyor_image = pygame.image.load('assets/conveyor.png')
        underground_image = pygame.image.load("assets/underground_belt.png")
        
        # Scale images
        assembler_image = pygame.transform.scale(assembler_image, (3 * CELL_SIZE, 3 * CELL_SIZE))
        inserter_image = pygame.transform.scale(inserter_image, (CELL_SIZE, CELL_SIZE))
        conveyor_image = pygame.transform.scale(conveyor_image, (CELL_SIZE, CELL_SIZE))
        underground_image = pygame.transform.scale(underground_image, (CELL_SIZE, CELL_SIZE))

        
        item_images = {}

        # Load all images from the assets folder
        assets_folder = 'assets'

        # Excluded images
        excluded_images = {'assembler.png', 'inserter.png', 'conveyor.png', 'underground_belt.png'}

        # Iterate through the assets folder
        for filename in os.listdir(assets_folder):
            if filename.endswith('.png') and filename not in excluded_images:
                # Load and scale the item image
                item_path = os.path.join(assets_folder, filename)
                image = pygame.image.load(item_path)
                item_images[filename[:-4]] = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))  # Remove .png and scale

        running = True
        # Main loop
        while running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # Draw the grid
            # Fill the screen with black
            window.fill(BLACK)
            # Draw the grid and place images with background colors
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    rect = pygame.Rect(side_panel_width + col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(window, WHITE, rect, 1)  # Draw grid lines

                    # Check if this cell is assigned as an input or output for any item
                    for item_index, item in enumerate(self.input_information):
                        if self.input_information[item]['input'] == (row, col):
                            # Draw green background for input
                            pygame.draw.rect(window, GREEN, rect)
                            window.blit(item_images[item], rect)
                        elif self.input_information[item]['output'] == (row, col):
                            # Draw red background for output
                            pygame.draw.rect(window, RED, rect)
                            window.blit(item_images[item], rect)
                            
                    for item_index, item in enumerate(self.output_information):
                        if self.output_information[item]['input'] == (row, col):
                            # Draw green background for input
                            pygame.draw.rect(window, GREEN, rect)
                            window.blit(item_images[item], rect)
                        elif self.output_information[item]['output'] == (row, col):
                            # Draw red background for output
                            pygame.draw.rect(window, RED, rect)
                            window.blit(item_images[item], rect)
                    
            # draw input items
            for item, data in self.input_information.items():
                grid = data['grid']
                direction_grid = data['direction_grid']
                if grid is not None:
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            if grid[col][row] == 2:  # Belt path marked with '2'
                                direction = direction_grid[col][row]
                                if direction in direction_angle_map:
                                    # Rotate the belt image based on direction
                                    rotated_belt_image = pygame.transform.rotate(conveyor_image, direction_angle_map[direction]+90)
                                            
                                    # Draw the belt image at the correct position
                                    window.blit(rotated_belt_image, (side_panel_width + col * CELL_SIZE, row * CELL_SIZE))
            # draw output items                      
            for item, data in self.output_information.items():
                grid = data['grid']
                direction_grid = data['direction_grid']
                if grid is not None:
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            if grid[col][row] == 2:  # Belt path marked with '2'
                                direction = direction_grid[col][row]
                                if direction in direction_angle_map:
                                    # Rotate the belt image based on direction
                                    rotated_belt_image = pygame.transform.rotate(conveyor_image, direction_angle_map[direction]+90)
                                            
                                    # Draw the belt image at the correct position
                                    window.blit(rotated_belt_image, (side_panel_width + col * CELL_SIZE, row * CELL_SIZE))
            
            # draw assembler inserter and belts
            # assembler is 3x3 and x,y is upper left corner of teh assembler
            # image is correctly scaled
            for assembler in self.placed_assembler:
                assembler_x = self.model.evaluate(assembler.x).as_long()
                assembler_y = self.model.evaluate(assembler.y).as_long() 
                
                # Draw the assembler at its position
                window.blit(assembler_image, (assembler_x, assembler_y))  # Draw the assembler image

                window.blit(item_images[assembler.item],(assembler_x, assembler_y))
                
                for inserter in assembler.inserters:
                    belt = inserter.belt
                    
                    inserter_x = self.model.evaluate(inserter.x).as_long()
                    inserter_y = self.model.evaluate(inserter.y).as_long() 
                    
                    window.blit(inserter_image, (inserter_x, inserter_y))
                    
                    belt_x = self.model.evaluate(belt.x).as_long()
                    belt_y = self.model.evaluate(belt.y).as_long() 
                    
                    window.blit(conveyor_image, (belt_x, belt_y))
                
                
                
            # Update display
            pygame.display.flip()

            # Cap the frame rate
            pygame.time.Clock().tick(30)
            
            
        pygame.quit()
        
    
        
def main():
    factorioProductionTree = FactorioProductionTree(10,10)
    total_requirements = factorioProductionTree.calculate_production("electronic-circuit", 20) #60
    
    #total_requirements = factorioProductionTree.calculate_production("copper-cable", 20) #60
    print(f"production data {total_requirements}")

    factorioProductionTree.manual_Input()
 
    factorioProductionTree.manual_Output("electronic-circuit")

   
    factorioProductionTree.add_manual_IO_constraints(total_requirements,sequential=False)
    
    #factorioProductionTree.calculate_minimal_grid_size(total_requirements)
    
    factorioProductionTree.solve(total_requirements,sequential=False)
    
    if(factorioProductionTree.build_belts("electronic-circuit",max_tries=20)):
        #factorioProductionTree.visualize_factory("")
        pass
        #print(factorioProductionTree.grid)
   
if __name__ == "__main__":
    main()
