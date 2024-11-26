#! .venv\Scripts\python.exe

import json
import logging
import pygame
import math
import sys, os
import csv
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
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

        # info for input items with belts
        self.input_items=[]
        self.input_information = None
        
        self.output_item = None
        self.output_information = None
        
        # init after calulation of production data
        self.z3_solver = None
        self.AStar = None
        
        self.obstacle_map = None
        
        self.production_data = {}
    
        
        
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
                "belts": 0  ,
                "capacity": 0
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
        
        
    def set_capacities(self,production_data):
        # create a reverse map of all the items in the production data
        reverse_mapping = {}
        
         # Collect ingredients used in production data
        for item_id, data in production_data.items():
            # Skip items that don't have recipes
            if "input_inserters" not in data:
                continue
            for ingredient in data["input_inserters"]:
                ingredient_id = ingredient["id"]
                
                # Only include items that are in the production data (i.e., their capacities will be calculated)
                if ingredient_id in production_data:
                    if ingredient_id not in reverse_mapping:
                        reverse_mapping[ingredient_id] = []
                    reverse_mapping[ingredient_id].append(item_id)
            
        # for each item calculate the capacity using the map as the number of its assemblers devided by the number of other assemblers
        # only do if both items are in the production data and the item is not the output item else set to 0 
        
        for item_id, data in production_data.items():
            # Skip if this item is already an output product (it shouldn't calculate capacity for itself)
            if "input_inserters" not in data:
                data['capacity'] = 0
                continue

            # Get the number of assemblers for this item
            item_assemblers = data.get("assemblers", 0)

            # Find the items that use this item as an ingredient (from the reverse mapping)
            if item_id in reverse_mapping:
                total_assemblers_needed = 0
                for product in reverse_mapping[item_id]:
                    if product in production_data:
                        total_assemblers_needed += production_data[product]["assemblers"]

                if total_assemblers_needed > 0:
                    data['capacity'] = round(item_assemblers / total_assemblers_needed)
                else:
                    data['capacity'] = 0
            else:
                # If no items are using this as an ingredient, set the capacity to 0
                data['capacity'] = 0
        return production_data



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
        for i,(item, x, y, _) in enumerate(belt_point_information):
            key = f"{item}_{i}"  # Unique key per occurrence of the item on the belt
            
            retrieval_points[key] = {
            'item': item,
            'destination': [],  # This is the destination point from belt_point_information
            'start_points': [(x,y)],  # List to store relevant start points
            'inserter_mapping':None
            }
            

            # Step 2: Check if item is in `input_information`
            if item in self.input_information:
                # Get all positions with the matching item from `input_information`
                input_point = self.input_information[item]['input']
                output_point = self.input_information[item]['output']
                input_point = (input_point[1], input_point[0])  # Swap the coordinates
                output_point = (output_point[1], output_point[0])  # Swap the coordinates
                
                # Add these points to the start_points list
                retrieval_points[key]['destination'].append(input_point)
                retrieval_points[key]['destination'].append(output_point)
                
                # Retrieve the grid for the item and locate all positions with '2' (indicating belt presence)
                item_grid = self.input_information[item]['grid']
                for row in range(len(item_grid)):
                    for col in range(len(item_grid[0])):
                        if item_grid[row][col] == 2:  # Belt presence for this item
                            retrieval_points[key]["destination"].append((row, col))
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
                                retrieval_points[key]["start_points"].append(position_pair[1])

        return retrieval_points

    
    def add_out_point_information(self,output_item,assembler_information):
 
        retrieval_points = {}
        
        # get all the possible output positions from the output information set by the user
        # same for all the output assemblers
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
        
        #retrieval_points[output_item]["destination"] = output_coords
        
        for i,(asm_item, assembler_x, assembler_y) in enumerate(assembler_information):
                if asm_item == output_item:
                    
                    # for each output assembler build own representation    
                    key = f"{output_item}_{i}"
                    
                    logging.info(f"build output path infomation for {key}")
            
                      
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
                        
                        [(assembler_x + 3, assembler_y), (assembler_x + 4, assembler_y)],  # Right up
                        [(assembler_x + 3, assembler_y + 1), (assembler_x + 4, assembler_y + 1)],  # Right middle
                        [(assembler_x + 3, assembler_y + 2), (assembler_x + 4, assembler_y + 2)]  # Right bottom
                    ]
                    
                
                    
                    retrieval_points[key] = {
                        'item':output_item,
                        'destination':  output_coords,  # This is the destination point from belt_point_information
                        'start_points': [],  # List to store relevant start points
                        'inserter_mapping':{}      
                    }

                    # Filter output positions to ensure no overlap with any existing structures
                    for position_pair in output_positions:
                            if self.is_position_available(position_pair[0]) and self.is_position_available(position_pair[1]):
                                
                                
                                retrieval_points[key]["inserter_mapping"][str(position_pair[1])] = position_pair[0]
                                
                                retrieval_points[key]["start_points"].append(position_pair[1])
                                
                                
                                
        return retrieval_points
    
    
    def rearrange_dict(self,input_dict, target_item):
        # Separate items based on the 'item' value
        target_items = {key: value for key, value in input_dict.items() if value.get('item') == target_item}
        other_items = {key: value for key, value in input_dict.items() if value.get('item') != target_item}
        
        # Combine the dictionaries with target items first
        rearranged_dict = {**target_items, **other_items}
        return rearranged_dict
    # need to solve once before you can execute this
    def build_belts(self,output_item,max_tries):
        
        
        for i in range(max_tries):
            
            print(f'Try Number: {i}')
            
            
            self.obstacle_map,belt_point_information,assembler_information,_ = self.z3_solver.build_map()
            
            print(self.obstacle_map)
            
            #print(belt_point_information)
            
            #print(f"assembler_information {assembler_information}")
            
            # get rid of belts that are already connected -> overlap with other input and output belts set by user or overlap with assembler -> direct insertion
            belt_point_information = [belt for belt in belt_point_information if not self.detect_belt_overlap(belt)]
            belt_point_information = [belt for belt in belt_point_information if not self.detect_assembler_overlap(belt,assembler_information)]
            
            #print(f"belt_point_information {belt_point_information}")
            
            retrieval_points = self.get_retrieval_points(belt_point_information,assembler_information)
            #print(f" retrieval points :{retrieval_points}")
            
        
            # add output belt if needed to form all possible to all possible
            retrieval_points.update(self.add_out_point_information(output_item,assembler_information))
            print(f"finished retrieval points :{retrieval_points}")
            
            try:
                
                # rearrange such that we first build paths for outputs
                retrieval_points = self.rearrange_dict(retrieval_points, output_item)
                
                
                astar_pathfinder = AStarPathFinder(self.obstacle_map,retrieval_points)
                paths , placed_inserter_information = astar_pathfinder.find_path_for_item()

                #print(paths)
                
                return paths,placed_inserter_information
            
            except:
                print(f"could not assign valid paths to that setup")
                logging.warning(f"coould not assign valid paths to that setup")
                
                # restrict the assembler and inserter positions to occur in teh same setup -> belts are not needed as they are bound by the inserter
                self.z3_solver.restrict_current_setup()
                self.z3_solver.solve()
                
                
                
        return None            
            

        
    def visualize_factory(self,paths,placed_inserter_information):
        #print(paths)
        
        _ ,belt_point_information,assembler_information,inserter_information = self.z3_solver.build_map()
        
        inserter_information = inserter_information + placed_inserter_information
        
        print(assembler_information)
        
        print(inserter_information)
        
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
        underground_belt_image = pygame.transform.scale(underground_image, (CELL_SIZE, CELL_SIZE))

        conveyor_image_rotated = {
            (0, 1): pygame.transform.rotate(conveyor_image, 0),     # Right
            (1, 0): pygame.transform.rotate(conveyor_image, 90),    # Down
            (0, -1): pygame.transform.rotate(conveyor_image, 180),  # Left
            (-1, 0): pygame.transform.rotate(conveyor_image, 270)   # Up
        }
        
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
            

            # Draw assemblers
            for assembler_item, assembler_x, assembler_y in assembler_information:
                # Convert grid coordinates to pixel coordinates
                pixel_x = side_panel_width + assembler_x * CELL_SIZE
                pixel_y = assembler_y * CELL_SIZE

                # Draw the assembler at its position
                window.blit(assembler_image, (pixel_x, pixel_y))
                window.blit(item_images[assembler_item], (pixel_x, pixel_y))

            # Draw inserters
            for inserter_item, inserter_x, inserter_y in inserter_information:
                # Convert grid coordinates to pixel coordinates
                pixel_x = side_panel_width + inserter_x * CELL_SIZE
                pixel_y = inserter_y * CELL_SIZE

                # Draw the inserter at its position
                window.blit(inserter_image, (pixel_x, pixel_y))

                # Scale the inserter_item image to a quarter of its size
                original_image = item_images[inserter_item]
                quarter_size = (original_image.get_width() // 2, original_image.get_height() // 2)
                scaled_image = pygame.transform.scale(original_image, quarter_size)

                # Calculate the position for the lower-right corner of the inserter
                inserter_width = inserter_image.get_width()
                inserter_height = inserter_image.get_height()
                scaled_width = scaled_image.get_width()
                scaled_height = scaled_image.get_height()
                corner_x = pixel_x + inserter_width - scaled_width
                corner_y = pixel_y + inserter_height - scaled_height

                # Draw the scaled-down inserter_item image at the calculated position
                window.blit(scaled_image, (corner_x, corner_y))
                    
            # Draw paths
            for item, path_info in paths.items():
                path = path_info['path']
                direction_grid = path_info['direction_grid']
                jump_markers = path_info['jump_markers']
                
                match = re.match(r"^[^_]+", item)
                if match:
                    item = match.group(0)
                            # Draw each point on the path
                for i, (x, y) in enumerate(path):
                    # Calculate position
                    center_pixel = (
                        side_panel_width + x * CELL_SIZE + CELL_SIZE // 2,
                        y * CELL_SIZE + CELL_SIZE // 2
                    )
                    
                    # Draw conveyor belt image based on direction
                    if direction_grid[y][x]:  # Only draw if direction is not None
                        dx, dy = direction_grid[y][x]
                        rotated_belt = conveyor_image_rotated[(dx, dy)]
                        window.blit(rotated_belt, (center_pixel[0] - rotated_belt.get_width() // 2, 
                                                center_pixel[1] - rotated_belt.get_height() // 2))

                    # Draw item image at the lower-right corner of each path cell
                    item_image = item_images[item]
                    if item_image:
                        
                        
                        #quarter_size = (item_image.get_width() // 2, item_image.get_height() // 2)
                        #scaled_image = pygame.transform.scale(item_image, quarter_size)

                        # Calculate the position for the lower-right corner of the belt
                        #belt_width = item_image.get_width()
                        #belt_height = item_image.get_height()
                        #scaled_width = scaled_image.get_width()
                        #scaled_height = scaled_image.get_height()
                        #corner_x = pixel_x + belt_width - scaled_width
                        #corner_y = pixel_y + belt_height - scaled_height

                        # Draw the scaled-down inserter_item image at the calculated position
                        #window.blit(scaled_image, (corner_x, corner_y))
                        
                        quarter_size = (item_image.get_width() // 2, item_image.get_height() // 2)
                        scaled_image = pygame.transform.scale(item_image, quarter_size)
                        window.blit(item_image, (center_pixel[0] + CELL_SIZE // 2 - item_image.get_width(), 
                                                 center_pixel[1] + CELL_SIZE // 2 - item_image.get_height()))
                        
                # Draw jump markers
                for marker_pair in jump_markers:
                    for marker in marker_pair:
                        jump_pixel = (
                            side_panel_width + marker[0] * CELL_SIZE + CELL_SIZE // 2,
                            marker[1] * CELL_SIZE + CELL_SIZE // 2
                        )
                        window.blit(underground_belt_image, (jump_pixel[0] - underground_belt_image.get_width() // 2, 
                                                            jump_pixel[1] - underground_belt_image.get_height() // 2))
            # Update display
            pygame.display.flip()

            # Cap the frame rate
            pygame.time.Clock().tick(30)
            
            
        pygame.quit()
        

 
def plot_csv_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure relevant columns are numeric
    df['Execution Time (seconds)'] = pd.to_numeric(df['Execution Time (seconds)'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Get unique items and methods
    items = df['Item'].unique()
    methods = df['Method'].unique()

    # Create a figure with subplots, one for each item
    fig, axes = plt.subplots(len(items), len(methods), figsize=(15, 10))

    # Iterate over each item
    for i, item in enumerate(items):
        for j, method in enumerate(methods):
            # Get the data for the current item and method
            item_method_data = df[(df['Item'] == item) & (df['Method'] == method)]
            
            # Set color based on 'Minimizer' value (1 for red, otherwise blue)
            colors = item_method_data['Minimizer'].apply(lambda x: 'red' if x == 1 else 'blue')
            
            # Select the axis for the current subplot
            ax = axes[i, j] if len(items) > 1 else axes[j]
            
            # Plot the data using scatter (no connecting lines), with conditional color
            ax.scatter(item_method_data['Amount'], item_method_data['Execution Time (seconds)'], c=colors)
            
            # Set subplot title and labels
            ax.set_title(f'{item} - {method}')
            ax.set_xlabel('Amount')
            ax.set_ylabel('Execution Time (seconds)')
            ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
 
# Function to log method execution times with additional information
def log_method_time(item, amount, minimizer, method_name, start_time, end_time):
    execution_time = end_time - start_time
    logging.info(f"Execution time for {method_name}: {execution_time:.4f} seconds.")
    
    # Open the CSV file and append the data
    try:
        with open("execution_times.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([item, amount, minimizer, method_name, execution_time])
    except Exception as e:
        logging.error(f"Error logging execution time for {method_name}: {e}")
        
        
def main():
    
    # Set up logging for better output in the console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    
    
    # Example item and amount
    item_to_produce = "electronic-circuit"
    amount_needed = 10
    
    # init 
    factorioProductionTree = FactorioProductionTree(14,14)
    production_data  = factorioProductionTree.calculate_production(item_to_produce,amount_needed) #60
    production_data = factorioProductionTree.set_capacities(production_data)
    minimizer = 1
    
    
    
    print(f"production data {production_data}")
    
    
    # Manual input and output
    factorioProductionTree.manual_Input()
    factorioProductionTree.manual_Output(item_to_produce)
    factorioProductionTree.add_manual_IO_constraints(production_data,sequential=False)
    

    # Track time for solving the problem
    start_time = time.perf_counter()
    factorioProductionTree.solve(production_data,sequential=False)
    end_time = time.perf_counter()
    log_method_time(item_to_produce, amount_needed, minimizer, "solve", start_time, end_time)

    
    
    start_time = time.perf_counter()
    paths, placed_inserter_information = factorioProductionTree.build_belts(item_to_produce,max_tries=2)
    end_time = time.perf_counter()
    log_method_time(item_to_produce, amount_needed, minimizer, "build_belts", start_time, end_time)
    
    if(paths):
        factorioProductionTree.visualize_factory(paths,placed_inserter_information)
        pass
        #print(factorioProductionTree.grid)
   
if __name__ == "__main__":
    # Prepare CSV file header if not exists
    if not os.path.exists("execution_times.csv"):
        try:
            with open("execution_times.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Item", "Amount", "Minimizer", "Method", "Execution Time (seconds)"])
        except Exception as e:
            logging.error(f"Error initializing CSV file: {e}")

    #plot_csv_data("execution_times.csv")
    main()
