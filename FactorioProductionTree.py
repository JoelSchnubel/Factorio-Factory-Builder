#! .venv\Scripts\python.exe

import json
import logging
import pygame
import math
import ast, os
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from z3 import And , Or
from z3Solver  import Z3Solver
import seaborn as sns
import numpy as np

from MultiAgentPathfinder import MultiAgentPathfinder,Splitter

# Define constants for colors
side_panel_width = 300
CELL_SIZE = 50  # Assuming a default cell size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  
RED = (255, 0, 0)  
BLUE = (0, 0, 255)
LIGHT_GREY = (211,211,211)


INSERTER_COLOR_MAP = {
    'input': (255, 165, 0),  # Orange 
    'output': (255, 20, 147) # Deep Pink
}

BELT_COLOR_MAP = {
    'start': (0, 128, 0),  # Green 
    'end': (255, 0, 0)     # Red
}


class FactorioProductionTree:
    def __init__(self,grid_width=15,grid_height=15) -> None:
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
        
        self.amount = 0
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

    
    def destroy_solver(self):
        self.z3_solver = None
    
    # Recursively calculate the production requirements, including assemblers, inserters, and belts.
 
    def calculate_production(self, item_id, items_per_minute,input_items=[],first=True):
        
        
        
        if first:
            self.output_item = item_id
            self.input_items += input_items
        item = self.item_lookup.get(item_id)
        
        # If the item doesn't exist or there's no recipe, return the required amount as it is.
        if not item or "recipe" not in item or item in input_items:
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
                "amount_per_minute": items_per_minute,
                "assemblers": math.ceil(self._calculate_assemblers(time_per_unit, recipe_runs_needed_per_minute)),
                "input_inserters": [],
                "belts": 0  ,
                "capacity": 0
            }
        }

        # Process each ingredient recursively and calculate belt requirements
        for ingredient in recipe["ingredients"]:

        

            ingredient_id = ingredient['id']
            ingredient_amount = ingredient["amount"]
            total_ingredient_needed_per_minute = ingredient_amount * recipe_runs_needed_per_minute

            # Calculate inserters needed for this ingredient
            # set upper bound for number inserters to 3
        
            inserters_needed = min(3, math.ceil(total_ingredient_needed_per_minute / (60 * self.machines_data['inserters']['ItemsPerSecond'])))
        
            
            total_requirements[item_id]["input_inserters"].append({
                "id": ingredient_id,
                "inserters": inserters_needed,
                "amount": ingredient_amount
            })
            
            if ingredient_id in input_items:
                # Add the ingredient directly if it's in input_items
                total_requirements[ingredient_id] = {
                    "amount_per_minute": total_ingredient_needed_per_minute,
                    "capacity": 0
                }
            else:
                sub_requirements = self.calculate_production(ingredient_id, total_ingredient_needed_per_minute,first=False)

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

    
    def manual_Output(self, Title="Manual Output"):
        side_panel_width = 300
        CELL_SIZE = 50  # Assuming a default cell size
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)  # Green for input
        RED = (255, 0, 0)  # Red for output
        
        output_image = pygame.image.load(f"assets/{self.output_item}.png")
        output_image = pygame.transform.scale(output_image, (CELL_SIZE, CELL_SIZE))
        
        input_information = self.input_information
        input_items = self.input_items
        # Load images for the items
        item_images = {item: pygame.image.load(f"assets/{item}.png") for item in input_items}
        
        conveyor_image = pygame.image.load("assets/conveyor.png")
        # Resize images to fit in the grid cells
        item_images = {item: pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE)) for item, image in item_images.items()}
        conveyor_image = pygame.transform.scale(conveyor_image, (CELL_SIZE, CELL_SIZE))

        output_information = {self.output_item:{'input': None, 'output': None, 'paths': None}}
        setting_input = True
        # build obstacle map
        grid_astar = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
                                        
        # Mark existing paths from input_information as obstacles
        for item, data in input_information.items():
            if data['paths'] is not None and item in data['paths']:
                for path_data in data['paths'][item]:
                    path = path_data['path']
                    
                    # Mark path positions as obstacles
                    for pos in path:
                        # Important: in path, positions are (x, y)
                        x, y = pos
                        if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                            grid_astar[y][x] = 1
            
            # Make sure to not cross existing belt start and end points
            if data['input']:
                y, x = data['input']  # y=row, x=col
                grid_astar[y][x] = 1

            if data['output']:
                y, x = data['output']  # y=row, x=col
                grid_astar[y][x] = 1

        pygame.init()

        # Set up the window size
        window_width = self.grid_width * CELL_SIZE + side_panel_width
        window_height = self.grid_height * CELL_SIZE
        window = pygame.display.set_mode((window_width, window_height))

        pygame.display.set_caption(Title)

        # Set up clock
        clock = pygame.time.Clock()
        running = True
        # Main loop
        while running:
            items = list(output_information.keys()) 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
                                    output_information[current_item]['input'] = (row, col)  # Store as (row, col) for consistency
                                    setting_input = False  # Switch to setting output next
                                elif not setting_input and output_information[current_item]['output'] is None:
                                    output_information[current_item]['output'] = (row, col)  # Store as (row, col) for consistency
                                    setting_input = True  # After setting output, switch to input for next item

                                    # Once both input and output are set, find the path and connect belts
                                    if output_information[current_item]['input'] and output_information[current_item]['output']:
                                        # Create points dictionary for MultiAgentPathfinder
                                        # Note: MultiAgentPathfinder expects (x, y) format, so swap the coordinates
                                        input_y, input_x = output_information[current_item]['input']
                                        output_y, output_x = output_information[current_item]['output']
                                        
                                        points = {
                                            current_item: {
                                                'item': current_item,
                                                'destination': [(output_x, output_y)],  # Swap to (col, row) for pathfinder
                                                'start_points': [(input_x, input_y)],  # Swap to (col, row) for pathfinder
                                                'inserter_mapping': None
                                            }
                                        }
                                        
                                        # Create the pathfinder object
                                        pathfinder = MultiAgentPathfinder(
                                            grid_astar, 
                                            points,
                                            allow_underground=False
                                        )
                                        
                                        # Find paths for all items
                                        paths, inserters = pathfinder.find_paths_for_all_items()
                                        
                                        # Store path information directly
                                        if current_item in paths and paths[current_item]:
                                            output_information[current_item]['paths'] = paths
                                        else:
                                            print(f"No valid path found for {current_item}")

                            # Right click to undo last placement (input or output)
                            if event.button == 3:
                                if not setting_input and output_information[current_item]['input'] is not None:
                                    # Undo input if input was set last
                                    output_information[current_item]['input'] = None
                                    setting_input = True  # Go back to setting input
                                    
                                    output_information[current_item]['paths'] = None
                                    
                                elif output_information[current_item]['output'] is not None:
                                    # Undo output if output was set last
                                    output_information[current_item]['output'] = None
                                    setting_input = False  # Go back to setting output
                                    
                                    output_information[current_item]['paths'] = None

                # Handle key press to change selected item
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:  # Right arrow to move to the next item
                        setting_input = True  # Reset to input when changing item
                    elif event.key == pygame.K_LEFT:  # Left arrow to move to the previous item
                        setting_input = True  # Reset to input when changing item

            # Fill the screen with white
            window.fill(WHITE)
            # Draw the grid and place images with background colors
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    rect = pygame.Rect(side_panel_width + col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(window, BLACK, rect, 1)  # Draw grid lines

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
            
            # Draw input paths
            for item, data in input_information.items():
                if data['paths'] is not None and item in data['paths']:
                    for path_data in data['paths'][item]:
                        path = path_data['path']
                        
                        # Draw belts for each segment of the path
                        for i in range(len(path) - 2):
                            current = path[i+1]
                            next_node = path[i + 2]
                            
                            # Calculate direction vector
                            dx = next_node[0] - current[0]
                            dy = next_node[1] - current[1]
                            
                            # Calculate angle based on direction
                            if dx == 1:
                                angle = 270  # Right
                            elif dx == -1:
                                angle = 90   # Left
                            elif dy == 1:
                                angle = 180  # Down
                            elif dy == -1:
                                angle = 0    # Up
                            else:
                                continue  # Skip if not a direct connection
                            
                            # Draw belt with correct rotation
                            rotated_belt = pygame.transform.rotate(conveyor_image, angle)
                            window.blit(rotated_belt, (side_panel_width + current[0] * CELL_SIZE, current[1] * CELL_SIZE))
            
            # Draw output paths
            for item, data in output_information.items():
                if data['paths'] is not None and item in data['paths']:
                    for path_data in data['paths'][item]:
                        path = path_data['path']
                        
                        # Draw belts for each segment of the path
                        for i in range(len(path) - 2):
                            current = path[i+1]
                            next_node = path[i + 2]
                            
                            # Calculate direction vector
                            dx = next_node[0] - current[0]
                            dy = next_node[1] - current[1]
                            
                            # Calculate angle based on direction
                            if dx == 1:
                                angle = 270  # Right
                            elif dx == -1:
                                angle = 90   # Left
                            elif dy == 1:
                                angle = 180  # Down
                            elif dy == -1:
                                angle = 0    # Up
                            else:
                                continue  # Skip if not a direct connection
                            
                            # Draw belt with correct rotation
                            rotated_belt = pygame.transform.rotate(conveyor_image, angle)
                            window.blit(rotated_belt, (side_panel_width + current[0] * CELL_SIZE, current[1] * CELL_SIZE))
                
            # Draw the side panel
            pygame.draw.rect(window, BLACK, (0, 0, side_panel_width, window_height))  # Side panel background
            font = pygame.font.Font(None, 36)
            item_text = f"Setting: {self.output_item}"
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
    
    def manual_Input(self, Title="Manual Input"):
        side_panel_width = 300
        CELL_SIZE = 50  # Assuming a default cell size
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)  # Green for input
        RED = (255, 0, 0)  # Red for output

        # Get all input items
        input_items = self.input_items
        
        # Load images for the items
        item_images = {item: pygame.image.load(f"assets/{item}.png") for item in input_items}
        conveyor_image = pygame.image.load("assets/conveyor.png")
        # Resize images to fit in the grid cells
        item_images = {item: pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE)) for item, image in item_images.items()}
        conveyor_image = pygame.transform.scale(conveyor_image, (CELL_SIZE, CELL_SIZE))

        # Track positions for inputs and outputs (item -> {'input': (row, col), 'output': (row, col)})
        input_information = {item: {'input': None, 'output': None, 'paths': None} for item in input_items}
        current_item_index = 0
        setting_input = True  # True when setting input position, False for output
        
        pygame.init()

        # Set up the window size
        window_width = self.grid_width * CELL_SIZE + side_panel_width
        window_height = self.grid_height * CELL_SIZE
        window = pygame.display.set_mode((window_width, window_height))

        pygame.display.set_caption(Title)

        # Set up clock
        clock = pygame.time.Clock()

        running = True
        # Main loop
        while running:
            items = list(input_information.keys()) 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
                                    # Store as (row, col) for consistency
                                    input_information[current_item]['input'] = (row, col)
                                    setting_input = False  # Switch to setting output next
                                elif not setting_input and input_information[current_item]['output'] is None:
                                    input_information[current_item]['output'] = (row, col)
                                    setting_input = True  # After setting output, switch to input for next item

                                    # Once both input and output are set, find the path and connect belts
                                    if input_information[current_item]['input'] and input_information[current_item]['output']:
                                        
                                        grid_astar = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
                                        
                                        # Mark existing paths as obstacles
                                        for item, data in input_information.items():
                                            if data['paths'] is not None and item in data['paths']:
                                                # Mark path positions as obstacles
                                                for path_data in data['paths'][item]:
                                                    path = path_data['path']
                                                    for pos in path:
                                                        # Important: in path, positions are (x, y) but we store as (row=y, col=x)
                                                        x, y = pos
                                                        if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                                                            grid_astar[y][x] = 1
                                                    
                                            # Make sure to not cross existing belt start and end points
                                            if data['input']:
                                                y, x = data['input']  # y=row, x=col
                                                grid_astar[y][x] = 1
                                                
                                            if data['output']:
                                                y, x = data['output']  # y=row, x=col
                                                grid_astar[y][x] = 1
                                        
                                        # Create points dictionary for MultiAgentPathfinder
                                        # Note: MultiAgentPathfinder expects (x, y) format, so swap the coordinates
                                        input_y, input_x = input_information[current_item]['input']
                                        output_y, output_x = input_information[current_item]['output']
                                        
                                        points = {
                                            current_item: {
                                                'item': current_item,
                                                'destination': [(output_x, output_y)],  # Swap to (col, row) for pathfinder
                                                'start_points': [(input_x, input_y)],  # Swap to (col, row) for pathfinder
                                                'inserter_mapping': None
                                            }
                                        }
                                        
                                        # Create the pathfinder object
                                        pathfinder = MultiAgentPathfinder(
                                            grid_astar, 
                                            points,
                                            allow_underground=True,  # You can enable underground belts if needed
                                            underground_length=3,
                                            allow_splitters=False,   # Disable splitters
                                            splitters={},            # Empty splitters dictionary
                                            find_optimal_paths=False # Don't need to find optimal paths
                                        )
                                        
                                      
                                        # Find paths for all items
                                        paths, _ = pathfinder.find_paths_for_all_items()
                                        
                                
                                        # Store path information directly
                                        if current_item in paths and paths[current_item]:
                                            input_information[current_item]['paths'] = paths
                                        else:
                                            print(f"No valid path found for {current_item}")

                            # Right click to undo last placement (input or output)
                            if event.button == 3:
                                if not setting_input and input_information[current_item]['input'] is not None:
                                    # Undo input if input was set last
                                    input_information[current_item]['input'] = None
                                    setting_input = True  # Go back to setting input
                                    
                                    input_information[current_item]['paths'] = None
                                    
                                elif input_information[current_item]['output'] is not None:
                                    # Undo output if output was set last
                                    input_information[current_item]['output'] = None
                                    setting_input = False  # Go back to setting output
                                    
                                    input_information[current_item]['paths'] = None

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

            # Draw paths for each item using the path data directly
            for item, data in input_information.items():
                if data['paths'] is not None and item in data['paths']:
                    for path_data in data['paths'][item]:
                        path = path_data['path']
                        
                        # Draw belts for each segment of the path
                        for i in range(len(path) - 2):
                            current = path[i+1]
                            next_node = path[i + 2]
                            
                            # Calculate direction vector
                            dx = next_node[0] - current[0]
                            dy = next_node[1] - current[1]
                            
                            # Calculate angle based on direction
                            if dx == 1:
                                angle = 270  # Right
                            elif dx == -1:
                                angle = 90   # Left
                            elif dy == 1:
                                angle = 180  # Down
                            elif dy == -1:
                                angle = 0    # Up
                            else:
                                continue  # Skip if not a direct connection
                            
                            # Draw belt with correct rotation
                            rotated_belt = pygame.transform.rotate(conveyor_image, angle)
                            window.blit(rotated_belt, (side_panel_width + current[0] * CELL_SIZE, current[1] * CELL_SIZE))
            
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
            
        # store input positions
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
        
    def add_manual_IO_constraints(self,production_data):
        if self.z3_solver is None:
            self.z3_solver = Z3Solver(self.grid_width,self.grid_height, production_data)
        
        
        
        self.z3_solver.add_manuel_IO_constraints(self.input_information,self.output_information)
    
    # TODO
    def detect_belt_overlap(self, belt):
        """
        Check if a belt position overlaps with any input/output belts or paths.
        
        Args:
            belt: A tuple of (item, x, y, _) where x and y are grid coordinates
            
        Returns:
            bool: True if there's an overlap, False otherwise
        """
        belt_x, belt_y = belt[1], belt[2]  # Extract x, y coordinates from belt tuple
        belt_coords = (belt_y, belt_x)  # Convert to (row, col) format
        
        # Extract input/output belt coordinates from paths
        input_coords = set()
        output_coords = set()

        # Gather all coordinates in input paths
        if self.input_information:
            for material, data in self.input_information.items():
                if data['paths'] is not None and material in data['paths']:
                    for path_data in data['paths'][material]:
                        path = path_data['path']
                        for pos in path:
                            # Path coordinates are in (x, y) format, convert to (row, col)
                            input_coords.add((pos[1], pos[0]))
                
                # Also add the input/output positions themselves (already in row, col format)
                if data['input']:
                    input_coords.add(data['input'])
                if data['output']:
                    input_coords.add(data['output'])

        # Gather all coordinates in output paths
        if self.output_information:
            for material, data in self.output_information.items():
                if data['paths'] is not None and material in data['paths']:
                    for path_data in data['paths'][material]:
                        path = path_data['path']
                        for pos in path:
                            # Path coordinates are in (x, y) format, convert to (row, col)
                            output_coords.add((pos[1], pos[0]))
                
                # Also add the input/output positions themselves (already in row, col format)
                if data['input']:
                    output_coords.add(data['input'])
                if data['output']:
                    output_coords.add(data['output'])
        
        # Check if the belt coordinates overlap with any input or output coordinates
        overlap = belt_coords in input_coords or belt_coords in output_coords
        
        # Debug output to help diagnose issues
        if overlap:
            if belt_coords in input_coords:
                print(f"Belt at {belt_coords} overlaps with input belt/path")
            if belt_coords in output_coords:
                print(f"Belt at {belt_coords} overlaps with output belt/path")
        
        return overlap

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
        if 0 <= y < len(self.obstacle_map) and 0 <= x < len(self.obstacle_map[0]):
            return self.obstacle_map[y][x] == 0
        return False  # Out-of-bounds positions are considered unavailable
    

    def get_retrieval_points(self, belt_point_information, assembler_information):
        retrieval_points = {}  # Dictionary to store all retrieval points for each item

        # Step 1: Check direct input points from `input_information`
        for i,(item, x, y, _) in enumerate(belt_point_information):
            key = f"{item}_{i}"  # Unique key per occurrence of the item on the belt
            
            retrieval_points[key] = {
            'item': item,
            'destination': [(x,y)],  # This is the destination point from belt_point_information
            'start_points': [],  # List to store relevant start points
            'inserter_mapping': None
            }
            
            # Step 2: Check if item is in `input_information`
            if item in self.input_information:
                # Get all positions with the matching item from `input_information`
                input_point = self.input_information[item]['input']
                output_point = self.input_information[item]['output']
                input_point = (input_point[1], input_point[0])  # Swap the coordinates
                output_point = (output_point[1], output_point[0])  # Swap the coordinates
                
                # Add these points to the start_points list
                retrieval_points[key]['start_points'].append(input_point)
                retrieval_points[key]['start_points'].append(output_point)
                
                # Process path data directly if it exists
                if self.input_information[item]['paths'] is not None and item in self.input_information[item]['paths']:
                    for path_data in self.input_information[item]['paths'][item]:
                        path = path_data['path']
                        
                        # Add all points in the path as potential destinations
                        for pos in path:
                            # Convert from (x, y) to (col, row) for consistency
                            retrieval_points[key]["start_points"].append(pos)
                
                continue  # Move to the next belt item after checking input information

            # Step 2: Check assembler output points from `assembler_information`
            for asm_item, assembler_x, assembler_y in assembler_information:
                if asm_item == item:
        
                    # Define assembler output positions based on `output_positions` template
                    output_positions = [
                        [(assembler_x, assembler_y - 1), (assembler_x, assembler_y - 2)],  # Upper left
                        [(assembler_x + 1, assembler_y - 1), (assembler_x + 1, assembler_y - 2)],  # Upper middle
                        [(assembler_x + 2, assembler_y - 1), (assembler_x + 2,assembler_y - 2)],  # Upper right
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
                                retrieval_points[key]["destination"].append(position_pair[1])

        return retrieval_points

    
    def add_out_point_information(self, output_item, assembler_information):
        retrieval_points = {}
        
        # Get all the possible output positions from the output information set by the user
        output_coords = []
        
        # Process path data directly if it exists
        if self.output_information[output_item]['paths'] is not None and output_item in self.output_information[output_item]['paths']:
            for path_data in self.output_information[output_item]['paths'][output_item]:
                path = path_data['path']
                
                # Add all points in the path as potential destinations
                for pos in path:
                    output_coords.append(pos)  # Positions are already in (x, y) format
                        
        # Add input and output points
        input_point = self.output_information[output_item]['input']
        output_point = self.output_information[output_item]['output']         
                        
        input_point = (input_point[1], input_point[0])  # Swap the coordinates (row, col) -> (x, y)
        output_point = (output_point[1], output_point[0])  # Swap the coordinates (row, col) -> (x, y)
        
        output_coords.append(input_point)
        output_coords.append(output_point)
        
        for i, (asm_item, assembler_x, assembler_y) in enumerate(assembler_information):
            if asm_item == output_item:
                
                # For each output assembler build own representation    
                key = f"{output_item}_{i}"
                
                logging.info(f"build output path infomation for {key}")
        
                    
                # Define assembler output positions based on `output_positions` template
                output_positions = [
                    [(assembler_x, assembler_y - 1), (assembler_x, assembler_y - 2)],  # Upper left
                    [(assembler_x + 1, assembler_y - 1), (assembler_x + 1, assembler_y - 2)],  # Upper middle
                    [(assembler_x + 2, assembler_y - 1), (assembler_x + 2,assembler_y - 2)],  # Upper right
                    
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
                    'item': output_item,
                    'destination': output_coords,  # This is the destination point from belt_point_information
                    'start_points': [],  # List to store relevant start points
                    'inserter_mapping': {}      
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
    
    def prepare_splitter_information(self, input_information, output_information):
        """
        Prepare splitter information and add orientation data to paths.
        Creates a splitter object for every point in the path.
        
        Returns:
            dict: Dictionary mapping each item to a list of Splitter objects
        """
        splitters = {}
        
        # Process input information
        for item, data in input_information.items():
            # Skip if no paths are available
            if data['paths'] is None or item not in data['paths']:
                continue
                
            # Initialize splitter list for this item if needed
            if item not in splitters:
                splitters[item] = []
            
            # Add orientation information to each path
            for path_data in data['paths'][item]:
                path = path_data['path']
                # Initialize or reset orientation data
                path_data['orientation'] = {}
                
                # Calculate orientation for each segment of the path
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    
                    # Calculate direction vector
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    # Store orientation
                    path_data['orientation'][current] = (dx, dy)
                
                # For the last point, use the same direction as the previous segment
                if len(path) > 1:
                    last = path[-1]
                    second_last = path[-2]
                    dx = last[0] - second_last[0]
                    dy = last[1] - second_last[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    path_data['orientation'][last] = (dx, dy)
                
                # Create a Splitter object for EVERY point in the path
                for point in path:
                    # Get the orientation for this point
                    orientation = path_data['orientation'].get(point, (0, 0))  # Default to (0, 0) if not found
                    
                    # Add as potential splitter using the Splitter class
                    splitters[item].append(Splitter(
                        item=item,
                        position=point,
                        direction=orientation
                    ))
        
        # Process output information using the same approach
        for item, data in output_information.items():
            # Skip if no paths are available
            if data['paths'] is None or item not in data['paths']:
                continue
                
            # Initialize splitter list for this item if needed
            if item not in splitters:
                splitters[item] = []
            
            # Add orientation information to each path
            for path_data in data['paths'][item]:
                path = path_data['path']
                # Initialize or reset orientation data
                path_data['orientation'] = {}
                
                # Calculate orientation for each segment of the path
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    
                    # Calculate direction vector
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    # Store orientation
                    path_data['orientation'][current] = (dx, dy)
                
                # For the last point, use the same direction as the previous segment
                if len(path) > 1:
                    last = path[-1]
                    second_last = path[-2]
                    dx = last[0] - second_last[0]
                    dy = last[1] - second_last[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    path_data['orientation'][last] = (dx, dy)
                
                # Create a Splitter object for EVERY point in the path
                for point in path:
                    # Get the orientation for this point
                    orientation = path_data['orientation'].get(point, (0, 0))  # Default to (0, 0) if not found
                    
                    # Add as potential splitter using the Splitter class
                    splitters[item].append(Splitter(
                        item=item,
                        position=point,
                        direction=orientation
                    ))
        
        return splitters
    
    
    # need to solve once before you can execute this
    def build_belts(self, max_tries):
        for i in range(max_tries):
            print(f'Try Number: {i}')
            
            
            self.obstacle_map,belt_point_information,assembler_information,_ = self.z3_solver.build_map()
          
             
            # get rid of belts that are already connected -> overlap with other input and output belts set by user or overlap with assembler -> direct insertion
            belt_point_information = [belt for belt in belt_point_information if not self.detect_belt_overlap(belt)]
            belt_point_information = [belt for belt in belt_point_information if not self.detect_assembler_overlap(belt, assembler_information)]

            
            retrieval_points = self.get_retrieval_points(belt_point_information,assembler_information)
            
            # add output belt if needed to form all possible to all possible
            retrieval_points.update(self.add_out_point_information(self.output_item,assembler_information))
            
            # rearrange such that we first build paths for outputs
            self.retrieval_points = self.rearrange_dict(retrieval_points, self.output_item)
            
            try:
                
                splitters = self.prepare_splitter_information(self.input_information,self.output_information)
 
         
                # Create the pathfinder
                pathfinder = MultiAgentPathfinder(
                    self.obstacle_map, 
                    retrieval_points,
                    allow_underground=True,
                    underground_length=3,
                    allow_splitters=True,
                    splitters=splitters,  
                    find_optimal_paths=True,
                    output_item=self.output_item,
                )

                # Find paths for all items
                paths, inserters = pathfinder.find_paths_for_all_items()
                
                
              
                return paths, inserters
            
            except Exception as e:
                print(f"could not assign valid paths to that setup: {e}")
                logging.warning(f"could not assign valid paths to that setup: {e}")
                
                # restrict the assembler and inserter positions to occur in the same setup -> belts are not needed as they are bound by the inserter
                self.z3_solver.restrict_current_setup()
                self.z3_solver.solve()
                
                
                
        return None            
    
    def load_data(self, file_path):
        try:
            # Read the data from the file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Restore the values from the loaded data
            self.output_item = data.get("output_item")
            self.amount = data.get("amount")
            self.max_output = data.get("max_ouput")
            self.production_data = data.get("production_data")
            self.grid_width = data.get("grid_width")
            self.grid_height = data.get("grid_height")
            self.input_items = data.get("input_items")
            
            # Load complex data structures
            
            # Handle input_information (with special handling for orientation dictionary)
            self.input_information = data.get("input_information", {})
            for item, item_data in self.input_information.items():
                if "paths" in item_data:
                    for path_item, paths_list in item_data["paths"].items():
                        for path_data in paths_list:
                            # Convert string keys back to tuples in orientation dictionary
                            if "orientation" in path_data:
                                converted_orientation = {}
                                for pos_str, direction in path_data["orientation"].items():
                                    # Convert from string to tuple
                                    pos_tuple = ast.literal_eval(pos_str)
                                    converted_orientation[pos_tuple] = tuple(direction)
                                path_data["orientation"] = converted_orientation
            
            # Handle output_information (similar to input_information)
            self.output_information = data.get("output_information", {})
            for item, item_data in self.output_information.items():
                if "paths" in item_data:
                    for path_item, paths_list in item_data["paths"].items():
                        for path_data in paths_list:
                            # Convert string keys back to tuples in orientation dictionary
                            if "orientation" in path_data:
                                converted_orientation = {}
                                for pos_str, direction in path_data["orientation"].items():
                                    # Convert from string to tuple
                                    pos_tuple = ast.literal_eval(pos_str)
                                    converted_orientation[pos_tuple] = tuple(direction)
                                path_data["orientation"] = converted_orientation
            
            # Load lists containing tuples
            self.inserter_information = data.get("inserter_information", [])
            self.belt_point_information = data.get("belt_point_information", [])
            self.assembler_information = data.get("assembler_information", [])
            
            # Handle retrieval_points with inserter_mapping
            self.retrieval_points = data.get("retrieval_points", {})
            for key, point_data in self.retrieval_points.items():
                if "inserter_mapping" in point_data and point_data["inserter_mapping"]:
                    # Convert string keys back to tuples
                    converted_mapping = {}
                    for pos_str, pos_value in point_data["inserter_mapping"].items():
                        if isinstance(pos_value, list):
                            converted_mapping[pos_str] = tuple(pos_value)
                        else:
                            converted_mapping[pos_str] = pos_value
                    point_data["inserter_mapping"] = converted_mapping
            
            self.obstacle_map = data.get("obstacle_map", [])
            self.paths = data.get("paths", {})

            print(f"Production tree data successfully loaded from {file_path}")
            return True

        except Exception as e:
            print(f"Failed to load production tree data: {e}")
            import traceback
            traceback.print_exc()
            return False
            
            
    def store_data(self, file_path, paths, placed_inserter_information):
        try:
            obstacle_map, belt_point_information, assembler_information, inserter_information = self.z3_solver.build_map()
            
            # Convert NumPy array to list if needed
            serializable_obstacle_map = []
            if isinstance(obstacle_map, np.ndarray):
                serializable_obstacle_map = obstacle_map.tolist()  # Convert NumPy array to list
            else:
                # Handle regular lists of lists
                serializable_obstacle_map = []
                for row in obstacle_map:
                    if isinstance(row, np.ndarray):
                        serializable_obstacle_map.append(row.tolist())
                    else:
                        serializable_obstacle_map.append(list(row))
            # Create serializable versions of complex objects
            serializable_input_info = {}
            if self.input_information:
                for item, data in self.input_information.items():
                    serializable_item_data = {
                        "input": data.get("input"),
                        "output": data.get("output"),
                        "paths": {}
                    }
                    
                    # Handle paths data
                    if data.get("paths"):
                        for path_item, paths_list in data["paths"].items():
                            serializable_item_data["paths"][path_item] = []
                            
                            for path_data in paths_list:
                                # Create serializable path data
                                serializable_path = {
                                    "path": path_data.get("path", []),
                                    "start": path_data.get("start"),
                                    "destination": path_data.get("destination"),
                                    "underground_segments": path_data.get("underground_segments", {}),
                                    "start_splitter": path_data.get("start_splitter") is not None,
                                    "dest_splitter": path_data.get("dest_splitter") is not None
                                }
                                
                                # Handle orientation dictionary (convert tuple keys to strings)
                                if "orientation" in path_data:
                                    serializable_orientation = {}
                                    for pos_tuple, direction in path_data["orientation"].items():
                                        # Convert tuple key to string representation
                                        pos_key = str(pos_tuple)
                                        serializable_orientation[pos_key] = direction
                                    serializable_path["orientation"] = serializable_orientation
                                    
                                serializable_item_data["paths"][path_item].append(serializable_path)
                                    
                    serializable_input_info[item] = serializable_item_data
            
            # Create serializable versions of output information
            serializable_output_info = {}
            if self.output_information:
                for item, data in self.output_information.items():
                    serializable_item_data = {
                        "input": data.get("input"),
                        "output": data.get("output"),
                        "paths": {}
                    }
                    
                    # Handle paths data
                    if data.get("paths"):
                        for path_item, paths_list in data["paths"].items():
                            serializable_item_data["paths"][path_item] = []
                            
                            for path_data in paths_list:
                                # Create serializable path data
                                serializable_path = {
                                    "path": path_data.get("path", []),
                                    "start": path_data.get("start"),
                                    "destination": path_data.get("destination"),
                                    "underground_segments": path_data.get("underground_segments", {}),
                                    "start_splitter": path_data.get("start_splitter") is not None,
                                    "dest_splitter": path_data.get("dest_splitter") is not None
                                }
                                
                                # Handle orientation dictionary (convert tuple keys to strings)
                                if "orientation" in path_data:
                                    serializable_orientation = {}
                                    for pos_tuple, direction in path_data["orientation"].items():
                                        # Convert tuple key to string representation
                                        pos_key = str(pos_tuple)
                                        serializable_orientation[pos_key] = direction
                                    serializable_path["orientation"] = serializable_orientation
                                    
                                serializable_item_data["paths"][path_item].append(serializable_path)
                                    
                    serializable_output_info[item] = serializable_item_data
            
            # Create serializable versions of paths data
            serializable_paths = {}
            for item_key, item_paths in paths.items():
                serializable_paths[item_key] = []
                
                for path_data in item_paths:
                    # Create a copy of the path data without problematic objects
                    serializable_path = {
                        "path": path_data.get("path", []),
                        "start": path_data.get("start"),
                        "destination": path_data.get("destination"),
                        "underground_segments": path_data.get("underground_segments", {}),
                        "start_splitter": path_data.get("start_splitter") is not None,
                        "dest_splitter": path_data.get("dest_splitter") is not None
                    }
                    
                    # Handle orientation dictionary (convert tuple keys to strings)
                    if "orientation" in path_data:
                        serializable_orientation = {}
                        for pos_tuple, direction in path_data["orientation"].items():
                            # Convert tuple key to string representation
                            pos_key = str(pos_tuple)
                            serializable_orientation[pos_key] = direction
                        serializable_path["orientation"] = serializable_orientation
                    
                    serializable_paths[item_key].append(serializable_path)
            
            # Create serializable versions of simple lists with tuples
            serializable_belt_points = []
            for belt in belt_point_information:
                serializable_belt_points.append(list(belt))
                
            serializable_assemblers = []
            for assembler in assembler_information:
                serializable_assemblers.append(list(assembler))
                
            serializable_inserters = []
            for inserter in inserter_information:
                serializable_inserters.append(list(inserter))
            
            # Process retrieval_points to make it serializable
            serializable_retrieval_points = {}
            if hasattr(self, 'retrieval_points') and self.retrieval_points:
                for key, data in self.retrieval_points.items():
                    serializable_retrieval_points[key] = {
                        "item": data.get("item"),
                        "destination": data.get("destination", []),
                        "start_points": data.get("start_points", [])
                    }
                    
                    # Handle inserter_mapping (convert tuple keys to strings)
                    if "inserter_mapping" in data and data["inserter_mapping"] is not None:
                        serializable_mapping = {}
                        for pos_str, pos_tuple in data["inserter_mapping"].items():
                            serializable_mapping[pos_str] = list(pos_tuple) if isinstance(pos_tuple, tuple) else pos_tuple
                        serializable_retrieval_points[key]["inserter_mapping"] = serializable_mapping
                    else:
                        serializable_retrieval_points[key]["inserter_mapping"] = None
            
            # Create serializable version of placed_inserter_information
            serializable_placed_inserters = {}
            if placed_inserter_information:
                for item_key, inserters in placed_inserter_information.items():
                    serializable_placed_inserters[item_key] = {}
                    for pos_str, pos_tuple in inserters.items():
                        serializable_placed_inserters[item_key][pos_str] = list(pos_tuple)
            
            # Create the data dictionary with all fields
            data = {
                "output_item": self.output_item,
                "max_ouput": self.calculate_max_output(),
                "amount": self.amount,
                "production_data": self.production_data,
                "grid_width": self.grid_width,
                "grid_height": self.grid_height,
                "input_items": self.input_items,
                "input_information": serializable_input_info,
                "output_information": serializable_output_info,
                "inserter_information": serializable_inserters,
                "belt_point_information": serializable_belt_points,
                "assembler_information": serializable_assemblers,
                "retrieval_points": serializable_retrieval_points,
                "obstacle_map": serializable_obstacle_map,
                "paths": serializable_paths,
                "placed_inserter_information": serializable_placed_inserters
            }
            
            # Ensure file path has a .json extension
            if not file_path.endswith('.json'):
                file_path += '.json'
                
            # Write data to the file as JSON
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            
            print(f"Production tree data successfully stored to {file_path}")
        
        except Exception as e:
            print(f"Failed to store production tree data: {e}")
            import traceback
            traceback.print_exc()
    
  
    
    def calculate_max_output(self):
        cycles_per_minute = 60 / self.item_lookup[self.output_item]["recipe"]["time"]
        return self.production_data[self.output_item]['assemblers']* self.machines_data["assemblers"]["crafting_speed"]* cycles_per_minute *self.item_lookup[self.output_item]["recipe"]["yield"]
    
    
    
    def count_assemblers(self,production_data):
        assembler_counts = {}
    
        for item, data in production_data.items():
            # Check if the item has an 'assemblers' field
            if 'assemblers' in data:
                assembler_counts[item] = data['assemblers']
            else:
                assembler_counts[item] = 0  # Assume 0 assemblers if the field is not present
        
        return assembler_counts
        
    def visualize_factory(self, paths=None, placed_inserter_information=None, cell_size=50, store=False, file_path=None):
        """
        Visualize the factory layout with paths, splitters, and underground belts.
        
        Args:
            paths: Dictionary of paths information from MultiAgentPathfinder
            placed_inserter_information: Additional inserters to draw
            cell_size: Size of each grid cell in pixels
            store: Whether to save the visualization to a file
            file_path: Path where to save the visualization if store is True
        """
        print(f'visualizing factory layout')
        
        # Initialize pygame
        pygame.init()
        window_width = self.grid_width * cell_size
        window_height = self.grid_height * cell_size
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Factory Layout Visualization')
        clock = pygame.time.Clock()
        
        # Get factory layout data
        _, belt_point_information, assembler_information, inserter_information = self.z3_solver.build_map()
        
        # Process additional inserters from placed_inserter_information
        if placed_inserter_information:
            for item_key, inserters in placed_inserter_information.items():
                # Extract base item name
                item_name = item_key.split('_')[0] if '_' in item_key else item_key
                
                for pos_str, inserter_pos in inserters.items():
                    # Add to inserter_information
                    inserter_information.append((item_name, inserter_pos[0], inserter_pos[1]))
        
        # Load and scale images
        images = self._load_factory_images(cell_size)
        item_images = self._load_item_images(cell_size)
        
        # Draw factory views
        # 1. Base view without paths
            
        self._draw_factory_base(window, assembler_information, inserter_information, item_images, images, cell_size)
        self._draw_io_paths(window, images,item_images, cell_size)
        if store and file_path:
            pygame.image.save(window, file_path.replace('.png', '_no_paths.png'))
        

        # 2. Full view with paths
        self._draw_factory_paths(window, paths, images, item_images, cell_size)
        if store and file_path:
            pygame.image.save(window, file_path)
        
        
        #pygame.quit()

    def _load_factory_images(self, cell_size):
        """Load and scale factory element images."""
        images = {}
        
        # Load basic factory elements
        image_files = {
            'assembler': 'assets/assembler.png',
            'inserter': 'assets/inserter.png',
            'conveyor': 'assets/conveyor.png',
            'underground': 'assets/underground_belt.png',
            'splitter': 'assets/splitter.png'
        }
        
        for key, path in image_files.items():
            image = pygame.image.load(path)
            if key == 'assembler':
                images[key] = pygame.transform.scale(image, (3 * cell_size, 3 * cell_size))
            elif key == 'splitter':
                # Splitter will be scaled when used based on orientation
                images[key] = image
            else:
                images[key] = pygame.transform.scale(image, (cell_size, cell_size))
        
        return images

    def _load_item_images(self, cell_size):
        """Load and scale item images."""
        item_images = {}
        assets_folder = 'assets'
        excluded_images = {'assembler.png', 'inserter.png', 'conveyor.png', 'underground_belt.png', 'splitter.png'}
        
        for filename in os.listdir(assets_folder):
            if filename.endswith('.png') and filename not in excluded_images:
                item_path = os.path.join(assets_folder, filename)
                image = pygame.image.load(item_path)
                item_images[filename[:-4]] = pygame.transform.scale(image, (cell_size, cell_size))
        
        return item_images

    def _draw_factory_base(self, window, assembler_information, inserter_information, item_images, images, cell_size):
        """Draw the base factory layout without paths."""
        # Fill background
        window.fill(WHITE)
        
        # Draw grid lines
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(window, BLACK, rect, 1)
        
        # Draw assemblers
        for assembler_item, assembler_x, assembler_y in assembler_information:
            pixel_x = assembler_x * cell_size
            pixel_y = assembler_y * cell_size
            window.blit(images['assembler'], (pixel_x, pixel_y))
            
            # Draw item icon on top of assembler
            if assembler_item in item_images:
                window.blit(item_images[assembler_item], (pixel_x, pixel_y))
        
        # Draw inserters
        for inserter_item, inserter_x, inserter_y in inserter_information:
            pixel_x = inserter_x * cell_size
            pixel_y = inserter_y * cell_size
            window.blit(images['inserter'], (pixel_x, pixel_y))
            
            # Draw small item icon on inserter
            if inserter_item in item_images:
                original_image = item_images[inserter_item]
                quarter_size = (original_image.get_width() // 2, original_image.get_height() // 2)
                scaled_image = pygame.transform.scale(original_image, quarter_size)
                
                # Position in bottom-right corner of inserter
                inserter_width = images['inserter'].get_width()
                inserter_height = images['inserter'].get_height()
                scaled_width = scaled_image.get_width()
                scaled_height = scaled_image.get_height()
                corner_x = pixel_x + inserter_width - scaled_width
                corner_y = pixel_y + inserter_height - scaled_height
                window.blit(scaled_image, (corner_x, corner_y))

    def _draw_factory_paths(self, window, paths, images, item_images, cell_size):
        """Draw paths on top of the base factory layout."""
        

    
        
        if not paths:
            return
        
        for item_key, item_paths in paths.items():
            # Extract base item name
            item_name = item_key.split('_')[0] if '_' in item_key else item_key
            
            # Determine if this is an output path
            is_output_path = item_name == self.output_item
            
            # Draw each path for this item
            for path_data in item_paths:
                path = path_data.get('path', [])
                if not path:
                    continue
                
                # Draw normal belt segments
                self._draw_belt_segments(window, path, path_data, is_output_path, images, cell_size)
                
                # Draw underground segments
                self._draw_underground_segments(window, path_data, is_output_path, images, cell_size)
                
                # Draw splitters
                self._draw_splitters(window, path_data, item_name, item_images, images, cell_size)
                

    def _draw_belt_segments(self, window, path, path_data, is_output_path, images, cell_size):
        """Draw normal belt segments of a path."""
        
        print(f"Drawing path: {path}")
        print(f"Underground segments: {path_data.get('underground_segments', {})}")
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Skip if this is part of an underground segment
            is_underground = False
            if 'underground_segments' in path_data and path_data['underground_segments']:
                for segment_id, segment in path_data['underground_segments'].items():
                    # Skip if current segment is an underground belt segment
                    if (current == segment['start'] and next_node == segment['end']) or \
                    (current == segment['end'] and next_node == segment['start']):
                        is_underground = True
                        break
                    
                    # Also skip intermediate points in the underground path
                    if segment['start'][0] == segment['end'][0]:  # Vertical underground
                        min_y = min(segment['start'][1], segment['end'][1])
                        max_y = max(segment['start'][1], segment['end'][1])
                        if current[0] == segment['start'][0] and min_y < current[1] < max_y:
                            is_underground = True
                            break
                    elif segment['start'][1] == segment['end'][1]:  # Horizontal underground
                        min_x = min(segment['start'][0], segment['end'][0])
                        max_x = max(segment['start'][0], segment['end'][0])
                        if current[1] == segment['start'][1] and min_x < current[0] < max_x:
                            is_underground = True
                            break
            
            if is_underground:
                print(f"Skipping underground segment: {current} -> {next_node}")
                continue
            
            # Calculate direction
            dx = next_node[0] - current[0]
            dy = next_node[1] - current[1]
            
            # Normalize direction
            magnitude = abs(dx) + abs(dy)
            if magnitude > 0:
                dx = dx // magnitude
                dy = dy // magnitude    
                
             # Calculate angle based on direction
            if dx == 1:
                angle = 270  # Right
            elif dx == -1:
                angle = 90   # Left
            elif dy == 1:
                angle = 180  # Down
            elif dy == -1:
                angle = 0    # Up
            else:
                # Default to 0 if no valid direction found
                continue  # Skip if not a direct connection
        
                # Flip direction for output paths
               
            rotated_belt = pygame.transform.rotate(images['conveyor'], angle)
            print(f"Drawing belt at: {current} with angle: {angle}")
            window.blit(rotated_belt, (current[0] * cell_size, current[1] * cell_size))
            
        # Also draw the last segment of the path - if it's not part of an underground
        if len(path) > 0:
            last_pos = path[-1]
            is_last_underground = False
            
            # Check if the last position is part of an underground segment
            if 'underground_segments' in path_data and path_data['underground_segments']:
                for segment_id, segment in path_data['underground_segments'].items():
                    if last_pos == segment['start'] or last_pos == segment['end']:
                        is_last_underground = True
                        break
            
            # Draw the final belt if it's not underground
            if not is_last_underground and len(path) > 1:
                second_last = path[-2]
                # Calculate direction from second-last to last
                dx = second_last[0] - last_pos[0]
                dy = second_last[1] - last_pos[1]
                
                if is_output_path:
                    dx = last_pos[0] - second_last[0]
                    dy = last_pos[1] - second_last[1]
                
                # Normalize
                magnitude = abs(dx) + abs(dy)
                if magnitude > 0:
                    dx = dx // magnitude
                    dy = dy // magnitude
                    
                # Calculate angle
                if dx == 1:
                    angle = 270  # Right
                elif dx == -1:
                    angle = 90   # Left
                elif dy == 1:
                    angle = 180  # Down
                elif dy == -1:
                    angle = 0    # Up
                else:
                    print(f"Invalid direction from {second_last} to {last_pos}")
                    return  # Skip if not a direct connection
                
                # Apply appropriate rotation based on whether it's an input or output path
                if is_output_path:
                    rotated_belt = pygame.transform.rotate(images['conveyor'], angle)
                else:
                    rotated_belt = pygame.transform.rotate(images['conveyor'], (angle + 180) % 360)
                
                print(f"Drawing final belt at: {last_pos} with angle: {angle}")
                window.blit(rotated_belt, (last_pos[0] * cell_size, last_pos[1] * cell_size))

    def _is_underground_segment(self, pos1, pos2, path_data):
        """Check if a segment between two positions is part of an underground belt."""
        if 'underground_segments' not in path_data or not path_data['underground_segments']:
            return False
        
        for segment_id, segment in path_data['underground_segments'].items():
            if pos1 == segment['start'] or pos2 == segment['end']:
                return True
        
        return False

    def _draw_underground_segments(self, window, path_data, is_output_path, images, cell_size):
        """Draw underground belt segments."""
        if 'underground_segments' not in path_data or not path_data['underground_segments']:
            return
        
        print(f"Drawing underground segments: {path_data['underground_segments']}")
        
        for segment_id, segment in path_data['underground_segments'].items():
            
            start_pos = segment['start']
            end_pos = segment['end']
            
            # Calculate direction
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
  
            # Normalize direction for non-zero components
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)

            # Determine angles based on direction
            if dx == 0 and dy == 1:    # Down
                belt_angle = 180       # Belt angle for down direction
                entrance_angle = 270
                exit_angle = 90
            elif dx == 1 and dy == 0:  # Right
                belt_angle = 270       # Belt angle for right direction
                entrance_angle = 0
                exit_angle = 180
            elif dx == 0 and dy == -1: # Up
                belt_angle = 0         # Belt angle for up direction
                entrance_angle = 90
                exit_angle = 270
            elif dx == -1 and dy == 0: # Left
                belt_angle = 90        # Belt angle for left direction
                entrance_angle = 180
                exit_angle = 0
            else:
                print(f"Invalid underground direction: dx={dx}, dy={dy}")
                continue  # Skip invalid directions
                
                
        
            # First draw conveyor belts underneath both entrance and exit
            conveyor_entrance = pygame.transform.rotate(images['conveyor'], belt_angle)
            conveyor_exit = pygame.transform.rotate(images['conveyor'], belt_angle)
            
            # Draw the belts underneath
            window.blit(conveyor_entrance, (start_pos[0] * cell_size, start_pos[1] * cell_size))
            window.blit(conveyor_exit, (end_pos[0] * cell_size, end_pos[1] * cell_size))
        
            # Draw entrance and exit
            entrance_belt = pygame.transform.rotate(images['underground'], entrance_angle)
            exit_belt = pygame.transform.rotate(images['underground'], exit_angle)
            exit_belt = pygame.transform.flip(exit_belt, False, True)  # Flip horizontally
            
            window.blit(entrance_belt, (start_pos[0] * cell_size, start_pos[1] * cell_size))
            window.blit(exit_belt, (end_pos[0] * cell_size, end_pos[1] * cell_size))

    def _draw_splitters(self, window, path_data, item_name, item_images, images, cell_size):
        """Draw splitters associated with a path."""
        # Draw start splitter if present
        if 'start_splitter' in path_data and path_data['start_splitter']:
            self._draw_single_splitter(window, path_data['start_splitter'], item_name, item_images, images, cell_size)
        
        # Draw destination splitter if present
        if 'dest_splitter' in path_data and path_data['dest_splitter']:
            self._draw_single_splitter(window, path_data['dest_splitter'], item_name, item_images, images, cell_size)


    def _draw_single_splitter(self, window, splitter, item_name, item_images, images, cell_size):
        """Draw a single splitter with correct orientation."""
        if not hasattr(splitter, 'position') or not hasattr(splitter, 'direction'):
            return
        
        pos = splitter.position
        direction = splitter.direction

        if pos is None or direction is None:
            return
        

            
        # Calculate angle based on direction vector
        # Default orientation of the splitter asset is horizontal (facing right to left)
        # Apply additional 180 degree rotation to match expected orientation
        if direction[0] == 1 and direction[1] == 0:  # Right
            angle = 0  # Horizontal facing right
        elif direction[0] == 0 and direction[1] == 1:  # Down
            angle = 90  # Vertical facing down
        elif direction[0] == 0 and direction[1] == -1:  # Up
            angle = 270  # Vertical facing up
        else:  # Left or default
            angle = 180  # Horizontal facing left
        
        # Create a scaled version of the splitter image
        splitter_img = pygame.transform.scale(images['splitter'], (cell_size, cell_size))
        
        # Rotate the scaled image
        rotated_splitter = pygame.transform.rotate(splitter_img, angle)
        
        # Get the size of the rotated image (might be different due to rotation)
        rot_width, rot_height = rotated_splitter.get_size()
        
        # Calculate pixel coordinates based on rotation
        if angle == 0 or angle == 180:  # Horizontal orientation
            pixel_x = pos[0] * cell_size
            pixel_y = pos[1] * cell_size
        else:  # Vertical orientation (90 or 270)
            # For vertical orientation, center the splitter on the position
            pixel_x = pos[0] * cell_size - cell_size // 2
            pixel_y = pos[1] * cell_size
        
        # Draw the splitter
        window.blit(rotated_splitter, (pixel_x, pixel_y))
        
        # Draw a small item icon on the splitter to indicate what item it handles
        if item_name in item_images:
            # Scale down the item image
            original_image = item_images[item_name]
            quarter_size = (original_image.get_width() // 2, original_image.get_height() // 2)
            scaled_image = pygame.transform.scale(original_image, quarter_size)
            
            # Position the item icon on the center of the splitter
            icon_x = pixel_x + (rot_width - scaled_image.get_width()) // 2
            icon_y = pixel_y + (rot_height - scaled_image.get_height()) // 2
            window.blit(scaled_image, (icon_x, icon_y))
            
            
    def _draw_io_paths(self, window, images,item_images, cell_size):

        
        if self.input_information:
                for row in range(self.grid_height):
                    for col in range(self.grid_width):
                        rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                        
                        # Draw input info points
                        for item in self.input_information:
                            if self.input_information[item]['input'] == (row, col):
                                # Draw green background for input
                                pygame.draw.rect(window, GREEN, rect)
                                window.blit(item_images[item], rect)
                            elif self.input_information[item]['output'] == (row, col):
                                # Draw red background for output
                                pygame.draw.rect(window, RED, rect)
                                window.blit(item_images[item], rect)
                        
                        # Draw output info points
                        if self.output_information:
                            for item in self.output_information:
                                if self.output_information[item]['input'] == (row, col):
                                    # Draw green background for input
                                    pygame.draw.rect(window, GREEN, rect)
                                    window.blit(item_images[item], rect)
                                elif self.output_information[item]['output'] == (row, col):
                                    # Draw red background for output
                                    pygame.draw.rect(window, RED, rect)
                                    window.blit(item_images[item], rect) 
        
        """Draw the manually defined input/output paths."""
        # Draw input paths
        if self.input_information:
            for item, data in self.input_information.items():
                if data['paths'] is not None and item in data['paths']:
                    for path_data in data['paths'][item]:
                        path = path_data['path']
                        if not path or len(path) < 2:
                            continue
                        
                        # Draw belts for each segment of the path
                        for i in range(len(path) - 2):
                            current = path[i+1]
                            next_node = path[i + 2]
                            
                            # Calculate direction vector
                            dx = next_node[0] - current[0]
                            dy = next_node[1] - current[1]
                            
                            
                            # Calculate angle based on direction
                            if dx == 1:
                                angle = 270  # Right
                            elif dx == -1:
                                angle = 90   # Left
                            elif dy == 1:
                                angle = 180  # Down
                            elif dy == -1:
                                angle = 0    # Up
                            else:
                                continue  # Skip if not a direct connection
                            
                            # For input paths, use the standard direction
                            rotated_belt = pygame.transform.rotate(images['conveyor'],angle)
                            window.blit(rotated_belt, (current[0] * cell_size, current[1] * cell_size))
        
        # Draw output paths
        if self.output_information:
            for item, data in self.output_information.items():
                if data['paths'] is not None and item in data['paths']:
                    for path_data in data['paths'][item]:
                        path = path_data['path']
                        if not path or len(path) < 2:
                            continue
                        
                        # Draw belts for each segment of the path
                        for i in range(len(path) - 2):
                            current = path[i+1]
                            next_node = path[i + 2]
                            
                            # Calculate direction vector
                            dx = next_node[0] - current[0]
                            dy = next_node[1] - current[1]
                            
                    
                            #   # Calculate angle based on direction
                            if dx == 1:
                                angle = 270  # Right
                            elif dx == -1:
                                angle = 90   # Left
                            elif dy == 1:
                                angle = 180  # Down
                            elif dy == -1:
                                angle = 0    # Up
                            else:
                                continue  # Skip if not a direct connection
                            
                            # For output paths, use the standard direction (not reversed)
                            rotated_belt = pygame.transform.rotate(images['conveyor'], angle)
                            window.blit(rotated_belt, (current[0] * cell_size, current[1] * cell_size))
                            

def plot_csv_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure relevant columns are numeric
    df['Execution Time (seconds)'] = pd.to_numeric(df['Execution Time (seconds)'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Parse the 'Assemblers' column as a dictionary (if it's a string representation of a dictionary)
    df['Assemblers'] = df['Assemblers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Extract the number of assemblers (e.g., counting the number of items in the dictionary)
    df['Assembler Count'] = df['Assemblers'].apply(lambda x: sum(x.values()) if isinstance(x, dict) else 0)

    # Get unique items and methods
    items = df['Item'].unique()
    methods = df['Method'].unique()

    # Create directory for saving plots
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each item
    for item in items:
        for method in methods:
            # Get the data for the current item and method
            item_method_data = df[(df['Item'] == item) & (df['Method'] == method)]
            
            # Create directory for saving plots for each item and method
            method_plot_dir = os.path.join(output_dir, item, method)
            os.makedirs(method_plot_dir, exist_ok=True)

            # Plot violin plot
            plt.figure(figsize=(8, 6))
            sns.violinplot(x='Assembler Count', y='Execution Time (seconds)', hue='Minimizer', data=item_method_data, palette={1: 'red', 0: 'blue'}, legend=False)
            method_type = 'Building Belts' if method.lower() == 'build_belts' else 'Solving'
            plt.title(f'{item} - {method_type} (Violin Plot)')
            plt.xlabel('Number of Assemblers')
            plt.ylabel('Execution Time (seconds)')
            plt.grid(True)
            
            # Save the violin plot
            violin_plot_path = os.path.join(method_plot_dir, f'{item}_{method}_violin_plot.png')
            plt.savefig(violin_plot_path)
            plt.close()  # Close the plot to prevent overlap with other subplots

            # Plot boxplot
            plt.figure(figsize=(8, 6))
            ax = sns.boxplot(x='Assembler Count', y='Execution Time (seconds)', hue='Minimizer', data=item_method_data, palette={1: 'red', 0: 'blue'}, legend=False)
            ax.set_yscale('log')  # Set y-axis to log scale plt.title(f'{item} - {method_type} (Boxplot)')
            plt.xlabel('Number of Assemblers')
            plt.ylabel('Execution Time (seconds)')
            plt.grid(True)
            
            # Save the boxplot
            box_plot_path = os.path.join(method_plot_dir, f'{item}_{method}_box_plot.png')
            plt.savefig(box_plot_path)
            plt.close()  # Close the plot to prevent overlap with other subplots

    print(f"Plots saved in {output_dir}")



# Function to log method execution times with additional information
def log_method_time(item, amount, minimizer, method_name, assembler_counts,start_time, end_time):
    execution_time = end_time - start_time
    logging.info(f"Execution time for {method_name}: {execution_time:.4f} seconds.")
    
    # Open the CSV file and append the data
    try:
        with open("execution_times.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([item, amount, minimizer, method_name,assembler_counts,execution_time])
    except Exception as e:
        logging.error(f"Error logging execution time for {method_name}: {e}")
        
        
def main():
    
    Simple_Run()
    
    #Eval_Runs("big-electric-pole",start=50,end=500,step=50,rep_per_step=10)
   

   
def Simple_Run():
    # Set up logging for better output in the console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    print("start")
    
    
    # Example item and amount
    item_to_produce = "electronic-circuit"
    amount_needed = 120
    
    
    input_items = ["iron-plate", "copper-cable"]  # Using explicit input items
    
    # init 
    factorioProductionTree = FactorioProductionTree(16,10)
    factorioProductionTree.amount = amount_needed
    production_data  = factorioProductionTree.calculate_production(item_to_produce,amount_needed,input_items=input_items) #60
    factorioProductionTree.production_data = production_data

    production_data = factorioProductionTree.set_capacities(production_data)
    
    
    # Manual input and output
    factorioProductionTree.manual_Input()
    factorioProductionTree.manual_Output()


    
    factorioProductionTree.add_manual_IO_constraints(production_data)
    
    assembler_counts = factorioProductionTree.count_assemblers(production_data)
        
    # Track time for solving the problem
    start_time = time.perf_counter()
    factorioProductionTree.solve(production_data,sequential=False)
    end_time = time.perf_counter()
    log_method_time(item_to_produce, 1, 1, "solve", assembler_counts, start_time, end_time)
    
    
    start_time = time.perf_counter()
    paths, placed_inserter_information = factorioProductionTree.build_belts(max_tries=2)
    end_time = time.perf_counter()
    log_method_time(item_to_produce, 1, 1, "build_belts", assembler_counts, start_time, end_time)
    
    print(f'paths: {paths}')
  
    
    if(paths):
        print("saving data")
        factorioProductionTree.store_data(f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module',paths,placed_inserter_information)
        
        factorioProductionTree.visualize_factory(paths,placed_inserter_information,store=True,file_path=f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module.png')
        pass

   

   
def Eval_Runs(item_to_produce, start, end, step, rep_per_step):
    # Initialize the production tree
    factorioProductionTree = FactorioProductionTree(15, 15)
    
    # Always start with a simulation for 1 unit
    production_data = factorioProductionTree.calculate_production(item_to_produce, 1)
    
    # Set manual input and output
    factorioProductionTree.manual_Input()
    factorioProductionTree.manual_Output()
    
    
    for rep in range(rep_per_step):
        factorioProductionTree.destroy_solver()
        
        print(f"\nRun {rep + 1}/{rep_per_step} for {1} unit of {item_to_produce}\n")
      
        # Set capacities and get initial assembler count
        production_data = factorioProductionTree.set_capacities(production_data)
        assembler_counts = factorioProductionTree.count_assemblers(production_data)
        
        factorioProductionTree.add_manual_IO_constraints(production_data, sequential=False)
        
        # Solve the production problem for amount_needed = 1
        start_time = time.perf.counter()
        factorioProductionTree.solve(production_data, sequential=False)
        end_time = time.perf.counter()
        log_method_time(item_to_produce, 1, 1, "solve", assembler_counts, start_time, end_time)
        
        # Build belts and visualize for amount_needed = 1
        start_time = time.perf.counter()
        paths, placed_inserter_information = factorioProductionTree.build_belts(max_tries=2)
        end_time = time.perf.counter()
        log_method_time(item_to_produce, 1, 1, "build_belts", assembler_counts, start_time, end_time)
    

    
    # Loop through different amounts
    for amount_needed in range(start, end, step):
        for rep in range(rep_per_step):
            print(f"\nRun {rep + 1}/{rep_per_step} for {amount_needed} units of {item_to_produce}\n")
            
            # Reset solver for a clean state
            factorioProductionTree.destroy_solver()
            
            # Recalculate production data with the new amount
            production_data = factorioProductionTree.calculate_production(item_to_produce, amount_needed)
            production_data = factorioProductionTree.set_capacities(production_data)
            assembler_counts = factorioProductionTree.count_assemblers(production_data)
            
            # Reapply manual IO constraints
            factorioProductionTree.add_manual_IO_constraints(production_data, sequential=False)
            
            # Solve the production problem
            start_time = time.perf.counter()
            factorioProductionTree.solve(production_data, sequential=False)
            end_time = time.perf.counter()
            log_method_time(item_to_produce, amount_needed, 1, "solve", assembler_counts, start_time, end_time)
            
            # Build belts and optionally visualize the factory
            start_time = time.perf.counter()
            paths, placed_inserter_information = factorioProductionTree.build_belts(max_tries=2)
            end_time = time.perf.counter()
            log_method_time(item_to_produce, amount_needed, 1, "build_belts", assembler_counts, start_time, end_time)
            
           
        
   
   
   
def test_build_belts():
    """Test function for the build_belts method of FactorioProductionTree."""
    # Set up logging for better output in the console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    print("Starting build_belts test")
    
    # Create a small test instance
    item_to_produce = "electronic-circuit"
    amount_needed = 120
    input_items = ["iron-plate", "copper-plate"]  # Using explicit input items

    
    # Initialize with a small grid for faster testing
    factorio_tree =  FactorioProductionTree(16,10)
    
    
    # Calculate production data
    production_data = factorio_tree.calculate_production(item_to_produce, amount_needed, input_items=input_items)
    factorio_tree.production_data = production_data
    production_data = factorio_tree.set_capacities(production_data)
    
    print("Production data:", production_data)
    
   
    # Setup minimal I/O points manually instead of using the GUI
    # This creates a simple test case with fixed input/output positions
    input_information = {
        "iron-plate": {
            "input": (0, 9),
            "output": (0, 0),
            "paths": {"iron-plate": [{
                "path": [(0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0)],
                "underground_segments": {},
                "start_splitter": None,
                "dest_splitter": None
            }]}
        },
        "copper-plate": {
            "input": (12, 9),
            "output": (12, 0),
            "paths": {"copper-plate": [{
                "path": [(12, 9), (12, 8), (12, 7), (12, 6), (12, 5), (12, 4), (12, 3), (12, 2), (12, 1), (12, 0)],
                "underground_segments": {},
                "start_splitter": None,
                "dest_splitter": None
            }]}
        }
    }

    output_information = {
        "electronic-circuit": {
            "input": (15, 0),
            "output": (15, 9),
            "paths": {"electronic-circuit": [{
                "path": [(15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9)],
                "underground_segments": {},
                "start_splitter": None,
                "dest_splitter": None
            }]}
        }
    }
    
    # Set the input and output information
    factorio_tree.input_information = input_information
    factorio_tree.output_information = output_information
    factorio_tree.input_items = input_items
    factorio_tree.output_item = item_to_produce
    
    # Add manual IO constraints
    factorio_tree.add_manual_IO_constraints(production_data)
    
    # Solve the initial layout problem
    factorio_tree.solve(production_data, sequential=False)
    
    # Test each component of build_belts separately for debugging
    print("\nTesting build_map...")
    obstacle_map, belt_point_information, assembler_information, _ = factorio_tree.z3_solver.build_map()
    print(f"Obstacle map shape: {len(obstacle_map)}x{len(obstacle_map[0])}")
    print(f"Belt points count: {len(belt_point_information)}")
    print(f"Assembler count: {len(assembler_information)}")
    
    print("\nTesting belt overlap detection...")
    filtered_belts = [belt for belt in belt_point_information if not factorio_tree.detect_belt_overlap(belt)]
    print(f"Belts after overlap filter: {len(filtered_belts)}")
    
    print("\nTesting assembler overlap detection...")
    filtered_belts = [belt for belt in filtered_belts if not factorio_tree.detect_assembler_overlap(belt, assembler_information)]
    print(f"Belts after assembler overlap filter: {len(filtered_belts)}")
    
    print("\nTesting retrieval points calculation...")
    retrieval_points = factorio_tree.get_retrieval_points(filtered_belts, assembler_information)
    print(f"Retrieval points count: {len(retrieval_points)}")
    
    print("\nTesting output point information...")
    retrieval_points.update(factorio_tree.add_out_point_information(item_to_produce, assembler_information))
    print(f"Total retrieval points after adding output: {len(retrieval_points)}")
    
    print("\nTesting dictionary rearrangement...")
    rearranged_points = factorio_tree.rearrange_dict(retrieval_points, item_to_produce)
    print(f"Rearranged retrieval points count: {len(rearranged_points)}")
    
    print("\nNow running the full build_belts method...")
    try:
        paths, inserters = factorio_tree.build_belts(max_tries=1)
        print("\nBuild belts completed successfully!")
        print(f"Paths: {paths}")
        print(f"Inserters: {inserters}")
        
        # Visualize the result if successful
        if paths:
            factorio_tree.visualize_factory(paths, inserters, store=True, file_path="test_build_belts_result.png")
            print("Visualization saved to test_build_belts_result.png")
        
        return True
    except Exception as e:
        print(f"ERROR in build_belts: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the test

if __name__ == "__main__":
    #test_build_belts()
    
    # Prepare CSV file header if not exists
    #if not os.path.exists("execution_times.csv"):
    #    try:
    #        with open("execution_times.csv", "w", newline="") as file:
    #            writer = csv.writer(file)
    #            writer.writerow(["Item", "Amount", "Minimizer", "Method","Assemblers", "Execution Time (seconds)"])
    #    except Exception as e:
    #        logging.error(f"Error initializing CSV file: {e}")

    #plot_csv_data("execution_times.csv")
    main()