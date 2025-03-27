#! .venv\Scripts\python.exe

import json
import logging
import pygame
import math
import ast, os
import csv
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from z3 import And , Or
from z3Solver  import Z3Solver
from AStarPathFinderold import AStarPathFinderold
import seaborn as sns

from AStarPathfinder import AStarPathFinder

# Define constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
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

    
    def manual_Output(self,Title="Manual Output"):
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
    

        output_information = {self.output_item:{'input': None, 'output': None,'grid': None, 'direction_grid': None}}
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

        pygame.display.set_caption(Title)

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
    
    def manual_Input(self,Title="Manual Input"):
        
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

        pygame.display.set_caption(Title)

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
            
            
            
            try:
                
                # rearrange such that we first build paths for outputs
                retrieval_points = self.rearrange_dict(retrieval_points, self.output_item)
               
                astar_pathfinder = AStarPathFinder(self.obstacle_map,retrieval_points,invert_paths=[self.output_item])
                paths , placed_inserter_information = astar_pathfinder.find_path_for_item()

                
                return paths, placed_inserter_information
            
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
            self.input_information = data.get("input_information")
            self.output_item = data.get("output_item")
            self.output_information = data.get("output_information")
            self.inserter_information = data.get("inserter_information", [])
            self.belt_point_information = data.get("belt_point_information", [])
            self.assembler_information = data.get("assembler_information", [])
            self.retrieval_points = data.get("retrieval_points", {})
            self.obstacle_map = data.get("obstacle_map",[])
            self.paths = data.get("paths", [])

            print(f"Production tree data successfully loaded from {file_path}")

        except Exception as e:
            print(f"Failed to load production tree data: {e}")      
            
    def store_data(self, file_path, paths, placed_inserter_information):
        try:
            _, belt_point_information, assembler_information, inserter_information = self.z3_solver.build_map()
            
            inserter_information = inserter_information + placed_inserter_information
            
            # Make sure paths is JSON serializable
            serializable_paths = {}
            for key, path_data in paths.items():
                serializable_paths[key] = {}
                for inner_key, inner_value in path_data.items():
                    if inner_key == "path":
                        # Make sure path is a list of lists
                        serializable_paths[key][inner_key] = [list(point) for point in inner_value]
                    elif inner_key == "direction_grid":
                        # Make sure direction_grid is a standard Python list
                        serializable_paths[key][inner_key] = [[list(d) if d else None for d in row] for row in inner_value]
                    elif inner_key == "underground_paths":
                        # Make sure underground_paths is serializable
                        serializable_underground = []
                        for entry in inner_value:
                            # Each entry might be (entrance, exit, direction)
                            if len(entry) == 3:
                                entrance, exit_point, direction = entry
                                serializable_underground.append([
                                    list(entrance), 
                                    list(exit_point), 
                                    list(direction) if direction else None
                                ])
                        serializable_paths[key][inner_key] = serializable_underground
                    else:
                        serializable_paths[key][inner_key] = inner_value
          
            # Create serializable retrieval points
            serializable_retrieval_points = None
            if hasattr(self, 'retrieval_points') and self.retrieval_points is not None:
                # Deep copy to avoid modifying the original
                serializable_retrieval_points = {}
                for key, value in self.retrieval_points.items():
                    serializable_retrieval_points[key] = {}
                    for inner_key, inner_value in value.items():
                        if isinstance(inner_value, list):
                            # Ensure list items are serializable
                            if inner_value and isinstance(inner_value[0], tuple):
                                serializable_retrieval_points[key][inner_key] = [list(item) for item in inner_value]
                            else:
                                serializable_retrieval_points[key][inner_key] = inner_value.copy()
                        elif isinstance(inner_value, dict):
                            # Handle dictionary with potential tuple keys
                            new_dict = {}
                            for k, v in inner_value.items():
                                if isinstance(v, tuple):
                                    new_dict[k] = list(v)
                                else:
                                    new_dict[k] = v
                            serializable_retrieval_points[key][inner_key] = new_dict
                        else:
                            serializable_retrieval_points[key][inner_key] = inner_value
            
            # Convert obstacle_map to a standard Python list if it's a NumPy array
            serializable_obstacle_map = None
            if self.obstacle_map is not None:
                import numpy as np
                if isinstance(self.obstacle_map, np.ndarray):
                    serializable_obstacle_map = self.obstacle_map.tolist()
                else:
                    # If it's already a list or some other serializable type, use it directly
                    serializable_obstacle_map = self.obstacle_map
            
            data = {
                "output_item": self.output_item,
                "max_ouput": self.calculate_max_output(),
                "amount": self.amount,
                "production_data": self.production_data,
                "grid_width": self.grid_width,
                "grid_height": self.grid_height,
                "input_items": self.input_items,
                "input_information": self.input_information,
                "output_item": self.output_item,
                "output_information": self.output_information,
                "inserter_information": inserter_information,
                "belt_point_information": belt_point_information,
                "assembler_information": assembler_information,
                "retrieval_points": serializable_retrieval_points,
                "obstacle_map": serializable_obstacle_map,
                "paths": serializable_paths
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
            logging.error(f"Failed to store production tree data: {e}")
            # Print the full traceback for debugging
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
        # Helper function to draw the factory layout
        
        def draw_factory(window, draw_paths=True):
            window.fill(WHITE)
            # Draw grid lines and pre-configured input/output positions
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    pygame.draw.rect(window, BLACK, rect, 1)  # Draw grid lines
            
            # Draw A* pathfinder paths first (below other elements)
            if draw_paths and paths:
                for item, path_info in paths.items():
                    if not path_info:
                        continue
                        
                    path = path_info.get('path', [])
                    direction_grid = path_info.get('direction_grid', [])
                    
                    if not path or not direction_grid:
                        continue
                    
   
                    # Draw belts for each node in the path
                    for i in range(len(path) - 1):  # For each segment of the path
                        current = path[i]
                        next_node = path[i + 1]
                        
                        # Calculate direction vector
                        dx = next_node[0] - current[0]
                        dy = next_node[1] - current[1]
                        
                        # Normalize direction to unit vector
                        magnitude = abs(dx) + abs(dy)
                        if magnitude > 0:
                            dx = dx // magnitude
                            dy = dy // magnitude
                        
                        # Draw the conveyor belt with correct direction
                        direction = (dx, dy)
                        if direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Basic directions
                            # Get the right rotation
                            if direction == (0, 1):  # down
                                angle = 180
                            elif direction == (1, 0):  # right
                                angle = 270
                            elif direction == (0, -1):  # up
                                angle = 0
                            elif direction == (-1, 0):  # left
                                angle = 90
                            
                            if item.split('_')[0]  != self.output_item:
                                rotated_belt = pygame.transform.rotate(conveyor_image,  (angle + 180) % 360)
                            else:
                                rotated_belt = pygame.transform.rotate(conveyor_image, angle)
                            window.blit(rotated_belt, (current[0] * cell_size, current[1] * cell_size))
                    
                    # Draw special markers for jumps (underground belts)
                    jump_markers = path_info.get('jump_markers', [])
                    for start, end in jump_markers:
                        # Calculate direction
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        magnitude = max(abs(dx), abs(dy))
                        
                        if magnitude > 0:
                            dx = dx // magnitude
                            dy = dy // magnitude
                        
                        # Rotate entrance belt
                        entrance_angle = 0
                        if (dx, dy) == (0, 1):
                            entrance_angle = 180
                        elif (dx, dy) == (1, 0):
                            entrance_angle = 270
                        elif (dx, dy) == (0, -1):
                            entrance_angle = 0
                        elif (dx, dy) == (-1, 0):
                            entrance_angle = 90
                        
                        # Rotate exit belt (opposite direction)
                        exit_angle = (entrance_angle + 180) % 360
                        
                        entrance_belt = pygame.transform.rotate(underground_image, entrance_angle)
                        exit_belt = pygame.transform.rotate(underground_image, exit_angle)
                        
                        #window.blit(entrance_belt, (start[0] * cell_size, start[1] * cell_size))
                        #window.blit(exit_belt, (end[0] * cell_size, end[1] * cell_size))
            
            # Draw input/output positions
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    for item_index, item in enumerate(self.input_information):
                        if self.input_information[item]['input'] == (row, col):
                            pygame.draw.rect(window, GREEN, rect)
                            window.blit(item_images[item], rect)
                        elif self.input_information[item]['output'] == (row, col):
                            pygame.draw.rect(window, RED, rect)
                            window.blit(item_images[item], rect)
                    for item_index, item in enumerate(self.output_information):
                        if self.output_information[item]['input'] == (row, col):
                            pygame.draw.rect(window, GREEN, rect)
                            window.blit(item_images[item], rect)
                        elif self.output_information[item]['output'] == (row, col):
                            pygame.draw.rect(window, RED, rect)
                            window.blit(item_images[item], rect)
            
            # Draw manually configured belts for input/output items
            for item, data in self.input_information.items():
                grid = data.get('grid')
                direction_grid = data.get('direction_grid')
                if grid is not None and direction_grid is not None:
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == 2:
                                direction = direction_grid[row][col]
                                if direction in direction_angle_map:
                                    angle = direction_angle_map[direction]
                                    rotated_belt_image = pygame.transform.rotate(conveyor_image, angle + 90)
                                    window.blit(rotated_belt_image, (col * cell_size, row * cell_size))
            
            for item, data in self.output_information.items():
                grid = data.get('grid')
                direction_grid = data.get('direction_grid')
                if grid is not None and direction_grid is not None:
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == 2:
                                direction = direction_grid[row][col]
                                if direction in direction_angle_map:
                                    angle = direction_angle_map[direction]
                                    rotated_belt_image = pygame.transform.rotate(conveyor_image, angle + 90)
                                    window.blit(rotated_belt_image, (col * cell_size, row * cell_size))
                
            # Draw assemblers
            for assembler_item, assembler_x, assembler_y in assembler_information:
                pixel_x = assembler_x * cell_size
                pixel_y = assembler_y * cell_size
                window.blit(assembler_image, (pixel_x, pixel_y))
                window.blit(item_images[assembler_item], (pixel_x, pixel_y))
                
            # Draw inserters
            for inserter_item, inserter_x, inserter_y in inserter_information:
                pixel_x = inserter_x * cell_size
                pixel_y = inserter_y * cell_size
                window.blit(inserter_image, (pixel_x, pixel_y))
                original_image = item_images[inserter_item]
                quarter_size = (original_image.get_width() // 2, original_image.get_height() // 2)
                scaled_image = pygame.transform.scale(original_image, quarter_size)
                inserter_width = inserter_image.get_width()
                inserter_height = inserter_image.get_height()
                scaled_width = scaled_image.get_width()
                scaled_height = scaled_image.get_height()
                corner_x = pixel_x + inserter_width - scaled_width
                corner_y = pixel_y + inserter_height - scaled_height
                window.blit(scaled_image, (corner_x, corner_y))

        # Initialize pygame and load assets
        _, belt_point_information, assembler_information, inserter_information = self.z3_solver.build_map()
        if placed_inserter_information:
            inserter_information = inserter_information + placed_inserter_information
        direction_angle_map = {'right': 90, 'down': 180, 'left': 270, 'up': 0}
        pygame.init()
        window_width = self.grid_width * cell_size
        window_height = self.grid_height * cell_size
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Factory Layout Visualization')
        clock = pygame.time.Clock()
        assembler_image = pygame.image.load('assets/assembler.png')
        inserter_image = pygame.image.load('assets/inserter.png')
        conveyor_image = pygame.image.load('assets/conveyor.png')
        underground_image = pygame.image.load("assets/underground_belt.png")

        assembler_image = pygame.transform.scale(assembler_image, (3 * cell_size, 3 * cell_size))
        inserter_image = pygame.transform.scale(inserter_image, (cell_size, cell_size))
        conveyor_image = pygame.transform.scale(conveyor_image, (cell_size, cell_size))
        underground_image = pygame.transform.scale(underground_image, (cell_size, cell_size))

        
        item_images = {}
        assets_folder = 'assets'
        excluded_images = {'assembler.png', 'inserter.png', 'conveyor.png', 'underground_belt_entrance.png', 'underground_belt_exit.png'}
        for filename in os.listdir(assets_folder):
            if filename.endswith('.png') and filename not in excluded_images:
                item_path = os.path.join(assets_folder, filename)
                image = pygame.image.load(item_path)
                item_images[filename[:-4]] = pygame.transform.scale(image, (cell_size, cell_size))
                
        # Draw the factory without paths
        draw_factory(window, draw_paths=False)
        if store and file_path:
            pygame.image.save(window, file_path.replace('.png', '_no_paths.png'))
            
        # Draw the factory with paths
        draw_factory(window, draw_paths=True)
        if store and file_path:
            pygame.image.save(window, file_path)
            
        # Draw the factory showing only input/output
        draw_factory(window, draw_paths=False)
        if store and file_path:
            pygame.image.save(window, file_path.replace('.png', '_input_output.png'))
            
        pygame.quit()

 
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
    item_to_produce = "copper-cable"
    amount_needed = 100
    
    
    input_items = []
    
    # init 
    factorioProductionTree = FactorioProductionTree(6,6)
    factorioProductionTree.amount = amount_needed
    production_data  = factorioProductionTree.calculate_production(item_to_produce,amount_needed,input_items=input_items) #60
    factorioProductionTree.production_data = production_data

    production_data = factorioProductionTree.set_capacities(production_data)

    production_data["copper-cable"]["input_inserters"] = [{'id': 'copper-plate', 'inserters': 4, 'amount': 1}]
    
    print(f"production data {production_data}")

   
   
    # Manual input and output
    factorioProductionTree.manual_Input()
    factorioProductionTree.manual_Output()
    factorioProductionTree.add_manual_IO_constraints(production_data,sequential=False)
   
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
    
    if(paths):
        factorioProductionTree.store_data(f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module',paths,placed_inserter_information)
        
        factorioProductionTree.visualize_factory(paths,placed_inserter_information,store=True,file_path=f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module.png')
        pass
        #print(factorioProductionTree.grid)
   

   
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
        start_time = time.perf_counter()
        factorioProductionTree.solve(production_data, sequential=False)
        end_time = time.perf_counter()
        log_method_time(item_to_produce, 1, 1, "solve", assembler_counts, start_time, end_time)
        
        # Build belts and visualize for amount_needed = 1
        start_time = time.perf_counter()
        paths, placed_inserter_information = factorioProductionTree.build_belts(max_tries=2)
        end_time = time.perf_counter()
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
            start_time = time.perf_counter()
            factorioProductionTree.solve(production_data, sequential=False)
            end_time = time.perf_counter()
            log_method_time(item_to_produce, amount_needed, 1, "solve", assembler_counts, start_time, end_time)
            
            # Build belts and optionally visualize the factory
            start_time = time.perf_counter()
            paths, placed_inserter_information = factorioProductionTree.build_belts(max_tries=2)
            end_time = time.perf_counter()
            log_method_time(item_to_produce, amount_needed, 1, "build_belts", assembler_counts, start_time, end_time)
            
           
        
        
    


   
if __name__ == "__main__":
    
    # Prepare CSV file header if not exists
    if not os.path.exists("execution_times.csv"):
        try:
            with open("execution_times.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Item", "Amount", "Minimizer", "Method","Assemblers", "Execution Time (seconds)"])
        except Exception as e:
            logging.error(f"Error initializing CSV file: {e}")

    #plot_csv_data("execution_times.csv")
    main()