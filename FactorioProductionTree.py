#! .venv\Scripts\python.exe

"""
Factorio Production Tree Module

This module provides a comprehensive framework for calculating, optimizing, and visualizing
Factorio production layouts. It handles the entire pipeline from recipe analysis to factory
visualization and blueprint generation.

Main Components:
- FactorioProductionTree: Core class for production planning and optimization
- Production calculation and recipe analysis
- SMT solver integration for layout optimization
- Multi-agent pathfinding for belt and pipe routing
- Pygame-based interactive GUI for manual I/O configuration
- Draftsman integration for Factorio blueprint generation
- PNG visualization and plotting capabilities

Key Features:
- Recursive production tree calculation with assembler, inserter, and belt requirements
- Factory layout optimization using Z3 SMT
- Automatic belt and pipe network generation with pathfinding
- Support for both fluid and solid item transport systems
- Power pole placement optimization
- Interactive manual configuration of input/output points
- Blueprint export compatible with Factorio game
- Performance analysis and execution time logging
- High-quality PNG plot generation for analysis and documentation
"""

import json
import pygame
import math
import ast, os
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from z3 import And , Or
from SMT_Solver  import SMTSolver
from GurobiSolver import GurobiSolver
import seaborn as sns
import numpy as np
from draftsman.data import entities
from draftsman.blueprintable import Blueprint
from draftsman.constants import Direction
from draftsman.entity import Inserter, AssemblingMachine, TransportBelt, UndergroundBelt , Pipe,UndergroundPipe, ElectricPole
from draftsman.utils import Vector
from draftsman.entity import Splitter as BlueprintSplitter

from MultiAgentPathfinder import MultiAgentPathfinder,Splitter
from logging_config import setup_logger

logger = setup_logger("FactorioProductionTree")

# Define constants for colors and GUI elements
side_panel_width = 300
CELL_SIZE = 50  # Default cell size for visualization
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
    """
    Main class for calculating, optimizing, and visualizing Factorio production layouts.
    
    This class handles the entire pipeline from recipe analysis to factory visualization:
    - Calculates production requirements for items
    - Optimizes factory layouts using SMT solvers
    - Generates belt networks and pathfinding
    - Creates visualizations and blueprints
    - Exports data for analysis
    
    Attributes:
        grid_width (int): Width of the factory grid
        grid_height (int): Height of the factory grid
        config (dict): Configuration settings loaded from config.json
        item_lookup (dict): Mapping of item IDs to their recipe data
        production_data (dict): Calculated production requirements
        z3_solver (SMTSolver): SMT solver for factory optimization
        AStar (MultiAgentPathfinder): Pathfinding algorithm for belt routing
        input_information (dict): Input/output points configuration
        output_information (dict): Output points configuration
    """
    
    def __init__(self,grid_width=15,grid_height=15) -> None:
        """
        Initialize the FactorioProductionTree with specified grid dimensions.
        
        Sets up the factory grid, loads configuration and recipe data, and initializes
        the necessary data structures for production planning.
        
        Args:
            grid_width (int, optional): Width of the factory grid. Defaults to 15.
            grid_height (int, optional): Height of the factory grid. Defaults to 15.
        """
        logger.info(f"Initializing FactorioProductionTree with grid size {grid_width}x{grid_height}")
        
        # Load configuration from config.json
        self.config = self.load_config()
        logger.info("Configuration loaded successfully")
        
        # Set grid dimensions, use config defaults if not specified
        self.grid_width = grid_width if grid_width is not None else self.config["grid"]["default_width"]
        self.grid_height = grid_height if grid_height is not None else self.config["grid"]["default_height"]
        logger.info(f"Grid dimensions set to {self.grid_width}x{self.grid_height}")
        
        # Load recipe and machine data from JSON files
        items_data = self.load_json("recipes.json")
        self.machines_data = self.load_json("machine_data.json")
        logger.info("Recipe and machine data loaded successfully")
        
        # Initialize the factory grid (2D array for layout planning)
        self.grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Create lookup dictionary for efficient item access
        self.item_lookup = {item["id"]: item for item in items_data}
        logger.info(f"Item lookup created with {len(self.item_lookup)} items")

        # Initialize production planning variables
        self.input_items=[]           # List of input items (raw materials)
        self.input_information = None # Input belt/pipe configurations
        
        self.amount = 0               # Target production amount
        self.output_item = None       # Main output item being produced
        self.output_information = None # Output belt/pipe configurations
        
        # Initialize solvers and pathfinding (created later when needed)
        self.z3_solver = None         # SMT solver for layout optimization
        self.AStar = None             # Multi-agent pathfinding algorithm
        
        self.obstacle_map = None      # Map of obstacles for pathfinding
        
        # Production data dictionary (populated by calculate_production)
        self.production_data = {}
        
        logger.info("FactorioProductionTree initialization complete")
    
    def load_config(self):
        """
        Load configuration settings from config.json file.
        
        Loads factory configuration including grid dimensions, machine types,
        belt settings, and other parameters. Falls back to default values
        if the config file is not found.
        
        Returns:
            dict: Configuration dictionary with all settings
            
        Raises:
            FileNotFoundError: If config.json is missing (handled gracefully)
        """
        logger.info("Loading configuration from config.json")
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
            logger.info("Configuration file loaded successfully")
            return config
        except FileNotFoundError:
            logger.warning("config.json not found, using default configuration")
            # Return default config if file doesn't exist
            default_config = {
                "grid": {"default_width": 16, "default_height": 10},
                "machines": {
                    "default_assembler": "assembling-machine-2",
                    "default_furnace": "electric-furnace",
                    "default_chemical_plant": "chemical-plant",
                    "default_refinery": "oil-refinery"
                },
                "inserters": {
                    "default_type": "inserter", 
                    "input_type": "fast-inserter",
                    "output_type": "fast-inserter"
                },
                "belts": {
                    "default_type": "transport-belt",
                    "underground_type": "underground-belt",
                    "underground_max_length": 4
                },
                "visualization": {
                    "cell_size": 50,
                    "show_grid_lines": True,
                    "save_images": True
                },
                "pathfinding": {
                    "allow_underground": True,
                    "allow_splitters": True,
                    "find_optimal_paths": True,
                    "max_tries": 3
                },
                "power": {
                    "place_power_poles": True
                }
            }
            logger.info("Default configuration created")
            return default_config
        
    def load_json(self, recipe_file):
        """
        Load JSON data from a specified file.
        
        Generic JSON loader used for recipes, machine data, and other
        configuration files.
        
        Args:
            recipe_file (str): Path to the JSON file to load
            
        Returns:
            list or dict: Parsed JSON data
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        logger.info(f"Loading JSON data from {recipe_file}")
        try:
            with open(recipe_file, "r") as file:
                recipes = json.load(file)
            logger.info(f"Successfully loaded {recipe_file}")
            return recipes
        except FileNotFoundError:
            logger.error(f"JSON file not found: {recipe_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {recipe_file}: {e}")
            raise
            
    def destroy_solver(self):
        """
        Clean up and destroy the current SMT solver instance.
        
        Used to reset the solver state between different optimization runs
        or when changing problem parameters.
        """
        logger.info("Destroying current solver instance")
        self.z3_solver = None
        logger.info("Solver instance destroyed")
    
    def calculate_production(self, item_id, items_per_minute, input_items=[], first=True):
        """
        Recursively calculate production requirements for a given item and production rate.
        
        This method analyzes the recipe tree for the target item and calculates the
        required production rates, machine counts, and material flows for the entire
        production chain.
        
        Args:
            item_id (str): ID of the item to produce
            items_per_minute (float): Target production rate in items per minute
            input_items (list, optional): List of items to treat as external inputs
            first (bool, optional): Whether this is the top-level call
            
        Returns:
            dict: Dictionary containing production data for all items in the chain,
                 with structure: {item_id: {amount_per_minute: float, assemblers: int, ...}}
        """
        logger.info(f"Calculating production for {item_id}: {items_per_minute} items/minute")
        
        # Set output item and input items list on first call
        if first:
            self.output_item = item_id
            self.input_items += input_items
            logger.info(f"Set output item to {item_id}, input items: {input_items}")
        
        # Look up the item in the recipe database
        item = self.item_lookup.get(item_id)
        
        # If the item doesn't exist, has no recipe, or is in input_items, treat as external input
        if not item or "recipe" not in item or item in input_items:
            logger.info(f"{item_id} treated as external input (no recipe or in input_items)")
            return {item_id: {"amount_per_minute": items_per_minute}}

        recipe = item["recipe"]
        time_per_unit = recipe.get("time")
        yield_per_recipe = recipe.get("yield")

        # If recipe has no ingredients or missing parameters, treat as base material
        if time_per_unit is None or yield_per_recipe is None or not recipe["ingredients"]:
            logger.info(f"{item_id} treated as base material (no ingredients or missing recipe data)")
            self.input_items.append(item_id)
            return {item_id: {"amount_per_minute": items_per_minute}}

        # Calculate how many recipe runs are needed per minute
        recipe_runs_needed_per_minute = items_per_minute / yield_per_recipe

        
        # Store total required amounts, including assemblers, inserters, and belts
        total_requirements = {
            item_id: {
                "amount_per_minute": items_per_minute,
                "assemblers": math.ceil(self._calculate_assemblers(time_per_unit, recipe_runs_needed_per_minute,item_id)),
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
            inserter_type = self.config["inserters"]["input_type"]
            inserters_needed = min(3, math.ceil(total_ingredient_needed_per_minute / (60 * self.machines_data['inserters'][inserter_type]['items_per_second'])))
        
            
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
        """
        Calculate production capacities for all items in the production chain.
        
        This method creates a reverse mapping of ingredient dependencies and calculates
        the capacity of each item based on the ratio of its assemblers to the total
        assemblers that consume it. This is used for belt balancing and throughput
        optimization.
        
        Args:
            production_data (dict): Dictionary containing production data for all items,
                                  including assembler counts and input/output inserters
        
        Returns:
            dict: Updated production_data with capacity values added to each item
        """
        logger.info("Calculating production capacities for all items")
        
        # Create a reverse map of all the items in the production data
        reverse_mapping = {}
        
        # Collect ingredients used in production data
        for item_id, data in production_data.items():
            # Skip items that don't have recipes (raw materials or outputs)
            if "input_inserters" not in data:
                continue

            # Process each ingredient that this item requires
            for ingredient in data["input_inserters"]:
                ingredient_id = ingredient["id"]
                
                # Only include items that are in the production data (i.e., their capacities will be calculated)
                if ingredient_id in production_data:
                    if ingredient_id not in reverse_mapping:
                        reverse_mapping[ingredient_id] = []
                    reverse_mapping[ingredient_id].append(item_id)
                    
        logger.info(f"Created reverse mapping for {len(reverse_mapping)} items")
        
        # For each item, calculate the capacity using the map as the number of its assemblers
        # divided by the number of other assemblers that consume it
        for item_id, data in production_data.items():
            # Skip if this item is already an output product (it shouldn't calculate capacity for itself)
            if "input_inserters" not in data:
                data['capacity'] = 0
                logger.debug(f"Set capacity to 0 for output item {item_id}")
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
                    
                logger.debug(f"Item {item_id}: {item_assemblers} assemblers, "
                           f"{total_assemblers_needed} total needed, capacity: {data['capacity']}")
            else:
                # If no items are using this as an ingredient, set the capacity to 0
                data['capacity'] = 0
                logger.debug(f"No consumers for item {item_id}, capacity set to 0")
                
        logger.info("Production capacities calculated successfully")
        return production_data

    def _calculate_assemblers(self, time_per_unit, recipe_runs_needed_per_minute,item_id):
        """
        Calculate the number of assemblers needed to produce the required amount per minute.
        
        This method determines the appropriate machine type for the recipe and calculates
        how many machines are needed based on the production rate and machine speed.
        
        Args:
            time_per_unit (float): Time in seconds required to produce one unit of the item
            recipe_runs_needed_per_minute (float): Number of recipe runs needed per minute
            item_id (str): ID of the item being produced
            
        Returns:
            float: Number of assemblers needed (may be fractional)
        """
        logger.debug(f"Calculating assemblers for {item_id}: {recipe_runs_needed_per_minute} runs/min")
        
        machine_type = self._get_machine_type_for_recipe(item_id)
        
        # Get machine info from the machine data
        if machine_type in self.machines_data["assemblers"]:
            machine_info = self.machines_data["assemblers"][machine_type]
            crafting_speed = machine_info["crafting_speed"]
            logger.debug(f"Using machine type {machine_type} with crafting speed {crafting_speed}")
        else:
            # Fall back to default if not found
            default_type = self.config["machines"]["default_assembler"]
            crafting_speed = self.machines_data["assemblers"][default_type]["crafting_speed"]
            logger.warning(f"Machine type {machine_type} not found, using default {default_type}")
    
        items_per_second_per_assembler = crafting_speed / time_per_unit
        items_per_minute_per_assembler = items_per_second_per_assembler * 60
        assemblers_needed = recipe_runs_needed_per_minute / items_per_minute_per_assembler
        
        logger.debug(f"Assemblers needed for {item_id}: {assemblers_needed}")
        return assemblers_needed
    
    
    def _get_machine_type_for_recipe(self, recipe_id):
        """
        Determine the appropriate machine type for a given recipe.
        
        This method checks the recipe-to-machine mapping and returns the correct
        machine type based on the recipe requirements and configuration.
        
        Args:
            recipe_id (str): ID of the recipe to get machine type for
            
        Returns:
            str: Machine type identifier (e.g., "assembling-machine-2", "chemical-plant")
        """
        logger.debug(f"Getting machine type for recipe {recipe_id}")
        
        # Check the recipe machine mapping in machine_data.json
        if "recipe_machine_mapping" in self.machines_data:
            recipes = self.machines_data["recipe_machine_mapping"].get("recipes", {})
            if recipe_id in recipes:
                machine_type = recipes[recipe_id]
                logger.debug(f"Found mapped machine type {machine_type} for recipe {recipe_id}")
                return machine_type
        
        # If no specific mapping, use the appropriate default based on config
        if recipe_id in self.machines_data.get("production_recipes", {}) and "required_machine" in self.machines_data["production_recipes"][recipe_id]:
            machine_type = self.machines_data["production_recipes"][recipe_id]["required_machine"]
            if machine_type == "oil-refinery":
                return self.config["machines"]["default_refinery"]
            elif machine_type == "chemical-plant":
                return self.config["machines"]["default_chemical_plant"]
            elif machine_type == "electric-furnace":
                return self.config["machines"]["default_furnace"]
        
        # Default to the standard assembler
        default_machine = self.config["machines"]["default_assembler"]
        logger.debug(f"Using default machine type {default_machine} for recipe {recipe_id}")
        return default_machine

    def _calculate_inserters(self, recipe_runs_needed_per_minute):
        """
        Calculate the number of inserters needed to move the required amount of items per minute.
        
        This method determines how many inserters are required to handle the material
        flow for a given production rate.
        
        Args:
            recipe_runs_needed_per_minute (float): Number of recipe runs needed per minute
            
        Returns:
            float: Number of inserters needed (may be fractional)
        """
        logger.debug(f"Calculating inserters for {recipe_runs_needed_per_minute} recipe runs/min")
        
        inserter_type = self.config["inserters"]["input_type"]
        items_per_second_per_inserter = self.machines_data["inserters"][inserter_type]["items_per_second"]
        items_per_minute_per_inserter = items_per_second_per_inserter * 60
        inserters_needed = recipe_runs_needed_per_minute / items_per_minute_per_inserter
        
        logger.debug(f"Inserters needed: {inserters_needed} (type: {inserter_type})")
        return inserters_needed
    
    def _calculate_belts(self, total_items_needed_per_minute):
        """
        Calculate the number of belts needed to move the required amount of items per minute.
        
        This method determines how many transport belts are required to handle the
        material throughput for a given production rate.
        
        Args:
            total_items_needed_per_minute (float): Total items that need to be transported per minute
            
        Returns:
            float: Number of belts needed (may be fractional)
        """
        logger.debug(f"Calculating belts for {total_items_needed_per_minute} items/min")
        
        belt_type = self.config["belts"]["default_type"]
        items_per_second_per_belt = self.machines_data["belts"][belt_type]["items_per_second"]
        items_per_minute_per_belt = items_per_second_per_belt * 60
        belts_needed = total_items_needed_per_minute / items_per_minute_per_belt
        
        logger.debug(f"Belts needed: {belts_needed} (type: {belt_type})")
        return belts_needed
    

    def calculate_minimal_grid_size(self,production_output):
        """
        Calculate the minimal grid size needed to accommodate all production facilities.
        
        This method estimates the minimum grid dimensions required based on the
        total number of assemblers, inserters, and belts needed for production.
        
        Args:
            production_output (dict): Dictionary containing production data with facility counts
            
        Returns:
            tuple: (width, height) of the minimal grid size needed
        """
        logger.info("Calculating minimal grid size for production layout")

        total_assemblers = 0
        total_inserters = 0
        total_belts = 0
        
        for item, requirements in production_output.items():
            total_assemblers += requirements.get('assemblers', 0)
            total_inserters += requirements.get('inserters', 0)
            total_belts += requirements.get('belts', 0)
        
        logger.debug(f"Total facilities: {total_assemblers} assemblers, {total_inserters} inserters, {total_belts} belts")
        
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
        
        logger.info(f"Calculated minimal grid size: {width}x{height}")
        return width, height

    
    
    def eval(self):
        """
        Evaluate the current factory layout for optimization purposes.
        
        This method calculates a fitness score for the current layout based on
        grid size and the number of occupied cells. Used by optimization algorithms.
        
        Returns:
            int: Fitness score (lower is better)
        """
        occupied_cells = sum(row.count(2) for row in self.grid)
        score = self.grid_height * self.grid_width + occupied_cells
        logger.debug(f"Layout evaluation: grid size {self.grid_width}x{self.grid_height}, "
                    f"occupied cells: {occupied_cells}, score: {score}")
        return score

    
    def manual_Output(self, Title="Manual Output"):
        """
        Interactive GUI for manually configuring output belt positions.
        
        This method provides a graphical interface for the user to select where
        output belts should be placed on the factory grid. The user can click
        on grid positions to place output points for the target item.
        
        Args:
            Title (str, optional): Title for the GUI window. Defaults to "Manual Output".
        """
        logger.info(f"Starting manual output configuration for {self.output_item}")
        
        # GUI configuration constants
        side_panel_width = 300
        CELL_SIZE = 50
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)  # Green for input
        RED = (255, 0, 0)  # Red for output
        
        # Load and scale output item image
        output_image = pygame.image.load(f"assets/{self.output_item}.png")
        output_image = pygame.transform.scale(output_image, (CELL_SIZE, CELL_SIZE))
        
        input_information = self.input_information
        input_items = self.input_items
        # Load images for the items
        item_images = {item: pygame.image.load(f"assets/{item}.png") for item in input_items}
        
        belt_type = self.config["belts"]["default_type"]
        conveyor_image = pygame.image.load(f"assets/{belt_type}.png")
        # Resize images to fit in the grid cells
        item_images = {item: pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE)) for item, image in item_images.items()}
        conveyor_image = pygame.transform.scale(conveyor_image, (CELL_SIZE, CELL_SIZE))

        output_information = {self.output_item:{'input': None, 'output': None, 'paths': None}}
        setting_input = True
        
        # Build obstacle map for pathfinding
        grid_astar = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        logger.debug("Creating obstacle map from existing input paths")
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
                                        paths, inserters = pathfinder.find_paths_for_all_items(IO_paths=True)
                                        
                                        # Store path information directly
                                        if current_item in paths and paths[current_item]:
                                            output_information[current_item]['paths'] = paths
                                        else:
                                            logger.info(f"No valid path found for {current_item}")

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
    
    def is_fluid_item(self, item_id):
        """
        Check if an item is a fluid based on its type in recipes data.
        
        This method determines whether an item should be transported via pipes
        instead of belts by checking its type in the recipe data.
        
        Args:
            item_id (str): ID of the item to check
            
        Returns:
            bool: True if the item is a fluid, False otherwise
        """
        # Look up item in the recipe data
        item = self.item_lookup.get(item_id, {})
        # Check if the item type is "Liquid"
        is_fluid = item.get("type", "") == "Liquid"
        logger.debug(f"Item {item_id} is fluid: {is_fluid}")
        return is_fluid
    
    def manual_Input(self, Title="Manual Input"):
        """
        Interactive GUI for manually configuring input belt positions.
        
        This method provides a graphical interface for the user to select where
        input belts should be placed on the factory grid for raw materials.
        The user can cycle through different input items and place both input
        and output points for each item.
        
        Args:
            Title (str, optional): Title for the GUI window. Defaults to "Manual Input".
        """
        logger.info(f"Starting manual input configuration for items: {self.input_items}")
        
        # GUI configuration constants
        side_panel_width = 300
        CELL_SIZE = 50
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)  # Green for input
        RED = (255, 0, 0)  # Red for output

        # Get all input items
        input_items = self.input_items
        
        # Load images for the items
        item_images = {item: pygame.image.load(f"assets/{item}.png") for item in input_items}
        
        # Load appropriate transport entity images based on item type
        transport_images = {}
        for item in input_items:
            if self.is_fluid_item(item):
                transport_images[item] = pygame.image.load("assets/pipe.png")
            else:
                belt_type = self.config["belts"]["default_type"]
                transport_images[item] = pygame.image.load(f"assets/{belt_type}.png")
        
        # Resize images to fit in the grid cells
        item_images = {item: pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE)) for item, image in item_images.items()}
        transport_images = {item: pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE)) for item, image in transport_images.items()}

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

                                    # Once both input and output are set, find the path and connect transport entities
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
                                                'inserter_mapping': None,
                                                'is_fluid': self.is_fluid_item(current_item)  # Flag for fluid handling
                                            }
                                        }
                                        
                                        # Create the pathfinder object with appropriate underground length
                                        is_fluid = self.is_fluid_item(current_item)
                                        underground_length = 10 if is_fluid else 3  # Pipes can go further underground
                                        
                                        pathfinder = MultiAgentPathfinder(
                                            grid_astar, 
                                            points,
                                            allow_underground=True,  # Enable underground entities
                                            underground_length=underground_length,
                                            allow_splitters=False,   # Disable splitters
                                            splitters={},            # Empty splitters dictionary
                                            find_optimal_paths=False # Don't need to find optimal paths
                                        )
                                        
                                        # Find paths for all items
                                        paths, _ = pathfinder.find_paths_for_all_items(IO_paths=True)
                                        
                                        # Store path information directly
                                        if current_item in paths and paths[current_item]:
                                            input_information[current_item]['paths'] = paths
                                        else:
                                            logger.info(f"No valid path found for {current_item}")

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
                    is_fluid = self.is_fluid_item(item)
                    transport_img = transport_images[item]
                    
                    for path_data in data['paths'][item]:
                        path = path_data['path']
                        
                        # Draw transport entities for each segment of the path
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
                            
                            # For fluids, only rotate if we're using pipes with directional graphics
                            if is_fluid:
                                # For pipes, we may not need to rotate if they connect automatically
                                rotated_entity = transport_img
                            else:
                                rotated_entity = pygame.transform.rotate(transport_img, angle)
                                
                            window.blit(rotated_entity, (side_panel_width + current[0] * CELL_SIZE, current[1] * CELL_SIZE))
            
            # Draw the side panel
            pygame.draw.rect(window, BLACK, (0, 0, side_panel_width, window_height))  # Side panel background
            font = pygame.font.Font(None, 36)
            item_text = f"Setting: {items[current_item_index]}"
            setting_text = "Input" if setting_input else "Output"
            is_fluid_text = "Type: Fluid" if self.is_fluid_item(items[current_item_index]) else "Type: Solid"
            text_surface_item = font.render(item_text, True, WHITE)
            text_surface_setting = font.render(setting_text, True, WHITE)
            text_surface_fluid = font.render(is_fluid_text, True, WHITE)
            window.blit(text_surface_item, (10, 50))  # Show the current item being set
            window.blit(text_surface_setting, (10, 100))  # Show whether we're setting input or output
            window.blit(text_surface_fluid, (10, 150))  # Show whether current item is fluid or solid

            # Update display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(30)
            
        # store input positions
        pygame.quit()
        self.input_information = input_information

    def solve(self, production_data, solver_type):
        """
        Solve the factory layout optimization problem using the specified solver.
        
        This method initializes the appropriate solver (Z3 SMT or Gurobi) and
        optimizes the placement of assemblers, inserters, and other factory
        components to minimize space usage while satisfying all constraints.
        
        Args:
            production_data (dict): Production requirements including assemblers,
                                  inserters, and material flows for each item
            solver_type (str): Type of solver to use ("z3" or "gurobi")
            
        Returns:
            None: Results are stored in the solver instance and can be accessed
                 through self.z3_solver
                 
        Raises:
            ValueError: If solver_type is not "z3" or "gurobi"
            RuntimeError: If solver fails to find a solution
        """
        logger.info(f"Starting factory layout optimization with {solver_type} solver")
        logger.info(f"Grid size: {self.grid_width}x{self.grid_height}")
        logger.info(f"Production items: {list(production_data.keys())}")
        
        # Initialize the appropriate solver based on type
        if self.z3_solver is None and solver_type == "z3":
            logger.info("Initializing Z3 SMT solver")
            self.z3_solver = SMTSolver(self.grid_width, self.grid_height, production_data, solver_type)
        
        elif self.z3_solver is None and solver_type == "gurobi":
            logger.info("Initializing Gurobi optimization solver")
            self.z3_solver = GurobiSolver(self.grid_width, self.grid_height, production_data)
        
        elif self.z3_solver is not None:
            logger.info(f"Using existing {solver_type} solver instance")
        
        else:
            logger.error(f"Invalid solver type: {solver_type}")
            raise ValueError(f"Solver type must be 'z3' or 'gurobi', got '{solver_type}'")

        # Build optimization constraints
        logger.info("Building optimization constraints")
        self.z3_solver.build_constraints()
        
        # Solve the optimization problem
        logger.info("Solving optimization problem")
        try:
            solution_found = self.z3_solver.solve()
            if solution_found:
                logger.info("Factory layout optimization completed successfully")
            else:
                logger.warning("Solver could not find a valid solution")
        except Exception as e:
            logger.error(f"Solver error: {e}")
            raise RuntimeError(f"Solver failed: {e}")
        
    def add_manual_IO_constraints(self, production_data, solver_type):
        """
        Add manually configured input/output constraints to the solver.
        
        This method integrates user-defined input and output points into the
        optimization problem, ensuring that material flows respect the
        manual I/O configuration.
        
        Args:
            production_data (dict): Production requirements data
            solver_type (str): Type of solver to use ("z3" or "gurobi")
            
        Returns:
            None: Constraints are added to the solver instance
        """
        logger.info(f"Adding manual I/O constraints with {solver_type} solver")
        
        # Initialize solver if not already created
        if self.z3_solver is None and solver_type == "z3":
            logger.info("Initializing Z3 solver for I/O constraints")
            self.z3_solver = SMTSolver(self.grid_width, self.grid_height, production_data, solver_type)
        
        elif self.z3_solver is None and solver_type == "gurobi":
            logger.info("Initializing Gurobi solver for I/O constraints")
            self.z3_solver = GurobiSolver(self.grid_width, self.grid_height, production_data)
        
        logger.info("Manual I/O constraints integration completed")
        
        
        self.z3_solver.add_manuel_IO_constraints(self.input_information,self.output_information)
    

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
                logger.info(f"Belt at {belt_coords} overlaps with input belt/path")
            if belt_coords in output_coords:
                logger.info(f"Belt at {belt_coords} overlaps with output belt/path")
        
        return overlap

    def detect_assembler_overlap(self,belt,assembler_information):
        """
        Check if a belt position overlaps with any assembler.
        
        This method determines whether a proposed belt position would conflict
        with existing assembler placements in the factory layout.
        
        Args:
            belt: A tuple of (item, x, y, _) representing the belt position
            assembler_information: List of assembler data with positions and dimensions
            
        Returns:
            bool: True if there's an overlap, False otherwise
        """
        belt_coords = (belt[1], belt[2])
        logger.debug(f"Checking belt at {belt_coords} for assembler overlap")
        
        for assembler in assembler_information:
            assembler_item, assembler_x, assembler_y, width, height, machine_type, orientation_idx = assembler
            
            # Check if the belt's coordinates overlap with the assembler's area
            for dx in range(width):  
                for dy in range(height):  
                    assembler_coords = (assembler_x + dx, assembler_y + dy)
                    
                    # If the belt's coordinates overlap with any of the assembler's coordinates
                    if belt_coords == assembler_coords:
                        logger.debug(f"Belt at {belt_coords} overlaps with assembler at {assembler_coords}")
                        return True  # There is an overlap
        return False  # No overlap detected
        

    def is_position_available(self, pos):
        """
        Check if a position is available (not occupied by obstacles).
        
        This method checks the obstacle map to determine if a given position
        is free for placement of factory components.
        
        Args:
            pos: Tuple of (x, y) coordinates to check
            
        Returns:
            bool: True if position is available, False otherwise
        """
        x, y = pos
        # Check if position is within bounds of the obstacle_map and if it's free (0)
        if 0 <= y < len(self.obstacle_map) and 0 <= x < len(self.obstacle_map[0]):
            return self.obstacle_map[y][x] == 0
        return False  # Out-of-bounds positions are considered unavailable
    
    def is_fluid_item(self, item_id):
        """
        Check if an item is a fluid based on its type in recipes data.
        
        This method determines whether an item should be transported via pipes
        instead of belts by checking its type in the recipe data.
        
        Args:
            item_id (str): ID of the item to check
            
        Returns:
            bool: True if the item is a fluid, False otherwise
        """
        # Look up item in the recipe data
        item = self.item_lookup.get(item_id, {})
        # Check if the item type is "Liquid"
        is_fluid = item.get("type", "") == "Liquid"
        logger.debug(f"Item {item_id} is fluid: {is_fluid}")
        return is_fluid
    
    def get_retrieval_points(self, belt_point_information, assembler_information):
        """
        Generate retrieval points for material transport pathfinding.
        
        This method creates a mapping of where materials need to be picked up
        and delivered, considering both input sources and assembler outputs.
        It handles both belt-based and pipe-based transport systems.
        
        Args:
            belt_point_information: List of belt positions and their associated items
            assembler_information: List of assembler data with positions and outputs
            
        Returns:
            dict: Dictionary mapping retrieval point keys to their configuration data
        """
        logger.info("Generating retrieval points for material transport")
        
        retrieval_points = {}  # Dictionary to store all retrieval points for each item
        
        # Track all belt positions to avoid overlaps
        occupied_belt_positions = set()
        for belt_item, belt_x, belt_y, _ in belt_point_information:
            occupied_belt_positions.add((belt_x, belt_y))
            
        logger.info(f"Occupied belt positions: {occupied_belt_positions}")

        # Step 1: Check direct input points from `input_information`
        for i,(item, x, y, _) in enumerate(belt_point_information):
            key = f"{item}_{i}"  # Unique key per occurrence of the item on the belt
            
            output_path_positions = set()
            if item in self.output_information:
                # Get all positions with the matching item from `input_information`
                input_point = self.output_information[item]['input']
                output_point = self.output_information[item]['output']
                input_point = (input_point[1], input_point[0])  # Swap the coordinates
                output_point = (output_point[1], output_point[0])  # Swap the coordinates
                
                
                output_path_positions.add(input_point)
                output_path_positions.add(output_point)
                
    
                # Process path data directly if it exists
                if self.output_information[item]['paths'] is not None and item in self.output_information[item]['paths']:
                    for path_data in self.output_information[item]['paths'][item]:
                        path = path_data['path']
                        
                        # Add all points in the path as potential destinations
                        for pos in path:
                            # Convert from (x, y) to (col, row) for consistency
                            output_path_positions.add(pos)

            
            retrieval_points[key] = {
            'item': item,
            'destination': [(x,y)],  # This is the destination point from belt_point_information
            'start_points': [],  # List to store relevant start points
            'inserter_mapping': None,
            'is_fluid': self.is_fluid_item(item)  # Check if the item is a fluid
            }
            
            # Step 2: Check if item is in `input_information`
            if item in self.input_information:
                # Get all positions with the matching item from `input_information`
                input_point = self.input_information[item]['input']
                output_point = self.input_information[item]['output']
                input_point = (input_point[1], input_point[0])  # Swap the coordinates
                output_point = (output_point[1], output_point[0])  # Swap the coordinates
                
                # Add these points to the start_points list only if they're not occupied by other belts
                if input_point not in occupied_belt_positions or input_point == (x, y):
                    retrieval_points[key]['start_points'].append(input_point)
                else:
                    logger.info(f"Skipping input point {input_point} for {item} as it's already occupied")
                
                if output_point not in occupied_belt_positions or output_point == (x, y):
                    retrieval_points[key]['start_points'].append(output_point)
                else:
                    logger.info(f"Skipping output point {output_point} for {item} as it's already occupied")
                  # Process path data directly if it exists
                if self.input_information[item]['paths'] is not None and item in self.input_information[item]['paths']:
                    for path_data in self.input_information[item]['paths'][item]:
                        path = path_data['path']
                        
                        # Add all points in the path as potential destinations
                        for pos in path:
                            # Skip if this position is occupied by another belt
                            if pos not in occupied_belt_positions or pos == (x, y):
                                # Convert from (x, y) to (col, row) for consistency
                                retrieval_points[key]["start_points"].append(pos)
                            else:
                                logger.info(f"Skipping path point {pos} for {item} as it's already occupied")
                
                continue  # Move to the next belt item after checking input information
                    
            
            fluid_connection_positions = set()
            _, _, _, _, fluid_connection_info, _ = self.z3_solver.build_map()
            
            for conn in fluid_connection_info:
                fluid_connection_positions.add(conn["position"])
                # Also add the position one step further in the same direction
                rel_pos = conn["relative_pos"]
                assembler_pos = None
                
                # Find the corresponding assembler position
                for asm_item, asm_x, asm_y, width, height, machine_type, orientation_idx in assembler_information:
                    if conn["assembler_id"].startswith(asm_item):
                        assembler_pos = (asm_x, asm_y)
                        break
                
                # Skip if we can't find the assembler
                if not assembler_pos:
                    continue
                    
                # Add positions adjacent to fluid connections (one step outward)
                if rel_pos[0] == 0:  # Left edge
                    fluid_connection_positions.add((conn["position"][0] - 1, conn["position"][1]))
                elif rel_pos[0] == width:  # Right edge
                    fluid_connection_positions.add((conn["position"][0] + 1, conn["position"][1]))
                elif rel_pos[1] == 0:  # Top edge
                    fluid_connection_positions.add((conn["position"][0], conn["position"][1] - 1))
                elif rel_pos[1] == height:  # Bottom edge
                    fluid_connection_positions.add((conn["position"][0], conn["position"][1] + 1))
              
                    
            for asm_item, assembler_x, assembler_y,width,height, machine_type, orientation_idx  in assembler_information:
                if asm_item == item:
                    output_positions = []
                                
                    # Top edge
                    for dx in range(width):
                        output_positions.append([(assembler_x + dx, assembler_y - 1), (assembler_x + dx, assembler_y - 2)])
                    
                    # Bottom edge
                    for dx in range(width):
                        output_positions.append([(assembler_x + dx, assembler_y + height), (assembler_x + dx, assembler_y + height + 1)])
                    
                    # Left edge
                    for dy in range(height):
                        output_positions.append([(assembler_x - 1, assembler_y + dy), (assembler_x - 2, assembler_y + dy)])
                    
                    # Right edge
                    for dy in range(height):
                        output_positions.append([(assembler_x + width, assembler_y + dy), (assembler_x + width + 1, assembler_y + dy)])

                    # Filter output positions to ensure no overlap with any existing structures or fluid connections
                    for position_pair in output_positions:
                        inserter_pos, belt_pos = position_pair
                        
                        # Skip positions that overlap with fluid connections
                        if inserter_pos in fluid_connection_positions or belt_pos in fluid_connection_positions:
                            continue
                            
                        logger.debug(f"Checking positions: {inserter_pos}, {belt_pos}")
                        
                        # Check if positions are available
                        if self.is_position_available(inserter_pos) and (self.is_position_available(belt_pos) or belt_pos in output_path_positions):
                            retrieval_points[key]["destination"].append(belt_pos)

        return retrieval_points

    
    def get_num_inserters(self, item):
        """
        Calculate the number of inserters needed for a given item based on throughput.
        
        This method determines how many inserters are required to handle the material
        flow for a specific item, considering both fluid connections and belt-based
        transport systems.
        
        Args:
            item (str): ID of the item to calculate inserter count for
            
        Returns:
            int: Number of inserters needed (1-3 for items, 1 for fluids)
        """
        logger.debug(f"Calculating inserter count for item {item}")
        
        # Check if the item is a fluid
        if self.is_fluid_item(item):
            logger.debug(f"Item {item} is a fluid, using 1 fluid connection")
            return 1
        
        # Get item production data
        if item not in self.production_data:
            logger.warning(f"No production data found for {item}, using default inserter count")
            return 1
        
        # Get items per minute being produced
        items_per_minute = self.production_data[item].get("amount_per_minute", 0)
        
        # Get inserter type and capacity from config
        inserter_type = self.config["inserters"]["output_type"]
        inserter_capacity = self.machines_data["inserters"][inserter_type]["items_per_second"] * 60  # Items per minute
        
        # Calculate number of inserters needed based on throughput
        inserters_needed = math.ceil(items_per_minute / inserter_capacity)
        
        # Cap at 1-3 inserters (practical limit for assembler connections)
        inserters_needed = max(1, min(3, inserters_needed))
        
        logger.debug(f"Calculated {inserters_needed} inserters for {item} (producing {items_per_minute} items/min)")
        return inserters_needed
    
    
    def add_out_point_information(self, output_item, assembler_information,belt_point_information):
        """
        Add output point information for the main factory output item.
        
        This method creates retrieval points for the factory's main output item,
        determining where output belts should connect to assemblers and where
        they should lead to external connections.
        
        Args:
            output_item (str): ID of the main output item
            assembler_information: List of assembler data with positions and outputs
            belt_point_information: List of belt positions and their associated items
            
        Returns:
            dict: Dictionary of retrieval points for the output item
        """
        logger.info(f"Adding output point information for {output_item}")
        
        retrieval_points = {}
        
        # Collect all the possible output positions from the output information
        output_coords = []
        output_path_positions = set()
        
        logger.info(f"Finding output points for {output_item}")
        
        # Extract coordinates from output path data
        if self.output_information[output_item]['paths'] is not None and output_item in self.output_information[output_item]['paths']:
            for path_data in self.output_information[output_item]['paths'][output_item]:
                path = path_data['path']
                logger.info(f"Found existing output path with {len(path)} points")
                
                # Add all points in the path as potential destinations
                for pos in path:
                    output_coords.append(pos)  # Add to ordered list
                    output_path_positions.add(tuple(pos))  # Add to fast lookup set
        
        # Add input and output points from specified I/O
        input_point = self.output_information[output_item]['input']
        output_point = self.output_information[output_item]['output']
        
        if input_point and output_point:
            logger.info(f"Using defined I/O points: input at {input_point}, output at {output_point}")
            
            input_point = (input_point[1], input_point[0])  # Swap the coordinates (row, col) -> (x, y)
            output_point = (output_point[1], output_point[0])  # Swap the coordinates
            
            output_coords.append(input_point)
            output_coords.append(output_point)
            
            output_path_positions.add(input_point)
            output_path_positions.add(output_point)
            
        fluid_connection_positions = set()
        _, _, _, _, fluid_connection_info, _ = self.z3_solver.build_map()            # Create a set of occupied belt positions to avoid overlaps
        occupied_belt_positions = set()
        for asm_info in assembler_information:
            for belt_item, belt_x, belt_y, _ in belt_point_information:
                occupied_belt_positions.add((belt_x, belt_y))
        
        logger.info(f"Occupied belt positions in output: {occupied_belt_positions}")
            
        # Process each assembler that produces this item
        for i, (asm_item, assembler_x, assembler_y, width, height, machine_type, orientation_idx) in enumerate(assembler_information):
            if asm_item == output_item:
                # Get the number of inserters needed for this assembler/item
                num_inserters = self.get_num_inserters(output_item)
                logger.info(f"Need {num_inserters} inserters for {output_item} at assembler {i}")
                
                # Generate potential output positions around the assembler
                output_positions = []
                
                # Define all possible positions around the assembler's edges
                # Top edge of assembler
                for dx in range(width):
                    output_positions.append([(assembler_x + dx, assembler_y - 1), (assembler_x + dx, assembler_y - 2)])
                
                # Bottom edge of assembler
                for dx in range(width):
                    output_positions.append([(assembler_x + dx, assembler_y + height), (assembler_x + dx, assembler_y + height + 1)])
                
                # Left edge of assembler
                for dy in range(height):
                    output_positions.append([(assembler_x - 1, assembler_y + dy), (assembler_x - 2, assembler_y + dy)])
                
                # Right edge of assembler
                for dy in range(height):
                    output_positions.append([(assembler_x + width, assembler_y + dy), (assembler_x + width + 1, assembler_y + dy)])
                
                # Rank positions by their proximity to destination and existing paths
                ranked_positions = []
                
                for position_pair in output_positions:
                    inserter_pos, belt_pos = position_pair
                    belt_pos_tuple = tuple(belt_pos)
                    
                    # Skip immediately if inserter position isn't available
                    if not self.is_position_available(inserter_pos):
                        logger.debug(f"Skipping inserter position {inserter_pos} - not available")
                        continue
                    
                    # HIGHEST PRIORITY: Direct connection to existing path
                    if belt_pos_tuple in output_path_positions:
                        # Found a perfect match - belt connects directly to existing path
                        logger.info(f"Found optimal position: inserter at {inserter_pos}, belt at {belt_pos} connects to existing path")
                        ranked_positions.append((0, inserter_pos, belt_pos))  # Priority 0 (highest)
                        continue
                    
                    # Check if belt position is available (if not connecting to existing path)
                    if not self.is_position_available(belt_pos):
                        logger.debug(f"Skipping belt position {belt_pos} - not available")
                        continue
                    
                    # Calculate minimum distance to any destination
                    min_distance = float('inf')
                    for dest in output_coords:
                        distance = abs(belt_pos[0] - dest[0]) + abs(belt_pos[1] - dest[1])
                        min_distance = min(min_distance, distance)
                    
                    logger.debug(f"Position {belt_pos} has distance {min_distance} to nearest destination")
                    ranked_positions.append((min_distance, inserter_pos, belt_pos))
                
                # Sort by distance (lowest first)
                ranked_positions.sort()
                
                # Create separate retrieval points for each inserter needed
                for inserter_idx in range(num_inserters):
                    # Create unique key for each inserter for this assembler
                    key = f"{output_item}_{i}_{inserter_idx}"
                    logger.info(f"Creating retrieval point {key}")
                    
                    # Initialize data structure for this assembler's output
                    retrieval_points[key] = {
                        'item': output_item,
                        'destination': list(set(output_coords)),  # Use copy of output coordinates
                        'start_points': [],
                        'inserter_mapping': {},
                        'is_fluid': self.is_fluid_item(output_item)
                    }
                            # Add these positions to the retrieval point
                    for _, inserter_pos, belt_pos in ranked_positions:
                        belt_pos_tuple = tuple(belt_pos)
                        
                        # Check if this belt position is already occupied by another belt
                        if belt_pos_tuple in occupied_belt_positions and belt_pos_tuple not in output_path_positions:
                            logger.info(f"Skipping belt position {belt_pos} - already occupied by another belt")
                            continue
                            
                        retrieval_points[key]["inserter_mapping"][str(belt_pos)] = inserter_pos
                        retrieval_points[key]["start_points"].append(belt_pos)
                        logger.debug(f"Added potential inserter at {inserter_pos} with belt at {belt_pos} to {key}")
                    
                    # Log total positions found
                    logger.info(f"Added {len(retrieval_points[key]['start_points'])} potential positions for {key}")
        
        return retrieval_points
        
    
    def rearrange_dict(self,input_dict, target_item):
        """
        Rearrange a dictionary to prioritize entries for a specific target item.
        
        This method reorders the retrieval points dictionary to place all entries
        for the target item (usually the main output) at the beginning, which
        helps with pathfinding prioritization.
        
        Args:
            input_dict (dict): Dictionary to rearrange
            target_item (str): Item ID to prioritize
            
        Returns:
            dict: Rearranged dictionary with target item entries first
        """
        logger.debug(f"Rearranging dictionary to prioritize {target_item}")
        
        # Separate items based on the 'item' value
        target_items = {key: value for key, value in input_dict.items() if value.get('item') == target_item}
        other_items = {key: value for key, value in input_dict.items() if value.get('item') != target_item}
        
        # Combine the dictionaries with target items first
        rearranged_dict = {**target_items, **other_items}
        
        logger.debug(f"Rearranged dictionary: {len(target_items)} target items, {len(other_items)} other items")
        return rearranged_dict
    
    def prepare_splitter_information(self, input_information, output_information):
        """
        Prepare splitter information for complex belt routing scenarios.
        
        This method analyzes input and output paths to identify locations where
        belt splitters can be placed to optimize material distribution. It creates
        orientation data for each path segment and generates potential splitter
        positions.
        
        Args:
            input_information (dict): Dictionary containing input belt configurations
            output_information (dict): Dictionary containing output belt configurations
            
        Returns:
            dict: Dictionary mapping item IDs to lists of potential splitter positions
        """
        logger.info("Preparing splitter information for belt routing")
        
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
    
    
    def build_belts(self, max_tries):
        """
        Build and optimize belt networks for material transport in the factory.
        
        This method constructs the belt and pipe networks that connect assemblers,
        inserters, and I/O points. It uses multi-agent pathfinding to route
        materials efficiently while avoiding conflicts and minimizing crossings.
        
        The method attempts multiple times to find optimal solutions, using
        different strategies and configurations if initial attempts fail.
        
        Args:
            max_tries (int): Maximum number of attempts to build successful belt networks
            
        Returns:
            tuple: (paths_dict, placed_inserter_information)
                - paths_dict: Dictionary mapping items to their transport paths
                - placed_inserter_information: Details of additional inserters placed
                
        Raises:
            RuntimeError: If belt building fails after max_tries attempts
            ValueError: If solver hasn't been run prior to calling this method
        """
        logger.info(f"Starting belt network construction with max {max_tries} attempts")
        
        if self.z3_solver is None:
            logger.error("Cannot build belts: solver not initialized or run")
            raise ValueError("Must call solve() before build_belts()")
            
        for attempt in range(max_tries):
            logger.info(f'Belt building attempt {attempt + 1}/{max_tries}')
            
            # Extract factory layout information from solver
            logger.info("Extracting factory layout from solver")
            layout_data = self.z3_solver.build_map()
            (self.obstacle_map, belt_point_information, assembler_information, 
             inserter_information, fluid_connection_info, power_pole_information) = layout_data
            
            logger.debug(f"Factory layout extracted:")
            logger.debug(f"  - Obstacles: {len(self.obstacle_map) if self.obstacle_map else 0} cells")
            logger.debug(f"  - Belt points: {len(belt_point_information)}")
            logger.debug(f"  - Assemblers: {len(assembler_information)}")
            logger.debug(f"  - Inserters: {len(inserter_information)}")
            
            # Filter out overlapping and redundant belt points
            logger.info("Filtering belt points for overlaps and conflicts")
            original_count = len(belt_point_information)
            
            # Remove belts that overlap with existing user-defined belts
            belt_point_information = [belt for belt in belt_point_information 
                                    if not self.detect_belt_overlap(belt)]
            
            # Remove belts that overlap with assemblers (direct insertion preferred)
            belt_point_information = [belt for belt in belt_point_information 
                                    if not self.detect_assembler_overlap(belt, assembler_information)]

            filtered_count = len(belt_point_information)
            logger.info(f"Filtered belt points: {original_count} -> {filtered_count}")
            
            # Generate material retrieval points for pathfinding
            logger.info("Generating material retrieval points")
            retrieval_points = self.get_retrieval_points(belt_point_information, assembler_information)
            
            # Add output points for the main factory output
            output_points = self.add_out_point_information(self.output_item, assembler_information, belt_point_information)
            retrieval_points.update(output_points)
            
            logger.info(f"Generated {len(retrieval_points)} retrieval point groups")
            logger.debug(f"Retrieval points: {list(retrieval_points.keys())}")
            
            # Prioritize output item paths for better routing
            self.retrieval_points = self.rearrange_dict(retrieval_points, self.output_item)
            
            try:
                # Prepare splitter information for complex routing scenarios
                logger.info("Preparing splitter configurations")
                splitters = self.prepare_splitter_information(self.input_information, self.output_information)
 
         
                # Create the pathfinder
                pathfinder = MultiAgentPathfinder(
                    self.obstacle_map, 
                    retrieval_points,
                    allow_underground=self.config["pathfinding"]["allow_underground"],
                    underground_length=self.config["belts"]["underground_max_length"],
                    allow_splitters=self.config["pathfinding"]["allow_splitters"],
                    splitters=splitters,  
                    find_optimal_paths=self.config["pathfinding"]["find_optimal_paths"],
                    output_item=self.output_item,
                    pipe_underground_length=self.config["belts"]["underground_max_length"],
                )
                                
                # Find paths for all items
                paths, inserters = pathfinder.find_paths_for_all_items()
                
                return paths, inserters
            
            except Exception as e:
                logger.info(f"could not assign valid paths to that setup: {e}")
                logger.warning(f"could not assign valid paths to that setup: {e}")
                
                # restrict the assembler and inserter positions to occur in the same setup -> belts are not needed as they are bound by the inserter
                self.z3_solver.restrict_current_setup()
                self.z3_solver.solve()
                
                
                
        return None            
    
    def load_data(self, file_path):
        """
        Load factory data from a JSON file.
        
        This method restores the complete factory state from a previously saved
        JSON file, including production data, layout information, paths, and
        all other factory configuration data.
        
        Args:
            file_path (str): Path to the JSON file containing factory data
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        logger.info(f"Loading factory data from {file_path}")
        
        try:
            # Read the data from the file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Restore the values from the loaded data
            self.output_item = data.get("output_item")
            self.amount = data.get("amount")
            self.max_output = data.get("max_ouput")  # Note: keeping original typo for compatibility
            self.production_data = data.get("production_data")
            self.grid_width = data.get("grid_width")
            self.grid_height = data.get("grid_height")
            self.input_items = data.get("input_items")
            self.placed_inserter_information = data.get("placed_inserter_information", [])
            self.power_pole_information = data.get("power_pole_information", [])
            
            logger.debug(f"Basic factory data loaded: output_item={self.output_item}, "
                        f"grid_size={self.grid_width}x{self.grid_height}")
            
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

            logger.info(f"Production tree data successfully loaded from {file_path}")
            logger.debug(f"Loaded {len(self.production_data)} production items, "
                        f"{len(self.paths)} paths, {len(self.retrieval_points)} retrieval points")
            return True

        except Exception as e:
            logger.error(f"Failed to load production tree data: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def place_power_poles(self,file_path):
        """
        Place power poles using the SMT solver to power assemblers and inserters.
        
        This method sets up power pole constraints in the SMT solver and finds
        an optimal placement that minimizes the number of power poles while
        ensuring all factory components are powered.
        
        Args:
            file_path (str): Path to save the power pole information JSON file
            
        Returns:
            bool: True if power poles were placed successfully, False otherwise
        """
        logger.info("Starting power pole placement optimization")
        
        # Check if SMT solver exists
        if not self.z3_solver:
            logger.warning("Cannot place power poles: No SMT solver initialized")
            return False
        
        logger.info("Setting up power poles using SMT solver")
        
        # Check if power poles are enabled in config
        power_config = self.config.get("power", {})
        if not power_config.get("place_power_poles", False):
            logger.info("Power pole placement is disabled in config")
            return False
    
        # Set up power poles in the SMT solver
        # This will create power pole variables and add constraints
        self.z3_solver.setup_power_poles()
        
        # Minimize the number of power poles while ensuring coverage
        logger.info("Minimizing number of power poles")
        result = self.z3_solver.minimize_power_poles()
        
        if result:
            logger.info("Power poles placed successfully")            # Get the latest map with power pole information
            _, _, _, _, _, power_pole_info = self.z3_solver.build_map()
            
            # Create serializable version of power pole information
            serializable_power_poles = []
            if power_pole_info:
                for pole in power_pole_info:
                    if isinstance(pole, tuple):
                        serializable_power_poles.append(list(pole))
                    else:
                        serializable_power_poles.append(pole)
                
                # Store serializable power pole information directly
                self.power_pole_information = serializable_power_poles
                
                logger.info(f"Retrieved and processed {len(serializable_power_poles)} power poles")
                for pole in serializable_power_poles:
                    logger.debug(f"Serialized Power pole: {pole}")
            else:
                logger.warning("No power pole information was retrieved")
                # Initialize with empty list to ensure proper JSON serialization
                self.power_pole_information = []
            
            # Save the power pole information directly to the JSON file if a file_path is provided
            if file_path:
                try:
                    # If file exists, read existing data first
                    data = {}
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, "r") as file:
                                data = json.load(file)
                                logger.info(f"Successfully loaded existing JSON data with {len(data.keys())} keys")
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse existing JSON file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Error reading existing JSON file: {e}")
                    else:
                        logger.warning(f"JSON file does not exist yet: {file_path}")
                    
                    # Update only the power_pole_information key
                    data["power_pole_information"] = self.power_pole_information
                    
                    # Ensure file path has a .json extension
                    if not file_path.endswith('.json'):
                        file_path += '.json'
                    
                    # Write the updated data back to the file
                    with open(file_path, "w") as file:
                        json.dump(data, file, indent=4)
                    
                    logger.info(f"Power pole information saved successfully to {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save power pole information to file: {e}")
                    import traceback
                    traceback.print_exc()
            
            return True
        else:
            logger.warning("Failed to place power poles optimally")
            return False
    
    def store_data(self, file_path, paths, placed_inserter_information):
        """
        Store complete factory data to a JSON file.
        
        This method serializes all factory data including production requirements,
        layout information, belt paths, inserter placements, and power pole
        configurations to a JSON file for later loading.
        
        Args:
            file_path (str): Path where the JSON file should be saved
            paths (dict): Dictionary containing all belt/pipe paths
            placed_inserter_information (list): List of inserter placement data
            
        Returns:
            bool: True if data was stored successfully, False otherwise
        """
        logger.info(f"Storing factory data to {file_path}")
        
        try:
            # We want to preserve the power_pole_information that was created in place_power_poles
            # So we'll store it temporarily if it exists
            existing_power_poles = None
            if hasattr(self, 'power_pole_information'):
                existing_power_poles = self.power_pole_information
                
            # Get latest map
            obstacle_map, belt_point_information, assembler_information, inserter_information, fluid_connection_info, latest_power_poles = self.z3_solver.build_map()
            
            # Restore existing power pole information if it was set, otherwise use the latest
            if existing_power_poles is not None:
                logger.info(f"Using pre-existing power pole information with {len(existing_power_poles)} poles")
            else:
                logger.info(f"Using latest power pole information with {len(latest_power_poles)} poles")
                self.power_pole_information = latest_power_poles
            
            logger.debug(f"Storing data to {file_path}")
            logger.debug(f"Belt point information: {belt_point_information}")
            logger.debug(f"Assembler information: {assembler_information}")
            logger.debug(f"Inserter information: {inserter_information}")
            
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
                        
            logger.debug(f"Prepared serializable obstacle map with {len(serializable_obstacle_map)} rows")
            
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
                                    "start_splitter": path_data.get("start_splitter").to_dict() if path_data.get("start_splitter") else None,
                                    "dest_splitter": path_data.get("dest_splitter").to_dict() if path_data.get("dest_splitter") else None
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
                        "start_splitter": path_data.get("start_splitter").to_dict() if path_data.get("start_splitter") else None,
                        "dest_splitter": path_data.get("dest_splitter").to_dict() if path_data.get("dest_splitter") else None
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
            
            # Create the data dictionary with all fields            # Create the data dictionary with all fields
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
                "placed_inserter_information": serializable_placed_inserters,
                # Use the power pole information that was already serialized in place_power_poles()
                "power_pole_information": self.power_pole_information if hasattr(self, 'power_pole_information') else []
            }
            
            # Ensure file path has a .json extension
            if not file_path.endswith('.json'):
                file_path += '.json'
                
            # Write data to the file as JSON
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            
            logger.info(f"Production tree data successfully stored to {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to store production tree data: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info(f"Factory data successfully stored to {file_path}")
        return True
            
    def visualize_factory(self, paths=None, placed_inserter_information=None, cell_size=50, store=False, file_path=None):
        """
        Create a visual representation of the factory layout using Pygame.
        
        This method generates a graphical visualization of the complete factory
        including assemblers, inserters, belts, pipes, and power poles. The
        visualization can be displayed interactively or saved to a file.
        
        Args:
            paths (dict, optional): Dictionary of belt/pipe paths to visualize
            placed_inserter_information (list, optional): List of additional inserter placements
            cell_size (int, optional): Size of each grid cell in pixels. Defaults to 50.
            store (bool, optional): Whether to save the visualization to a file. Defaults to False.
            file_path (str, optional): Path to save the visualization image
            
        Returns:
            None
        """
        logger.info(f'Starting factory layout visualization with cell size {cell_size}')
        
        # Initialize pygame
        pygame.init()
        window_width = self.grid_width * cell_size
        window_height = self.grid_height * cell_size
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Module Layout Visualization')
        clock = pygame.time.Clock()
        
        # Always use stored data for visualization to ensure consistency
        belt_point_information = self.belt_point_information if hasattr(self, 'belt_point_information') else []
        assembler_information = self.assembler_information if hasattr(self, 'assembler_information') else []
        inserter_information = self.inserter_information if hasattr(self, 'inserter_information') else []
        fluid_connection_info = []  # Not stored in JSON
        
        # Make a copy of inserter_information to avoid modifying the original
        inserter_information = list(inserter_information)
        
        # If we don't have stored data yet and the solver is available, get it from there
        if (not belt_point_information or not assembler_information or not inserter_information) and self.z3_solver is not None:
            logger.info("No stored data found, fetching from solver")
            # Before getting data from the solver, check if we should preserve existing inserters
            existing_inserters = {}
            if hasattr(self, 'inserter_information') and self.inserter_information:
                for inserter_data in self.inserter_information:
                    if len(inserter_data) >= 3:
                        pos = (inserter_data[1], inserter_data[2])
                        existing_inserters[pos] = inserter_data
            
            # Get data from solver
            _, belt_point_information, assembler_information, new_inserter_info, fluid_connection_info, power_pole_information = self.z3_solver.build_map()
            
            # Merge with existing inserters if applicable
            if existing_inserters:
                for inserter_data in new_inserter_info:
                    if len(inserter_data) >= 3:
                        pos = (inserter_data[1], inserter_data[2])
                        existing_inserters.pop(pos, None)  # Remove from existing if found in new
                
                # Add back any existing inserters that weren't in the new data
                for pos, inserter_data in existing_inserters.items():
                    new_inserter_info.append(inserter_data)
                    logger.info(f"Preserving existing inserter: {inserter_data[0]} at position {pos}")
            
            # Store data for future use
            self.belt_point_information = belt_point_information
            self.assembler_information = assembler_information
            self.inserter_information = new_inserter_info            # Store power pole information for drawing if not already set
            # Always preserve existing power pole information if it exists
            if not hasattr(self, 'power_pole_information') or not self.power_pole_information:
                logger.info(f"Using new power pole information with {len(power_pole_information)} poles")
                self.power_pole_information = power_pole_information
            else:
                logger.info(f"Preserving existing power pole information with {len(self.power_pole_information)} poles")
          # Process additional inserters from placed_inserter_information
        if placed_inserter_information:
            # Track existing inserter positions to avoid duplicates
            existing_positions = set()
            for inserter in inserter_information:
                if len(inserter) >= 3:
                    existing_positions.add((inserter[1], inserter[2]))
            
            for item_key, inserters in placed_inserter_information.items():
                # Extract base item name
                item_name = item_key.split('_')[0] if '_' in item_key else item_key
                
                for pos_str, inserter_pos in inserters.items():
                    # Only add if position is not already occupied
                    if (inserter_pos[0], inserter_pos[1]) not in existing_positions:
                        inserter_information.append((item_name, inserter_pos[0], inserter_pos[1]))
                        existing_positions.add((inserter_pos[0], inserter_pos[1]))
                        logger.info(f"Added placed inserter for {item_name} at position ({inserter_pos[0]}, {inserter_pos[1]})")
                    else:
                        logger.info(f"Skipped duplicate inserter at position ({inserter_pos[0]}, {inserter_pos[1]})")
        
        # Load and scale images
        images = self._load_factory_images(cell_size)
        item_images = self._load_item_images(cell_size)
        
        # Draw factory views
        # 1. Base view without paths
            
        self._draw_factory_base(window, assembler_information, inserter_information, item_images, images, cell_size)
        self._draw_io_points(window, item_images, cell_size)
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
            'assembler': f'assets/{self.config["machines"]["default_assembler"]}.png',
            'inserter': f'assets/{self.config["inserters"]["input_type"]}.png',
            'conveyor': f"assets/{self.config['belts']['default_type']}.png",
            'underground': f"assets/{self.config['belts']['underground_type']}.png",
            'splitter': 'assets/splitter.png',
            'pipe': 'assets/pipe.png',  
            'pipe-underground': 'assets/pipe-to-ground.png',  
            'chemical-plant': 'assets/chemical-plant.png',  
            'oil-refinery': 'assets/oil-refinery.png',
            'small-electric-pole': 'assets/small-electric-pole.png',
            'medium-electric-pole': 'assets/medium-electric-pole.png',
            'big-electric-pole': 'assets/big-electric-pole.png',
            'substation': 'assets/substation.png'
        }
        
        for key, path in image_files.items():
            try:
                image = pygame.image.load(path)
                if key == 'assembler':
                    images[key] = pygame.transform.scale(image, (3 * cell_size, 3 * cell_size))
                elif key == 'splitter':
                    # Splitter will be scaled when used based on orientation
                    images[key] = image
                else:
                    images[key] = pygame.transform.scale(image, (cell_size, cell_size))
            except Exception as e:
                logger.info(f"Could not load image for {key}: {e}")
                # Create fallback image
                if "pole" in key:
                    fallback = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                    pygame.draw.rect(fallback, (100, 100, 100, 255), (cell_size//4, cell_size//4, cell_size//2, cell_size//2))
                    pygame.draw.line(fallback, (50, 50, 50, 255), (cell_size//2, 0), (cell_size//2, cell_size), 2)
                    pygame.draw.line(fallback, (50, 50, 50, 255), (0, cell_size//2), (cell_size, cell_size//2), 2)
                    images[key] = fallback
        
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
        for assembler_item, assembler_x, assembler_y,width,height,machine_type,orientation_idx  in assembler_information:
            pixel_x = assembler_x * cell_size
            pixel_y = assembler_y * cell_size
            
            # Choose the correct image based on machine_type
            if machine_type == "chemical-plant" and "chemical-plant" in images:
                machine_image = images["chemical-plant"]
            elif machine_type == "oil-refinery" and "oil-refinery" in images:
                machine_image = images["oil-refinery"]
            elif machine_type in images:
                # Use specific machine image if available
                machine_image = images[machine_type]
            else:
                # Fall back to default assembler
                machine_image = images['assembler']
        
            window.blit(machine_image, (pixel_x, pixel_y))
            
            # Draw item icon on top of assembler
            if assembler_item in item_images:
                window.blit(item_images[assembler_item], (pixel_x, pixel_y))
                
        # Draw power poles if they exist
        if hasattr(self, 'power_pole_information') and self.power_pole_information:
            # Load power pole image if it doesn't exist yet
            if 'power-pole' not in images:
                try:
                    pole_img = pygame.image.load("assets/medium-electric-pole.png")
                    images['power-pole'] = pygame.transform.scale(pole_img, (cell_size, cell_size))
                except:
                    # Create a basic pole image if asset is missing
                    pole_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                    pygame.draw.rect(pole_surface, (100, 100, 100, 255), (cell_size//4, cell_size//4, cell_size//2, cell_size//2))
                    pygame.draw.line(pole_surface, (50, 50, 50, 255), (cell_size//2, 0), (cell_size//2, cell_size), 2)
                    pygame.draw.line(pole_surface, (50, 50, 50, 255), (0, cell_size//2), (cell_size, cell_size//2), 2)
                    images['power-pole'] = pole_surface
            
            for pole_type, pole_x, pole_y in self.power_pole_information:
                pixel_x = pole_x * cell_size
                pixel_y = pole_y * cell_size
                
                # Choose specific pole image if available, otherwise use generic
                pole_key = pole_type if pole_type in images else 'power-pole'
                pole_img = images[pole_key]
                  # First, draw a colored grid to show power coverage area
                pole_radius = self.z3_solver.power_pole_radius if hasattr(self.z3_solver, 'power_pole_radius') else 2.5
                # Get the center of the pole
                center_x_cell = pole_x + self.z3_solver.power_pole_width // 2 if hasattr(self.z3_solver, 'power_pole_width') else pole_x
                center_y_cell = pole_y + self.z3_solver.power_pole_height // 2 if hasattr(self.z3_solver, 'power_pole_height') else pole_y
                
                # Calculate the grid cells affected by this power pole
                for dx in range(-int(pole_radius), int(pole_radius) + 1):
                    for dy in range(-int(pole_radius), int(pole_radius) + 1):
                        cell_x = center_x_cell + dx
                        cell_y = center_y_cell + dy
                        
                        # Skip if out of bounds
                        if cell_x < 0 or cell_x >= self.grid_width or cell_y < 0 or cell_y >= self.grid_height:
                            continue
                        
                        # Create a rectangle for this cell
                        cell_rect = pygame.Rect(cell_x * cell_size, cell_y * cell_size, cell_size, cell_size)
                        
                        # Color the cell with translucent blue to show power coverage
                        coverage_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                        pygame.draw.rect(coverage_surface, (0, 100, 255, 40), (0, 0, cell_size, cell_size))
                        window.blit(coverage_surface, (cell_x * cell_size, cell_y * cell_size))
                        
                        # Draw a border around the coverage cell
                        #pygame.draw.rect(window, (0, 100, 255, 100), cell_rect, 1)
                
                # Now draw the power pole itself on top
                window.blit(pole_img, (pixel_x, pixel_y))
                
                # Draw a square outline to show the exact coverage area
                radius_pixels = int(pole_radius * cell_size)
                center_x = pixel_x + cell_size 
                center_y = pixel_y + cell_size 
                coverage_rect = pygame.Rect(
                    center_x - radius_pixels,
                    center_y - radius_pixels,
                    radius_pixels * 2,
                    radius_pixels * 2
                )
                #pygame.draw.rect(window, (0, 0, 139, 150) , coverage_rect, 2)
                
        # Draw inserters
        for inserter_data in inserter_information:
            if len(inserter_data) >= 3:  # Ensure we have at least item, x, y
                inserter_item = inserter_data[0]
                inserter_x = inserter_data[1]
                inserter_y = inserter_data[2]
                # Direction may be missing, default to "north" if not provided
                direction = inserter_data[3] if len(inserter_data) >= 4 else "north"
                
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
        if '_' in path_data.get('item', ''):
            item_name = path_data['item'].split('_')[0]
        else:
            item_name = path_data.get('item', '')
        
        # Check if this is a fluid item
        is_fluid = self.is_fluid_item(item_name) if item_name else False
        
        logger.info(f"Drawing path: {path}")
        logger.info(f"Underground segments: {path_data.get('underground_segments', {})}")
        logger.info(f"Is fluid: {is_fluid}")
        
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
                logger.info(f"Skipping underground segment: {current} -> {next_node}")
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
        
             # Choose the appropriate image based on fluid vs solid
            if is_fluid:
                # For fluids, use pipe image
                belt_img = images['pipe']
            else:
                # For solids, use transport belt with rotation
                belt_img = images['conveyor']
                belt_img = pygame.transform.rotate(belt_img, angle)
         
            logger.info(f"Drawing {'pipe' if is_fluid else 'belt'} at: {current}")
            window.blit(belt_img, (current[0] * cell_size, current[1] * cell_size))
            
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
                    logger.info(f"Invalid direction from {second_last} to {last_pos}")
                    return  # Skip if not a direct connection
                
                if is_fluid:
                    # For fluids, use pipe image (may not need rotation)
                    final_img = images['pipe']
                    logger.info(f"Drawing final pipe at: {last_pos}")
                else:
                    # For solids, use transport belt with rotation
                    final_img = images['conveyor']
                    
                    # Apply appropriate rotation based on whether it's an input or output path
                    if is_output_path:
                        final_img = pygame.transform.rotate(final_img, angle)
                    else:
                        final_img = pygame.transform.rotate(final_img, (angle + 180) % 360)
                    logger.info(f"Drawing final belt at: {last_pos} with angle: {angle}")
                
                window.blit(final_img, (last_pos[0] * cell_size, last_pos[1] * cell_size))

    def _is_underground_segment(self, pos1, pos2, path_data):
        """Check if a segment between two positions is part of an underground belt."""
        if 'underground_segments' not in path_data or not path_data['underground_segments']:
            return False
        
        for segment_id, segment in path_data['underground_segments'].items():
            if pos1 == segment['start'] or pos2 == segment['end']:
                return True
        
        return False

    def _draw_underground_segments(self, window, path_data, is_output_path, images, cell_size):
  
        if 'underground_segments' not in path_data or not path_data['underground_segments']:
            return
        
        if '_' in path_data.get('item', ''):
            item_name = path_data['item'].split('_')[0]
        else:
            item_name = path_data.get('item', '')
            
        # Check if this is a fluid item
        is_fluid = self.is_fluid_item(item_name) if item_name else False
        logger.info(f"Drawing underground segments: {path_data['underground_segments']}")
        logger.info(f"Is fluid: {is_fluid}")
        
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
                logger.info(f"Invalid underground direction: dx={dx}, dy={dy}")
                continue  # Skip invalid directions
                
            if is_fluid:
                # For fluids, use pipe-underground
                conveyor_img = images['pipe']
                underground_img = images['pipe-underground']
                entrance_text = "underground pipe entrance"
                exit_text = "underground pipe exit"
            else:
                # For solids, use transport belt and underground belt
                conveyor_img = images['conveyor']
                underground_img = images['underground']
                entrance_text = "underground belt entrance"
                exit_text = "underground belt exit"
        
            # First draw base images underneath both entrance and exit
            if not is_fluid:
                conveyor_entrance = pygame.transform.rotate(conveyor_img, belt_angle)
                conveyor_exit = pygame.transform.rotate(conveyor_img, belt_angle)
            else:
                conveyor_entrance = conveyor_img  # Pipes don't rotate
                conveyor_exit = conveyor_img
            
            # Draw the base images underneath
            window.blit(conveyor_entrance, (start_pos[0] * cell_size, start_pos[1] * cell_size))
            window.blit(conveyor_exit, (end_pos[0] * cell_size, end_pos[1] * cell_size))
        
            # Draw entrance and exit
            if not is_fluid:
                # For belts, rotate the underground image
                entrance_img = pygame.transform.rotate(underground_img, entrance_angle)
                exit_img = pygame.transform.rotate(underground_img, exit_angle)
                exit_img = pygame.transform.flip(exit_img, False, True)  # Flip horizontally
            else:
                # For pipes, handle differently if needed
                entrance_img = pygame.transform.rotate(underground_img, entrance_angle)
                exit_img = pygame.transform.rotate(underground_img, exit_angle)
            
            logger.info(f"Drawing {entrance_text} at: {start_pos}")
            window.blit(entrance_img, (start_pos[0] * cell_size, start_pos[1] * cell_size))
            
            logger.info(f"Drawing {exit_text} at: {end_pos}")
            window.blit(exit_img, (end_pos[0] * cell_size, end_pos[1] * cell_size))

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
  
  
    def _draw_io_points(self, window, item_images, cell_size):
        """Draw input/output points with colored backgrounds."""
        logger.info("Drawing input/output points")
        
        # Process input information
        if self.input_information:
            logger.info(f"Input information available for {len(self.input_information)} items: {list(self.input_information.keys())}")
            
            for item in self.input_information:
                # Correctly access input position array
                if self.input_information[item].get('input'):
                    input_pos = self.input_information[item]['input']
                    row, col = input_pos[0], input_pos[1]  # Correctly unpack the array
                    
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    # Draw green background for input
                    logger.info(f"Drawing INPUT GREEN rect for {item} at ({row},{col})")
                    pygame.draw.rect(window, GREEN, rect)
                    if item in item_images:
                        logger.info(f"Drawing item image for {item}")
                        window.blit(item_images[item], rect)
                    else:
                        logger.warning(f"Item image for {item} not found")
                
                # Correctly access output position array
                if self.input_information[item].get('output'):
                    output_pos = self.input_information[item]['output']
                    row, col = output_pos[0], output_pos[1]  # Correctly unpack the array
                    
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    # Draw red background for output
                    logger.info(f"Drawing INPUT RED rect for {item} at ({row},{col})")
                    pygame.draw.rect(window, RED, rect)
                    if item in item_images:
                        logger.info(f"Drawing item image for {item}")
                        window.blit(item_images[item], rect)
                    else:
                        logger.warning(f"Item image for {item} not found")
        else:
            logger.info("No input_information available")
            
        # Process output information
        if self.output_information:
            logger.info(f"Output information available for {len(self.output_information)} items: {list(self.output_information.keys())}")
            
            # Load output image if needed
            output_image = None
            if self.output_item in item_images:
                output_image = item_images[self.output_item]
                logger.info(f"Output image loaded for {self.output_item}")
            else:
                logger.warning(f"Output image for {self.output_item} not found in item_images")
            
            for item in self.output_information:
                # Correctly access input position array for output items
                if self.output_information[item].get('input'):
                    input_pos = self.output_information[item]['input']
                    row, col = input_pos[0], input_pos[1]  # Correctly unpack the array
                    
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    # Draw green background for input
                    logger.info(f"Drawing OUTPUT GREEN rect for {item} at ({row},{col})")
                    pygame.draw.rect(window, GREEN, rect)
                    if output_image:
                        logger.info(f"Drawing output image for {item}")
                        window.blit(output_image, rect)
                    else:
                        logger.warning("No output image available")
                        
                # Correctly access output position array for output items
                if self.output_information[item].get('output'):
                    output_pos = self.output_information[item]['output']
                    row, col = output_pos[0], output_pos[1]  # Correctly unpack the array
                    
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    # Draw red background for output
                    logger.info(f"Drawing OUTPUT RED rect for {item} at ({row},{col})")
                    pygame.draw.rect(window, RED, rect)
                    if output_image:
                        logger.info(f"Drawing output image for {item}")
                        window.blit(output_image, rect)
                    else:
                        logger.warning("No output image available")
        else:
            logger.info("No output_information available")
  
  
    def calculate_max_output(self):
        """
        Calculate the maximum output rate for the current factory configuration.
        
        This method determines the theoretical maximum production rate based on
        the number of assemblers, machine type, and recipe parameters for the
        main output item.
        
        Returns:
            float: Maximum output rate in items per minute
        """
        logger.info(f"Calculating maximum output for {self.output_item}")
        
        # Get the item recipe
        recipe = self.item_lookup[self.output_item].get("recipe", {})
        
        if not recipe:
            logger.warning(f"No recipe found for {self.output_item}")
            return 0
        
        # Get the time per unit and yield per recipe
        time_per_unit = recipe.get("time", 1)
        yield_per_recipe = recipe.get("yield", 1)
        
        # Get the machine type for this recipe
        machine_type = self._get_machine_type_for_recipe(self.output_item)
        
        # Get the crafting speed for this machine type
        if machine_type in self.machines_data["assemblers"]:
            crafting_speed = self.machines_data["assemblers"][machine_type].get("crafting_speed", 1)
        else:
            # Fallback to default assembler if machine type not found
            default_type = self.config["machines"]["default_assembler"]
            crafting_speed = self.machines_data["assemblers"][default_type].get("crafting_speed", 1)
        
        # Calculate cycles per minute
        cycles_per_minute = 60 / time_per_unit
        
        # Get the number of assemblers
        assembler_count = self.production_data.get(self.output_item, {}).get('assemblers', 0)
        
        # Calculate and return the maximum output
        max_output = assembler_count * crafting_speed * cycles_per_minute * yield_per_recipe
        logger.info(f"Maximum output calculated: {max_output} items/min "
                   f"({assembler_count} assemblers × {crafting_speed} speed × {cycles_per_minute} cycles/min × {yield_per_recipe} yield)")
        return max_output
        
    def count_assemblers(self,production_data):
        """
        Count the number of assemblers for each item in the production data.
        
        This method extracts assembler counts from the production data and
        returns them in a simplified dictionary format.
        
        Args:
            production_data (dict): Dictionary containing production data for all items
            
        Returns:
            dict: Dictionary mapping item IDs to their assembler counts
        """
        logger.debug("Counting assemblers for all production items")
        
        assembler_counts = {}
    
        for item, data in production_data.items():
            # Check if the item has an 'assemblers' field
            if 'assemblers' in data:
                assembler_counts[item] = data['assemblers']
            else:
                assembler_counts[item] = 0  # Assume 0 assemblers if the field is not present
                
        logger.debug(f"Assembler counts: {assembler_counts}")
        return assembler_counts
    
    def create_blueprint(self, json_path, output_path):
        """
        Create a Factorio blueprint from saved factory data using Draftsman.
        
        This method loads factory data from a JSON file and converts it into
        a Factorio blueprint format that can be imported into the game.
        The blueprint includes all assemblers, inserters, belts, pipes, and
        power poles with correct positioning and connections.
        
        Args:
            json_path (str): Path to the JSON file containing factory data
            output_path (str): Path where the blueprint file should be saved
            
        Returns:
            bool: True if blueprint was created successfully, False otherwise
        """
        logger.info(f"Creating Factorio blueprint from {json_path}")
        
        try:
            # Load the saved factory data
            if not self.load_data(json_path):
                logger.error(f"Failed to load factory data from {json_path}")
                return False
                
            logger.info(f"Creating blueprint from {json_path}")
            
            # Create a new blueprint with a name based on output item and amount
            blueprint = Blueprint()
            blueprint.label = f"{self.output_item}_{self.amount}_per_minute"
            blueprint.description = f"Factory producing {self.amount} {self.output_item} per minute"
            
            logger.info(f"Blueprint created with label: {blueprint.label}")
            
            # Track occupied positions to avoid overlap
            occupied_positions = set()
            
            # Track assembler positions for orienting inserters
            assembler_positions = {}  # Maps (x, y) to assembler object
            
            logger.info("1. Placing assembling machines...")
            # Place assembling machines from assembler_information
            if hasattr(self, 'assembler_information'):
                for item, x, y,width,height,machine_type,orientation_idx in self.assembler_information:
                    # Create assembling machine
                    center_x = x + int(width/2)
                    center_y = y + int(height/2)
                    
                    logger.info(f"  - Placing assembler for {item} at ({center_x},{center_y}), facing {orientation_idx}")
                    
                    direction_mapping = {
                        0: Direction.WEST,  
                        1: Direction.NORTH,   
                        2: Direction.EAST,  
                        3: Direction.SOUTH    
                    }
                    
                    assembler = AssemblingMachine(
                        name=machine_type, 
                        position=(center_x, center_y),
                        recipe=item,  # Set the recipe to the item it produces
                        direction=direction_mapping.get(orientation_idx)
                        
                    )
                    blueprint.entities.append(assembler)
                    
                    # Store the assembler in our mapping
                    for dx in range(width):
                        for dy in range(height):
                            assembler_positions[(x + dx, y + dy)] = (item, (x, y), (center_x, center_y))
                    
                    # Mark the 3x3 area as occupied
                    for dx in range(width):
                        for dy in range(height):
                            occupied_positions.add((x + dx, y + dy))


          
            logger.info("2. Placing inserters...")
            # Place inserters from inserter_information - check for overlaps and orient properly
            if hasattr(self, 'inserter_information'):
                for item, x, y , direction in self.inserter_information:
                    
                    
                    # Skip if position is occupied by an assembler
                    if (x, y) in occupied_positions:
                        logger.info(f"  - Skipping inserter at ({x},{y}) due to overlap with assembler")
                        continue
                    
                    logger.info(direction)
                    
                    if self.is_fluid_item(item):
                        logger.info(f"  - Skipping inserter for fluid item {item} at ({x},{y}) - using pipe")
                        
                        # Add pipe instead (no direction needed)
                        pipe = Pipe(  #
                            name="pipe",
                            position=(x, y)
                        )
                        blueprint.entities.append(pipe)
                        occupied_positions.add((x, y))
                        continue
                    
                    # Convert direction string to draftsman Direction constant # rotate by 180 degress
                    if direction == "north":
                        blueprint_direction = Direction.SOUTH
                    elif direction == "east":
                        blueprint_direction = Direction.WEST
                    elif direction == "south":
                        blueprint_direction = Direction.NORTH
                    elif direction == "west":
                        blueprint_direction = Direction.EAST
                    else:
                        blueprint_direction = Direction.NORTH  # Default
                    
                    # Create inserter
                    logger.info(f"  - Placing inserter for {item} at ({x},{y}) facing {blueprint_direction}")
                    inserter = Inserter(
                        name="inserter",
                        position=(x, y),
                        direction=blueprint_direction
                    )
                    blueprint.entities.append(inserter)
                    occupied_positions.add((x, y))
                    
                          
            logger.info("3. Adding additional inserters...")
            # Add placed output inserters from the blueprint data
            if hasattr(self, 'placed_inserter_information') and self.placed_inserter_information:
                logger.info("  - Adding placed output inserters...")
                for item_key, inserter_positions in self.placed_inserter_information.items():
                    logger.info(f"    - Processing output inserters for {item_key}")
                    for dest_str, src_pos in inserter_positions.items():
                        # Parse dest_pos if it's a string
                        if isinstance(dest_str, str):
                            try:
                                dest_pos = ast.literal_eval(dest_str)
                            except (ValueError, SyntaxError):
                                logger.info(f"    - Skipping inserter due to invalid destination: {dest_str}")
                                continue
                        else:
                            dest_pos = dest_str                        # Skip if position is occupied
                        src_tuple = tuple(src_pos) if isinstance(src_pos, list) else src_pos
                        if src_tuple in occupied_positions:
                            logger.info(f"    - Skipping output inserter at {src_pos} due to overlap")
                            continue
                        
                        # Calculate direction from source to destination
                        dx = dest_pos[0] - src_pos[0]
                        dy = dest_pos[1] - src_pos[1]
                        
                        # Normalize direction
                        if dx != 0:
                            dx = dx // abs(dx)
                        if dy != 0:
                            dy = dy // abs(dy)
                        
                        # Determine inserter direction
                        
                        if dx == 1:
                            direction = Direction.WEST
                        elif dx == -1:
                            direction = Direction.EAST
                        elif dy == 1:
                            direction = Direction.NORTH
                        else:
                            direction = Direction.SOUTH
                          # Create and place inserter
                        logger.info(f"    - Placing output inserter for {item_key} at {src_pos} facing {direction}")
                        inserter = Inserter(
                            name="inserter",
                            position=src_pos,
                            direction=direction
                            
                        )   
                        blueprint.entities.append(inserter)
                        occupied_positions.add(src_tuple)

            # Place belts from paths generated by the pathfinder
            logger.info("4. Adding path belts...")
            self._add_path_belts_to_blueprint(blueprint, occupied_positions)
           
            # Place belts for I/O paths with more detailed logger
            logger.info("5. Adding I/O belts...")
            self._add_io_belts_to_blueprint(blueprint, occupied_positions)
            
            # In create_blueprint method, after placing inserters
            logger.info("6. Placing power poles...")
            if hasattr(self, 'power_pole_information'):
                for pole_type, x, y in self.power_pole_information:
                    logger.info(f"  - Placing {pole_type} at ({x},{y})")
                    power_pole = ElectricPole(
                        name=pole_type,
                        position=(x, y)
                    )
                    blueprint.entities.append(power_pole)
                    occupied_positions.add((x, y))

            # Add this code after placing all power poles in create_blueprint method

            logger.info("6b. Connecting power poles...")
            # Create a list to store all placed power poles for easier access
            power_poles = []
            for entity in blueprint.entities:
                if isinstance(entity, ElectricPole):
                    power_poles.append(entity)

            # Connect each pole to nearby poles within its connection radius
            if power_poles:
                logger.info(f"  - Found {len(power_poles)} power poles to connect")
                
                # For each pole, find other poles within reach and connect them
                for i, pole in enumerate(power_poles):
                    # Get pole connection radius from entity data
                    pole_name = pole.name
                    connection_distance = entities.raw.get(pole_name, {}).get("maximum_wire_distance", 9)
                    x, y = pole.position
                    pole_pos = Vector(x, y)
                    
                    
                    logger.info(f"  - Processing pole {i+1}/{len(power_poles)} at {pole_pos} with range {connection_distance}")
                    
                    # Find nearby poles to connect to
                    connected = 0
                    for other_pole in power_poles:
                        # Skip connecting to self
                        if other_pole is pole:
                            continue
                            
                        
                        other_x, other_y = other_pole.position
                        other_pos = Vector(other_x, other_y)
                        
                        # Calculate distance between poles
                        distance = ((pole_pos - other_pos).x ** 2 + (pole_pos - other_pos).y ** 2) ** 0.5
                        
                        # If in range and not already connected, connect them
                        if distance <= connection_distance:
                            try:
                                # Check if they're already connected to avoid duplicates
                                already_connected = False
                                if hasattr(pole, 'connections') and pole.connections:
                                    # Check if there's already a connection to this pole
                                    if 'copper' in pole.connections:
                                        for connection in pole.connections['copper']:
                                            if 'entity_id' in connection and connection['entity_id'] == other_pole.entity_number:
                                                already_connected = True
                                                break
                                
                                if not already_connected:
                                    pole.connect_power(other_pole)
                                    connected += 1
                                    logger.debug(f"    - Connected to pole at {other_pos} (distance: {distance:.2f})")
                            except Exception as e:
                                logger.warning(f"    - Failed to connect poles: {e}")
                    
                    logger.info(f"  - Pole at {pole_pos} connected to {connected} other poles")
    
            # Export the blueprint to a file
            logger.info("7. Exporting blueprint...")
            with open(output_path, "w") as f:
                f.write(blueprint.to_string())
            
            
            logger.info(f"Blueprint successfully exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating blueprint: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _add_belt_path_to_blueprint(self, blueprint, path_data, item, occupied_positions):
        
        # Determine if this is an output path
        is_output_path = item in self.output_information if hasattr(self, 'output_information') else False
        
        is_fluid = self.is_fluid_item(item)
        
        if is_fluid:
            belt_type = self.config.get("pipes", {}).get("default_type", "pipe")
            underground_type = self.config.get("pipes", {}).get("underground_type", "pipe-to-ground")
        else:
            belt_type = self.config["belts"]["default_type"]
            underground_type = self.config["belts"]["underground_type"]
            
        """Add a specific belt path to the blueprint, avoiding overlaps"""
        path = path_data.get('path', [])
        if not path:
            return
        
        logger.info(f"  - Processing {'output' if is_output_path else 'input/regular'} belt path for {item} with {len(path)} points")
        
        # Get orientation information
        has_orientation = 'orientation' in path_data and path_data['orientation']
        
        # Handle splitters - check if they exist either as objects or boolean values
        start_splitter = path_data.get('start_splitter')
        dest_splitter = path_data.get('dest_splitter')
        
        # check if we have a dest splitter if so palce a tarnsport belt at the destination point, facing the splitter
        if dest_splitter:
            logger.info(f"  - Found destination splitter for {item}: {dest_splitter}")
            dest_point = path_data.get('destination', None)
            if dest_point:
                logger.info(f"  - Destination point for {item} is {dest_point}")
                dest_splitter_pos = dest_splitter.get('position', None)
                dx = dest_splitter_pos[0] - dest_point[0]
                dy = dest_splitter_pos[1] - dest_point[1]
                
                print(f"  - Destination splitter direction vector (dx={dx}, dy={dy})")
                
                if dx > 0:
                    des_direction= Direction.SOUTH
                elif dx < 0:
                    des_direction = Direction.NORTH
                elif dy > 0:
                    des_direction = Direction.EAST
                elif dy < 0:
                    des_direction = Direction.WEST
                else:
                    des_direction = Direction.EAST
                
                
                belt = TransportBelt(
                    name=belt_type,
                    position=tuple(dest_point),
                    direction=des_direction
                )
                blueprint.entities.append(belt)
                occupied_positions.add(tuple(dest_point))
                    
                
        
        blueprint_direction_start = None
        blueprint_direction_dest = None
        
        # Skip positions where splitters will be placed
        splitter_positions = set()
        
        # Track splitter information for later placement
        if not is_fluid and start_splitter is not None:
            # Check if start_splitter is a dictionary/object with position info or just a boolean
            if isinstance(start_splitter, dict) and 'position' in start_splitter:
                splitter_pos = start_splitter['position']
                direction = start_splitter['direction']
                
                splitter_tuple = tuple(splitter_pos) if isinstance(splitter_pos, list) else splitter_pos
                splitter_positions.add(splitter_tuple)
            
                dx, dy = direction
                if dx == 1 and dy == 0:  # Right
                    blueprint_direction_start = Direction.EAST
                elif dx == -1 and dy == 0:  # Left
                    blueprint_direction_start = Direction.WEST
                elif dx == 0 and dy == 1:  # Down
                    blueprint_direction_start = Direction.SOUTH
                elif dx == 0 and dy == -1:  # Up
                    blueprint_direction_start = Direction.NORTH
                else:
                    # Default to EAST if we can't determine direction
                    blueprint_direction_start = Direction.EAST

                # place the splitter in the blueprint
                logger.info(f"  - Adding start splitter for {item} at {splitter_pos} facing {direction}")
                start_splitter = BlueprintSplitter(
                    name="splitter",
                    position=splitter_pos,
                    direction=blueprint_direction_start
                )
                blueprint.entities.append(start_splitter)
                occupied_positions.add(splitter_tuple)
        
                
        if not is_fluid and dest_splitter is not None:
            # Check if dest_splitter is a dictionary/object with position info or just a boolean
            if isinstance(dest_splitter, dict) and 'position' in dest_splitter:
                splitter_pos = dest_splitter['position']
                direction = dest_splitter['direction']
                
                # Debug output for dest_splitter processing
                is_output_path = item in self.output_information if hasattr(self, 'output_information') else False
                logger.info(f"  - DEBUG: Processing destination splitter for {'output path' if is_output_path else 'input/regular path'}")
                logger.info(f"  - DEBUG: Destination splitter position: {splitter_pos}, direction: {direction}")
                
                splitter_tuple = tuple(splitter_pos) if isinstance(splitter_pos, list) else splitter_pos
                splitter_positions.add(splitter_tuple)
                
                dx, dy = direction
                logger.info(f"  - DEBUG: Destination splitter direction vector (dx={dx}, dy={dy})")
                
                if dx == 1 and dy == 0:  # Right
                    blueprint_direction_dest = Direction.EAST
                    logger.info(f"  - DEBUG: Setting blueprint_direction_dest to EAST")
                elif dx == -1 and dy == 0:  # Left
                    blueprint_direction_dest = Direction.WEST
                    logger.info(f"  - DEBUG: Setting blueprint_direction_dest to WEST")
                elif dx == 0 and dy == 1:  # Down
                    blueprint_direction_dest = Direction.SOUTH
                    logger.info(f"  - DEBUG: Setting blueprint_direction_dest to SOUTH")
                elif dx == 0 and dy == -1:  # Up
                    blueprint_direction_dest = Direction.NORTH
                    logger.info(f"  - DEBUG: Setting blueprint_direction_dest to NORTH")
                else:
                    # Default to EAST if we can't determine direction
                    blueprint_direction_dest = Direction.EAST
                    logger.info(f"  - DEBUG: Setting blueprint_direction_dest to default EAST")
           
               
                # place the splitter in the blueprint
                logger.info(f"  - Adding destination splitter for {item} at {splitter_pos} facing {direction}")
                dest_splitter = BlueprintSplitter(
                    name="splitter",
                    position=splitter_pos,
                    direction=blueprint_direction_dest
                )
                blueprint.entities.append(dest_splitter)
                
                occupied_positions.add(splitter_tuple)
        
        # Process underground segments
        underground_positions = set()
        if 'underground_segments' in path_data and path_data['underground_segments']:
            logger.info(f"  - Found {len(path_data['underground_segments'])} underground segments")
            for segment_id, segment in path_data['underground_segments'].items():
                start_pos = segment['start']
                end_pos = segment['end']
                
                # Skip if positions are occupied or are splitter positions
                start_tuple = tuple(start_pos) if isinstance(start_pos, list) else tuple(start_pos)
                end_tuple = tuple(end_pos) if isinstance(end_pos, list) else tuple(end_pos)
        
                # Calculate direction
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                
                # Determine primary direction
                if abs(dx) > abs(dy):
                    direction = Direction.EAST if dx > 0 else Direction.WEST
                else:
                    direction = Direction.SOUTH if dy > 0 else Direction.NORTH
                if not is_fluid:
                    logger.info(f"  - Adding underground belt segment from {start_pos} to {end_pos} facing {direction}")
                    
                    # Create and place underground belt entrance
                    entrance = UndergroundBelt(
                        name=underground_type,
                        position=start_pos,
                        direction=direction,
                        type="input"  # This is the entrance
                    )
                    blueprint.entities.append(entrance)
                    occupied_positions.add(start_tuple)
                    underground_positions.add(start_tuple)
                    
                    # Create and place underground belt exit
                    exit_belt = UndergroundBelt(
                        name=underground_type,
                        position=end_pos,
                        direction=direction,
                        type="output"  # This is the exit
                    )
                    blueprint.entities.append(exit_belt)
                    occupied_positions.add(end_tuple)
                    underground_positions.add(end_tuple)
                else:
                    logger.info(f"  - Adding underground belt from {start_pos} to {end_pos} facing {direction}")
                    
                    # Create and place underground pipe entrance
                    entrance = UndergroundPipe (
                        name=underground_type,
                        position=start_pos,
                        direction=direction,
                        type="input"  # This is the entrance
                    )
                    blueprint.entities.append(entrance)
                    occupied_positions.add(start_tuple)
                    underground_positions.add(start_tuple)
                    
                    # Create and place underground pipe exit
                    exit_belt = UndergroundPipe(
                        name=underground_type,
                        position=end_pos,
                        direction=direction,
                        type="output"  # This is the exit
                    )
                    blueprint.entities.append(exit_belt)
                    occupied_positions.add(end_tuple)
                    underground_positions.add(end_tuple)
            
        # Process regular belt segments
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # Skip if this position is part of underground belt or a splitter
            current_tuple = tuple(current) if isinstance(current, list) else tuple(current)
            if current_tuple in underground_positions or current_tuple in splitter_positions:
                continue
            
            # Skip if position is occupied
            if current_tuple in occupied_positions:
                logger.info(f"  - Skipping belt at {current} due to overlap")
                continue
            
            # Calculate direction
            if has_orientation and path_data['orientation']:
                # Convert current position to tuple if it's a list (to make it hashable)
                current_key = str(tuple(current)) if isinstance(current, list) else str(current)
                
                # Check if this position has an orientation
                if current_key in path_data['orientation']:
                    direction = self._get_belt_direction(path_data['orientation'][current_key])
                else:
                    # Calculate direction from current to next position
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    direction = self._get_belt_direction((dx, dy))
            else:
                # Calculate direction from current to next position
                dx = next_pos[0] - current[0]
                dy = next_pos[1] - current[1]
                direction = self._get_belt_direction((dx, dy))
            
            # Create and place transport belt
            logger.info(f"  - Adding transport belt at {current} facing {direction}")
            if not is_fluid:
              
                belt = TransportBelt(
                    name=belt_type,
                    position=current,
                    direction=direction
                )
                blueprint.entities.append(belt)
                occupied_positions.add(current_tuple)
                
            else:
                pipe = Pipe(
                    name=belt_type,
                    position=current,
                    direction=direction
                )
                blueprint.entities.append(pipe)
                occupied_positions.add(current_tuple)
                
            # Process the last belt in the path if it's not an underground belt or splitter
            if len(path) > 1:
                
                logger.debug(f"  - Processing last belt in path for {item}")
                
                last_pos = path[-1]
                last_tuple = tuple(last_pos) if isinstance(last_pos, list) else tuple(last_pos)
                
                # Check if we need to place a belt at the destination point
                place_dest_belt = True
                
                # Check if this position is occupied
                if last_tuple in occupied_positions:
                    logger.info(f"  - Skipping last belt at {last_pos} due to overlap")
                    place_dest_belt = False
                
                # Check if this is an underground entrance/exit
                if last_tuple in underground_positions:
                    logger.info(f"  - Skipping last belt at {last_pos} due to underground position")
                    place_dest_belt = False                # Check if this is a splitter position
                if last_tuple in splitter_positions:
                    # Need to place a belt before the splitter to connect to it
                    logger.info(f"  - Last point at {last_pos} is a splitter position")
                    # We will place the belt, but with a direction facing the splitter
                    # This is handled later in the special case for dest_splitter
                
                if place_dest_belt:
                    # Determine the direction for the last belt
                    
                    # Special case: If we have a destination splitter,
                    # orient the last belt to face the splitter's input
                    if dest_splitter is not None and isinstance(dest_splitter, dict):
                        # Get the splitter position and calculate direction toward it
                        splitter_pos = dest_splitter.get('position')
                        if splitter_pos:
                            # Calculate vector from last belt to splitter
                            dx = splitter_pos[0] - last_pos[0]
                            dy = splitter_pos[1] - last_pos[1]
                            
                            # Enhanced debug logging
                            logger.info(f"  - DEBUG: Last belt position: {last_pos}")
                            logger.info(f"  - DEBUG: Destination splitter position: {splitter_pos}")
                            logger.info(f"  - DEBUG: dx={dx}, dy={dy}")
                            logger.info(f"  - DEBUG: Path is from {'output_information' if item in self.output_information else 'input_information or paths'}")
                            
                            # Set direction toward splitter
                            if dx > 0:  # Splitter is to the right
                                direction = Direction.EAST
                                logger.info(f"  - DEBUG: Setting direction to EAST (splitter is to the right)")
                            elif dx < 0:  # Splitter is to the left
                                direction = Direction.WEST
                                logger.info(f"  - DEBUG: Setting direction to WEST (splitter is to the left)")
                            elif dy > 0:  # Splitter is below
                                direction = Direction.SOUTH
                                logger.info(f"  - DEBUG: Setting direction to SOUTH (splitter is below)")
                            elif dy < 0:  # Splitter is above
                                direction = Direction.NORTH
                                logger.info(f"  - DEBUG: Setting direction to NORTH (splitter is above)")
                            else:
                                # Fallback to splitter direction if positions are the same
                                direction = blueprint_direction_dest or Direction.EAST
                                logger.info(f"  - DEBUG: Setting direction to fallback {direction}")
                                
                            logger.info(f"  - Adding last transport belt at {last_pos} facing toward splitter at {splitter_pos}")
                        else:
                            # If we can't determine splitter position, use blueprint direction
                            direction = blueprint_direction_dest or Direction.EAST
                            logger.info(f"  - DEBUG: Using blueprint_direction_dest due to missing splitter position: {direction}")
                            logger.info(f"  - Adding last transport belt at {last_pos} with splitter direction {direction}")
                    else:
                        # Normal case: calculate direction based on path
                        second_last = path[-2]
                        
                        # Calculate direction for last belt
                        if has_orientation and path_data['orientation']:
                            # Convert last position to tuple if it's a list
                            last_pos_key = str(tuple(last_pos)) if isinstance(last_pos, list) else str(last_pos)
                            
                            if last_pos_key in path_data['orientation']:
                                direction = self._get_belt_direction(path_data['orientation'][last_pos_key])
                            else:
                                # Calculate direction from second-last to last
                                dx = last_pos[0] - second_last[0]
                                dy = last_pos[1] - second_last[1]
                                direction = self._get_belt_direction((dx, dy))
                        else:
                            # Calculate direction from second-last to last
                            dx = last_pos[0] - second_last[0]
                            dy = last_pos[1] - second_last[1]
                            direction = self._get_belt_direction((dx, dy))
                    
                    # Create and place last transport belt
                    logger.info(f"  - Adding last transport belt at {last_pos} facing {direction}")
                    if not is_fluid:
                        belt = TransportBelt(
                            name=belt_type,
                            position=last_pos,
                            direction=direction
                        )
                        blueprint.entities.append(belt)
                        occupied_positions.add(last_tuple)
                    else:
                        pipe = Pipe(
                            name=belt_type,
                            position=last_pos,
                            direction=direction
                        )
                        blueprint.entities.append(pipe)  
                        occupied_positions.add(last_tuple)

    def _add_io_belts_to_blueprint(self, blueprint, occupied_positions):
        """Add user-defined I/O belts to the blueprint, avoiding overlaps"""
        # Process input belts
        if self.input_information:
            logger.info(f"  - Processing {len(self.input_information)} input belt routes")
            for item, data in self.input_information.items():
                if data['paths'] is not None and item in data['paths']:
                    for path_data in data['paths'][item]:
                        self._add_belt_path_to_blueprint(blueprint, path_data, item, occupied_positions)
        else:
            logger.info("  - No input belt routes found")
        
        # Process output belts
        if self.output_information:
            logger.info(f"  - Processing {len(self.output_information)} output belt routes")
            for item, data in self.output_information.items():
                if data['paths'] is not None and item in data['paths']:
                    for path_data in data['paths'][item]:
                        self._add_belt_path_to_blueprint(blueprint, path_data, item, occupied_positions)
        else:
            logger.info("  - No output belt routes found")

    def _add_path_belts_to_blueprint(self, blueprint, occupied_positions):
        """Add pathfinder-generated belts to the blueprint, avoiding overlaps"""
        if not hasattr(self, 'paths'):
            logger.info("  - No path data found")
            return
        
        path_count = sum(len(item_paths) for item_paths in self.paths.values())
        logger.info(f"  - Processing {path_count} pathfinder-generated belt paths for {len(self.paths)} items")
        
        for item_key, item_paths in self.paths.items():
            # Extract base item name
            item_name = item_key.split('_')[0] if '_' in item_key else item_key
            logger.info(f"  - Processing {len(item_paths)} paths for {item_name}")
            
            for path_data in item_paths:
                self._add_belt_path_to_blueprint(blueprint, path_data, item_name, occupied_positions)


    def _get_belt_direction(self, orientation):
        """Convert orientation tuple to Draftsman direction constant"""
        dx, dy = orientation
        
        if dx == 0 and dy == -1:  # Up
            return Direction.NORTH
        elif dx == 1 and dy == 0:  # Right
            return Direction.EAST
        elif dx == 0 and dy == 1:  # Down
            return Direction.SOUTH
        elif dx == -1 and dy == 0:  # Left
            return Direction.WEST
        else:
            # Default direction if orientation is invalid
            return Direction.NORTH

        

def plot_csv_data(file_path):
    """
    Generate PNG plots from execution time CSV data.
    
    This function reads CSV data containing execution times for different items and methods,
    then generates violin plots and box plots for analysis. The plots are saved in PNG format
    for easy viewing and integration into various document formats.
    
    Args:
        file_path (str): Path to the CSV file containing execution time data.
                        Expected columns: Item, Amount, Method, Assemblers, Execution Time (seconds)
    
    Returns:
        None: Plots are saved to the 'Plots' directory in PNG format
        
    Example:
        >>> plot_csv_data("execution_times.csv")
        # Generates plots in Plots/{item}/{method}/ directories
    """
    logger.info(f"Starting plot generation from {file_path}")
    
    # Configure matplotlib for high-quality PNG output
    logger.info("Configuring matplotlib for PNG output")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,  # High DPI for crisp PNG images
        'savefig.dpi': 300,  # High DPI for saved figures
        'savefig.format': 'png'
    })
    
    # Read and validate CSV data
    logger.info(f"Reading CSV data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows of data")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    # Ensure relevant columns are numeric and handle conversion errors
    logger.info("Processing and cleaning data columns")
    df['Execution Time (seconds)'] = pd.to_numeric(df['Execution Time (seconds)'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Parse the 'Assemblers' column as a dictionary (if it's a string representation)
    df['Assemblers'] = df['Assemblers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Extract the number of assemblers by summing dictionary values
    df['Assembler Count'] = df['Assemblers'].apply(lambda x: sum(x.values()) if isinstance(x, dict) else 0)
    logger.info(f"Data processing complete. Assembler counts range: {df['Assembler Count'].min()}-{df['Assembler Count'].max()}")

    # Get unique items and methods for plot generation
    items = df['Item'].unique()
    methods = df['Method'].unique()
    logger.info(f"Found {len(items)} unique items and {len(methods)} unique methods")

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
            
            # Save the violin plot as PNG
            violin_plot_path = os.path.join(method_plot_dir, f'{item}_{method}_violin_plot.png')
            plt.savefig(violin_plot_path, format='png', bbox_inches='tight', dpi=300)
            plt.close()  # Close the plot to prevent overlap with other subplots

            # Plot boxplot
            plt.figure(figsize=(8, 6))
            ax = sns.boxplot(x='Assembler Count', y='Execution Time (seconds)', hue='Minimizer', data=item_method_data, palette={1: 'red', 0: 'blue'}, legend=False)
            ax.set_yscale('log')  # Set y-axis to log scale
            plt.title(f'{item} - {method_type}')
            plt.xlabel('Number of Assemblers')
            plt.ylabel('Execution Time (seconds)')
            plt.grid(True)
            
            # Save the boxplot as PNG
            box_plot_path = os.path.join(method_plot_dir, f'{item}_{method}_box_plot.png')
            plt.savefig(box_plot_path, format='png', bbox_inches='tight', dpi=300)
            plt.close()  # Close the plot to prevent overlap with other subplots
            
            logger.info(f"Generated plots for {item} - {method}: violin and box plots saved")

    logger.info(f"All plots saved in {output_dir} directory")
    logger.info("Plot generation complete")



def log_method_time(item, amount, method_name, assembler_counts,start_time, end_time, solver_type):
    """
    Log method execution times to a CSV file for performance analysis.
    
    This function records the execution time of factory optimization methods
    along with relevant metadata to a CSV file for later analysis and plotting.
    
    Args:
        item (str): Name of the item being produced
        amount (int): Target production amount
        method_name (str): Name of the method being timed
        assembler_counts (dict): Dictionary of assembler counts for each item
        start_time (float): Start time timestamp
        end_time (float): End time timestamp
        solver_type (str): Type of solver used ("z3" or "gurobi")
        
    Returns:
        None: Data is written to "execution_times.csv" file
    """
    execution_time = end_time - start_time
    logger.info(f"Execution time for {method_name}: {execution_time:.4f} seconds")
    
    # Open the CSV file and append the data
    try:
        with open("execution_times.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([item, amount, method_name, assembler_counts, execution_time, solver_type])
        
        logger.debug(f"Logged execution time for {method_name}: {execution_time:.4f}s")
    except Exception as e:
        logger.error(f"Error logging execution time for {method_name}: {e}")
        
     
       
def main():
    """
    Main function for running factory optimization examples.
    
    This function demonstrates the usage of the FactorioProductionTree class
    with various production scenarios. It can be configured to run simple
    examples or evaluation runs for performance testing.
    """
    #factory = FactorioProductionTree(16,10)
    #factory.create_blueprint("Modules/electronic-circuit_120_[]_module.json", "Blueprints/electronic-circuit_120_[]_module.txt")

    Simple_Run()
    
    #Eval_Runs("electronic-circuit",start=1,end=1,step=1,rep_per_step=10)
   

def Simple_Run():
    """
    Run a simple factory optimization example.
    
    This function demonstrates basic usage of the FactorioProductionTree class
    by creating a factory to produce electronic circuits with specific input
    constraints and solver configuration.
    """
    logger.info("Starting simple factory optimization run")
    
    # Example item and amount
    item_to_produce = "electronic-circuit"  # "copper-cable" or "electronic-circuit"
    amount_needed = 120
    solver_type = "z3"  # "gurobi" or "z3"
    input_items = ['copper-cable']  # Using explicit input items
    
    logger.info(f"Configuration: {item_to_produce} @ {amount_needed}/min using {solver_type} solver")
    
    # Initialize factory with specified grid size
    #factorioProductionTree = FactorioProductionTree(10,8)
    factorioProductionTree = FactorioProductionTree(16,10)
    logger.info("Factory initialized with 16x10 grid")
    factorioProductionTree.amount = amount_needed
    production_data  = factorioProductionTree.calculate_production(item_to_produce,amount_needed,input_items=input_items) #60
    factorioProductionTree.production_data = production_data

        
    production_data = factorioProductionTree.set_capacities(production_data)
    
    
    logger.info(f"Production data for {production_data}:")
    
    # Manual input and output
    factorioProductionTree.manual_Input()
    factorioProductionTree.manual_Output()


    
    factorioProductionTree.add_manual_IO_constraints(production_data,solver_type=solver_type)
    
    assembler_counts = factorioProductionTree.count_assemblers(production_data)
        
    # Track time for solving the problem
    start_time = time.perf_counter()
    factorioProductionTree.solve(production_data,solver_type=solver_type)
    end_time = time.perf_counter()
    log_method_time(item_to_produce, 1, "solve", assembler_counts, start_time, end_time,solver_type)
    
    
    start_time = time.perf_counter()
    paths, placed_inserter_information = factorioProductionTree.build_belts(max_tries=2)
    end_time = time.perf_counter()
    log_method_time(item_to_produce, 1, "build_belts", assembler_counts, start_time, end_time,solver_type)
    
    
    
    logger.info("saving data")
    factorioProductionTree.store_data(f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module',paths,placed_inserter_information)
    
    factorioProductionTree.place_power_poles(f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module.json')
        
    factorioProductionTree.create_blueprint(f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module.json',f'Blueprints/{item_to_produce}_{amount_needed}_{input_items}_module.txt')
        
    factorioProductionTree.visualize_factory(paths,placed_inserter_information,store=True,file_path=f'Modules/{item_to_produce}_{amount_needed}_{input_items}_module.png')
        

   

   
def Eval_Runs(item_to_produce, start, end, step, rep_per_step):
    # Initialize the production tree
    factorioProductionTree = FactorioProductionTree(8, 8)
    
    # Always start with a simulation for 1 unit
    production_data = factorioProductionTree.calculate_production(item_to_produce, 300)
    factorioProductionTree.production_data = production_data

        
    production_data = factorioProductionTree.set_capacities(production_data)
    #production_data['copper-cable']['input_inserters'][0]['inserters'] = 4
    
    print(production_data)
    

    # Set manual input and output
    factorioProductionTree.manual_Input()
    factorioProductionTree.manual_Output()
    
    # Store timing data
    solve_times = []
    belt_times = []
    combined_times = []
    
    for rep in range(rep_per_step):
        factorioProductionTree.destroy_solver()
        
        logger.info(f"\nRun {rep + 1}/{rep_per_step} for {1} unit of {item_to_produce}\n")
      
        # Set capacities and get initial assembler count
        production_data = factorioProductionTree.set_capacities(production_data)
        assembler_counts = factorioProductionTree.count_assemblers(production_data)
        
        factorioProductionTree.add_manual_IO_constraints(production_data,solver_type="z3")
        
       # Solve the production problem for amount_needed = 1
        start_time = time.perf_counter()
        factorioProductionTree.solve(production_data, solver_type="z3")
        solve_end_time = time.perf_counter()
        solve_time = solve_end_time - start_time
        solve_times.append(solve_time)
        log_method_time(item_to_produce, 1, "solve", assembler_counts, start_time, solve_end_time, "z3")
        
        # Build belts and visualize for amount_needed = 1
        belt_start_time = time.perf_counter()
        paths, placed_inserter_information = factorioProductionTree.build_belts(max_tries=2)
        belt_end_time = time.perf_counter()
        belt_time = belt_end_time - belt_start_time
        belt_times.append(belt_time)
        log_method_time(item_to_produce, 1, "build_belts", assembler_counts, belt_start_time, belt_end_time, "z3")
        
        # Calculate and store combined time
        combined_time = solve_time + belt_time
        combined_times.append(combined_time)
        log_method_time(item_to_produce, 1, "combined", assembler_counts, start_time, belt_end_time, "z3")

        if paths or placed_inserter_information:
            factorioProductionTree.visualize_factory(paths,placed_inserter_information,store=True,file_path=f'Modules/3x5.png')
        
    

    if solve_times:
        avg_solve_time = sum(solve_times) / len(solve_times)
        logger.info(f"\nSolve times for {item_to_produce}:")
        for i, time_value in enumerate(solve_times):
            logger.info(f"Run {i+1}: {time_value:.4f} seconds")
        logger.info(f"Average solve time: {avg_solve_time:.4f} seconds")
    
    if belt_times:
        avg_belt_time = sum(belt_times) / len(belt_times)
        logger.info(f"\nBelt building times for {item_to_produce}:")
        for i, time_value in enumerate(belt_times):
            logger.info(f"Run {i+1}: {time_value:.4f} seconds")
        logger.info(f"Average belt building time: {avg_belt_time:.4f} seconds")
    
    if combined_times:
        avg_combined_time = sum(combined_times) / len(combined_times)
        logger.info(f"\nCombined execution times for {item_to_produce}:")
        for i, time_value in enumerate(combined_times):
            logger.info(f"Run {i+1}: {time_value:.4f} seconds")
        logger.info(f"Average combined execution time: {avg_combined_time:.4f} seconds")
        
        print(f"Average combined execution time: {avg_combined_time:.4f} seconds")
    
    return
    # Loop through different amounts
    for amount_needed in range(start, end, step):
        for rep in range(rep_per_step):
            logger.info(f"\nRun {rep + 1}/{rep_per_step} for {amount_needed} units of {item_to_produce}\n")
            
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
            
           
        
 


if __name__ == "__main__":

    
    # Prepare CSV file header if not exists
    #if not os.path.exists("execution_times.csv"):
    #    try:
    #        with open("execution_times.csv", "w", newline="") as file:
    #            writer = csv.writer(file)
    #            writer.writerow(["Item", "Amount", "Minimizer", "Method","Assemblers", "Execution Time (seconds)"])
    #    except Exception as e:
    #        logger.error(f"Error initializing CSV file: {e}")

    plot_csv_data("execution_times.csv")
    #main()