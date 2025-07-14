#! .venv\Scripts\python.exe

from FactorioProductionTree import FactorioProductionTree
from FactoryZ3Solver import FactoryZ3Solver
from MultiAgentPathfinder import MultiAgentPathfinder
import pygame
import json
import os
import time
import csv
from math import ceil
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from draftsman.blueprintable import Blueprint
from draftsman.constants import Direction
from draftsman.entity import Inserter, AssemblingMachine, TransportBelt, UndergroundBelt , Pipe,UndergroundPipe, ElectricPole, ConstantCombinator
from draftsman.entity import Splitter as BlueprintSplitter
import traceback
    
from logging_config import setup_logger
logger = setup_logger("FactoryBuilder")


# Define constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Color for input gates
BLUE = (0, 0, 255)  # Color for output gates
GREEN = (0, 255, 0)  # Color for blocks

class FactoryBuilder:
    """
    Multi-module factory builder for complex Factorio production systems.
    
    This class manages the creation of large-scale factories by breaking down
    complex production chains into smaller, manageable modules. Each module
    is optimized individually and then connected together to form a complete
    production system.
    
    The class handles:
    - Splitting complex recipes into smaller sub-factories
    - Managing inter-module connections and material flows
    - Optimizing overall factory layout using SMT solvers
    - Generating blueprints and visualizations for multi-module systems
    
    Attributes:
        output_item (str): Target item to produce
        amount (float): Target production amount
        max_assembler_per_blueprint (int): Maximum assemblers per module
        start_width (int): Initial width for module layouts
        start_height (int): Initial height for module layouts
        block_data (dict): Data for each production module
        final_blocks (dict): Optimized block positions and connections
        z3_solver (FactoryZ3Solver): SMT solver for multi-module optimization
    """
    
    def __init__(self, output_item, amount, max_assembler_per_blueprint, start_width, start_height, load_modules) -> None:
        """
        Initialize the FactoryBuilder with production parameters and constraints.
        
        Args:
            output_item (str): The item to produce (e.g., "electronic-circuit")
            amount (float): Target production amount per minute
            max_assembler_per_blueprint (int): Maximum assemblers allowed per module
            start_width (int): Initial grid width for module layouts
            start_height (int): Initial grid height for module layouts
            load_modules (bool): Whether to load existing modules or create new ones
        """
        logger.info(f"Initializing FactoryBuilder for {output_item}: {amount} units/min")
        logger.info(f"Max assemblers per module: {max_assembler_per_blueprint}")
        logger.info(f"Initial module grid size: {start_width}x{start_height}")
        
        # Production target parameters
        self.output_item = output_item
        self.amount = amount
        self.max_assembler_per_blueprint = max_assembler_per_blueprint
        
        # Module layout parameters  
        self.start_width = start_width
        self.start_height = start_height
        
        # Factory layout coordinates
        self.output_point = (0,0)    # Main factory output point
    
        # Solvers and algorithms (initialized when needed)
        self.z3_solver = None        # SMT solver for multi-module layout
        self.AStar = None            # Pathfinding for inter-module connections
        
        # Module and factory data
        self.block_data = {}         # Data for each production module
        
        # Load recipe data for production calculations
        self.items_data = self.load_json("recipes.json")
        logger.info(f"Loaded {len(self.items_data)} recipes")
        
        # Final optimized layout results
        self.final_x = None              # Final factory width
        self.final_y = None              # Final factory height  
        self.final_blocks = None         # Optimized block positions
        self.gate_connections = None     # Inter-module connections
        self.inter_block_paths = None    # Paths between modules
  
        # Visualization assets
        self.images = {}
        
        # Configuration flags
        self.load_modules = load_modules  # Whether to load existing module files
        
        # External I/O configuration
        self.external_io = None
        self.module_input_points = []
        self.module_output_points = []
        
        logger.info("FactoryBuilder initialization complete")


    def load_json(self, recipe_file):
        """
        Load and parse JSON recipe data.
        
        Args:
            recipe_file (str): Path to the recipe JSON file
            
        Returns:
            dict: Dictionary mapping item IDs to recipe data
            
        Raises:
            FileNotFoundError: If the recipe file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        logger.info(f"Loading recipe data from {recipe_file}")
        try:
            with open(recipe_file, "r") as file:
                recipes = json.load(file)
            logger.info(f"Successfully loaded {len(recipes)} recipes")
            return {item["id"]: item for item in recipes}
        except FileNotFoundError:
            logger.error(f"Recipe file not found: {recipe_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in recipe file: {e}")
            raise
        

    def eval_split(self, production_data, input_items):
        """
        Evaluate if production data needs to be split into multiple modules.
        
        This method checks if the current production requirements exceed the
        maximum assembler limit per module. If so, it recursively splits the
        production amount in half until each module is within limits.
        
        Args:
            production_data (dict): Current production requirements
            input_items (list): List of external input items
            
        Returns:
            tuple: (updated_production_data, number_of_modules_needed)
        """
        logger.info(f"Evaluating split for {self.count_assembler(production_data)} assemblers")
        logger.info(f"Maximum allowed per module: {self.max_assembler_per_blueprint}")
        
        num_factories = 1
        factorioProductionTree = FactorioProductionTree(
            grid_width=self.start_width, 
            grid_height=self.start_height
        )
        
        amount = self.amount
        
        # Keep splitting until assembler count is within limits
        while self.count_assembler(production_data) > self.max_assembler_per_blueprint:
            logger.info(f"Splitting: {self.count_assembler(production_data)} > {self.max_assembler_per_blueprint}")
            
            amount = amount/2
            logger.info(f"Reduced amount to {amount}")
            
            # Recalculate production data with reduced amount
            production_data = factorioProductionTree.calculate_production(
                self.output_item, amount, input_items
            )
            production_data = factorioProductionTree.set_capacities(production_data)
            num_factories += 1 

        logger.info(f"Split evaluation complete: {num_factories} modules needed")
        logger.info(f"Final assembler count per module: {self.count_assembler(production_data)}")
        
        return production_data, num_factories
    
    
    def count_assembler(self, production_data) -> int:
        """
        Count the total number of assemblers required in production data.
        
        Args:
            production_data (dict): Production requirements with assembler counts
            
        Returns:
            int: Total number of assemblers needed
        """
        total_assemblers = 0
        for key, value in production_data.items():
            if 'assemblers' in value:
                total_assemblers += value['assemblers']
        
        logger.debug(f"Total assemblers counted: {total_assemblers}")
        return total_assemblers
    def count_assembler(self,production_data) -> int:
        total_assemblers = 0
        for key, value in production_data.items():
            if 'assemblers' in value:
                total_assemblers += value['assemblers']
        
        return total_assemblers
    

    def get_input_items(self,output,partial_items=[]):
        if not partial_items:
            return []
        
        def collect_ingredients(item_id, excluded_items, visited=set()):
            """
            Recursively collects ingredients for the given item ID, discarding those in excluded_items.
            """
            if item_id in visited:
                return set()  # Prevent infinite loops in case of circular recipes
            visited.add(item_id)
            
            item = self.items_data.get(item_id)  
            if not item or "recipe" not in item or "ingredients" not in item["recipe"]:
                return set()  # No recipe or ingredients, so nothing to add

            ingredients = set()
            for ingredient in item["recipe"]["ingredients"]:
                ingredient_id = ingredient["id"]
                if ingredient_id not in excluded_items:
                    ingredients.add(ingredient_id)
                    # Recursively collect ingredients for sub-items
                    ingredients.update(collect_ingredients(ingredient_id, excluded_items, visited))
            return ingredients

        # Collect ingredients, excluding partial_items
        result = collect_ingredients(output, set(partial_items))
        return list(result)

    
    # allows the user to split recipies at any given point
    # from that rebuild the production data for each subfactory and let the user design all the subfactories in and outputs
    # build a button for each item and let user click -> highlight green if clicked
    def split_recipies(self):
        
        factorioProductionTree = FactorioProductionTree(grid_width=self.start_width,grid_height=self.start_height)
        production_data  = factorioProductionTree.calculate_production(self.output_item , self.amount) 
        production_data = factorioProductionTree.set_capacities(production_data)
            
        # Filter out basic input items and the output item
        selectable_items = {
            item: data
            for item, data in production_data.items()
            if 'assemblers' in data and item != self.output_item
        }
        
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Split Recipes")
        
        running = True
        selected_items = set()  # Track selected item
        
        while running:
            screen.fill((0, 0, 0))  # Clear screen
            
            # Display items as buttons
            y_pos = 50
            for item in selectable_items.keys():
                rect = pygame.Rect(100, y_pos, 400, 40)
                pygame.draw.rect(screen, (0, 255, 0) if item in selected_items else (255, 255, 255), rect)
        
                
                font = pygame.font.Font(None, 36)
                text = font.render(item, True, (0, 0, 0))
                screen.blit(text, (110, y_pos + 5))
                y_pos += 50
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if a button was clicked
                    x, y = event.pos
                    y_pos = 50
                    for item in selectable_items.keys():
                        if pygame.Rect(100, y_pos, 200, 40).collidepoint(x, y):
                            if item in selected_items:
                                selected_items.remove(item)  # Deselect if already selected
                            else:
                                selected_items.add(item)  # Select item
                            logger.info(f"Selected items: {selected_items}")
                        y_pos += 50

        pygame.quit()
            
        logger.info(f"Selected items: {selected_items}")
        
        selected_items.add(self.output_item)
        
        if self.load_modules:
            
            for item in selected_items:
                
                # ask to select module for item
                
                # for every selected item pick an modules txt file to recreate the factorioProductionTree
                file_path = self.select_module_for_item(item)
                
                if file_path and os.path.exists(file_path):
                    factorioProductionTree = self.load_module(file_path)
                    
                    
            
                    num_factories = ceil(production_data[item]["amount_per_minute"] / factorioProductionTree.calculate_max_output())
                    
                    self.block_data[item] = {
                    "tree": factorioProductionTree,
                    "production_data": factorioProductionTree.production_data,
                    "num_factories": num_factories,
                    "png":file_path.replace(".json", ".png"),
                    "json":file_path
                    }
                
            
            return
            

        factorioProductionTree = FactorioProductionTree(grid_width=self.start_width,grid_height=self.start_height)
        
        # Always include the output item
        input_items = self.get_input_items(self.output_item,list(selected_items)) + list(selected_items)


        production_data  = factorioProductionTree.calculate_production(self.output_item , self.amount, input_items) 
        production_data = factorioProductionTree.set_capacities(production_data)
        
        production_data,num_factories = self.eval_split(production_data, list(selected_items))
        
        factorioProductionTree.manual_Input(Title=f"Setting Manual Input for {self.output_item}")
        factorioProductionTree.manual_Output(Title=f"Setting Manual Output for {self.output_item}")
        factorioProductionTree.add_manual_IO_constraints(production_data,sequential=False)
        
        if self.output_item not in self.block_data:
            self.block_data[self.output_item] = {}
        
        self.block_data[self.output_item]["tree"]=factorioProductionTree
        self.block_data[self.output_item]["production_data"]=production_data
        self.block_data[self.output_item]["num_factories"]=num_factories
        
        
        # Print input information (keys with input and output values)
        logger.info("Input Information:")
        for key, value in self.block_data[self.output_item]["tree"].input_information.items():
            if 'input' in value and 'output' in value:
                logger.info(f"{key}: input={value['input']}, output={value['output']}")

        # Print output information (keys with input and output values)
        logger.info("Output Information:")
        for key, value in self.block_data[self.output_item]["tree"].output_information.items():
            if 'input' in value and 'output' in value:
                logger.info(f"{key}: input={value['input']}, output={value['output']}")

       
        for item in selected_items:
            
            logger.info(f"building block for subitem {item}")

            if item not in self.block_data:
                self.block_data[item] = {}
            
            factorioProductionTree = FactorioProductionTree(grid_width=self.start_width,grid_height=self.start_height)

            input_items = self.get_input_items(item,list(selected_items)) + list(selected_items)

            new_data = factorioProductionTree.calculate_production(item, production_data[item]['amount_per_minute'],input_items)
            new_data = factorioProductionTree.set_capacities(new_data)
            
            new_data,num_factories = self.eval_split(new_data,input_items)
            
                 
            factorioProductionTree.manual_Input(Title=f"Setting Manual Input for {item}")
            factorioProductionTree.manual_Output(Title=f"Setting Manual Output for {item}")
            factorioProductionTree.add_manual_IO_constraints(new_data,sequential=False)
           
            self.block_data[item]["tree"]=factorioProductionTree
            self.block_data[item]["production_data"]=new_data
            self.block_data[item]["num_factories"]=num_factories
            
    
         
         
    # Example method to ask for the module file using tkinter
    def select_module_for_item(self, item):
        # Create a tkinter root window and immediately hide it (we only need the file dialog)
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Ask the user to select a file using the file picker dialog
        file_path = filedialog.askopenfilename(
            title=f"Select module for {item}",
            filetypes=(("Json files", "*.json"),)
        )

        # Return the selected file path (it may be empty if the user cancels)
        return file_path
                
    
    def load_module(self,file_path):
        factorioProductionTree = FactorioProductionTree()
        factorioProductionTree.load_data(file_path)
        
        return factorioProductionTree
    

    
    def solve_small_blocks(self, visualize):
        
        if self.load_modules:
            return
        
        
        for item in self.block_data.keys():
            self.block_data[item]["tree"].solve(self.block_data[item]["production_data"],sequential=False)
            paths, placed_inserter_information = self.block_data[item]["tree"].build_belts(max_tries=2)
            self.block_data[item]["paths"] = paths
            self.block_data[item]["placed_inserter_information"]=placed_inserter_information

            if visualize:
                self.block_data[item]["tree"].visualize_factory(paths,placed_inserter_information)

        
            
    def solve_factory(self):
        logger.info("solving factory")
        
        # Define external I/O points before creating the solver
        self.define_factory_io_points()
        
        logger.info("block data:")
        logger.info(self.block_data)
        
        self.z3_solver = FactoryZ3Solver(self.block_data, self.output_point)
        num_factories = 0
        for i, key in enumerate(self.block_data.keys()):
            num_factories += self.block_data[key]["num_factories"]
            
        logger.info(f'total number of modules: {num_factories}')
        
        # Apply I/O constraints to solver
        self.apply_io_constraints_to_solver()
            
        self.z3_solver.build_constraints()
        
        # Store gate connections along with block positions
        self.final_blocks, self.final_x, self.final_y, self.gate_connections = self.z3_solver.solve()
        
        
        logger.debug(f"Gate connections: {self.gate_connections}")
        
        # Plan paths between connected gates
        if self.final_blocks:
            path = f"Factorys/factory_{self.output_item}_{self.amount}.json"
            factory_data,_ = self.create_json(output_json_path=path)
            inter_block_paths , _ = self.plan_inter_block_paths(factory_data)
            self.add_paths_to_json(path,inter_block_paths)
            
            blueprint_path= f"Blueprints/blueprint_{self.output_item}_{self.amount}.txt"
            self.create_blueprint_from_json(path, output_path=blueprint_path)
            
            factory_img_path= f"Factorys/factory_{self.output_item}_{self.amount}.png"
            self.visualize_factory(json_path=path, save_path=factory_img_path)

        logger.info(f"Factory dimensions: {self.final_x} x {self.final_y}")
        logger.info(f"Final blocks: {self.final_blocks}")
        logger.info(f"Gate connections: {self.gate_connections}")


    def add_paths_to_json(self, json_path,inter_block_paths):
        """
        Add the inter-block paths to an existing factory JSON file.
        
        Args:
            json_path (str): Path to the existing factory JSON file
            inter_block_paths (dict): Dictionary of paths between blocks
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the existing JSON file
            with open(json_path, 'r') as f:
                factory_data = json.load(f)
            
            logger.info(f"Adding {len(inter_block_paths)} inter-block paths to {json_path}")
            
            # Add the inter-block paths to the JSON data
            factory_data["inter_block_paths"] = []
            
            # Process each path and format it for the JSON
            for item_key, paths_data_list in inter_block_paths.items():
                for path_data in paths_data_list:
                    # Create a copy of path_data to avoid modifying the original
                    path_entry = {
                        "item": item_key.split('_')[0],  # Extract the base item name
                        "connection_id": item_key,
                        "path": path_data.get("path", []),
                    }
                    
                    # Add underground segments if present
                    if "underground_segments" in path_data:
                        path_entry["underground_segments"] = path_data["underground_segments"]
                    
                    # Add source and target gate information if available
                    if "source_id" in path_data:
                        path_entry["source_gate"] = path_data["source_id"]
                    if "target_id" in path_data:
                        path_entry["target_gate"] = path_data["target_id"]
                    
                    factory_data["inter_block_paths"].append(path_entry)
            
            # Save the updated JSON back to the file
            with open(json_path, 'w') as f:
                json.dump(factory_data, f, indent=2)
            
            logger.info(f"Successfully added {len(factory_data['inter_block_paths'])} inter-block paths to {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paths to JSON: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    # create realtive positons foe all modules and add them to the json file
    def create_json(self, output_json_path=None):
        if not self.final_blocks:
            logger.error("Cannot create JSON: No final blocks available")
            return
        
        # Initialize the factory data structure
        factory_data = {
            "factory_dimensions": {
                "width": self.final_x,
                "height": self.final_y
            },
            "blocks": {},
            "entities": {
                "assemblers": [],
                "inserters": [],
                "belts": [],
                "underground_belts": [],
                "splitters": [],
                "power_poles": [],
            },
            "io_points": {
                "inputs": [],
                "outputs": []
            },
            "inter_block_paths": [],
        }
        
        # Add external I/O points defined by the user
        if hasattr(self, 'external_io') and self.external_io:
            # Add external input gates to outputs (they provide items to the factory)
            for input_gate in self.external_io.get("input_gates", []):
                factory_data["io_points"]["outputs"].append({
                    "item": input_gate["item"],
                    "position": self._get_absolute_position(input_gate),
                    "block_id": input_gate["id"],
                    "external": True
                })
                logger.info(f"Added external input gate for {input_gate['item']} at {input_gate['position']} as an output point")
            
            # Add external output gates to inputs (they receive items from the factory)
            for output_gate in self.external_io.get("output_gates", []):
                factory_data["io_points"]["inputs"].append({
                    "item": output_gate["item"],
                    "position": self._get_absolute_position(output_gate),
                    "block_id": output_gate["id"],
                    "external": True
                })
                logger.info(f"Added external output gate for {output_gate['item']} at {output_gate['position']} as an input point")
                
        # Process each block
        for block_id, block_info in self.final_blocks.items():
            block_x = block_info['x']
            block_y = block_info['y']
            block_width = block_info['width']
            block_height = block_info['height']
            block_type = block_info.get('block_type')
            
            if not block_type:
                # Extract type from block_id 
                block_parts = block_id.split("_")
                block_type = block_parts[1] if len(block_parts) >= 3 else block_id
            
            # Add block metadata to the factory data
            factory_data["blocks"][block_id] = {
                "type": block_type,
                "position": [block_x, block_y],
                "dimensions": [block_width, block_height]
            }
            
            # Get the module JSON path
            module_json_path = block_info.get('module_json')
            
            # If no JSON path in block_info, try to find it in block_data
            if not module_json_path and block_type in self.block_data:
                module_json_path = self.block_data[block_type].get('json')
            
            # Skip this block if we can't find its module JSON
            if not module_json_path or not os.path.exists(module_json_path):
                logger.warning(f"Module JSON not found for {block_id} (type: {block_type}), skipping")
                #raise Exception(f"Module JSON not found for {block_id} (type: {block_type}), skipping")
                
            # Load module JSON
            try:
                with open(module_json_path, 'r') as f:
                    module_data = json.load(f)
                
                logger.info(f"Processing module data from {module_json_path} for block {block_id}")
                
                # Process assemblers
                if "assembler_information" in module_data:
                    for assembler in module_data["assembler_information"]:
                        item, rel_x, rel_y, w, h, machine_type = assembler[0], assembler[1], assembler[2], assembler[3], assembler[4], assembler[5]
                        # Convert coordinates
                        abs_x = block_x + rel_x
                        abs_y = block_y + rel_y 
                        
                        factory_data["entities"]["assemblers"].append({
                            "item": item,
                            "position": [abs_x, abs_y],
                            "block_id": block_id,
                            "type": machine_type,
                            "dimensions": [w, h]
                        })
                        
                        logger.info(f"Added assembler {item} at {[abs_x, abs_y]} for block {block_id}")
                
                # Process inserters
                if "inserter_information" in module_data:
                    for inserter in module_data["inserter_information"]:
                        item, rel_x, rel_y, direction = inserter[0], inserter[1], inserter[2], inserter[3]
                        # Convert coordinates
                        abs_x = block_x + rel_x
                        abs_y = block_y + rel_y
                        
                        factory_data["entities"]["inserters"].append({
                            "item": item,
                            "position": [abs_x, abs_y],
                            "direction": direction,
                            "block_id": block_id
                        })
                        logger.info(f"Added inserter {item} at {[abs_x, abs_y]} facing {direction} for block {block_id}")
                            
                if "placed_inserter_information" in module_data:
                    for item_id, inserters in module_data["placed_inserter_information"].items():
                        # Extract the item name from the ID (e.g., "electronic-circuit_0" -> "electronic-circuit")
                        item = item_id.split('_')[0]
                    
                        for pos_str, target_pos in inserters.items():
                            # Parse the source position "(10, 2)" -> [10, 2]
                            pos_str = pos_str.strip('()')
                            source_x, source_y = map(int, pos_str.split(','))
                            
                            # Get the target position
                            target_x, target_y = target_pos
                            
                            # Determine direction based on relative positions
                            dx = source_x - target_x  # Reversed
                            dy = source_y - target_y  # Reversed
                            
                            direction = "east"  # default
                            if abs(dx) > abs(dy):
                                direction = "east" if dx > 0 else "west"
                            else:
                                direction = "south" if dy > 0 else "north"
                            
                            # Convert to absolute coordinates
                            abs_x = block_x + target_x
                            abs_y = block_y + target_y
                            
                            # Add to corrected entities
                            factory_data["entities"]["inserters"].append({
                                "item": item,
                                "position": [abs_x, abs_y],
                                "direction": direction,
                                "block_id": block_id
                            })
                            
                            logger.info(f"Added placed inserter for {item} at {[abs_x, abs_y]} facing {direction}")

                logger.debug(f"Module data for block {block_id}: {module_data}")
                
                if "input_information" in module_data:
                    for item, data in module_data["input_information"].items():
                        if 'paths' in data and data['paths'] is not None and item in data['paths']:
                            for path_data in data['paths'][item]:
                                if "path" in path_data:
                                    path = path_data["path"]
                                    
                                    for i in range(len(path)):
                                        current = path[i]
                                        
                                        # Convert to tuple if it's a list
                                        current = tuple(current) if isinstance(current, list) else current
                                        
                                        # Determine direction based on next position
                                        direction = None
                                        if i < len(path) - 1:
                                            next_pos = path[i + 1]
                                            next_pos = tuple(next_pos) if isinstance(next_pos, list) else next_pos
                                            
                                            dx = next_pos[0] - current[0]
                                            dy = next_pos[1] - current[1]
                                            if dx > 0:
                                                direction = "east"
                                            elif dx < 0:
                                                direction = "west"
                                            elif dy > 0:
                                                direction = "south"
                                            elif dy < 0:
                                                direction = "north"
                                        elif i > 0:
                                            # Use the same direction as the previous segment for the last point
                                            prev_pos = path[i - 1]
                                            prev_pos = tuple(prev_pos) if isinstance(prev_pos, list) else prev_pos
                                            
                                            dx = current[0] - prev_pos[0]
                                            dy = current[1] - prev_pos[1]
                                            
                                            if dx > 0:
                                                direction = "east"
                                            elif dx < 0:
                                                direction = "west"
                                            elif dy > 0:
                                                direction = "south"
                                            elif dy < 0:
                                                direction = "north"
                                        
                                        # Convert coordinates
                                        abs_x = block_x + current[0]
                                        abs_y = block_y + current[1]
                                        
                                        # Add to belts if not already added (avoid duplicates)
                                        position_exists = False
                                        for belt in factory_data["entities"]["belts"]:
                                            if belt["position"][0] == abs_x and belt["position"][1] == abs_y:
                                                position_exists = True
                                                break
                                        
                                        if not position_exists and direction:
                                            factory_data["entities"]["belts"].append({
                                                "item": item,
                                                "position": [abs_x, abs_y],
                                                "direction": direction,
                                                "block_id": block_id
                                            })
                            logger.info(f"Added input belt route for {item} at {[abs_x, abs_y]} facing {direction}")
                            
                        
                        if 'input' in data and 'output' in data:
                            input_pos = data.get("input")
                            output_pos = data.get("output")
                        
                            # For input_information: 
                            # - "input" position is where this module receives the item (external connection point)
                            # - "output" position is where the item flows into the module's internal systems
                            
                            # Validate coordinates before adding
                            input_abs_x = block_x + input_pos[1]
                            input_abs_y = block_y + input_pos[0]
                            output_abs_x = block_x + output_pos[1]
                            output_abs_y = block_y + output_pos[0]
                            
                            # Check bounds
                            if (0 <= input_abs_x < self.final_x and 0 <= input_abs_y < self.final_y and
                                0 <= output_abs_x < self.final_x and 0 <= output_abs_y < self.final_y):
                                
                                # Add input connection point (where the module receives the item from external)
                                factory_data["io_points"]["inputs"].append({
                                        "item": item,
                                        "position": [input_abs_x, input_abs_y],
                                        "block_id": block_id,
                                        "external": False
                                    })
                                logger.info(f"Added input connection point for {item} at {[input_abs_x, input_abs_y]} for block {block_id}")
                                
                                # Add output connection point (where the item flows out of this connection)
                                factory_data["io_points"]["outputs"].append({
                                    "item": item,
                                    "position": [output_abs_x, output_abs_y],
                                    "block_id": block_id,
                                    "external": False
                                })
                                logger.info(f"Added output connection point for {item} at {[output_abs_x, output_abs_y]} for block {block_id}")
                            else:
                                logger.warning(f"Skipping out-of-bounds I/O points for {item} in block {block_id}: input={[input_abs_x, input_abs_y]}, output={[output_abs_x, output_abs_y]}, bounds=[{self.final_x}, {self.final_y}]")

                if "output_information" in module_data:
                    for item, data in module_data["output_information"].items():
                        if 'paths' in data and data['paths'] is not None and item in data['paths']:
                            for path_data in data['paths'][item]:
                                if "path" in path_data:
                                    path = path_data["path"]
                                    
                                    for i in range(len(path)):
                                        current = path[i]
                                        
                                        # Convert to tuple if it's a list
                                        current = tuple(current) if isinstance(current, list) else current
                                        
                                        # Determine direction based on next position
                                        direction = None
                                        if i < len(path) - 1:
                                            next_pos = path[i + 1]
                                            next_pos = tuple(next_pos) if isinstance(next_pos, list) else next_pos
                                            
                                            dx = next_pos[0] - current[0]
                                            dy = next_pos[1] - current[1]
                                            
                                            if dx > 0:
                                                direction = "east"
                                            elif dx < 0:
                                                direction = "west"
                                            elif dy > 0:
                                                direction = "south"
                                            elif dy < 0:
                                                direction = "north"
                                        elif i > 0:
                                            # Use the same direction as the previous segment for the last point
                                            prev_pos = path[i - 1]
                                            prev_pos = tuple(prev_pos) if isinstance(prev_pos, list) else prev_pos
                                            
                                            dx = current[0] - prev_pos[0]
                                            dy = current[1] - prev_pos[1]
                                            
                                            if dx > 0:
                                                direction = "east"
                                            elif dx < 0:
                                                direction = "west"
                                            elif dy > 0:
                                                direction = "south"
                                            elif dy < 0:
                                                direction = "north"
                                        
                                        # Convert coordinates
                                        abs_x = block_x + current[0]
                                        abs_y = block_y + current[1]
                                        
                                        # Add to belts if not already added (avoid duplicates)
                                        position_exists = False
                                        for belt in factory_data["entities"]["belts"]:
                                            if belt["position"][0] == abs_x and belt["position"][1] == abs_y:
                                                position_exists = True
                                                break
                                        
                                        if not position_exists and direction:
                                            factory_data["entities"]["belts"].append({
                                                "item": item,
                                                "position": [abs_x, abs_y],
                                                "direction": direction,
                                                "block_id": block_id
                                            })
                            logger.info(f"Added output belt route for {item} at {[abs_x, abs_y]} facing {direction}")
                        
                       
                        if 'input' in data and 'output' in data:
                            input_pos = data.get("input")
                            output_pos = data.get("output")
                            
                            # For output_information:
                            # - "input" position is where the item comes from within the module 
                            # - "output" position is where the item exits the module (external connection point)
                            
                            # Validate coordinates before adding
                            input_abs_x = block_x + input_pos[1]
                            input_abs_y = block_y + input_pos[0]
                            output_abs_x = block_x + output_pos[1]
                            output_abs_y = block_y + output_pos[0]
                            
                            # Check bounds
                            if (0 <= input_abs_x < self.final_x and 0 <= input_abs_y < self.final_y and
                                0 <= output_abs_x < self.final_x and 0 <= output_abs_y < self.final_y):
                                
                                # Add input connection point (where the item comes from within the module)
                                factory_data["io_points"]["inputs"].append({
                                        "item": item,
                                        "position": [input_abs_x, input_abs_y],
                                        "block_id": block_id,
                                        "external": False
                                    })
                                logger.info(f"Added internal input point for {item} at {[input_abs_x, input_abs_y]} for block {block_id}")
                                
                                # Add output connection point (where the item exits the module)
                                factory_data["io_points"]["outputs"].append({
                                    "item": item,
                                    "position": [output_abs_x, output_abs_y],
                                    "block_id": block_id,
                                    "external": False
                                })
                                logger.info(f"Added external output point for {item} at {[output_abs_x, output_abs_y]} for block {block_id}")
                            else:
                                logger.warning(f"Skipping out-of-bounds I/O points for {item} in block {block_id}: input={[input_abs_x, input_abs_y]}, output={[output_abs_x, output_abs_y]}, bounds=[{self.final_x}, {self.final_y}]")
                
                # Process paths which include underground belts and splitters
                if "paths" in module_data:
                    for item_key, paths_list in module_data["paths"].items():
                        item_name = item_key.split("_")[0]  # Extract item name from key
                        
                        for path_data in paths_list:
                            # Process regular belts
                            if "path" in path_data:
                                path = path_data["path"]
                                has_dest_splitter = False
                                splitter_direction = None
                                
                                
                                if "dest_splitter" in path_data and path_data["dest_splitter"] and "direction" in path_data["dest_splitter"]:
                                    has_dest_splitter = True
                                    dx, dy = path_data["dest_splitter"]["direction"]
                                
                                    if dx > 0:
                                        splitter_direction = "east"
                                    elif dx < 0:
                                        splitter_direction = "west" 
                                    elif dy > 0:
                                        splitter_direction = "south"
                                    elif dy < 0:
                                        splitter_direction = "north"
                                    
                                    
                                for i, (rel_x, rel_y) in enumerate(path):
                                    # Convert coordinates
                                    abs_x = block_x + rel_x
                                    abs_y = block_y + rel_y
                                    
                                    # Determine belt direction
                                    direction = None
                                    if i < len(path) - 1:
                                        # For all belts except the last one, get direction from next position
                                        next_x, next_y = path[i+1]
                                        dx = next_x - rel_x
                                        dy = next_y - rel_y
                                        
                                        # Convert to compass directions
                                        if dx > 0:
                                            direction = "east"
                                        elif dx < 0:
                                            direction = "west"
                                        elif dy > 0:
                                            direction = "south"
                                        elif dy < 0:
                                            direction = "north"
                                    elif i > 0:
                                        # For the last belt, use the same direction as the previous belt
                                        prev_x, prev_y = path[i-1]
                                        dx = rel_x - prev_x
                                        dy = rel_y - prev_y
                                        
                                        # Convert to compass directions
                                        if dx > 0:
                                            direction = "east"
                                        elif dx < 0:
                                            direction = "west"
                                        elif dy > 0:
                                            direction = "south"
                                        elif dy < 0:
                                            direction = "north"

                                        if has_dest_splitter:
                                            # If there's a destination splitter, set the direction to that
                                            direction = splitter_direction
                                        
                                    # Add to belts if not already added (avoid duplicates)
                                    position_exists = False
                                    for belt in factory_data["entities"]["belts"]:
                                        if belt["position"][0] == abs_x and belt["position"][1] == abs_y:
                                            position_exists = True
                                            break
                                    
                                    if not position_exists and direction:
                                        factory_data["entities"]["belts"].append({
                                            "item": item_name,
                                            "position": [abs_x, abs_y],
                                            "direction": direction,
                                            "block_id": block_id
                                        })
                                        logger.info(f"Added belt {item_name} at {[abs_x, abs_y]} facing {direction} for block {block_id}")
                            
                            # Process underground belts
                            if "underground_segments" in path_data:
                                for segment_id, segment in path_data["underground_segments"].items():
                                    if "start" in segment and "end" in segment:
                                        start_rel_x, start_rel_y = segment["start"]
                                        end_rel_x, end_rel_y = segment["end"]
                                        
                                        # Convert coordinates
                                        start_abs_x = block_x + start_rel_x
                                        start_abs_y = block_y +  start_rel_y
                                        end_abs_x = block_x + end_rel_x
                                        end_abs_y = block_y + end_rel_y
                                        
                                        # Determine direction
                                        dx = end_rel_x - start_rel_x
                                        dy = end_rel_y - start_rel_y
                                        
                                        if dx > 0:
                                            direction = "east"
                                        elif dx < 0:
                                            direction = "west"
                                        elif dy > 0:
                                            direction = "south"
                                        elif dy < 0:
                                            direction = "north"
                                        
                                        # Add entrance
                                        factory_data["entities"]["underground_belts"].append({
                                            "item": item_name,
                                            "position": [start_abs_x, start_abs_y],
                                            "type": "entrance",
                                            "direction": direction,
                                            "block_id": block_id
                                        })
                                        
                                        # Add exit
                                        factory_data["entities"]["underground_belts"].append({
                                            "item": item_name,
                                            "position": [end_abs_x, end_abs_y],
                                            "type": "exit",
                                            "direction": direction,
                                            "block_id": block_id
                                        })
                                        logger.info(f"Added underground belt segment {item_name} from {[start_abs_x, start_abs_y]} to {[end_abs_x, end_abs_y]} for block {block_id}")
                            
                            # Process splitters
                            for splitter_type in ["start_splitter", "dest_splitter"]:
                                if splitter_type in path_data and path_data[splitter_type]:
                                    splitter_data = path_data[splitter_type]
                                    
                                    if "position" in splitter_data:
                                        rel_x, rel_y = splitter_data["position"]
                                        
                                        # Convert coordinates
                                        abs_x = block_x + rel_x
                                        abs_y = block_y + rel_y
                                        
                                        # Determine direction
                                        direction = "right"  # Default
                                        if "direction" in splitter_data:
                                            dx, dy = splitter_data["direction"]
                                            if dx > 0:
                                                direction = "east"
                                            elif dx < 0:
                                                direction = "west"
                                            elif dy > 0:
                                                direction = "south"
                                            elif dy < 0:
                                                direction = "north"
                                        
                                        factory_data["entities"]["splitters"].append({
                                            "item": item_name,
                                            "position": [abs_x, abs_y],
                                            "direction": direction,
                                            "block_id": block_id
                                        })
                                        logger.info(f"Added splitter {item_name} at {[abs_x, abs_y]} facing {direction} for block {block_id}")
                
                if "power_pole_information" in module_data:
                    for power_pole in module_data["power_pole_information"]:
                        item, rel_x, rel_y = power_pole[0], power_pole[1], power_pole[2]
                        # Convert coordinates
                        abs_x = block_x + rel_x
                        abs_y = block_y + rel_y
                        
                        factory_data["entities"]["power_poles"].append({
                            "item": item,
                            "position": [abs_x, abs_y],
                            "block_id": block_id
                        })
                        logger.info(f"Added power pole {item} at {[abs_x, abs_y]} for block {block_id}")
                    
                
                
            except Exception as e:
                logger.error(f"Error processing module JSON for {block_id}: {e}")
                import traceback
                traceback.print_exc()
        
        
            
         

        # If no path specified, use default
        if not output_json_path:
            output_json_path = f"Factorys/factory_{self.output_item}_{self.amount}.json"            # Save to file
        with open(output_json_path, 'w') as f:
            json.dump(factory_data, f, indent=2)
        
        logger.info(f"Factory JSON data saved to {output_json_path}")
      
        
        return factory_data, output_json_path
    
    def determine_gate_connections(self):
        """Determine which gates should be connected between blocks"""
        connections = []
        
        # Group gates by item type
        gates_by_item = {}
        for block_id, block_info in self.final_blocks.items():
            # Process output gates
            for gate in block_info['output_points']:
                item = gate['id']
                if item not in gates_by_item:
                    gates_by_item[item] = {"input": [], "output": []}
                
                gate_data = {
                    "block_id": block_id,
                    "gate_id": gate['id'],
                    "item": item,
                    "x": block_info['x'] + gate['x'],
                    "y": block_info['y'] + gate['y']
                }
                gates_by_item[item]['output'].append(gate_data)
            
            # Process input gates
            for gate in block_info['input_points']:
                item = gate['id']
                if item not in gates_by_item:
                    gates_by_item[item] = {"input": [], "output": []}
                
                gate_data = {
                    "block_id": block_id,
                    "gate_id": gate['id'],
                    "item": item,
                    "x": block_info['x'] + gate['x'],
                    "y": block_info['y'] + gate['y']
                }
                gates_by_item[item]['input'].append(gate_data)
        
        # For each item type, connect output gates to input gates
        for item, item_gates in gates_by_item.items():
            outputs = item_gates["output"]
            inputs = item_gates["input"]
            
            # Find nearest connections using greedy approach
            for output in outputs:
                # Only connect to inputs from different blocks
                valid_inputs = [input_gate for input_gate in inputs 
                            if input_gate["block_id"] != output["block_id"]]
                
                # Find closest valid input gate
                best_distance = float('inf')
                best_input = None
                
                for input_gate in valid_inputs:
                    dist = manhattan_distance(
                        (output["x"], output["y"]),
                        (input_gate["x"], input_gate["y"])
                    )
                    if dist < best_distance:
                        best_distance = dist
                        best_input = input_gate
                
                if best_input:
                    connections.append((output, best_input))
                    inputs = [i for i in inputs if i != best_input]
        
        logger.info(f"Found {len(connections)} connections between different blocks")
        return connections

    def load_images(self):
        """Load images from the assets folder based on block names."""
        for block_key in self.final_blocks.keys():
            # Extract the base name of the block (e.g., 'electronic-circuit')
            base_name = block_key.split('_')[1]
            image_path = os.path.join('assets', f'{base_name}.png')
            
          
            if os.path.exists(image_path):
                self.images[block_key] = pygame.image.load(image_path)
            else:
                logger.info(f"Image not found for {base_name} at {image_path}")

 
    
    def get_num_subfactories(self):
        if self.final_blocks is None:
            return 0
        return len(self.final_blocks)
    
        
    def load_item_images(self):
        """Load and cache item images from the assets folder."""
        self.item_images = {}
        for block_info in self.final_blocks.values():
            # Load images for input points
            for gate in block_info['input_points']:
                item = gate['item']
                if item not in self.item_images:
                    image_path = os.path.join('assets', f'{item}.png')
                    if os.path.exists(image_path):
                        self.item_images[item] = pygame.image.load(image_path)
                        self.item_images[item] = pygame.transform.scale(
                            self.item_images[item], 
                            (20, 20)  # Default size, will be scaled by cell_size
                        )
                    else:
                        logger.info(f"Warning: Image not found for {item} at {image_path}")
            
            # Load images for output points
            for gate in block_info['output_points']:
                item = gate['item']
                if item not in self.item_images:
                    image_path = os.path.join('assets', f'{item}.png')
                    if os.path.exists(image_path):
                        self.item_images[item] = pygame.image.load(image_path)
                        self.item_images[item] = pygame.transform.scale(
                            self.item_images[item], 
                            (20, 20)  # Default size, will be scaled by cell_size
                        )
                    else:
                        logger.info(f"Warning: Image not found for {item} at {image_path}")
                    
    def define_factory_io_points(self):
        """
        Define external input/output points for the factory on the edges.
        This determines which items need external connections and where they should be placed.
        """
        logger.info("Defining factory I/O points")
        
        # Collect all unique items that need external I/O
        external_inputs = set()  # Items that need to be imported from outside
        external_outputs = set()  # Items that need to be exported outside
        internal_items = set()    # Items that are produced and consumed internally
        
        # Analyze all blocks to determine which items are purely internal vs external
        all_produced_items = set()  # Items produced by any block
        all_consumed_items = set()  # Items consumed by any block
        
        for block_key, block_data in self.block_data.items():
            # Check output information - these are produced items
            for item, data in block_data["tree"].output_information.items():
                all_produced_items.add(item)
                
            # Check input information - these are consumed items  
            for item, data in block_data["tree"].input_information.items():
                all_consumed_items.add(item)
        
        # Items produced but not consumed are external outputs
        external_outputs = all_produced_items - all_consumed_items
        
        # Items consumed but not produced are external inputs
        external_inputs = all_consumed_items - all_produced_items
        
        # Items both produced and consumed are internal
        internal_items = all_produced_items.intersection(all_consumed_items)
        
        logger.info(f"External inputs: {external_inputs}")
        logger.info(f"External outputs: {external_outputs}")
        logger.info(f"Internal items: {internal_items}")
        
        # Now create I/O points for external items
        self.external_io = {
            "input_gates": [],    # List of fixed input gate objects
            "output_gates": []    # List of fixed output gate objects
        }
        
        # Let the user choose edge placement for each external I/O
        self._choose_io_placement(external_inputs, "input")
        self._choose_io_placement(external_outputs, "output")
        
        return self.external_io

    def _choose_io_placement(self, items, io_type):
        """
        Allow the user to choose edge placement for I/O points using a scrollable 1D interface:
        1. Select a factory edge
        2. Select a position along that edge using a scrollable line interface
        
        Args:
            items (set): Set of items to place
            io_type (str): Either "input" or "output"
        """
        if not items:
            return
        
        pygame.init()
        screen_width, screen_height = 800, 800
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Select {io_type.capitalize()} Placement")
        
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        tiny_font = pygame.font.Font(None, 18)
        
        # Dictionary to store item placements
        placements = {}
        
        # Initial grid size estimate (will be adjusted in Z3 solver)
        grid_width = 30  # Larger initial grid to allow more flexibility
        grid_height = 30
        cell_size = 20
        
        # Factory display parameters
        factory_x = 100
        factory_y = 100
        factory_width = grid_width * cell_size
        factory_height = grid_height * cell_size
        
        # Edge thickness for selection
        edge_thickness = 15
        
        # Process one item at a time
        for item in items:
            # STEP 1: Select which edge to place the item on
            selected_edge = None
            
            while selected_edge is None:
                # Draw screen showing factory outline
                screen.fill((240, 240, 240))
                
                # Draw factory outline
                factory_rect = pygame.Rect(factory_x, factory_y, factory_width, factory_height)
                pygame.draw.rect(screen, (200, 200, 200), factory_rect)
                pygame.draw.rect(screen, (100, 100, 100), factory_rect, 2)
                
                # Draw instructions
                instructions1 = font.render(f"Step 1: Select an edge for {item} {io_type}", True, (0, 0, 0))
                instructions2 = small_font.render("Click on any edge of the factory", True, (0, 0, 0))
                screen.blit(instructions1, (100, 50))
                screen.blit(instructions2, (100, 80))
                
                # Draw item image if available
                image_path = os.path.join('assets', f'{item}.png')
                if os.path.exists(image_path):
                    item_image = pygame.image.load(image_path)
                    item_image = pygame.transform.scale(item_image, (32, 32))
                    screen.blit(item_image, (50, 50))
                
                # Define and draw the four edges with different colors
                north_edge = pygame.Rect(factory_x, factory_y, factory_width, edge_thickness)
                east_edge = pygame.Rect(factory_x + factory_width - edge_thickness, factory_y, 
                                    edge_thickness, factory_height)
                south_edge = pygame.Rect(factory_x, factory_y + factory_height - edge_thickness, 
                                    factory_width, edge_thickness)
                west_edge = pygame.Rect(factory_x, factory_y, edge_thickness, factory_height)
                
                # Draw edges with different colors and labels
                edge_colors = {
                    "North": (0, 150, 0),    # Green
                    "East": (0, 0, 150),     # Blue
                    "South": (150, 150, 0),  # Yellow
                    "West": (150, 0, 0)      # Red
                }
                
                pygame.draw.rect(screen, edge_colors["North"], north_edge)
                pygame.draw.rect(screen, edge_colors["East"], east_edge)
                pygame.draw.rect(screen, edge_colors["South"], south_edge)
                pygame.draw.rect(screen, edge_colors["West"], west_edge)
                
                # Add edge labels
                label_north = small_font.render("North", True, (255, 255, 255))
                label_east = small_font.render("East", True, (255, 255, 255))
                label_south = small_font.render("South", True, (255, 255, 255))
                label_west = small_font.render("West", True, (255, 255, 255))
                
                # Rotate east and west labels
                label_east = pygame.transform.rotate(label_east, 270)
                label_west = pygame.transform.rotate(label_west, 90)
                
                # Position labels
                screen.blit(label_north, (factory_x + factory_width//2 - label_north.get_width()//2, factory_y + 2))
                screen.blit(label_east, (factory_x + factory_width - edge_thickness + 2, 
                                        factory_y + factory_height//2 - label_east.get_height()//2))
                screen.blit(label_south, (factory_x + factory_width//2 - label_south.get_width()//2, 
                                        factory_y + factory_height - label_south.get_height() - 2))
                screen.blit(label_west, (factory_x + 2, factory_y + factory_height//2 - label_west.get_height()//2))
                
                # Draw existing I/O points on edges
                for existing_item, placement in placements.items():
                    pos = placement["position"]
                    edge = placement["edge"]
                    
                    # Determine display position based on edge
                    if edge == "North":
                        pt_x = factory_x + pos[0] * cell_size + cell_size // 2
                        pt_y = factory_y + edge_thickness // 2
                    elif edge == "East":
                        pt_x = factory_x + factory_width - edge_thickness // 2
                        pt_y = factory_y + pos[1] * cell_size + cell_size // 2
                    elif edge == "South":
                        pt_x = factory_x + pos[0] * cell_size + cell_size // 2
                        pt_y = factory_y + factory_height - edge_thickness // 2
                    elif edge == "West":
                        pt_x = factory_x + edge_thickness // 2
                        pt_y = factory_y + pos[1] * cell_size + cell_size // 2
                    
                    # Draw a marker for each existing point
                    pygame.draw.circle(screen, (255, 255, 255), (pt_x, pt_y), 4)
                    
                    # Draw item name in tiny font
                    item_label = tiny_font.render(existing_item, True, (255, 255, 255))
                    
                    if edge == "North":
                        screen.blit(item_label, (pt_x - item_label.get_width()//2, pt_y - 15))
                    elif edge == "East":
                        screen.blit(item_label, (pt_x + 7, pt_y - item_label.get_height()//2))
                    elif edge == "South":
                        screen.blit(item_label, (pt_x - item_label.get_width()//2, pt_y + 7))
                    elif edge == "West":
                        screen.blit(item_label, (pt_x - item_label.get_width() - 7, pt_y - item_label.get_height()//2))
                
                pygame.display.flip()
                
                # Wait for user to select an edge
                waiting_for_edge = True
                while waiting_for_edge:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_pos = event.pos
                            
                            if north_edge.collidepoint(mouse_pos):
                                selected_edge = "North"
                                waiting_for_edge = False
                            elif east_edge.collidepoint(mouse_pos):
                                selected_edge = "East"
                                waiting_for_edge = False
                            elif south_edge.collidepoint(mouse_pos):
                                selected_edge = "South"
                                waiting_for_edge = False
                            elif west_edge.collidepoint(mouse_pos):
                                selected_edge = "West"
                                waiting_for_edge = False
            
            # STEP 2: New scrollable 1D interface for positioning
            # Create a new pygame window for the precise position selection
            position_screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption(f"Select Position for {item} on {selected_edge} Edge")
            
            # Define scrollable line parameters
            visible_cells = 15  # Number of cells visible at once
            cell_size = 40      # Size of each cell
            
            # Start at position 0 (no negative values allowed)
            center_index = 0
            offset = 0          # Offset for scrolling
            
            # Define scroll speed and limits - only positive values allowed
            scroll_speed = 1
            min_offset = 0      # No negative positions allowed
            max_offset = 50
            
            # Get existing positions on this edge
            existing_positions = []
            for existing_item, placement in placements.items():
                if placement["edge"] == selected_edge:
                    if selected_edge in ["North", "South"]:
                        existing_positions.append((placement["position"][0], existing_item))
                    else:  # East, West
                        existing_positions.append((placement["position"][1], existing_item))
            
            selected_position = None
            
            while selected_position is None:
                position_screen.fill((240, 240, 240))
                
                # Draw instructions
                instructions = font.render(f"Select position for {item} on {selected_edge} edge", True, (0, 0, 0))
                scroll_info = small_font.render("Use arrow keys to scroll, click to select position", True, (0, 0, 0))
                position_info = small_font.render("Only positive positions (0 and above) are allowed", True, (0, 0, 0))
                position_screen.blit(instructions, (20, 20))
                position_screen.blit(scroll_info, (20, 60))
                position_screen.blit(position_info, (20, 90))
                
                # Draw item image if available
                if os.path.exists(image_path):
                    position_screen.blit(item_image, (20, 120))
                
                # Draw the scrollable line
                line_y = 200
                if selected_edge in ["North", "South"]:
                    # Horizontal line
                    pygame.draw.line(position_screen, (100, 100, 100), (50, line_y), (750, line_y), 2)
                    
                    # Draw cell marks with indices
                    for i in range(visible_cells):
                        cell_index = i + offset  # Only positive indices
                        cell_x = 50 + i * cell_size
                        
                        # Draw cell
                        cell_rect = pygame.Rect(cell_x - cell_size//2, line_y - cell_size//2, cell_size, cell_size)
                        
                        # Highlight center cell (0 position)
                        if cell_index == 0:
                            pygame.draw.rect(position_screen, (200, 200, 255), cell_rect)
                        else:
                            pygame.draw.rect(position_screen, (220, 220, 220), cell_rect)
                            
                        pygame.draw.rect(position_screen, (100, 100, 100), cell_rect, 1)
                        
                        # Draw index
                        index_text = small_font.render(str(cell_index), True, (0, 0, 0))
                        position_screen.blit(index_text, (cell_x - index_text.get_width()//2, line_y + cell_size//2 + 5))
                        
                        # Draw existing items at their positions
                        for pos, existing in existing_positions:
                            if pos == cell_index:
                                pygame.draw.circle(position_screen, (255, 0, 0), (cell_x, line_y), 8)
                                item_label = tiny_font.render(existing, True, (0, 0, 0))
                                position_screen.blit(item_label, (cell_x - item_label.get_width()//2, line_y - 25))
                else:
                    # Vertical line
                    pygame.draw.line(position_screen, (100, 100, 100), (400, 100), (400, 600), 2)
                    
                    # Draw cell marks with indices
                    for i in range(visible_cells):
                        cell_index = i + offset  # Only positive indices
                        cell_y = 100 + i * cell_size
                        
                        # Draw cell
                        cell_rect = pygame.Rect(400 - cell_size//2, cell_y - cell_size//2, cell_size, cell_size)
                        
                        # Highlight center cell (0 position)
                        if cell_index == 0:
                            pygame.draw.rect(position_screen, (200, 200, 255), cell_rect)
                        else:
                            pygame.draw.rect(position_screen, (220, 220, 220), cell_rect)
                            
                        pygame.draw.rect(position_screen, (100, 100, 100), cell_rect, 1)
                        
                        # Draw index
                        index_text = small_font.render(str(cell_index), True, (0, 0, 0))
                        position_screen.blit(index_text, (400 + cell_size//2 + 5, cell_y - index_text.get_height()//2))
                        
                        # Draw existing items at their positions
                        for pos, existing in existing_positions:
                            if pos == cell_index:
                                pygame.draw.circle(position_screen, (255, 0, 0), (400, cell_y), 8)
                                item_label = tiny_font.render(existing, True, (0, 0, 0))
                                position_screen.blit(item_label, (400 - item_label.get_width() - 10, cell_y - 8))
                
                pygame.display.flip()
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT or event.key == pygame.K_UP:
                            offset = max(min_offset, offset - scroll_speed)  # Don't go below 0
                        elif event.key == pygame.K_RIGHT or event.key == pygame.K_DOWN:
                            offset = min(max_offset, offset + scroll_speed)
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_x, mouse_y = event.pos
                        
                        # Determine which cell was clicked
                        if selected_edge in ["North", "South"]:
                            for i in range(visible_cells):
                                cell_index = i + offset  # Only positive indices
                                cell_x = 50 + i * cell_size
                                cell_rect = pygame.Rect(cell_x - cell_size//2, line_y - cell_size//2, cell_size, cell_size)
                                
                                if cell_rect.collidepoint(mouse_x, mouse_y):
                                    if selected_edge == "North":
                                        selected_position = (cell_index, 0)
                                    else:  # South
                                        selected_position = (cell_index, grid_height - 1)
                                    break
                        else:
                            for i in range(visible_cells):
                                cell_index = i + offset  # Only positive indices
                                cell_y = 100 + i * cell_size
                                cell_rect = pygame.Rect(400 - cell_size//2, cell_y - cell_size//2, cell_size, cell_size)
                                
                                if cell_rect.collidepoint(mouse_x, mouse_y):
                                    if selected_edge == "East":
                                        selected_position = (grid_width - 1, cell_index)
                                    else:  # West
                                        selected_position = (0, cell_index)
                                    break
            
            # Store the placement info
            placements[item] = {"edge": selected_edge, "position": selected_position}
            logger.debug(f"Selected {selected_edge} at position {selected_position} for {item}")
        
        pygame.quit()
        
        # Create fixed gate objects for each placement
        for item, placement in placements.items():
            gate_id = f"external_{io_type}_{item}"
            position = placement["position"]
            
            # Create a gate object with fixed position
            fixed_gate = {
                "id": gate_id,
                "item": item,
                "type": io_type,
                "position": position,
                "edge": placement["edge"]
            }
            
            # Add to the appropriate list
            if io_type == "input":
                self.external_io["input_gates"].append(fixed_gate)
            else:
                self.external_io["output_gates"].append(fixed_gate)
            
            logger.info(f"Created fixed {io_type} gate for {item} at position {position} on {placement['edge']} edge")

    def apply_io_constraints_to_solver(self):
        """
        Apply the external I/O constraints to the factory Z3 solver.
        Instead of constraining existing gates, this adds fixed I/O gates to the solver.
        """
        if not hasattr(self, 'external_io') or not self.external_io:
            logger.warning("No external I/O defined, skipping I/O constraints")
            return
        
        if not hasattr(self, 'z3_solver') or not self.z3_solver:
            logger.error("Z3 solver not initialized, cannot apply I/O constraints")
            return
        
        # Add fixed external input gates
        for input_gate in self.external_io["input_gates"]:
            # Create a Z3 fixed gate
            fixed_gate = self.z3_solver.add_fixed_gate(
                gate_id=input_gate["id"],
                item=input_gate["item"],
                position=input_gate["position"],
                gate_type="input",
                edge=input_gate["edge"]
            )
            
            logger.info(f"Added fixed input gate {input_gate['id']} at position {input_gate['position']}")
        
        # Add fixed external output gates
        for output_gate in self.external_io["output_gates"]:
            # Create a Z3 fixed gate
            fixed_gate = self.z3_solver.add_fixed_gate(
                gate_id=output_gate["id"],
                item=output_gate["item"],
                position=output_gate["position"],
                gate_type="output",
                edge=output_gate["edge"]
            )
            
            logger.info(f"Added fixed output gate {output_gate['id']} at position {output_gate['position']}")

    def create_point_connections(self, factory_data):
        """
        Create connections between output and input points with the same item type.
        Prioritize the closest connections until all valid pairs are created.
        
        Rules:
        1. Connect OUTPUT points (sources) to INPUT points (destinations) with the same item
        2. Once points are in a pair, they cannot be connected again
        3. Points from the same block cannot be connected
        4. No output-to-output or input-to-input connections allowed
        
        Args:
            factory_data (dict): Factory data with io_points
            
        Returns:
            list: List of connection pairs, each containing source and target points
        """
        logger.info("Creating point connections for inter module paths")
        
        # Extract all input and output points
        input_points = factory_data["io_points"]["inputs"]
        output_points = factory_data["io_points"]["outputs"]
        
        # Filter out points at the top row (y=0) and other boundary conditions
        def is_valid_connection_point(point):
            pos = point.get("position", [])
            if not isinstance(pos, list) or len(pos) < 2:
                return False
            
            # Skip points at the top row (y=0)
            if pos[1] == 0:
                logger.debug(f"Skipping point at top row: {point['item']} at {pos}")
                return False
            
            return True
        
        # Filter the points
        filtered_input_points = [p for p in input_points if is_valid_connection_point(p)]
        filtered_output_points = [p for p in output_points if is_valid_connection_point(p)]
        
        logger.info(f"Filtered points: {len(input_points)} -> {len(filtered_input_points)} inputs, {len(output_points)} -> {len(filtered_output_points)} outputs")
        
        # Group points by item type
        inputs_by_item = {}
        outputs_by_item = {}
        
        for input_point in filtered_input_points:
            item = input_point["item"]
            if item not in inputs_by_item:
                inputs_by_item[item] = []
            inputs_by_item[item].append(input_point)
        
        for output_point in filtered_output_points:
            item = output_point["item"]
            if item not in outputs_by_item:
                outputs_by_item[item] = []
            outputs_by_item[item].append(output_point)
        
        # Create connections by finding the closest valid pairs
        connections = []
        
        # Track existing connections between blocks for each item to prevent duplicates
        existing_block_connections = {}  # Format: {item: {(output_block, input_block): distance}}
        
        # For each item type, connect OUTPUT points (sources) to INPUT points (destinations)
        for item, output_list in outputs_by_item.items():
            # Skip if no inputs for this item
            if item not in inputs_by_item:
                logger.warning(f"No input points found for item {item}")
                continue
            
            available_inputs = inputs_by_item[item].copy()
            available_outputs = output_list.copy()
            
            # Initialize tracking for this item
            existing_block_connections[item] = {}
            
            logger.debug(f"Processing {item}: {len(available_outputs)} outputs, {len(available_inputs)} inputs")
            
            # Keep finding pairs until no more valid ones can be created
            while available_inputs and available_outputs:
                closest_pair = None
                closest_distance = float('inf')
                
                # Find the closest valid pair: OUTPUT (source) -> INPUT (destination)
                for output_point in available_outputs:
                    output_block = output_point["block_id"]
                    output_pos = output_point["position"]
                    is_output_external = output_point.get("external", False)
                    
                    for input_point in available_inputs:
                        input_block = input_point["block_id"]
                        input_pos = input_point["position"]
                        is_input_external = input_point.get("external", False)
                        
                        # VALIDATION RULES:
                        
                        # Rule 1: Skip if points are in the same block
                        if output_block == input_block:
                            logger.debug(f"Skipping same-block connection: {output_block}")
                            continue
                        
                        # Rule 2: Skip connections between two external points
                        if is_output_external and is_input_external:
                            logger.debug(f"Skipping external-to-external connection for {item}")
                            continue
                        
                        # Rule 3: Check if we already have a connection between these two blocks for this item (in either direction)
                        block_pair_forward = (output_block, input_block)
                        block_pair_reverse = (input_block, output_block)
                        
                        # Check both directions to prevent any duplicate connections between the same blocks
                        existing_connection_key = None
                        existing_distance = float('inf')
                        
                        if block_pair_forward in existing_block_connections[item]:
                            existing_connection_key = block_pair_forward
                            existing_distance = existing_block_connections[item][block_pair_forward]
                        elif block_pair_reverse in existing_block_connections[item]:
                            existing_connection_key = block_pair_reverse
                            existing_distance = existing_block_connections[item][block_pair_reverse]
                        
                        if existing_connection_key:
                            # Calculate current distance to compare
                            try:
                                current_distance = abs(output_pos[0] - input_pos[0]) + abs(output_pos[1] - input_pos[1])
                                if current_distance >= existing_distance:
                                    logger.debug(f"Skipping connection {output_block} -> {input_block} for {item}: existing shorter connection between these blocks (dist {existing_distance}) vs current (dist {current_distance})")
                                    continue
                                else:
                                    logger.debug(f"Found shorter connection {output_block} -> {input_block} for {item}: new (dist {current_distance}) vs existing (dist {existing_distance})")
                                    # We'll update this connection if it becomes the closest pair
                            except Exception as e:
                                logger.error(f"Error calculating distance for duplicate check: {e}")
                                continue
                        
                        # Rule 4: Complexity check - only allow connections from lower/equal complexity to higher/equal complexity
                        output_item_type = self.get_block_item_type(output_block)
                        input_item_type = self.get_block_item_type(input_block)
                        
                        if output_item_type and input_item_type:
                            output_complexity = self.calculate_item_complexity(output_item_type)
                            input_complexity = self.calculate_item_complexity(input_item_type)
                            
                            # Only allow connections where output complexity <= input complexity
                            if output_complexity > input_complexity:
                                logger.debug(f"Skipping connection: {output_item_type}(complexity:{output_complexity}) -> {input_item_type}(complexity:{input_complexity})")
                                continue
                        
                        # Rule 5: Ensure positions are valid lists
                        if isinstance(output_pos, tuple):
                            output_pos = list(output_pos)
                        if isinstance(input_pos, tuple):
                            input_pos = list(input_pos)
                        
                        # Rule 6: Make sure we're dealing with numeric positions
                        if not (isinstance(output_pos, list) and len(output_pos) >= 2 and 
                                isinstance(input_pos, list) and len(input_pos) >= 2):
                            logger.warning(f"Invalid position format: output={output_pos}, input={input_pos}")
                            continue

                       
                       
                        try:
                            # Calculate Manhattan distance
                            distance = abs(output_pos[0] - input_pos[0]) + abs(output_pos[1] - input_pos[1])
                            
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_pair = (output_point, input_point)
                                
                        except Exception as e:
                            logger.error(f"Error calculating distance: {e} for output_pos={output_pos}, input_pos={input_pos}")
                            continue
                
                # If a valid pair was found, add it to connections and remove from available points
                if closest_pair:
                    source_output, target_input = closest_pair
                    
                    # Validate that we have OUTPUT -> INPUT connection
                    if source_output in available_outputs and target_input in available_inputs:
                        output_block = source_output["block_id"]
                        input_block = target_input["block_id"]
                        block_pair_forward = (output_block, input_block)
                        block_pair_reverse = (input_block, output_block)
                        
                        # Check if we need to replace an existing connection (in either direction)
                        connection_to_replace = None
                        existing_connection_key = None
                        
                        # Check for existing connection in forward direction
                        if block_pair_forward in existing_block_connections[item]:
                            existing_connection_key = block_pair_forward
                            for idx, existing_conn in enumerate(connections):
                                if (existing_conn["item"] == item and 
                                    existing_conn["source"]["block_id"] == output_block and 
                                    existing_conn["target"]["block_id"] == input_block):
                                    connection_to_replace = idx
                                    break
                        
                        # Check for existing connection in reverse direction
                        elif block_pair_reverse in existing_block_connections[item]:
                            existing_connection_key = block_pair_reverse
                            for idx, existing_conn in enumerate(connections):
                                if (existing_conn["item"] == item and 
                                    ((existing_conn["source"]["block_id"] == output_block and existing_conn["target"]["block_id"] == input_block) or
                                     (existing_conn["source"]["block_id"] == input_block and existing_conn["target"]["block_id"] == output_block))):
                                    connection_to_replace = idx
                                    break
                        
                        # Create the new connection
                        new_connection = {
                            "item": item,
                            "source": {
                                "position": source_output["position"],
                                "block_id": source_output["block_id"],
                                "external": source_output.get("external", False),
                                "type": "output"  # Source is an output point
                            },
                            "target": {
                                "position": target_input["position"],
                                "block_id": target_input["block_id"],
                                "external": target_input.get("external", False),
                                "type": "input"   # Target is an input point
                            },
                            "distance": closest_distance
                        }
                        
                        if connection_to_replace is not None:
                            # Replace the existing connection
                            old_connection = connections[connection_to_replace]
                            connections[connection_to_replace] = new_connection
                            logger.info(f"Replaced bidirectional connection for {item}: {output_block} -> {input_block}, old distance: {old_connection['distance']}, new distance: {closest_distance}")
                            
                            # Remove the old tracking entry and add the new one
                            if existing_connection_key:
                                del existing_block_connections[item][existing_connection_key]
                        else:
                            # Add new connection
                            connections.append(new_connection)
                            logger.info(f"Created valid OUTPUT->INPUT connection for {item}: {output_block} -> {input_block}, distance: {closest_distance}")
                        
                        # Update the tracking with the new forward direction
                        existing_block_connections[item][block_pair_forward] = closest_distance
                        
                        available_outputs.remove(source_output)
                        available_inputs.remove(target_input)
                    else:
                        logger.error(f"Invalid pair found - points not in available lists!")
                        break
                else:
                    # No more valid pairs can be created
                    logger.debug(f"No more valid pairs for {item}")
                    break
        
        logger.info(f"Created {len(connections)} valid OUTPUT->INPUT point connections")
        return connections

    def plan_inter_block_paths(self, factory_data):
        logger.info("Planning inter-block paths with detailed obstacles...")
        
        if not self.final_blocks or not self.gate_connections:
            logger.error("Cannot plan paths: No blocks or gate connections available")
            return {}, {}
        
        # Calculate the overall grid size based on final_x and final_y
        grid_width = factory_data["factory_dimensions"]["width"]
        grid_height = factory_data["factory_dimensions"]["height"]
        
        # Create an obstacle map (0 = free, 1 = obstacle)
        obstacle_map = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Mark all entities in the factory data as obstacles
        if "entities" in factory_data:
           
            for assembler in factory_data["entities"].get("assemblers", []):
                pos_x, pos_y = assembler["position"]
                # Assemblers are nxm
                size_x, size_y = assembler.get("dimension", [3, 3])

                for dy in range(size_y):
                    for dx in range(size_x):
                        x, y = pos_x + dx, pos_y + dy
                        if 0 <= x < grid_width and 0 <= y < grid_height:
                            obstacle_map[y][x] = 1
            
            # Mark transport belts
            for belt in factory_data["entities"].get("belts", []):
                pos_x, pos_y = belt["position"]
                if 0 <= pos_x < grid_width and 0 <= pos_y < grid_height:
                    obstacle_map[pos_y][pos_x] = 1
            
            # Mark inserters
            for inserter in factory_data["entities"].get("inserters", []):
                pos_x, pos_y = inserter["position"]
                if 0 <= pos_x < grid_width and 0 <= pos_y < grid_height:
                    obstacle_map[pos_y][pos_x] = 1
            
            # Mark underground belts
            for ug_belt in factory_data["entities"].get("underground_belts", []):
                pos_x, pos_y = ug_belt["position"]
                if 0 <= pos_x < grid_width and 0 <= pos_y < grid_height:
                    obstacle_map[pos_y][pos_x] = 1
            
            # Mark splitters (2x1 or 1x2 entities)
            for splitter in factory_data["entities"].get("splitters", []):
                pos_x, pos_y = splitter["position"]
                direction = splitter.get("direction", "north")
                
                # Mark the primary position
                if 0 <= pos_x < grid_width and 0 <= pos_y < grid_height:
                    obstacle_map[pos_y][pos_x] = 1
                
                # Mark the secondary position based on direction
                if direction in ["north", "south", "up", "down"]:
                    # Horizontal splitter (2 tiles wide)
                    second_x, second_y = pos_x + 1, pos_y
                else:
                    # Vertical splitter (2 tiles tall)
                    second_x, second_y = pos_x, pos_y + 1
                
                if 0 <= second_x < grid_width and 0 <= second_y < grid_height:
                    obstacle_map[second_y][second_x] = 1
        

            for power_pole in factory_data["entities"].get("power_poles", []):
                pos_x, pos_y = power_pole["position"]
                # Power poles occupy a 1x1 area
                if 0 <= pos_x < grid_width and 0 <= pos_y < grid_height:
                    obstacle_map[pos_y][pos_x] = 1
            
            # Prepare connection points for the pathfinder
            connection_points = {}
        
            # Create connections from the point connection data
            point_connections = self.create_point_connections(factory_data)
            
            logger.info(f"Created {len(point_connections)} point connections for pathfinding")
            logger.debug(f"Point connections: {point_connections}")
            
            # Process each connection to add to connection_points
            for idx, connection in enumerate(point_connections):
                item = connection["item"]
                source = connection["source"]
                target = connection["target"]
                
                # Get source and target positions and ensure they are lists
                source_pos = source["position"]
                target_pos = target["position"]
                
                # Convert positions to tuples if they are lists (pathfinder needs tuples as hashable keys)
                if isinstance(source_pos, list):
                    source_pos = tuple(source_pos)
                if isinstance(target_pos, list):
                    target_pos = tuple(target_pos)
                
                # Create a unique key for this connection
                conn_key = f"{item}_{idx}"
                
                # Add to connection_points in the format expected by MultiAgentPathfinder
                connection_points[conn_key] = {
                    "item": item,
                    "start_points": [source_pos],  # MultiAgentPathfinder expects this key
                    "destination": [target_pos],   # MultiAgentPathfinder expects this key
                    "source_external": source.get("external", False),
                    "target_external": target.get("external", False),
                    "is_fluid": False  # Default to false, can be updated based on item type
                }
                
                logger.info(f"Added connection point for {item}: {source_pos} -> {target_pos}")
        
        if not connection_points:
            logger.warning("No valid connections to route! Please check your gate connections.")
            return {}, {}

        
        # Create and run the MultiAgentPathfinder
        try:
            pathfinder = MultiAgentPathfinder(
                obstacle_map=obstacle_map,
                points=connection_points,
                allow_underground=True,
                underground_length=5,
                find_optimal_paths=True,
                output_item=self.output_item
            )
              # Find paths for all connections
            paths, inserters = pathfinder.find_paths_for_all_items()
            if paths:
                logger.info(f"Found paths for {len(paths)} connections out of {len(connection_points)} requested")
            else:
                logger.warning(f"No paths were found by the pathfinder. Check if the gates are positioned properly.")
            
            # Save the paths for later use
            self.inter_block_paths = paths
            self.inter_block_inserters = inserters
            
            # Generate path visualizations
            try:
                pass
                #pathfinder.visualize_grid(filename="factory_detailed_obstacle_map_with_paths.png")
                #pathfinder.visualize_paths(filename_template="inter_block_path_detailed_{}.png")
                #logger.info("Path visualizations saved to disk")
            except Exception as e:
                logger.error(f"Failed to visualize paths: {e}")
            
            return paths, inserters
            
        except Exception as e:
            logger.error(f"Error in pathfinding: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}

    def calculate_factory_bounds(self):
        """
        Calculate the minimum and maximum x and y coordinates of factory components.
        This helps determine the occupied area of the factory for visualization.
        
        Returns:
            tuple: (min_x, max_x, min_y, max_y) bounds of the factory
        """
        min_x = self.final_x
        max_x = 0
        min_y = self.final_y
        max_y = 0
        
        # Check block positions
        for block_id, block_info in self.final_blocks.items():
            block_x = block_info['x']
            block_y = block_info['y']
            block_width = block_info['width']
            block_height = block_info['height']
            
            # Update bounds
            min_x = min(min_x, block_x)
            max_x = max(max_x, block_x + block_width)
            min_y = min(min_y, block_y)
            max_y = max(max_y, block_y + block_height)
            
            # Check gate positions
            for gate in block_info.get('input_points', []) + block_info.get('output_points', []):
                gate_x = block_x + gate['x']
                gate_y = block_y + gate['y']
                
                min_x = min(min_x, gate_x)
                max_x = max(max_x, gate_x + 1)
                min_y = min(min_y, gate_y)
                max_y = max(max_y, gate_y + 1)
        
        # Check path positions
        if hasattr(self, 'inter_block_paths') and self.inter_block_paths:
            # Handle both list and dict formats
            if isinstance(self.inter_block_paths, list):
                for path_data in self.inter_block_paths:
                    if 'path' in path_data:
                        for x, y in path_data['path']:
                            min_x = min(min_x, x)
                            max_x = max(max_x, x + 1)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y + 1)
            else:
                # Original dict format
                for item_key, path_data_list in self.inter_block_paths.items():
                    for path_data in path_data_list:
                        if 'path' in path_data:
                            for x, y in path_data['path']:
                                min_x = min(min_x, x)
                                max_x = max(max_x, x + 1)
                                min_y = min(min_y, y)
                                max_y = max(max_y, y + 1)
                    
                    if 'underground_segments' in path_data:
                        for segment_id, segment in path_data['underground_segments'].items():
                            if 'start' in segment and 'end' in segment:
                                start_x, start_y = segment['start']
                                end_x, end_y = segment['end']
                                
                                min_x = min(min_x, start_x, end_x)
                                max_x = max(max_x, start_x + 1, end_x + 1)
                                min_y = min(min_y, start_y, end_y)
                                max_y = max(max_y, start_y + 1, end_y + 1)
        
        # Ensure we have at least some area even if no blocks are found
        if min_x > max_x or min_y > max_y:
            return 0, self.final_x, 0, self.final_y
        
        return min_x, max_x, min_y, max_y    
    
    def _get_absolute_position(self, gate):
        # For external gates, we need to calculate position based on edge
        if "edge" in gate:
            edge = gate["edge"]
            position = gate["position"]
            
            # Ensure position is a number
            if isinstance(position, (list, tuple)):
                position = position[0]  # Use first element if it's a list/tuple
            
            # Get factory dimensions
            grid_width = self.final_x
            grid_height = self.final_y
            
            # Calculate position based on edge
            if edge == "North":
                return [int(position), 0]  # Top edge
            elif edge == "South":
                return [int(position), grid_height - 1]  # Bottom edge
            elif edge == "East":
                return [grid_width - 1, int(position)]  # Right edge
            elif edge == "West":
                return [0, int(position)]  # Left edge
            else:
                logger.error(f"Unknown edge type: {edge}")
                return [0, 0]
        
        # For normal gates, return position directly - this shouldn't be called
        elif "position" in gate:
            pos = gate["position"]
            # Convert to list if it's a tuple
            if isinstance(pos, tuple):
                return list(pos)
            elif isinstance(pos, list):
                return pos
            else:
                logger.error(f"Unexpected position format: {pos}")
                return [0, 0]
        else:
            logger.error(f"Malformed gate object: {gate}")
            return [0, 0]
    
    def calculate_item_complexity(self, item_name):
        """
        Calculate the complexity of an item based on its position in the recipe tree.
        Base materials (no recipe) have complexity 0.
        Each level of crafting adds 1 to complexity.
        
        Args:
            item_name (str): The name of the item
            
        Returns:
            int: The complexity level of the item (0 = base material, higher = more complex)
        """
        if not hasattr(self, '_complexity_cache'):
            self._complexity_cache = {}
            
        if item_name in self._complexity_cache:
            return self._complexity_cache[item_name]
        
        try:
            recipes = load_json("recipes.json")
            
            # Find the recipe for this item
            item_recipe = None
            for recipe in recipes:
                if recipe.get("id") == item_name:
                    item_recipe = recipe
                    break
            
            # If no recipe found, it's a base material
            if not item_recipe:
                self._complexity_cache[item_name] = 0
                return 0
            
            # If recipe has no ingredients, it's also a base material
            ingredients = item_recipe.get("ingredients", [])
            if not ingredients:
                self._complexity_cache[item_name] = 0
                return 0
            
            # Calculate complexity as 1 + max complexity of ingredients
            max_ingredient_complexity = 0
            for ingredient in ingredients:
                ingredient_name = ingredient.get("id") if isinstance(ingredient, dict) else ingredient
                if ingredient_name and ingredient_name != item_name:  # Avoid circular references
                    ingredient_complexity = self.calculate_item_complexity(ingredient_name)
                    max_ingredient_complexity = max(max_ingredient_complexity, ingredient_complexity)
            
            complexity = max_ingredient_complexity + 1
            self._complexity_cache[item_name] = complexity
            return complexity
            
        except Exception as e:
            logger.error(f"Error calculating complexity for {item_name}: {e}")
            # Default to complexity 0 on error
            self._complexity_cache[item_name] = 0
            return 0

    def get_block_item_type(self, block_id):
        """
        Extract the item type from a block ID.
        
        Args:
            block_id (str): The block ID (e.g., "Block_copper-cable_0_0")
            
        Returns:
            str: The item type or None if it can't be determined
        """
        # Handle external blocks
        if "external_input_" in block_id:
            return block_id.replace("external_input_", "")
        elif "external_output_" in block_id:
            return block_id.replace("external_output_", "")
        
        # Handle regular blocks (format: "Block_<item>_<x>_<y>")
        if block_id.startswith("Block_"):
            parts = block_id.split("_")
            if len(parts) >= 2:
                return parts[1]  # Return the item type part
        
        return None
    
    
    
def manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        
def main():
    
    
    #visualize_factory(json_path="Factorys/factory_electronic-circuit_100.json",save_path="Factorys/factory_electronic-circuit_100.png")
    
    output_item = "electronic-circuit"
    amount = 100
    max_assembler_per_blueprint = 5
    
    start_width = 15
    start_height = 15

    builder = FactoryBuilder(output_item, amount, max_assembler_per_blueprint, start_width, start_height, load_modules=True)

    # num_factories, production_data = builder.eval_split()
    # logger.info(f"Number of factories required: {num_factories}")
    
    builder.split_recipies()
    
    start_time = time.perf_counter()       
    print(start_time)
    builder.solve_factory()
    end_time = time.perf_counter()
    print(end_time)

    
    log_method_time(item=output_item,amount=amount,method_name="solve",assemblers_per_recipie=max_assembler_per_blueprint,num_subfactories=builder.get_num_subfactories(),start_time=start_time,end_time=end_time)
    

    #builder.visualize_factory(save_path=f"Factorys/{output_item}_factory.png")


def log_method_time(item, amount, method_name,assemblers_per_recipie,num_subfactories,start_time, end_time):
    execution_time = end_time - start_time
    logger.info(f"Execution time for {method_name}: {execution_time:.4f} seconds.")
    
    # Open the CSV file and append the data
    try:
        with open("execution_times_big_factory.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([item, amount, method_name,assemblers_per_recipie,num_subfactories,execution_time])
    except Exception as e:
        logger.error(f"Error logger execution time for {method_name}: {e}")
        
 
def plot_csv_data(file_path):
    # Configure matplotlib for LaTeX output with better font handling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'pgf.rcfonts': True,  # Use matplotlib's font system
        'pgf.preamble': r'\usepackage{amsmath}\usepackage{amssymb}'
    })

    df = pd.read_csv(file_path, header=None, names=["item", "steps", "action", "param1", "param2", "solve_time"])

    # Convert steps to categorical for better visualization
    df["steps"] = (df["steps"]//100).astype(str)

    # Define plot directory
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Set the style
    sns.set_style("whitegrid")

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    ax =sns.boxplot(
        x="steps",
        y="solve_time",
        hue="param1",  # Use param1 to differentiate if needed, or remove if not applicable
        data=df,
        palette = {5: "red", 0: "blue"}, 
        legend=False,
    )
    ax.set_yscale('log')

    # Labels and title
    plt.title("Electronic Circuit - Solve Time")
    plt.xlabel("Number of Modules")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)

    # Save the boxplot as PGF for LaTeX
    box_plot_path = os.path.join(plot_dir, "electronic_circuit_solve_box_plot.pgf")
    plt.savefig(box_plot_path, format='pgf', bbox_inches='tight')
    plt.close()  # Close the plot to prevent overlap with other subplots

    logger.info(f"Boxplot saved at: {box_plot_path}")
      
      
      
def create_blueprint_from_json(json_path, output_path=None):
    """
    Create a Factorio blueprint directly from a factory JSON file.
    
    Args:
        json_path (str): Path to the factory JSON file
        output_path (str, optional): Path where to save the blueprint file. If None, will use same name as input with .txt extension
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            factory_data = json.load(f)
        
        logger.info(f"Creating blueprint from {json_path}")
        
        # Set output path if not provided
        if output_path is None:
            output_path = os.path.splitext(json_path)[0] + ".txt"
        
        # Create a new blueprint
        blueprint = Blueprint()
        
        # Track occupied positions to avoid overlap
        occupied_positions = set()
        
        # Track assembler positions for orienting inserters
        assembler_positions = {}  # Maps (x, y) to assembler object
        
        logger.info("1. Placing assembling machines...")
        # Place assembling machines from the JSON data
        if "entities" in factory_data and "assemblers" in factory_data["entities"]:
            for assembler_data in factory_data["entities"]["assemblers"]:
                item = assembler_data["item"]
                position = assembler_data["position"]
                
                # Assemblers are 3x3 and centered on their position in Factorio
                # In JSON, position is top-left corner, we need to offset by (1,1) for center
                center_x = position[0] + 1
                center_y = position[1] + 1
                
                logger.info(f"  - Placing assembler for {item} at ({center_x},{center_y})")
                
                # Create assembler entity
                assembler = AssemblingMachine(
                    name="assembling-machine-1",
                    position=(center_x, center_y),
                    recipe=item  # Set the recipe to the item it produces
                )
                blueprint.entities.append(assembler)
                
                # Store the assembler in our mapping and mark the 3x3 area as occupied
                for dx in range(3):
                    for dy in range(3):
                        pos = (position[0] + dx, position[1] + dy)
                        assembler_positions[pos] = assembler
                        occupied_positions.add(pos)
        
        logger.info("2. Placing inserters...")
        # Place inserters from the JSON data
        if "entities" in factory_data and "inserters" in factory_data["entities"]:
            for inserter_data in factory_data["entities"]["inserters"]:
                item = inserter_data["item"]
                position = tuple(inserter_data["position"])
                direction = inserter_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    logger.info(f"  - Skipping inserter at {position} due to overlap")
                    continue
                
                # Convert direction string to draftsman Direction constant
                # Rotate by 180 degrees as Factorio and the JSON use different conventions
                if direction == "north":
                    blueprint_direction = Direction.SOUTH
                elif direction == "east":
                    blueprint_direction = Direction.WEST
                elif direction == "south":
                    blueprint_direction = Direction.NORTH
                elif direction == "west":
                    blueprint_direction = Direction.EAST
                else:
                    # Handle common direction names
                    direction_map = {
                        "up": Direction.NORTH,
                        "right": Direction.EAST,
                        "down": Direction.SOUTH,
                        "left": Direction.WEST
                    }
                    blueprint_direction = direction_map.get(direction, Direction.NORTH)
                
                # Create inserter entity
                logger.info(f"  - Placing inserter for {item} at {position} facing {blueprint_direction}")
                inserter = Inserter(
                    name="inserter",
                    position=position,
                    direction=blueprint_direction
                )
                blueprint.entities.append(inserter)
                occupied_positions.add(position)
        
        
        
        logger.info("3. Placing underground belts...")
        # Place underground belts from the JSON data
        if "entities" in factory_data and "underground_belts" in factory_data["entities"]:
            for ug_data in factory_data["entities"]["underground_belts"]:
                item = ug_data["item"]
                position = tuple(ug_data["position"])
                belt_type = ug_data["type"]  # "entrance" or "exit"
                direction = ug_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    logger.info(f"  - Skipping underground belt at {position} due to overlap")
                    continue
                
                # Convert direction string to draftsman Direction constant
                direction_map = {
                    "north": Direction.NORTH,
                    "east": Direction.EAST,
                    "south": Direction.SOUTH,
                    "west": Direction.WEST
                }
                blueprint_direction = direction_map.get(direction, Direction.NORTH)
                
                # Determine underground belt type
                ug_type = "input" if belt_type == "entrance" else "output"
                
                # Create underground belt entity
                logger.info(f"  - Placing underground belt for {item} at {position} facing {blueprint_direction}, type: {ug_type}")
                ug_belt = UndergroundBelt(
                    name="underground-belt",
                    position=position,
                    direction=blueprint_direction,
                    type=ug_type
                )
                blueprint.entities.append(ug_belt)
                occupied_positions.add(position)
        
        logger.info("4. Placing splitters...")
        # Place splitters from the JSON data
        if "entities" in factory_data and "splitters" in factory_data["entities"]:
            for splitter_data in factory_data["entities"]["splitters"]:
                item = splitter_data["item"]
                position = tuple(splitter_data["position"])
                direction = splitter_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    logger.info(f"  - Skipping splitter at {position} due to overlap")
                    continue
                
                # Convert direction string to draftsman Direction constant
                direction_map = {
                    "north": Direction.NORTH,
                    "east": Direction.EAST,
                    "south": Direction.SOUTH,
                    "west": Direction.WEST
                }
                blueprint_direction = direction_map.get(direction, Direction.NORTH)
                
                # Create splitter entity
                logger.info(f"  - Placing splitter for {item} at {position} facing {blueprint_direction}")
                splitter = BlueprintSplitter(
                    name="splitter",
                    position=position,
                    direction=blueprint_direction
                )
                blueprint.entities.append(splitter)
                occupied_positions.add(position)
                
                # Splitters are 2x1, so mark the second tile as occupied too
                # The orientation depends on the direction
                if blueprint_direction in [Direction.NORTH, Direction.SOUTH]:
                    # Horizontal splitter (takes up two horizontal tiles)
                    occupied_positions.add((position[0] + 1, position[1]))
                else:
                    # Vertical splitter (takes up two vertical tiles)
                    occupied_positions.add((position[0], position[1] + 1))
        
        
                
        logger.info("5. Placing transport belts...")
        # Place regular belts from the JSON data
        if "entities" in factory_data and "belts" in factory_data["entities"]:
            for belt_data in factory_data["entities"]["belts"]:
                item = belt_data["item"]
                position = tuple(belt_data["position"])
                direction = belt_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    logger.info(f"  - Skipping belt at {position} due to overlap")
                    continue
                
                # Convert direction string to draftsman Direction constant
                direction_map = {
                    "north": Direction.NORTH,
                    "east": Direction.EAST,
                    "south": Direction.SOUTH,
                    "west": Direction.WEST
                }
                blueprint_direction = direction_map.get(direction, Direction.NORTH)
                
                # Create transport belt entity
                logger.info(f"  - Placing transport belt for {item} at {position} facing {blueprint_direction}")
                belt = TransportBelt(
                    name="transport-belt",
                    position=position,
                    direction=blueprint_direction
                )
                blueprint.entities.append(belt)
                occupied_positions.add(position)
        
        logger.info("6. Placing external I/O points as constant combinators...")
        if "io_points" in factory_data and "inputs" in factory_data["io_points"]:
                for io_data in factory_data["io_points"]["inputs"]:
                    if io_data["external"]:
                        item = io_data["item"]
                        position = tuple(io_data["position"])
                        
                        # Skip if position is occupied
                        if position in occupied_positions:
                            logger.info(f"  - Skipping input point at {position} due to overlap")
                            continue
                    
                        logger.info(f"  - Placing input point for {item} at {position}")
                        constant_combinator = ConstantCombinator()
                        constant_combinator.tile_position = position
                        constant_combinator.direction = Direction.SOUTH
                        section = constant_combinator.add_section()
                        section.filters = [{
                                    "index": 1,
                                    "name": item,  # The name should be at top level, not nested in signal
                                    "count": 1,
                                    "comparator": "=",
                        }]
                        constant_combinator.id = "output_" + item
                        blueprint.entities.append(constant_combinator)
                        
                        occupied_positions.add(position)
            
        if "io_points" in factory_data and "outputs" in factory_data["io_points"]:
                for io_data in factory_data["io_points"]["outputs"]:
                    if io_data["external"]:
                        item = io_data["item"]
                        position = tuple(io_data["position"])
                        
                        # Skip if position is occupied
                        if position in occupied_positions:
                            logger.info(f"  - Skipping output point at {position} due to overlap")
                            continue
        
                        logger.info(f"  - Placing output point for {item} at {position}")
                        constant_combinator = ConstantCombinator()
                        constant_combinator.tile_position = position
                        constant_combinator.direction = Direction.SOUTH
                        section = constant_combinator.add_section()
                        section.filters = [{
                                    "index": 1,
                                    "name": item,  # The name should be at top level, not nested in signal
                                    "count": 1,
                                    "comparator": "="
                        }]
                        constant_combinator.id = "output_" + item
                        
                        blueprint.entities.append(constant_combinator)
                        occupied_positions.add(position)
            
        logger.info("7. Placing power poles...")
        if "entities" in factory_data and "power_poles" in factory_data["entities"]:
            for power_pole_data in factory_data["entities"]["power_poles"]:
                # Get power pole type and position
                pole_type = power_pole_data["item"]
                x, y = power_pole_data["position"]
         
                logger.info(f"  - Placing {pole_type} at ({x},{y})")
                power_pole = ElectricPole(
                    name=pole_type,
                    position=(x, y)
                )
                logger.info(f"  - Placed {pole_type} at ({x},{y})")
                blueprint.entities.append(power_pole)

        logger.info("8. Placing inter-module paths...")
        # Process inter-block paths
        if "inter_block_paths" in factory_data:
            _add_path_belts_to_blueprint(blueprint, occupied_positions,factory_data["inter_block_paths"])

        # Export the blueprint to a file
        logger.info(f"9. Exporting blueprint to {output_path}...")
        with open(output_path, "w") as f:
            f.write(blueprint.to_string())
            
        logger.info(f"Blueprint successfully exported to {output_path}")
        return True
 
    except Exception as e:
        logger.info(f"Error creating blueprint: {e}")
        import traceback
        traceback.print_exc()
        return False  
    

def _add_path_belts_to_blueprint(blueprint, occupied_positions, paths):
        """Add pathfinder-generated belts to the blueprint, avoiding overlaps"""

        # Handle both list and dict formats for paths
        if isinstance(paths, list):
            path_count = len(paths)
            logger.info(f"  - Processing {path_count} pathfinder-generated belt paths")
            
            for path_data in paths:
                item_name = path_data.get("item", "unknown")
                logger.info(f"  - Processing path for {item_name}")
                _add_belt_path_to_blueprint(blueprint, path_data, item_name, occupied_positions)
        else:
            # Original dict format
            path_count = sum(len(item_paths) for item_paths in paths.values())
            logger.info(f"  - Processing {path_count} pathfinder-generated belt paths for {len(paths)} items")

            for item_key, item_paths in paths.items():
                # Extract base item name
                item_name = item_key.split('_')[0] if '_' in item_key else item_key
                logger.info(f"  - Processing {len(item_paths)} paths for {item_name}")
                
                for path_data in item_paths:
                    _add_belt_path_to_blueprint(blueprint, path_data, item_name, occupied_positions)
                _add_belt_path_to_blueprint(blueprint, path_data, item_name, occupied_positions)


def _add_belt_path_to_blueprint(blueprint, path_data, item, occupied_positions):
    
        config = load_config()

        items_data = load_json("recipes.json")
 
        # Create a lookup dictionary for items by their ID
        item_lookup = {item["id"]: item for item in items_data}
        item = item_lookup.get(item, {})
   
        is_fluid = item.get("type", "") == "Liquid"
        
        if is_fluid:
            belt_type = config.get("pipes", {}).get("default_type", "pipe")
            underground_type = config.get("pipes", {}).get("underground_type", "pipe-to-ground")
        else:
            belt_type = config["belts"]["default_type"]
            underground_type = config["belts"]["underground_type"]
            
        """Add a specific belt path to the blueprint, avoiding overlaps"""
        path = path_data.get('path', [])
        if not path:
            return
        
    
        # Get orientation information
        has_orientation = 'orientation' in path_data and path_data['orientation']
        
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
                    if dx > 0:  # Moving right (east)
                        direction = Direction.EAST
                    else:  # Moving left (west)
                        direction = Direction.WEST
                else:
                    if dy > 0:  # Moving down (south)
                        direction = Direction.SOUTH
                    else:  # Moving up (north)
                        direction = Direction.NORTH
                
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
            if current_tuple in underground_positions:
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
                    direction = _get_belt_direction(path_data['orientation'][current_key])
                else:
                    # Calculate direction from current to next position
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    direction = _get_belt_direction((dx, dy))
            else:
                # Calculate direction from current to next position
                dx = next_pos[0] - current[0]
                dy = next_pos[1] - current[1]
                direction = _get_belt_direction((dx, dy))
            
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
                
                # Skip if this position is part of underground belt or a splitter
                if last_tuple in underground_positions or last_tuple in occupied_positions:
                    logger.info(f"  - Skipping last belt at {last_pos} due to overlap or underground belt")
                    return
                    
                second_last = path[-2]

                # Calculate direction for last belt
                direction = None
                if has_orientation and path_data['orientation']:
                    # Convert last position to tuple if it's a list
                    last_pos_key = str(tuple(last_pos)) if isinstance(last_pos, list) else str(last_pos)
                    
                    if last_pos_key in path_data['orientation']:
                        direction = _get_belt_direction(path_data['orientation'][last_pos_key])
                    else:
                        # Calculate direction from second-last to last
                        dx = last_pos[0] - second_last[0]
                        dy = last_pos[1] - second_last[1]
                        direction = _get_belt_direction((dx, dy))
                else:
                    # Calculate direction from second-last to last
                    dx = last_pos[0] - second_last[0]
                    dy = last_pos[1] - second_last[1]
                    direction = _get_belt_direction((dx, dy))
                
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


def visualize_factory(json_path, cell_size=20, save_path=None):
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Initialize Pygame
        pygame.init()
        
        # Get original factory dimensions
        width = data["factory_dimensions"]["width"]
        height = data["factory_dimensions"]["height"]
        
        # Create grid occupancy maps to identify empty rows and columns
        column_occupancy = [False] * width
        row_occupancy = [False] * height
        
        # Load images
        images = load_factory_images(cell_size)
        item_images = load_item_images(cell_size)
        
        # Mark occupied cells from all entities
        def mark_occupied(x, y, w=1, h=1):
            for dx in range(w):
                for dy in range(h):
                    col = x + dx
                    row = y + dy
                    if 0 <= col < width and 0 <= row < height:
                        column_occupancy[col] = True
                        row_occupancy[row] = True
        
        # Mark blocks
        for block_id, block in data["blocks"].items():
            x = block["position"][0]
            y = block["position"][1]
            w = block["dimensions"][0]
            h = block["dimensions"][1]
            mark_occupied(x, y, w, h)
        
        # Mark entities
        if "entities" in data:
            # Mark assemblers
            for assembler in data["entities"].get("assemblers", []):
                pos = assembler.get("position")
                w, h = assembler.get("dimensions")
                
                # Handle both list and single value position formats
                if isinstance(pos, list) and len(pos) >= 2:
                    mark_occupied(pos[0], pos[1], w, h)
                else:
                    # Assuming pos might be a single number in some cases
                    logger.warning(f"Found unusual assembler position: {pos}")
                    
            # Mark inserters
            for inserter in data["entities"].get("inserters", []):
                if "position" in inserter:
                    pos = inserter.get("position")
                    if isinstance(pos, list) and len(pos) >= 2:
                        mark_occupied(pos[0], pos[1])
                
            # Mark belts
            for belt in data["entities"].get("belts", []):
                if "position" in belt:
                    pos = belt["position"]
                    if isinstance(pos, list) and len(pos) >= 2:
                        mark_occupied(pos[0], pos[1])
                
            # Mark underground belts
            for underground_belt in data["entities"].get("underground_belts", []):
                if "position" in underground_belt:
                    pos = underground_belt["position"]
                    if isinstance(pos, list) and len(pos) >= 2:
                        mark_occupied(pos[0], pos[1])
                
            # Mark power poles
            for power_pole in data["entities"].get("power_poles", []):
                if "position" in power_pole:
                    pos = power_pole["position"]
                    if isinstance(pos, list) and len(pos) >= 2:
                        mark_occupied(pos[0], pos[1])
                
            # Mark splitters (2x1 or 1x2)
            for splitter in data["entities"].get("splitters", []):
                if "position" in splitter:
                    pos = splitter["position"]
                    if isinstance(pos, list) and len(pos) >= 2:
                        direction = splitter.get("direction", "north")
                        if direction in ["east", "west"]:
                            mark_occupied(pos[0], pos[1], 1, 2)
                        else:
                            mark_occupied(pos[0], pos[1], 2, 1)
        
        # Mark I/O points
        if "io_points" in data:
            inputs = data["io_points"].get("inputs")
            outputs = data["io_points"].get("outputs")
            
            if inputs:
                for input_point in inputs:
                    if "position" in input_point:
                        pos = input_point["position"]
                        mark_occupied(pos[0], pos[1])
                      
            if outputs:
                for output_point in outputs:
                    if "position" in output_point:
                        pos = output_point["position"]
                        mark_occupied(pos[0], pos[1])
        
        # Mark paths in inter-block paths
        if "inter_block_paths" in data:
            # Handle both list and dict formats
            if isinstance(data["inter_block_paths"], list):
                for path_data in data["inter_block_paths"]:
                    if 'path' in path_data:
                        for position in path_data['path']:
                            if isinstance(position, list) and len(position) >= 2:
                                mark_occupied(position[0], position[1])
            else:
                # Original dict format
                for item_key, path_data_list in data["inter_block_paths"].items():
                    for path_data in path_data_list:
                        if 'path' in path_data:
                            for position in path_data['path']:
                                if isinstance(position, list) and len(position) >= 2:
                                    mark_occupied(position[0], position[1])
        
        # Create mapping from original to compressed coordinates
        x_map = []
        compressed_x = 0
        for x in range(width):
            if column_occupancy[x]:
                x_map.append(compressed_x)
                compressed_x += 1
            else:
                x_map.append(-1)  # Mark skipped columns
        
        y_map = []
        compressed_y = 0
        for y in range(height):
            if row_occupancy[y]:
                y_map.append(compressed_y)
                compressed_y += 1
            else:
                y_map.append(-1)  # Mark skipped rows
        
        # Calculate new dimensions
        compressed_width = sum(1 for x in column_occupancy if x)
        compressed_height = sum(1 for y in row_occupancy if y)
        
        # Create the display surface with the compressed dimensions
        screen_width = compressed_width * cell_size 
        screen_height = compressed_height * cell_size
        
        # Create a surface with the proper dimensions to draw on
        screen = pygame.Surface((screen_width, screen_height))
        screen.fill((255, 255, 255))  # White background
        
        # Draw grid lines if needed
        for y in range(compressed_height):
            for x in range(compressed_width):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)
        
        # Map to store (compressed_x, compressed_y) to (original_x, original_y)
        reverse_map = {}
        for orig_x, comp_x in enumerate(x_map):
            if comp_x != -1:
                for orig_y, comp_y in enumerate(y_map):
                    if comp_y != -1:
                        reverse_map[(comp_x, comp_y)] = (orig_x, orig_y)

        # Draw inter-block paths
        if "inter_block_paths" in data:
            
            paths = data.get("inter_block_paths")
            
            # Draw inter-block paths
            for path_data in paths:
                item = path_data.get("item")
                path = path_data.get("path", [])
                underground_segments = path_data.get("underground_segments", {})
                
                # Skip if path is empty
                if not path:
                    continue
                
                # Create a set of positions that are part of underground segments
                underground_positions = set()
                for segment_id, segment in underground_segments.items():
                    start_pos = tuple(segment["start"])
                    end_pos = tuple(segment["end"])
                    underground_positions.add(start_pos)
                    underground_positions.add(end_pos)
                    
                    # Also add intermediate positions for underground segments
                    if start_pos[0] == end_pos[0]:  # Vertical underground
                        min_y = min(start_pos[1], end_pos[1])
                        max_y = max(start_pos[1], end_pos[1])
                        for y in range(min_y + 1, max_y):
                            underground_positions.add((start_pos[0], y))
                    elif start_pos[1] == end_pos[1]:  # Horizontal underground
                        min_x = min(start_pos[0], end_pos[0])
                        max_x = max(start_pos[0], end_pos[0])
                        for x in range(min_x + 1, max_x):
                            underground_positions.add((x, start_pos[1]))
                
                # Draw regular belt segments
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    current_tuple = tuple(current)
                    
                    # Skip if position is part of an underground belt segment
                    if current_tuple in underground_positions:
                        continue
                    
                    # Skip if position is outside the grid or in a skipped row/column
                    orig_x, orig_y = current
                    if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    compressed_x = x_map[orig_x]
                    compressed_y = y_map[orig_y]
                    
                    pixel_x = compressed_x * cell_size
                    pixel_y = compressed_y * cell_size
                    
                    # Calculate direction from current to next
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    
                    # Calculate angle based on direction
                    if dx == 1 and dy == 0:     # Right
                        angle = 270
                    elif dx == -1 and dy == 0:   # Left
                        angle = 90
                    elif dx == 0 and dy == 1:    # Down
                        angle = 180
                    elif dx == 0 and dy == -1:   # Up
                        angle = 0
                    else:
                        continue  # Skip diagonal connections
                    
                    # Use pipe image for fluid items, conveyor for others
                    if is_fluid_item(item):
                        belt_image = pygame.transform.rotate(images['pipe'], angle)
                    else:
                        belt_image = pygame.transform.rotate(images['conveyor'], angle)
                    
                    # Draw the belt
                    screen.blit(belt_image, (pixel_x, pixel_y))
                    
                    # Draw small item icon on the belt if available
                    if item in item_images:
                        small_size = (item_images[item].get_width() // 3, item_images[item].get_height() // 3)
                        small_item = pygame.transform.scale(item_images[item], small_size)
                        corner_x = pixel_x + (cell_size - small_size[0]) // 2
                        corner_y = pixel_y + (cell_size - small_size[1]) // 2
                        screen.blit(small_item, (corner_x, corner_y))
                
                # Draw the last position in the path
                if path:
                    last_pos = path[-1]
                    last_tuple = tuple(last_pos)
                    
                    # Skip if position is part of an underground belt segment
                    if last_tuple in underground_positions:
                        continue
                    
                    # Skip if position is outside the grid or in a skipped row/column
                    orig_x, orig_y = last_pos
                    if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    compressed_x = x_map[orig_x]
                    compressed_y = y_map[orig_y]
                    
                    pixel_x = compressed_x * cell_size
                    pixel_y = compressed_y * cell_size
                    
                    # Calculate direction based on second-last to last position
                    if len(path) > 1:
                        second_last = path[-2]
                        dx = last_pos[0] - second_last[0]
                        dy = last_pos[1] - second_last[1]
                        
                        if dx == 1 and dy == 0:     # Right
                            angle = 270
                        elif dx == -1 and dy == 0:   # Left
                            angle = 90
                        elif dx == 0 and dy == 1:    # Down
                            angle = 180
                        elif dx == 0 and dy == -1:   # Up
                            angle = 0
                        else:
                            angle = 0  # Default
                            
                        # Use pipe image for fluid items, conveyor for others
                        if is_fluid_item(item):
                            belt_image = pygame.transform.rotate(images['pipe'], angle)
                        else:
                            belt_image = pygame.transform.rotate(images['conveyor'], angle)
                        
                        # Draw the belt
                        screen.blit(belt_image, (pixel_x, pixel_y))
                        
                        # Draw small item icon on the belt if available
                        if item in item_images:
                            small_size = (item_images[item].get_width() // 3, item_images[item].get_height() // 3)
                            small_item = pygame.transform.scale(item_images[item], small_size)
                            corner_x = pixel_x + (cell_size - small_size[0]) // 2
                            corner_y = pixel_y + (cell_size - small_size[1]) // 2
                            screen.blit(small_item, (corner_x, corner_y))
                
                # Draw underground belt segments
                for segment_id, segment in underground_segments.items():
                    start_pos = segment["start"]
                    end_pos = segment["end"]
                    
                    # Skip if positions are outside the grid or in skipped rows/columns
                    start_x, start_y = start_pos
                    end_x, end_y = end_pos
                    
                    # Skip if start position is invalid
                    if start_x >= len(x_map) or start_y >= len(y_map) or x_map[start_x] == -1 or y_map[start_y] == -1:
                        continue
                        
                    # Skip if end position is invalid
                    if end_x >= len(x_map) or end_y >= len(y_map) or x_map[end_x] == -1 or y_map[end_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    start_compressed_x = x_map[start_x]
                    start_compressed_y = y_map[start_y]
                    end_compressed_x = x_map[end_x]
                    end_compressed_y = y_map[end_y]
                    
                    start_pixel_x = start_compressed_x * cell_size
                    start_pixel_y = start_compressed_y * cell_size
                    end_pixel_x = end_compressed_x * cell_size
                    end_pixel_y = end_compressed_y * cell_size
                    
                    # Calculate direction
                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    
                    # Determine direction and rotation angles for entrance and exit
                    # Underground belts point in the direction of travel
                    if abs(dx) > abs(dy):
                        if dx > 0:  # Moving right (east)
                            start_angle = 270  # Entrance faces east
                            end_angle = 270    # Exit faces east
                            direction = "east"
                        else:  # Moving left (west)
                            start_angle = 90   # Entrance faces west
                            end_angle = 90     # Exit faces west
                            direction = "west"
                    else:
                        if dy > 0:  # Moving down (south)
                            start_angle = 180  # Entrance faces south
                            end_angle = 180    # Exit faces south
                            direction = "south"
                        else:  # Moving up (north)
                            start_angle = 0    # Entrance faces north
                            end_angle = 0      # Exit faces north
                            direction = "north"
                            
                    # Choose the appropriate images
                    if is_fluid_item(item):
                        # For fluids, use pipe-underground
                        underground_img = images['pipe-underground']
                    else:
                        # For solids, use underground belt
                        underground_img = images['underground']
                    
                    # Draw the underground belt entrance
                    entrance_img = pygame.transform.rotate(underground_img, start_angle)
                    screen.blit(entrance_img, (start_pixel_x, start_pixel_y))
                    
                    # Draw the underground belt exit
                    exit_img = pygame.transform.rotate(underground_img, end_angle)
                    screen.blit(exit_img, (end_pixel_x, end_pixel_y))
                    
                    # Draw item icons on underground belts
                    if item in item_images:
                        small_size = (item_images[item].get_width() // 3, item_images[item].get_height() // 3)
                        small_item = pygame.transform.scale(item_images[item], small_size)
                        
                        # Entrance icon
                        corner_x = start_pixel_x + (cell_size - small_size[0]) // 2
                        corner_y = start_pixel_y + (cell_size - small_size[1]) // 2
                        screen.blit(small_item, (corner_x, corner_y))
                        
                        # Exit icon
                        corner_x = end_pixel_x + (cell_size - small_size[0]) // 2
                        corner_y = end_pixel_y + (cell_size - small_size[1]) // 2
                        screen.blit(small_item, (corner_x, corner_y))
            # Draw entities on the compressed grid
        # Draw assemblers
        if "entities" in data:
            for assembler in data["entities"].get("assemblers", []):
                pos = assembler.get("position")
                item = assembler.get("item")
                machine_type = assembler.get("machine_type")
                
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                
                # Skip if position is outside the grid or in a skipped row/column
                orig_x, orig_y = pos[0], pos[1]
                if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                    continue
                
                # Get compressed coordinates
                compressed_x = x_map[orig_x]
                compressed_y = y_map[orig_y]
                
                pixel_x = compressed_x * cell_size
                pixel_y = compressed_y * cell_size
                
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
                
                screen.blit(machine_image, (pixel_x, pixel_y))
                
                # Draw item icon on top of assembler
                if item in item_images:
                    screen.blit(item_images[item], (pixel_x, pixel_y))
            
            # Draw inserters
            for inserter in data["entities"].get("inserters", []):
                if "position" in inserter:
                    pos = inserter.get("position")
                    if not isinstance(pos, list) or len(pos) < 2:
                        continue
                    
                    # Skip if position is outside the grid or in a skipped row/column
                    orig_x, orig_y = pos[0], pos[1]
                    if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    compressed_x = x_map[orig_x]
                    compressed_y = y_map[orig_y]
                    
                    pixel_x = compressed_x * cell_size
                    pixel_y = compressed_y * cell_size
                    
                    item = inserter.get("item")
                    
                    screen.blit(images['inserter'], (pixel_x, pixel_y))
                    if item in item_images:
                        original_image = item_images[item]
                        quarter_size = (original_image.get_width() // 2, original_image.get_height() // 2)
                        scaled_image = pygame.transform.scale(original_image, quarter_size)
                        
                        # Position in bottom-right corner of inserter
                        inserter_width = images['inserter'].get_width()
                        inserter_height = images['inserter'].get_height()
                        scaled_width = scaled_image.get_width()
                        scaled_height = scaled_image.get_height()
                        corner_x = pixel_x + inserter_width - scaled_width
                        corner_y = pixel_y + inserter_height - scaled_height
                        screen.blit(scaled_image, (corner_x, corner_y))
            
            # Draw belts
            for belt in data["entities"].get("belts", []):
                if "position" in belt:
                    pos = belt["position"]
                    if not isinstance(pos, list) or len(pos) < 2:
                        continue
                    
                    # Skip if position is outside the grid or in a skipped row/column
                    orig_x, orig_y = pos[0], pos[1]
                    if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    compressed_x = x_map[orig_x]
                    compressed_y = y_map[orig_y]
                    
                    pixel_x = compressed_x * cell_size
                    pixel_y = compressed_y * cell_size
                    
                    direction = belt.get("direction", "north")
                    item = belt.get("item")
                    
                    angle = get_angle_from_orientation(direction)
                    
                    if is_fluid_item(item):
                        # Use pipe image for fluid items
                        belt_image = pygame.transform.rotate(images['pipe'], angle)
                    else:
                        belt_image = pygame.transform.rotate(images['conveyor'], angle)
                    
                    screen.blit(belt_image, (pixel_x, pixel_y))
            
            # Draw underground belts
            for underground_belt in data["entities"].get("underground_belts", []):
                if "position" in underground_belt:
                    pos = underground_belt["position"]
                    if not isinstance(pos, list) or len(pos) < 2:
                        continue
                    
                    # Skip if position is outside the grid or in a skipped row/column
                    orig_x, orig_y = pos[0], pos[1]
                    if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    compressed_x = x_map[orig_x]
                    compressed_y = y_map[orig_y]
                    
                    pixel_x = compressed_x * cell_size
                    pixel_y = compressed_y * cell_size
                    
                    direction = underground_belt.get("direction")
                    belt_type = underground_belt.get("type", "entrance")
                    
                    # Get the correct rotation angle based on direction
                    angle = get_angle_from_orientation(direction)
                    belt_image = pygame.transform.rotate(images['underground'], angle)
                    
                    # Draw the underground belt
                    screen.blit(belt_image, (pixel_x, pixel_y))
                    
                    # Add visual indicator for entrance vs exit
                    if belt_type == "entrance":
                        # Draw a small green circle for entrance
                        pygame.draw.circle(screen, (0, 255, 0), 
                                         (pixel_x + cell_size//4, pixel_y + cell_size//4), 3)
                    else:  # exit
                        # Draw a small red circle for exit
                        pygame.draw.circle(screen, (255, 0, 0), 
                                         (pixel_x + cell_size//4, pixel_y + cell_size//4), 3)
            
            # Draw power poles
            for power_pole in data["entities"].get("power_poles", []):
                if "position" in power_pole:
                    pos = power_pole["position"]
                    if not isinstance(pos, list) or len(pos) < 2:
                        continue
                    
                    # Skip if position is outside the grid or in a skipped row/column
                    orig_x, orig_y = pos[0], pos[1]
                    if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    compressed_x = x_map[orig_x]
                    compressed_y = y_map[orig_y]
                    
                    pixel_x = compressed_x * cell_size
                    pixel_y = compressed_y * cell_size
                    
                    pole_type = power_pole.get("item")
                    
                    if pole_type in images:
                        screen.blit(images[pole_type], (pixel_x, pixel_y))
                    else:
                        logger.warning(f"Power pole image not found for {pole_type}")
            
            # Draw splitters (2x1 entities)
            for splitter in data["entities"].get("splitters", []):
                if "position" in splitter:
                    pos = splitter["position"]
                    if not isinstance(pos, list) or len(pos) < 2:
                        continue
                    
                    # Skip if position is outside the grid or in a skipped row/column
                    orig_x, orig_y = pos[0], pos[1]
                    if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                        continue
                    
                    # Get compressed coordinates
                    compressed_x = x_map[orig_x]
                    compressed_y = y_map[orig_y]
                    
                    pixel_x = compressed_x * cell_size
                    pixel_y = compressed_y * cell_size
                    
                    direction = splitter.get("direction", "north")
                    item = splitter.get("item")
                    
                    # Splitters are 2x1 entities - determine their size and orientation correctly
                    # In Factorio: north/south splitters are 1 wide, 2 tall; east/west splitters are 2 wide, 1 tall
                    if direction in ["north", "south"]:
                        # Vertical orientation: splitter is 1 wide, 2 tall
                        splitter_width = cell_size
                        splitter_height = 2 * cell_size
                        angle = 0  # Default orientation for vertical splitters
                    else:  # east or west
                        # Horizontal orientation: splitter is 2 wide, 1 tall
                        splitter_width = 2 * cell_size
                        splitter_height = cell_size
                        angle = 90  # Rotate 90 degrees for horizontal splitters
                    
                    # Create and scale the splitter image first
                    base_splitter = pygame.transform.scale(images['splitter'], (splitter_width, splitter_height))
                    
                    # Apply rotation if needed
                    if angle != 0:
                        rotated_splitter = pygame.transform.rotate(base_splitter, angle)
                    else:
                        rotated_splitter = base_splitter
                    
                    # Calculate the actual size after rotation
                    actual_width, actual_height = rotated_splitter.get_size()
                    
                    # Position the splitter at the given coordinates
                    splitter_x = pixel_x
                    splitter_y = pixel_y
                    
                    # Draw the splitter
                    screen.blit(rotated_splitter, (splitter_x, splitter_y))
                    
                    # Draw item icon on the splitter
                    if item in item_images:
                        # Scale down the item image
                        original_image = item_images[item]
                        quarter_size = (original_image.get_width() // 2, original_image.get_height() // 2)
                        scaled_image = pygame.transform.scale(original_image, quarter_size)
                        
                        # Position the item icon in the center of the splitter
                        icon_x = splitter_x + (actual_width - scaled_image.get_width()) // 2
                        icon_y = splitter_y + (actual_height - scaled_image.get_height()) // 2
                        screen.blit(scaled_image, (icon_x, icon_y))
        
        # Draw I/O points
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)

            
            
        if "io_points" in data:
            # Draw inputs
            inputs = data["io_points"].get("inputs")
            if inputs:
                for input_point in inputs:
                    if "position" in input_point:
                        pos = input_point["position"]
                        item = input_point.get("item")
                        
                        # Skip if position is outside the grid or in a skipped row/column
                        orig_x, orig_y = pos[0], pos[1]
                        if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                            continue
                        
                        # Get compressed coordinates
                        compressed_x = x_map[orig_x]
                        compressed_y = y_map[orig_y]
                        
                        pixel_x = compressed_x * cell_size
                        pixel_y = compressed_y * cell_size
                        
                        # Draw green background for input
                        rect = pygame.Rect(pixel_x, pixel_y, cell_size, cell_size)
                        pygame.draw.rect(screen, GREEN, rect)
                        
                        # Draw item icon
                        if item in item_images:
                            screen.blit(item_images[item], (pixel_x, pixel_y))
            
            # Draw outputs
            outputs = data["io_points"].get("outputs")
            if outputs:
                for output_point in outputs:
                    if "position" in output_point:
                        pos = output_point["position"]
                        item = output_point.get("item")
                        
                        # Skip if position is outside the grid or in a skipped row/column
                        orig_x, orig_y = pos[0], pos[1]
                        if orig_x >= len(x_map) or orig_y >= len(y_map) or x_map[orig_x] == -1 or y_map[orig_y] == -1:
                            continue
                        
                        # Get compressed coordinates
                        compressed_x = x_map[orig_x]
                        compressed_y = y_map[orig_y]
                        
                        pixel_x = compressed_x * cell_size
                        pixel_y = compressed_y * cell_size
                        
                        # Draw red background for output
                        rect = pygame.Rect(pixel_x, pixel_y, cell_size, cell_size)
                        pygame.draw.rect(screen, RED, rect)
                        
                        # Draw item icon
                        if item in item_images:
                            screen.blit(item_images[item], (pixel_x, pixel_y))
        
        

        
        
        # Create visible window for display
        window = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Factory Visualization: {os.path.basename(json_path)}")
        window.blit(screen, (0, 0))
        pygame.display.flip()
        
        # Save if requested
        if save_path:
            pygame.image.save(screen, save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        # Main loop to display the visualization
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.flip()
        
        pygame.quit()
        return True
        
    except Exception as e:
        traceback.print_exc()
        logger.info(f"Error visualizing JSON: {e}")
        return False



def load_factory_images(cell_size):
        """Load and scale factory element images."""
        images = {}
        config = load_config()
        
        # Load basic factory elements
        image_files = {
            'assembler': f'assets/{config["machines"]["default_assembler"]}.png',
            'inserter': f'assets/{config["inserters"]["input_type"]}.png',
            'conveyor': f"assets/{config['belts']['default_type']}.png",
            'underground': f"assets/{config['belts']['underground_type']}.png",
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
                else:
                    # Generic fallback for other entities
                    fallback = pygame.Surface((cell_size, cell_size))
                    fallback.fill((200, 200, 200))  # Light gray
                    pygame.draw.rect(fallback, (100, 100, 100), (0, 0, cell_size, cell_size), 1)
                    images[key] = fallback
        
        return images

def load_item_images(cell_size):
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


def load_config():
        """Load the configuration file"""
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
            return config
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return {
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


def load_json(recipe_file):
        with open(recipe_file, "r") as file:
                recipes = json.load(file)
                return recipes


def _get_belt_direction(orientation):

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
    
def get_angle_from_orientation(orientation):
    """Convert orientation to angle in degrees."""
    if orientation == "north":
        return 0    # Up direction in Pygame (0 degrees)
    elif orientation == "east":
        return 270  # Right direction (270 degrees clockwise)
    elif orientation == "south":
        return 180  # Down direction (180 degrees)
    elif orientation == "west":
        return 90   # Left direction (90 degrees)
    else:
        return 0    # Default to north/up


def is_fluid_item(item_id):
    
        items_data = load_json("recipes.json")
        
        item_lookup = {item["id"]: item for item in items_data}
        
        # Look up item in the recipe data
        item = item_lookup.get(item_id, {})
   
        # Check if the item type is "Liquid"
        return item.get("type", "") == "Liquid"
    
    
if __name__ == "__main__":
    
    #plot_csv_data("execution_times_big_factory.csv")
    main()


