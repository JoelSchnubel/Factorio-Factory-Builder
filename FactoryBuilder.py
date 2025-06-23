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
from draftsman.entity import Inserter, AssemblingMachine, TransportBelt, UndergroundBelt,ConstantCombinator
from draftsman.entity import Splitter as BlueprintSplitter
from logging_config import setup_logger
logger = setup_logger("FactoryBuilder")


# Define constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Color for input gates
BLUE = (0, 0, 255)  # Color for output gates
GREEN = (0, 255, 0)  # Color for blocks

class FactoryBuilder:
    
    def __init__(self,output_item,amount,max_assembler_per_blueprint,start_width,start_height,load_modules) -> None:
        
        self.output_item = output_item
        self.amount = amount
        self.max_assembler_per_blueprint = max_assembler_per_blueprint
        
        self.start_width = start_width
        self.start_height = start_height
        
        self.output_point = (0,0)
    
        self.z3_solver = None
        self.AStar = None
        
        self.block_data = {}

        self.items_data = self.load_json("recipes.json")
        
        self.final_x = None
        self.final_y = None
        self.final_blocks = None
        self.gate_connections = None
        self.inter_block_paths = None
  
        
        self.images = {}
        
        self.load_modules = load_modules
        
        self.external_io = None
        


    def load_json(self,recipe_file):
        with open(recipe_file, "r") as file:
                recipes = json.load(file)
                return {item["id"]: item for item in recipes}
        

    # gets a list of production_data and evals each for number of assemblers
    # if number of assemblers > than limit -> split the production data in half 
    def eval_split(self,production_data,input_items):
        num_factories = 1
        factorioProductionTree = FactorioProductionTree(grid_width=self.start_width,grid_height=self.start_height)
        
        amount = self.amount
        
        while self.count_assembler(production_data) > self.max_assembler_per_blueprint:
            
            amount = amount/2
            
            # split in half and check again 
            production_data  = factorioProductionTree.calculate_production(self.output_item , amount,input_items) 
            
            production_data = factorioProductionTree.set_capacities(production_data)
            num_factories +=1 


        
        return production_data,num_factories
    
    
    # count the number of assemblers in the production data
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
                            print(f"Selected items: {selected_items}")
                        y_pos += 50

        pygame.quit()
            
        print(f"Selected items: {selected_items}")
        
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
        print("Input Information:")
        for key, value in self.block_data[self.output_item]["tree"].input_information.items():
            if 'input' in value and 'output' in value:
                print(f"{key}: input={value['input']}, output={value['output']}")

        # Print output information (keys with input and output values)
        print("Output Information:")
        for key, value in self.block_data[self.output_item]["tree"].output_information.items():
            if 'input' in value and 'output' in value:
                print(f"{key}: input={value['input']}, output={value['output']}")

       
        for item in selected_items:
            
            print(f"building block for subitem {item}")

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
        print("solving factory")
        
        # Define external I/O points before creating the solver
        self.define_factory_io_points()
        
        print("block data:")
        print(self.block_data)
        
        self.z3_solver = FactoryZ3Solver(self.block_data, self.output_point)
        num_factories = 0
        for i, key in enumerate(self.block_data.keys()):
            num_factories += self.block_data[key]["num_factories"]
            
        print(f'total number of modules: {num_factories}')
        
        # Apply I/O constraints to solver
        self.apply_io_constraints_to_solver()
            
        self.z3_solver.build_constraints()
        
        # Store gate connections along with block positions
        self.final_blocks, self.final_x, self.final_y, self.gate_connections = self.z3_solver.solve()
        
        # Plan paths between connected gates
        if self.final_blocks:
            path = f"Factorys/factory_{self.output_item}_{self.amount}.json"
            factory_data,_ = self.create_json(output_json_path=path)
            inter_block_paths , _ = self.plan_inter_block_paths(factory_data)
            self.add_paths_to_json(path,inter_block_paths)
            
            blueprint_path= f"Blueprints/blueprint_{self.output_item}_{self.amount}.txt"
            create_blueprint_from_json(path, output_path=blueprint_path)
            
            factory_img_path= f"Factorys/factory_{self.output_item}_{self.amount}.png"
            self.visualize_factory(save_path=factory_img_path)
            
        
        print(f"Factory dimensions: {self.final_x} x {self.final_y}")
        print(f"Final blocks: {self.final_blocks}")
        print(f"Gate connections: {self.gate_connections}")


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
                "splitters": []
            },
            "io_points": {
                "inputs": [],
                "outputs": []
            },
            "inter_block_paths": [],
            "external_io": []
        }
        
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
                        
                            item, rel_x, rel_y = assembler[0], assembler[1], assembler[2]
                            # Convert coordinates
                            abs_x = block_x + rel_x
                            abs_y = block_y + rel_y 
                            
                            factory_data["entities"]["assemblers"].append({
                                "item": item,
                                "position": [abs_x, abs_y],
                                "block_id": block_id
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
                            # Parse the source position string "(10, 2)" -> [10, 2]
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

                # add I/O paths to the json file
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
                else:
                    logger.info("  - No input belt routes found")

                # Process output information paths similarly
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
                else:
                    logger.info("  - No output belt routes found")
                            
                
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
                
            except Exception as e:
                logger.error(f"Error processing module JSON for {block_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Add external I/O points if available
        if hasattr(self, 'external_io') and self.external_io:
            for input_gate in self.external_io.get("input_gates", []):
                factory_data["io_points"]["inputs"].append({
                    "item": input_gate["item"],
                    "position": list(input_gate["position"]),
                    "edge": input_gate["edge"],
                    "gate_id": input_gate["id"],
                    "external": True
                })
                logger.info(f"Added external input gate {input_gate['item']} at {input_gate['position']}")
            
            for output_gate in self.external_io.get("output_gates", []):
                factory_data["io_points"]["outputs"].append({
                    "item": output_gate["item"],
                    "position": list(output_gate["position"]),
                    "edge": output_gate["edge"],
                    "gate_id": output_gate["id"],
                    "external": True
                })
                logger.info(f"Added external output gate {output_gate['item']} at {output_gate['position']}")
        

        # If no path specified, use default
        if not output_json_path:
            output_json_path = f"Factorys/factory_{self.output_item}_{self.amount}.json"
        
        # Save to file
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
                item = gate['item']
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
                item = gate['item']
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
                print(f"Image not found for {base_name} at {image_path}")

 
    
    def get_num_subfactories(self):
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
                        print(f"Warning: Image not found for {item} at {image_path}")
            
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
                        print(f"Warning: Image not found for {item} at {image_path}")
                        
                        
    def visualize_factory(self, cell_size=20, save_path=None):
        """Visualize the factory layout with blocks and inter-block paths"""
        pygame.init()
        
        # Calculate window size with padding
        window_width = (self.final_x + 2) * cell_size
        window_height = (self.final_y + 2) * cell_size
        
        # Create Pygame window
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Factory Layout with Paths')
        
        # Load block images
        block_images = {}
        for block_id, block_info in self.final_blocks.items():
            # Extract the base item name from block_id
            parts = block_id.split("_")
            if len(parts) >= 3:
                block_type = parts[1]  # Extract the item name
            else:
                block_type = block_id
            
            # Try to find image in the block data
            if block_type in self.block_data and 'png' in self.block_data[block_type]:
                image_path = self.block_data[block_type]['png']
                if os.path.exists(image_path):
                    image = pygame.image.load(image_path)
                    # Scale image to block size
                    image = pygame.transform.scale(
                        image,
                        (block_info['width'] * cell_size, block_info['height'] * cell_size)
                    )
                    block_images[block_id] = image
        
        # Load item images for gates
        self.load_item_images()
        
        # Define colors for belt paths
        BELT_COLOR = (255, 165, 0)  # Orange
        UNDERGROUND_COLOR = (139, 69, 19)  # Brown
        
        # Clear the screen
        window.fill(WHITE)
        
        # Draw grid lines
        for x in range(0, (self.final_x + 1) * cell_size, cell_size):
            pygame.draw.line(window, (200, 200, 200), (x, 0), (x, window_height))
        for y in range(0, (self.final_y + 1) * cell_size, cell_size):
            pygame.draw.line(window, (200, 200, 200), (0, y), (window_width, y))        # Draw inter-block paths first so they appear behind blocks
        if self.inter_block_paths:
            logger.info(f"Found {len(self.inter_block_paths)} inter-block path items to visualize")
            for item_key, path_data_list in self.inter_block_paths.items():
                logger.info(f"Drawing paths for item: {item_key}, paths: {len(path_data_list)}")
                for path_data in path_data_list:
                    # Log what's in the path_data
                    logger.info(f"  Path data keys: {list(path_data.keys())}")
                    
                    # First check if there's a path key
                    if 'path' in path_data:
                        path = path_data['path']
                    # If not, try to use start and destination
                    elif 'start' in path_data and 'destination' in path_data:
                        path = [path_data['start'], path_data['destination']]
                        logger.info(f"  Using start/destination for path: {path}")
                    else:
                        path = []
                        logger.warning(f"  No valid path found for {item_key}")
                    
                    underground_segments = path_data.get('underground_segments', {})
                    
                    # Draw regular path segments
                    for i in range(len(path) - 1):
                        start_x, start_y = path[i]
                        end_x, end_y = path[i + 1]
                        # Skip if part of underground segment
                        is_underground = False
                        for segment_id, segment in underground_segments.items():
                            # Some segments have 'path' others have 'start'/'end'
                            if 'path' in segment:
                                segment_path = segment['path']
                                if (start_x, start_y) in segment_path and (end_x, end_y) in segment_path:
                                    idx1 = segment_path.index((start_x, start_y))
                                    idx2 = segment_path.index((end_x, end_y))
                                    if abs(idx1 - idx2) == 1:  # Adjacent in segment
                                        is_underground = True
                                        break
                            elif 'start' in segment and 'end' in segment:
                                # Check if this segment contains our points
                                start_seg = segment['start']
                                end_seg = segment['end']
                                if ((start_x, start_y) == tuple(start_seg) and (end_x, end_y) == tuple(end_seg)) or \
                                   ((start_x, start_y) == tuple(end_seg) and (end_x, end_y) == tuple(start_seg)):
                                    is_underground = True
                                    break
                        
                        if not is_underground:
                            start_pos = (start_x * cell_size + cell_size // 2, start_y * cell_size + cell_size // 2)
                            end_pos = (end_x * cell_size + cell_size // 2, end_y * cell_size + cell_size // 2)
                            pygame.draw.line(window, BELT_COLOR, start_pos, end_pos, 3)
                      # Draw underground segments
                    for segment_id, segment in underground_segments.items():
                        # Some formats store start/end directly, others use a path
                        if 'start' in segment and 'end' in segment:
                            start = segment['start']
                            end = segment['end']
                        elif 'path' in segment and len(segment['path']) >= 2:
                            start = segment['path'][0]
                            end = segment['path'][-1]
                        else:
                            logger.warning(f"Skipping segment with invalid format: {segment}")
                            continue
                        
                        # Draw underground entry
                        entry_rect = pygame.Rect(
                            start[0] * cell_size + cell_size // 4,
                            start[1] * cell_size + cell_size // 4,
                            cell_size // 2,
                            cell_size // 2
                        )
                        pygame.draw.rect(window, UNDERGROUND_COLOR, entry_rect)
                        
                        # Draw underground exit
                        exit_rect = pygame.Rect(
                            end[0] * cell_size + cell_size // 4,
                            end[1] * cell_size + cell_size // 4,
                            cell_size // 2,
                            cell_size // 2
                        )
                        pygame.draw.rect(window, UNDERGROUND_COLOR, exit_rect)
                        
                        # Draw dotted line connecting entry to exit
                        start_pos = (start[0] * cell_size + cell_size // 2, start[1] * cell_size + cell_size // 2)
                        end_pos = (end[0] * cell_size + cell_size // 2, end[1] * cell_size + cell_size // 2)
                        
                        # Draw dashed line
                        dash_length = 5
                        dash_gap = 5
                        dx = end_pos[0] - start_pos[0]
                        dy = end_pos[1] - start_pos[1]
                        dist = max(1, abs(dx) + abs(dy))  # Prevent division by zero
                        dx, dy = dx/dist, dy/dist
                        
                        # Draw dashed line
                        pos = start_pos
                        dash_on = True
                        distance = 0
                        while distance < dist:
                            end = (pos[0] + dash_length * dx, pos[1] + dash_length * dy)
                            if dash_on:
                                pygame.draw.line(window, UNDERGROUND_COLOR, pos, end, 2)
                            pos = end
                            distance += dash_length
                            dash_on = not dash_on
        
        # Draw blocks
        for block_id, block_info in self.final_blocks.items():
            block_x = block_info['x']
            block_y = block_info['y']
            block_width = block_info['width']
            block_height = block_info['height']
            
            # Draw block with image if available
            if block_id in block_images:
                window.blit(block_images[block_id], (block_x * cell_size, block_y * cell_size))
            else:
                # Draw block as rectangle
                block_rect = pygame.Rect(
                    block_x * cell_size,
                    block_y * cell_size,
                    block_width * cell_size,
                    block_height * cell_size
                )
                pygame.draw.rect(window, GREEN, block_rect)
                pygame.draw.rect(window, BLACK, block_rect, 2)  # Border
                
                # Draw block ID text
                font = pygame.font.Font(None, 24)
                parts = block_id.split("_")
                if len(parts) >= 3:
                    display_text = parts[1]  # Extract item name
                else:
                    display_text = block_id
                text = font.render(display_text, True, BLACK)
                text_rect = text.get_rect(center=(
                    block_x * cell_size + (block_width * cell_size) // 2,
                    block_y * cell_size + (block_height * cell_size) // 2
                ))
                window.blit(text, text_rect)
            
            # Draw gates
            for gate in block_info['input_points']:
                gate_x = block_x + gate['x']
                gate_y = block_y + gate['y']
                
                gate_rect = pygame.Rect(
                    gate_x * cell_size,
                    gate_y * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(window, RED, gate_rect)
                
                # Draw item image if available
                if gate['item'] in self.item_images:
                    scaled_image = pygame.transform.scale(
                        self.item_images[gate['item']], 
                        (cell_size, cell_size)
                    )
                    window.blit(scaled_image, (gate_x * cell_size, gate_y * cell_size))
            
            for gate in block_info['output_points']:
                gate_x = block_x + gate['x']
                gate_y = block_y + gate['y']
                
                gate_rect = pygame.Rect(
                    gate_x * cell_size,
                    gate_y * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(window, BLUE, gate_rect)
                
                # Draw item image if available
                if gate['item'] in self.item_images:
                    scaled_image = pygame.transform.scale(
                        self.item_images[gate['item']], 
                        (cell_size, cell_size)
                    )
                    window.blit(scaled_image, (gate_x * cell_size, gate_y * cell_size))
        
        # Update the display
        pygame.display.flip()
        
        # Save the image if requested
        if save_path:
            pygame.image.save(window, save_path)
            logger.info(f"Factory visualization saved to {save_path}")
          # Wait for user to close the window or run for a limited time
        waiting = True
        clock = pygame.time.Clock()
        start_time = time.time()
        max_display_time = 3.0  # Auto-close after 3 seconds if not testing
        
        if save_path:
            # Save screenshot if a save path is provided
            pygame.image.save(window, save_path)
            logger.info(f"Factory visualization saved to: {save_path}")
        
        while waiting:
            clock.tick(60)  # 60 FPS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
            
            # Auto-close after max_display_time seconds
            if time.time() - start_time > max_display_time:
                waiting = False
        
        pygame.quit()

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
                edge=input_gate["edge"]
            )
            
            logger.info(f"Added fixed output gate {output_gate['id']} at position {output_gate['position']}")

    def plan_inter_block_paths(self, factory_data):
        """
        Plan paths between connected gates using the MultiAgentPathfinder and detailed factory data.
        
        Args:
            factory_data (dict): The factory JSON data containing entity positions
            
        Returns:
            tuple: (paths, inserters) - dictionaries of paths and inserters
        """
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
            # Mark assemblers (3x3 entities)
            for assembler in factory_data["entities"].get("assemblers", []):
                pos_x, pos_y = assembler["position"]
                # Assemblers are 3x3
                for dy in range(3):
                    for dx in range(3):
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
        
        # Create a visualization of the obstacle map for debugging
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(obstacle_map, cmap='binary', interpolation='none')
            plt.title("Factory Layout - Detailed Obstacle Map")
            plt.colorbar(label="Obstacle (1) vs Free (0)")
            plt.grid(True, alpha=0.3)
            plt.savefig("factory_detailed_obstacle_map.png")
            plt.close()
            logger.info("Saved detailed obstacle map visualization")
        except Exception as e:
            logger.warning(f"Could not create obstacle map visualization: {e}")
        
        # Prepare connection points for the pathfinder
        connection_points = {}
        
        # Process all gate connections
        for i, connection in enumerate(self.gate_connections):
            
            logger.debug(f"Processing connection {i}: {connection}")
            
            # Ensure we have a valid connection tuple
            if isinstance(connection, tuple) and len(connection) == 2:
                source_gate, target_gate = connection
                
                try:
                    # Handle both dictionary and Gate object formats
                    if hasattr(source_gate, 'x') and hasattr(source_gate, 'y'):  # Gate object
                        # Extract from Z3 solver model
                        source_x = self.final_blocks[source_gate.id.split("_output")[0]]["x"] + source_gate.relative_x
                        source_y = self.final_blocks[source_gate.id.split("_output")[0]]["y"] + source_gate.relative_y
                    
                        item = source_gate.item
                        source_id = source_gate.id
                    elif isinstance(source_gate, dict):  # Dictionary
                        source_x = int(source_gate['x'])
                        source_y = int(source_gate['y'])
                        item = source_gate['item']
                        source_id = source_gate.get('gate_id', f'unknown_source_{i}')
                    else:
                        logger.error(f"Source gate has unsupported type: {type(source_gate)}")
                        continue
                    
                    # Handle both dictionary and Gate object formats for target
                    if hasattr(target_gate, 'x') and hasattr(target_gate, 'y'):  # Gate object
                        # Extract from Z3 solver model
                        target_x = self.final_blocks[target_gate.id.split("_input")[0]]["x"] + target_gate.relative_x
                        target_y = self.final_blocks[target_gate.id.split("_input")[0]]["y"] + target_gate.relative_y
                        target_id = target_gate.id
                    elif isinstance(target_gate, dict):  # Dictionary
                        target_x = int(target_gate['x'])
                        target_y = int(target_gate['y'])
                        target_id = target_gate.get('gate_id', f'unknown_target_{i}')
                    else:
                        logger.error(f"Target gate has unsupported type: {type(target_gate)}")
                        continue
                    
                    # Create connection ID
                    connection_id = f"{item}_{i}"
                    
                    # Clear gate positions in the obstacle map for pathfinding
                    if 0 <= source_x < grid_width and 0 <= source_y < grid_height:
                        obstacle_map[source_y][source_x] = 0
                    
                    if 0 <= target_x < grid_width and 0 <= target_y < grid_height:
                        obstacle_map[target_y][target_x] = 0
                    
                    # Create a path request with start/end points
                    connection_points[connection_id] = {
                        'item': item,
                        'start_points': [(source_x, source_y)],
                        'destination': [(target_x, target_y)],
                        'source_id': source_id,
                        'target_id': target_id
                    }
                    
                    logger.debug(f"Added connection {connection_id}: {item} from {source_id} at ({source_x}, {source_y}) " +
                                f"to {target_id} at ({target_x}, {target_y})")
                    
                except (TypeError, KeyError, AttributeError) as e:
                    logger.error(f"Error processing connection {i}: {e}")
                    continue
            else:
                logger.warning(f"Invalid connection format at index {i}: {connection}")
        
        # Process external I/O points
        if "io_points" in factory_data:
            # Clear I/O points in the obstacle map
            for io_type in ["inputs", "outputs"]:
                for point in factory_data["io_points"].get(io_type, []):
                    pos_x, pos_y = point["position"]
                    if 0 <= pos_x < grid_width and 0 <= pos_y < grid_height:
                        obstacle_map[pos_y][pos_x] = 0
        
        if not connection_points:
            logger.warning("No valid connections to route!")
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
            logger.info(f"Found paths for {len(paths)} connections out of {len(connection_points)} requested")
            
            # Save the paths for later use
            self.inter_block_paths = paths
            self.inter_block_inserters = inserters
            
            # Generate path visualizations
            try:
                pathfinder.visualize_grid(filename="factory_detailed_obstacle_map_with_paths.png")
                pathfinder.visualize_paths(filename_template="inter_block_path_detailed_{}.png")
                logger.info("Path visualizations saved to disk")
            except Exception as e:
                logger.error(f"Failed to visualize paths: {e}")
            
            return paths, inserters
            
        except Exception as e:
            logger.error(f"Error in pathfinding: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}

def manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        
def main():
    
    output_item = "electronic-circuit"
    amount = 10
    max_assembler_per_blueprint = 5
    
    start_width = 15
    start_height = 15

    
    builder = FactoryBuilder(output_item,amount,max_assembler_per_blueprint,start_width,start_height,load_modules=True)
    
    #num_factories, production_data = builder.eval_split()
    #print(f"Number of factories required: {num_factories}")
    
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
    plt.title("Electronic Circuit - Solve Time (Boxplot)")
    plt.xlabel("Number of Modules")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)

    # Save the boxplot
    box_plot_path = os.path.join(plot_dir, "electronic_circuit_solve_box_plot.png")
    plt.savefig(box_plot_path)
    plt.close()  # Close the plot to prevent overlap with other subplots

    print(f"Boxplot saved at: {box_plot_path}")
      
      
      
      
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
        
        print("1. Placing assembling machines...")
        # Place assembling machines from the JSON data
        if "entities" in factory_data and "assemblers" in factory_data["entities"]:
            for assembler_data in factory_data["entities"]["assemblers"]:
                item = assembler_data["item"]
                position = assembler_data["position"]
                
                # Assemblers are 3x3 and centered on their position in Factorio
                # In JSON, position is top-left corner, we need to offset by (1,1) for center
                center_x = position[0] + 1
                center_y = position[1] + 1
                
                print(f"  - Placing assembler for {item} at ({center_x},{center_y})")
                
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
        
        print("2. Placing inserters...")
        # Place inserters from the JSON data
        if "entities" in factory_data and "inserters" in factory_data["entities"]:
            for inserter_data in factory_data["entities"]["inserters"]:
                item = inserter_data["item"]
                position = tuple(inserter_data["position"])
                direction = inserter_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    print(f"  - Skipping inserter at {position} due to overlap")
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
                print(f"  - Placing inserter for {item} at {position} facing {blueprint_direction}")
                inserter = Inserter(
                    name="inserter",
                    position=position,
                    direction=blueprint_direction
                )
                blueprint.entities.append(inserter)
                occupied_positions.add(position)
        
        
        
        print("3. Placing underground belts...")
        # Place underground belts from the JSON data
        if "entities" in factory_data and "underground_belts" in factory_data["entities"]:
            for ug_data in factory_data["entities"]["underground_belts"]:
                item = ug_data["item"]
                position = tuple(ug_data["position"])
                belt_type = ug_data["type"]  # "entrance" or "exit"
                direction = ug_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    print(f"  - Skipping underground belt at {position} due to overlap")
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
                print(f"  - Placing underground belt for {item} at {position} facing {blueprint_direction}, type: {ug_type}")
                ug_belt = UndergroundBelt(
                    name="underground-belt",
                    position=position,
                    direction=blueprint_direction,
                    type=ug_type
                )
                blueprint.entities.append(ug_belt)
                occupied_positions.add(position)
        
        print("4. Placing splitters...")
        # Place splitters from the JSON data
        if "entities" in factory_data and "splitters" in factory_data["entities"]:
            for splitter_data in factory_data["entities"]["splitters"]:
                item = splitter_data["item"]
                position = tuple(splitter_data["position"])
                direction = splitter_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    print(f"  - Skipping splitter at {position} due to overlap")
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
                print(f"  - Placing splitter for {item} at {position} facing {blueprint_direction}")
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
        
        
                
        print("5. Placing transport belts...")
        # Place regular belts from the JSON data
        if "entities" in factory_data and "belts" in factory_data["entities"]:
            for belt_data in factory_data["entities"]["belts"]:
                item = belt_data["item"]
                position = tuple(belt_data["position"])
                direction = belt_data.get("direction", "north")  # Default to north if not specified
                
                # Skip if position is occupied
                if position in occupied_positions:
                    print(f"  - Skipping belt at {position} due to overlap")
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
                print(f"  - Placing transport belt for {item} at {position} facing {blueprint_direction}")
                belt = TransportBelt(
                    name="transport-belt",
                    position=position,
                    direction=blueprint_direction
                )
                blueprint.entities.append(belt)
                occupied_positions.add(position)
        
        print("6. Placing external I/O points as constant combinators...")
        if "io_points" in factory_data and "inputs" in factory_data["io_points"]:
            for io_data in factory_data["io_points"]["inputs"]:
                if io_data["external"]:
                    item = io_data["item"]
                    position = tuple(io_data["position"])
                    
                    # Skip if position is occupied
                    if position in occupied_positions:
                        print(f"  - Skipping input point at {position} due to overlap")
                        continue
                
                    print(f"  - Placing input point for {item} at {position}")
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
                        print(f"  - Skipping output point at {position} due to overlap")
                        continue
    
                    print(f"  - Placing output point for {item} at {position}")
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
        
        
        # Export the blueprint to a file
        print(f"7. Exporting blueprint to {output_path}...")
        with open(output_path, "w") as f:
            f.write(blueprint.to_string())
            
        logger.info(f"Blueprint successfully exported to {output_path}")
        return True
 
    except Exception as e:
        print(f"Error creating blueprint: {e}")
        import traceback
        traceback.print_exc()
        return False  
    


def visualize_json(json_path, cell_size=20, save_path=None):
    """
    Visualize a factory JSON file using Pygame
    
    Args:
        json_path (str): Path to the factory JSON file
        cell_size (int): Size of each grid cell in pixels
        save_path (str, optional): Path to save the visualization image
    """
    import json
    import os
    import pygame
    
    # Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLOCK_COLOR = (200, 230, 200)  # Light green for blocks
    BLOCK_BORDER = (0, 100, 0)     # Dark green for block borders
    ASSEMBLER_COLOR = (150, 150, 200)  # Light blue for assemblers
    BELT_COLOR = (255, 165, 0)     # Orange for belts
    INSERTER_COLOR = (255, 100, 100)  # Pink for inserters
    UNDERGROUND_IN_COLOR = (139, 69, 19)  # Brown for underground belt entrances
    UNDERGROUND_OUT_COLOR = (160, 82, 45)  # Sienna for underground belt exits
    SPLITTER_COLOR = (255, 215, 0)  # Gold for splitters
    INPUT_COLOR = (200, 0, 0)     # Red for input points
    OUTPUT_COLOR = (0, 0, 200)    # Blue for output points
    
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Initialize Pygame
        pygame.init()
        
        # Get factory dimensions
        width = data["factory_dimensions"]["width"]
        height = data["factory_dimensions"]["height"]
        
        # Create the display surface
        screen_width = width * cell_size + 200  # Extra space for legend
        screen_height = height * cell_size
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Factory Visualization: {os.path.basename(json_path)}")
        
        # Fill the background
        screen.fill(WHITE)
        
        # Draw grid
        for x in range(0, width * cell_size, cell_size):
            pygame.draw.line(screen, (220, 220, 220), (x, 0), (x, height * cell_size))
        for y in range(0, height * cell_size, cell_size):
            pygame.draw.line(screen, (220, 220, 220), (0, y), (width * cell_size, y))
        
        # Draw blocks
        for block_id, block in data["blocks"].items():
            x = block["position"][0] * cell_size
            y = block["position"][1] * cell_size
            w = block["dimensions"][0] * cell_size
            h = block["dimensions"][1] * cell_size
            
            # Draw block background
            pygame.draw.rect(screen, BLOCK_COLOR, (x, y, w, h))
            pygame.draw.rect(screen, BLOCK_BORDER, (x, y, w, h), 2)
            
            # Draw block label
            font = pygame.font.SysFont(None, 24)
            text = font.render(block["type"], True, BLACK)
            text_rect = text.get_rect(center=(x + w/2, y + h/2))
            screen.blit(text, text_rect)
        
        # Draw entities
        if "entities" in data:
            # Draw assemblers
            for assembler in data["entities"].get("assemblers", []):
                x = assembler["position"][0] * cell_size
                y = assembler["position"][1] * cell_size
                pygame.draw.rect(screen, ASSEMBLER_COLOR, (x, y, cell_size*3, cell_size*3))
                pygame.draw.rect(screen, BLACK, (x, y, cell_size*3, cell_size*3), 1)
                
                # Label with item
                small_font = pygame.font.SysFont(None, 18)
                text = small_font.render(assembler["item"], True, BLACK)
                screen.blit(text, (x + 5, y + 5))
            
            # Draw belts
            for belt in data["entities"].get("belts", []):
                x = belt["position"][0] * cell_size
                y = belt["position"][1] * cell_size
                
                direction = belt.get("direction", "north")
                arrow_points = []
                
                # Calculate arrow points based on direction
                if direction == "north" or direction == "up":
                    arrow_points = [
                        (x + cell_size/2, y + cell_size/4),
                        (x + cell_size/4, y + 3*cell_size/4),
                        (x + 3*cell_size/4, y + 3*cell_size/4)
                    ]
                elif direction == "east" or direction == "right":
                    arrow_points = [
                        (x + 3*cell_size/4, y + cell_size/2),
                        (x + cell_size/4, y + cell_size/4),
                        (x + cell_size/4, y + 3*cell_size/4)
                    ]
                elif direction == "south" or direction == "down":
                    arrow_points = [
                        (x + cell_size/2, y + 3*cell_size/4),
                        (x + cell_size/4, y + cell_size/4),
                        (x + 3*cell_size/4, y + cell_size/4)
                    ]
                elif direction == "west" or direction == "left":
                    arrow_points = [
                        (x + cell_size/4, y + cell_size/2),
                        (x + 3*cell_size/4, y + cell_size/4),
                        (x + 3*cell_size/4, y + 3*cell_size/4)
                    ]
                
                # Draw belt background and arrow
                pygame.draw.rect(screen, BELT_COLOR, (x, y, cell_size, cell_size))
                pygame.draw.polygon(screen, BLACK, arrow_points)
            
            # Draw inserters
            for inserter in data["entities"].get("inserters", []):
                x = inserter["position"][0] * cell_size
                y = inserter["position"][1] * cell_size
                direction = inserter.get("direction", "north")
                
                # Draw inserter base
                pygame.draw.rect(screen, INSERTER_COLOR, (x, y, cell_size, cell_size))
                
                # Draw arrow indicating direction
                if direction == "north" or direction == "up":
                    pygame.draw.line(screen, BLACK, (x + cell_size/2, y + 3*cell_size/4), 
                                     (x + cell_size/2, y + cell_size/4), 2)
                    pygame.draw.polygon(screen, BLACK, [(x + cell_size/2, y + cell_size/4), 
                                                      (x + cell_size/3, y + cell_size/2), 
                                                      (x + 2*cell_size/3, y + cell_size/2)])
                elif direction == "east" or direction == "right":
                    pygame.draw.line(screen, BLACK, (x + cell_size/4, y + cell_size/2), 
                                     (x + 3*cell_size/4, y + cell_size/2), 2)
                    pygame.draw.polygon(screen, BLACK, [(x + 3*cell_size/4, y + cell_size/2), 
                                                      (x + cell_size/2, y + cell_size/3), 
                                                      (x + cell_size/2, y + 2*cell_size/3)])
                elif direction == "south" or direction == "down":
                    pygame.draw.line(screen, BLACK, (x + cell_size/2, y + cell_size/4), 
                                     (x + cell_size/2, y + 3*cell_size/4), 2)
                    pygame.draw.polygon(screen, BLACK, [(x + cell_size/2, y + 3*cell_size/4), 
                                                      (x + cell_size/3, y + cell_size/2), 
                                                      (x + 2*cell_size/3, y + cell_size/2)])
                elif direction == "west" or direction == "left":
                    pygame.draw.line(screen, BLACK, (x + 3*cell_size/4, y + cell_size/2), 
                                     (x + cell_size/4, y + cell_size/2), 2)
                    pygame.draw.polygon(screen, BLACK, [(x + cell_size/4, y + cell_size/2), 
                                                      (x + cell_size/2, y + cell_size/3), 
                                                      (x + cell_size/2, y + 2*cell_size/3)])
            
            # Draw underground belts
            for ug in data["entities"].get("underground_belts", []):
                x = ug["position"][0] * cell_size
                y = ug["position"][1] * cell_size
                belt_type = ug["type"]
                color = UNDERGROUND_IN_COLOR if belt_type == "entrance" else UNDERGROUND_OUT_COLOR
                
                # Draw underground belt
                pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))
                
                # Draw symbol based on type
                if belt_type == "entrance":
                    pygame.draw.rect(screen, BLACK, (x + cell_size/4, y + cell_size/4, cell_size/2, cell_size/2), 1)
                    pygame.draw.line(screen, BLACK, (x + cell_size/4, y + cell_size/2), 
                                     (x + 3*cell_size/4, y + cell_size/2), 2)
                else:  # exit
                    pygame.draw.rect(screen, BLACK, (x + cell_size/4, y + cell_size/4, cell_size/2, cell_size/2), 1)
                    pygame.draw.circle(screen, BLACK, (x + cell_size/2, y + cell_size/2), cell_size/6)
            
            # Draw splitters
            for splitter in data["entities"].get("splitters", []):
                x = splitter["position"][0] * cell_size
                y = splitter["position"][1] * cell_size
                direction = splitter.get("direction", "north")
                
                # Splitters are 2x1 or 1x2 based on orientation
                if direction in ["north", "south", "up", "down"]:
                    # Horizontal splitter (takes up two horizontal tiles)
                    pygame.draw.rect(screen, SPLITTER_COLOR, (x, y, cell_size*2, cell_size))
                    pygame.draw.rect(screen, BLACK, (x, y, cell_size*2, cell_size), 1)
                    # Draw splitter divider line
                    pygame.draw.line(screen, BLACK, (x + cell_size, y), (x + cell_size, y + cell_size), 1)
                else:
                    # Vertical splitter (takes up two vertical tiles)
                    pygame.draw.rect(screen, SPLITTER_COLOR, (x, y, cell_size, cell_size*2))
                    pygame.draw.rect(screen, BLACK, (x, y, cell_size, cell_size*2), 1)
                    # Draw splitter divider line
                    pygame.draw.line(screen, BLACK, (x, y + cell_size), (x + cell_size, y + cell_size), 1)
        
        # Draw I/O points
        if "io_points" in data:
            # Draw inputs
            for input_point in data["io_points"].get("inputs", []):
                x = input_point["position"][0] * cell_size
                y = input_point["position"][1] * cell_size
                pygame.draw.rect(screen, INPUT_COLOR, (x, y, cell_size, cell_size))
                
                # Draw small item label
                tiny_font = pygame.font.SysFont(None, 16)
                text = tiny_font.render(input_point["item"], True, WHITE)
                screen.blit(text, (x + 2, y + 2))
            
            # Draw outputs
            for output_point in data["io_points"].get("outputs", []):
                x = output_point["position"][0] * cell_size
                y = output_point["position"][1] * cell_size
                pygame.draw.rect(screen, OUTPUT_COLOR, (x, y, cell_size, cell_size))
                
                # Draw small item label
                tiny_font = pygame.font.SysFont(None, 16)
                text = tiny_font.render(output_point["item"], True, WHITE)
                screen.blit(text, (x + 2, y + 2))
        
        # Draw connections between I/O points
        if "connections" in data:
            for conn in data["connections"]:
                source = conn["source"]["position"]
                target = conn["target"]["position"]
                
                # Calculate pixel positions (center of cells)
                start_x = source[0] * cell_size + cell_size/2
                start_y = source[1] * cell_size + cell_size/2
                end_x = target[0] * cell_size + cell_size/2
                end_y = target[1] * cell_size + cell_size/2
                
                # Draw dotted line for connection
                dash_length = 5
                space_length = 5
                dx = end_x - start_x
                dy = end_y - start_y
                steps = max(1, int((abs(dx) + abs(dy)) / (dash_length + space_length)))
                
                for i in range(steps):
                    t1 = i / steps
                    t2 = min(1, (i + 0.5) / steps)
                    
                    x1 = start_x + dx * t1
                    y1 = start_y + dy * t1
                    x2 = start_x + dx * t2
                    y2 = start_y + dy * t2
                    
                    pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 1)
        
        # Draw legend
        legend_x = width * cell_size + 20
        legend_y = 20
        legend_font = pygame.font.SysFont(None, 24)
        
        # Legend title
        title = legend_font.render("Legend", True, BLACK)
        screen.blit(title, (legend_x, legend_y))
        legend_y += 30
        
        # Block
        pygame.draw.rect(screen, BLOCK_COLOR, (legend_x, legend_y, cell_size*2, cell_size))
        pygame.draw.rect(screen, BLOCK_BORDER, (legend_x, legend_y, cell_size*2, cell_size), 2)
        text = legend_font.render("Block", True, BLACK)
        screen.blit(text, (legend_x + cell_size*2 + 10, legend_y))
        legend_y += 30
        
        # Assembler
        pygame.draw.rect(screen, ASSEMBLER_COLOR, (legend_x, legend_y, cell_size*2, cell_size))
        text = legend_font.render("Assembler", True, BLACK)
        screen.blit(text, (legend_x + cell_size*2 + 10, legend_y))
        legend_y += 30
        
        # Belt
        pygame.draw.rect(screen, BELT_COLOR, (legend_x, legend_y, cell_size*2, cell_size))
        text = legend_font.render("Belt", True, BLACK)
        screen.blit(text, (legend_x + cell_size*2 + 10, legend_y))
        legend_y += 30
        
        # Inserter
        pygame.draw.rect(screen, INSERTER_COLOR, (legend_x, legend_y, cell_size*2, cell_size))
        text = legend_font.render("Inserter", True, BLACK)
        screen.blit(text, (legend_x + cell_size*2 + 10, legend_y))
        legend_y += 30
        
        # Underground Belt
        pygame.draw.rect(screen, UNDERGROUND_IN_COLOR, (legend_x, legend_y, cell_size, cell_size))
        pygame.draw.rect(screen, UNDERGROUND_OUT_COLOR, (legend_x + cell_size, legend_y, cell_size, cell_size))
        text = legend_font.render("Underground", True, BLACK)
        screen.blit(text, (legend_x + cell_size*2 + 10, legend_y))
        legend_y += 30
        
        # Splitter
        pygame.draw.rect(screen, SPLITTER_COLOR, (legend_x, legend_y, cell_size*2, cell_size))
        text = legend_font.render("Splitter", True, BLACK)
        screen.blit(text, (legend_x + cell_size*2 + 10, legend_y))
        legend_y += 30
        
        # I/O Points
        pygame.draw.rect(screen, INPUT_COLOR, (legend_x, legend_y, cell_size, cell_size))
        pygame.draw.rect(screen, OUTPUT_COLOR, (legend_x + cell_size, legend_y, cell_size, cell_size))
        text = legend_font.render("I/O Points", True, BLACK)
        screen.blit(text, (legend_x + cell_size*2 + 10, legend_y))
        
        # Save if requested
        if save_path:
            pygame.image.save(screen, save_path)
            print(f"Visualization saved to {save_path}")
        
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
        import traceback
        traceback.print_exc()
        print(f"Error visualizing JSON: {e}")
        return False


if __name__ == "__main__":
    #plot_csv_data("execution_times_big_factory.csv")
    main()
    
    json_file_path = "Factorys/factory_electronic-circuit_200.json"
    output_file_path = "Blueprints/factory_electronic-circuit_200.txt"
    #save_image_path = "Factorys/Factory_visualization.png"
    #create_blueprint_from_json(json_file_path, output_file_path)
    #visualize_json(json_file_path, cell_size=30, save_path=save_image_patsh)

