#! .venv\Scripts\python.exe

from FactorioProductionTree import FactorioProductionTree
from FactoryZ3Solver import FactoryZ3Solver
from MultiAgentPathfinder import MultiAgentPathfinder
import pygame
import json
import os
import time
import logging
import csv
from math import ceil
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Configure logging to handle Unicode characters
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for detailed information, change to INFO for less verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("factoryBuilder_log.log", mode='w'),  # Specify the log file name
    ]
)

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
            self.inter_block_paths, inserters = self.plan_inter_block_paths()
            self.create_json()
            
        
        print(f"Factory dimensions: {self.final_x} x {self.final_y}")
        print(f"Final blocks: {self.final_blocks}")
        print(f"Gate connections: {self.gate_connections}")


    def create_json(self, path=None):
        """
        Create a comprehensive factory JSON file by extracting and combining data from all module JSON files.
        Adjusts positions to account for:
        1. Final blocks are positioned by upper left corner
        2. Module JSONs use (0,0) as lower left corner
        
        Args:
            path (str): Path where to save the JSON file
        """
        if not self.final_blocks:
            logging.error("Cannot create JSON: No final blocks available")
            return
        
        # Initialize the factory data structure
        factory_data = {
            "factory_dimensions": {
                "width": self.final_x + 1,
                "height": self.final_y + 1
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
            "connections": [],
            "inter_block_paths": []
        }
        
        # Process each block
        for block_id, block_info in self.final_blocks.items():
            block_x = block_info['x']
            block_y = block_info['y']
            block_width = block_info['width']
            block_height = block_info['height']
            block_type = block_info.get('block_type')
            
            if not block_type:
                # Extract type from block_id if not explicitly provided
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
            
            # If still no JSON path, try to find the file based on naming patterns
            if not module_json_path:
                possible_paths = [
                    f"Modules/{block_type}_module.json",
                    f"Modules/{block_type}_120_[]_module.json",
                    f"Modules/{block_type}_{self.amount}_[]_module.json"
                ]
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        module_json_path = possible_path
                        break
            
            # Skip this block if we can't find its module JSON
            if not module_json_path or not os.path.exists(module_json_path):
                logging.warning(f"Module JSON not found for {block_id} (type: {block_type}), skipping")
                continue
                
            # Load module JSON
            try:
                with open(module_json_path, 'r') as f:
                    module_data = json.load(f)
                
                logging.info(f"Processing module data from {module_json_path} for block {block_id}")
                
                # Process assemblers
                if "assembler_information" in module_data:
                    for assembler in module_data["assembler_information"]:
                        if len(assembler) >= 3:  # Ensure we have enough data
                            item, rel_x, rel_y = assembler[0], assembler[1], assembler[2]
                            # Convert module coordinates (0,0 at bottom left) to factory coordinates (0,0 at top left)
                            # y-coordinates need to be flipped relative to the block's height
                            abs_x = block_x + rel_x
                            abs_y = block_y + (block_height - 1) - rel_y
                            
                            factory_data["entities"]["assemblers"].append({
                                "item": item,
                                "position": [abs_x, abs_y],
                                "block_id": block_id
                            })
                
                # Process inserters
                if "inserter_information" in module_data:
                    for inserter in module_data["inserter_information"]:
                        if len(inserter) >= 4:  # Ensure we have enough data
                            item, rel_x, rel_y, direction = inserter[0], inserter[1], inserter[2], inserter[3]
                            # Convert coordinates
                            abs_x = block_x + rel_x
                            abs_y = block_y + (block_height - 1) - rel_y
                            
                            factory_data["entities"]["inserters"].append({
                                "item": item,
                                "position": [abs_x, abs_y],
                                "direction": direction,
                                "block_id": block_id
                            })
                
                # Process belt points
                if "belt_point_information" in module_data:
                    for belt_point in module_data["belt_point_information"]:
                        if len(belt_point) >= 4:  # Ensure we have enough data
                            item, rel_x, rel_y, point_type = belt_point[0], belt_point[1], belt_point[2], belt_point[3]
                            # Convert coordinates
                            abs_x = block_x + rel_x
                            abs_y = block_y + (block_height - 1) - rel_y
                            
                            factory_data["entities"]["belts"].append({
                                "item": item,
                                "position": [abs_x, abs_y],
                                "type": point_type,
                                "block_id": block_id
                            })
                
                # Process paths which include underground belts and splitters
                if "paths" in module_data:
                    for item_key, paths_list in module_data["paths"].items():
                        item_name = item_key.split("_")[0]  # Extract item name from key
                        
                        for path_data in paths_list:
                            # Process regular belts
                            if "path" in path_data:
                                path = path_data["path"]
                                for i, (rel_x, rel_y) in enumerate(path):
                                    # Convert coordinates
                                    abs_x = block_x + rel_x
                                    abs_y = block_y + (block_height - 1) - rel_y
                                    
                                    # Determine belt direction
                                    direction = None
                                    if i < len(path) - 1:
                                        next_x, next_y = path[i+1]
                                        dx = next_x - rel_x
                                        dy = next_y - rel_y
                                        
                                        # Get orientation
                                        if dx > 0:
                                            direction = "right"
                                        elif dx < 0:
                                            direction = "left"
                                        elif dy > 0:
                                            direction = "down"
                                        elif dy < 0:
                                            direction = "up"
                                    
                                    # Add to belts if not already added (avoid duplicates)
                                    belt_pos_str = f"{abs_x},{abs_y}"
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
                            
                            # Process underground belts
                            if "underground_segments" in path_data:
                                for segment_id, segment in path_data["underground_segments"].items():
                                    if "start" in segment and "end" in segment:
                                        start_rel_x, start_rel_y = segment["start"]
                                        end_rel_x, end_rel_y = segment["end"]
                                        
                                        # Convert coordinates
                                        start_abs_x = block_x + start_rel_x
                                        start_abs_y = block_y + (block_height - 1) - start_rel_y
                                        end_abs_x = block_x + end_rel_x
                                        end_abs_y = block_y + (block_height - 1) - end_rel_y
                                        
                                        # Determine direction
                                        dx = end_rel_x - start_rel_x
                                        dy = end_rel_y - start_rel_y
                                        
                                        if abs(dx) > abs(dy):  # Horizontal
                                            direction = "right" if dx > 0 else "left"
                                        else:  # Vertical
                                            direction = "down" if dy > 0 else "up"
                                        
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
                            
                            # Process splitters
                            for splitter_type in ["start_splitter", "dest_splitter"]:
                                if splitter_type in path_data and path_data[splitter_type]:
                                    splitter_data = path_data[splitter_type]
                                    
                                    if "position" in splitter_data:
                                        rel_x, rel_y = splitter_data["position"]
                                        
                                        # Convert coordinates
                                        abs_x = block_x + rel_x
                                        abs_y = block_y + (block_height - 1) - rel_y
                                        
                                        # Determine direction
                                        direction = "right"  # Default
                                        if "direction" in splitter_data:
                                            dx, dy = splitter_data["direction"]
                                            if dx > 0:
                                                direction = "right"
                                            elif dx < 0:
                                                direction = "left"
                                            elif dy > 0:
                                                direction = "down"
                                            elif dy < 0:
                                                direction = "up"
                                        
                                        factory_data["entities"]["splitters"].append({
                                            "item": item_name,
                                            "position": [abs_x, abs_y],
                                            "direction": direction,
                                            "block_id": block_id
                                        })
                
                # Process I/O points
                # Input points
                for gate in block_info["input_points"]:
                    gate_x = gate["x"]
                    gate_y = gate["y"]
                    
                    factory_data["io_points"]["inputs"].append({
                        "item": gate["item"],
                        "position": [gate_x, gate_y],
                        "block_id": block_id,
                        "gate_id": gate["id"]
                    })
                
                # Output points
                for gate in block_info["output_points"]:
                    gate_x = gate["x"]
                    gate_y = gate["y"]
                    
                    factory_data["io_points"]["outputs"].append({
                        "item": gate["item"],
                        "position": [gate_x, gate_y],
                        "block_id": block_id,
                        "gate_id": gate["id"]
                    })
                        
                # Add inter-block paths if available
                if hasattr(self, 'inter_block_paths') and self.inter_block_paths:
                    for path_id, path_data_list in self.inter_block_paths.items():
                        item = path_id.split("_")[0]  # Extract item name from path ID
                        
                        for i, path_data in enumerate(path_data_list):
                            # Basic path data
                            path = path_data.get('path', [])
                            
                            # Prepare path entry
                            path_entry = {
                                "id": f"{path_id}_{i}",
                                "item": item,
                                "path": path,
                                "belts": []
                            }
                            
                            # Process regular belt segments
                            for j in range(len(path) - 1):
                                start_point = path[j]
                                end_point = path[j + 1]
                                
                                # Calculate direction
                                dx = end_point[0] - start_point[0]
                                dy = end_point[1] - start_point[1]
                                
                                direction = None
                                if dx > 0:
                                    direction = "right"
                                elif dx < 0:
                                    direction = "left"
                                elif dy > 0:
                                    direction = "down"
                                elif dy < 0:
                                    direction = "up"
                                
                                # Check if this segment is part of an underground
                                is_underground = False
                                if 'underground_segments' in path_data:
                                    for segment_id, segment in path_data['underground_segments'].items():
                                        segment_path = segment.get('path', [])
                                        if start_point in segment_path and end_point in segment_path:
                                            idx1 = segment_path.index(start_point)
                                            idx2 = segment_path.index(end_point)
                                            if abs(idx1 - idx2) == 1:
                                                is_underground = True
                                                break
                                
                                # Add belt if not underground
                                if not is_underground and direction:
                                    path_entry["belts"].append({
                                        "position": start_point,
                                        "direction": direction
                                    })
                            
                            # Process underground segments
                            if 'underground_segments' in path_data:
                                path_entry["underground_belts"] = []
                                
                                for segment_id, segment in path_data['underground_segments'].items():
                                    start = segment.get('start', None)
                                    end = segment.get('end', None)
                                    
                                    if start and end:
                                        # Calculate direction
                                        dx = end[0] - start[0]
                                        dy = end[1] - start[1]
                                        
                                        direction = None
                                        if abs(dx) > abs(dy):  # Horizontal
                                            direction = "right" if dx > 0 else "left"
                                        else:  # Vertical
                                            direction = "down" if dy > 0 else "up"
                                        
                                        # Add entrance
                                        path_entry["underground_belts"].append({
                                            "position": start,
                                            "type": "entrance",
                                            "direction": direction
                                        })
                                        
                                        # Add exit
                                        path_entry["underground_belts"].append({
                                            "position": end,
                                            "type": "exit",
                                            "direction": direction
                                        })
                            
                            # Process splitters
                            if ('start_splitter' in path_data and path_data['start_splitter']) or \
                            ('dest_splitter' in path_data and path_data['dest_splitter']):
                                path_entry["splitters"] = []
                                
                                for splitter_type in ["start_splitter", "dest_splitter"]:
                                    if splitter_type in path_data and path_data[splitter_type]:
                                        splitter_data = path_data[splitter_type]
                                        position = splitter_data.get('position', None)
                                        
                                        if position:
                                            # Get direction
                                            direction = "right"  # Default
                                            if 'direction' in splitter_data:
                                                dx, dy = splitter_data['direction']
                                                if dx > 0:
                                                    direction = "right"
                                                elif dx < 0:
                                                    direction = "left"
                                                elif dy > 0:
                                                    direction = "down"
                                                elif dy < 0:
                                                    direction = "up"
                                            
                                            # Add splitter
                                            path_entry["splitters"].append({
                                                "position": position,
                                                "direction": direction,
                                                "type": splitter_type
                                            })
                            
                            # Add the complete path entry
                            factory_data["inter_block_paths"].append(path_entry)     
                    
                    
            except Exception as e:
                logging.error(f"Error processing module JSON for {block_id}: {e}")
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
            
            for output_gate in self.external_io.get("output_gates", []):
                factory_data["io_points"]["outputs"].append({
                    "item": output_gate["item"],
                    "position": list(output_gate["position"]),
                    "edge": output_gate["edge"],
                    "gate_id": output_gate["id"],
                    "external": True
                })
        
        # Add gate connections from Z3 solver
        if self.gate_connections:
            for connection in self.gate_connections:
                source_gate, target_gate = connection
                
                # Handle different gate object types
                try:
                    # Check if source_gate is a dictionary, a Gate object, or another type
                    if isinstance(source_gate, dict):
                        source_data = {
                            "block_id": source_gate.get("block_id", ""),
                            "gate_id": source_gate.get("gate_id", ""),
                            "position": [source_gate.get("x", 0), source_gate.get("y", 0)],
                            "item": source_gate.get("item", "")
                        }
                    elif hasattr(source_gate, "block_id") and hasattr(source_gate, "gate_id"):
                        # It's a Gate object with attributes
                        source_data = {
                            "block_id": getattr(source_gate, "block_id", ""),
                            "gate_id": getattr(source_gate, "gate_id", ""),
                            "position": [getattr(source_gate, "x", 0), getattr(source_gate, "y", 0)],
                            "item": getattr(source_gate, "item", "")
                        }
                    else:
                        # Fallback with minimal information
                        logging.warning(f"Unknown source gate format: {type(source_gate)}")
                        source_data = {
                            "block_id": str(source_gate),
                            "gate_id": str(source_gate),
                            "position": [0, 0],
                            "item": ""
                        }
                    
                    # Same check for target_gate
                    if isinstance(target_gate, dict):
                        target_data = {
                            "block_id": target_gate.get("block_id", ""),
                            "gate_id": target_gate.get("gate_id", ""),
                            "position": [target_gate.get("x", 0), target_gate.get("y", 0)],
                            "item": target_gate.get("item", "")
                        }
                    elif hasattr(target_gate, "block_id") and hasattr(target_gate, "gate_id"):
                        # It's a Gate object with attributes
                        target_data = {
                            "block_id": getattr(target_gate, "block_id", ""),
                            "gate_id": getattr(target_gate, "gate_id", ""),
                            "position": [getattr(target_gate, "x", 0), getattr(target_gate, "y", 0)],
                            "item": getattr(target_gate, "item", "")
                        }
                    else:
                        # Fallback with minimal information
                        logging.warning(f"Unknown target gate format: {type(target_gate)}")
                        target_data = {
                            "block_id": str(target_gate),
                            "gate_id": str(target_gate),
                            "position": [0, 0],
                            "item": ""
                        }
                    
                    factory_data["connections"].append({
                        "source": source_data,
                        "target": target_data
                    })
                except Exception as e:
                    logging.error(f"Error processing gate connection: {e}")
                    continue
        
        print(path)
        
        # If no path specified, use default
        if not path:
            path = f"Factorys/{self.output_item}_factory.json"
        
        # Fix the path if it's a list
        if isinstance(path, list):
            path = str(path[0]) if path else f"Factorys/{self.output_item}_factory.json"
            
                
                # Ensure directory exists
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(factory_data, f, indent=2)
        
        logging.info(f"Factory JSON data saved to {path}")
        
        return factory_data
    
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
        
        logging.info(f"Found {len(connections)} connections between different blocks")
        return connections



    def load_images(self):
        """Load images from the assets folder based on block names."""
        for block_key in self.final_blocks.keys():
            # Extract the base name of the block (e.g., 'electronic-circuit')
            base_name = block_key.split('_')[1]
            image_path = os.path.join('assets', f'{base_name}.png')
            
            
            print(image_path)
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
            pygame.draw.line(window, (200, 200, 200), (0, y), (window_width, y))
        
            # Draw inter-block paths first so they appear behind blocks
        if False:
            for item_key, path_data_list in self.inter_block_paths.items():
                for path_data in path_data_list:
                    path = path_data.get('path', [])
                    underground_segments = path_data.get('underground_segments', {})
                    
                    # Draw regular path segments
                    for i in range(len(path) - 1):
                        start_x, start_y = path[i]
                        end_x, end_y = path[i + 1]
                        
                        # Skip if part of underground segment
                        is_underground = False
                        for segment_id, segment in underground_segments.items():
                            segment_path = segment['path']
                            if (start_x, start_y) in segment_path and (end_x, end_y) in segment_path:
                                idx1 = segment_path.index((start_x, start_y))
                                idx2 = segment_path.index((end_x, end_y))
                                if abs(idx1 - idx2) == 1:  # Adjacent in segment
                                    is_underground = True
                                    break
                        
                        if not is_underground:
                            start_pos = (start_x * cell_size + cell_size // 2, start_y * cell_size + cell_size // 2)
                            end_pos = (end_x * cell_size + cell_size // 2, end_y * cell_size + cell_size // 2)
                            pygame.draw.line(window, BELT_COLOR, start_pos, end_pos, 3)
                    
                    # Draw underground segments
                    for segment_id, segment in underground_segments.items():
                        start = segment['start']
                        end = segment['end']
                        
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
            print(f"Factory visualization saved to {save_path}")
        
        # Wait for user to close the window or run for a limited time
        waiting = True
        clock = pygame.time.Clock()
        while waiting:
            clock.tick(60)  # 60 FPS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
        
        pygame.quit()

    def define_factory_io_points(self):
        """
        Define external input/output points for the factory on the edges.
        This determines which items need external connections and where they should be placed.
        """
        logging.info("Defining factory I/O points")
        
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
        
        logging.info(f"External inputs: {external_inputs}")
        logging.info(f"External outputs: {external_outputs}")
        logging.info(f"Internal items: {internal_items}")
        
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
            logging.debug(f"Selected {selected_edge} at position {selected_position} for {item}")
        
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
            
            logging.info(f"Created fixed {io_type} gate for {item} at position {position} on {placement['edge']} edge")

    def apply_io_constraints_to_solver(self):
        """
        Apply the external I/O constraints to the factory Z3 solver.
        Instead of constraining existing gates, this adds fixed I/O gates to the solver.
        """
        if not hasattr(self, 'external_io') or not self.external_io:
            logging.warning("No external I/O defined, skipping I/O constraints")
            return
        
        if not hasattr(self, 'z3_solver') or not self.z3_solver:
            logging.error("Z3 solver not initialized, cannot apply I/O constraints")
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
            
            logging.info(f"Added fixed input gate {input_gate['id']} at position {input_gate['position']}")
        
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
            
            logging.info(f"Added fixed output gate {output_gate['id']} at position {output_gate['position']}")

    def plan_inter_block_paths(self):
        """
        Plan paths between connected gates using the MultiAgentPathfinder.
        
        1. Create an obstacle map marking all blocks and entities
        2. Identify connection points between blocks
        3. Use MultiAgentPathfinder to create paths between connected gates
        """
        logging.info("Planning inter-block paths...")
        
        if not self.final_blocks or not self.gate_connections:
            logging.error("Cannot plan paths: No blocks or gate connections available")
            return {}, {}
        
        # Calculate the overall grid size based on final_x and final_y
        grid_width = self.final_x + 1
        grid_height = self.final_y + 1
        
        # Create an obstacle map (0 = free, 1 = obstacle)
        obstacle_map = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Mark all blocks as obstacles
        for block_id, block_info in self.final_blocks.items():
            block_x = block_info['x']
            block_y = block_info['y']
            block_width = block_info['width']
            block_height = block_info['height']
            
            # Mark the entire block area as obstacles
            for y in range(block_y, block_y + block_height):
                for x in range(block_x, block_x + block_width):
                    # Make sure we're within the grid bounds
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        obstacle_map[y][x] = 1
        
        # Prepare connection points for the pathfinder
        connection_points = {}
        
        # Process all gate connections
        for i, connection in enumerate(self.gate_connections):
            # Ensure we have a valid connection tuple
            if isinstance(connection, tuple) and len(connection) == 2:
                source_gate, target_gate = connection
                
                # Both source and target should be dictionary-like objects at this point
                # Extract necessary data with appropriate error handling
                try:
                    # Source gate data
                    if isinstance(source_gate, dict):
                        source_x = source_gate['x']
                        source_y = source_gate['y']
                        item = source_gate['item']
                    else:
                        logging.error(f"Source gate is not a dictionary: {source_gate}")
                        continue
                    
                    # Target gate data
                    if isinstance(target_gate, dict):
                        target_x = target_gate['x']
                        target_y = target_gate['y']
                    else:
                        logging.error(f"Target gate is not a dictionary: {target_gate}")
                        continue
                    
                    # Ensure we have concrete integer values
                    source_x = int(source_x)
                    source_y = int(source_y)
                    target_x = int(target_x)
                    target_y = int(target_y)
                    
                    # Connection identifier
                    connection_id = f"{item}_{i}"
                    
                    # Clear gate positions in the obstacle map for pathfinding
                    if 0 <= source_x < grid_width and 0 <= source_y < grid_height:
                        obstacle_map[source_y][source_x] = 0
                    
                    if 0 <= target_x < grid_width and 0 <= target_y < grid_height:
                        obstacle_map[target_y][target_x] = 0
                    
                    # Prepare connection information for pathfinder
                    connection_points[connection_id] = {
                        'item': item,
                        'start_points': [(source_x, source_y)],
                        'destination': [(target_x, target_y)]
                    }
                    
                    logging.debug(f"Added connection {connection_id}: {item} from ({source_x}, {source_y}) to ({target_x}, {target_y})")
                    
                except (TypeError, KeyError) as e:
                    logging.error(f"Error processing connection {i}: {e}")
                    continue
            else:
                logging.warning(f"Invalid connection format at index {i}: {connection}")
        
        # Process external I/O points if available
        if hasattr(self, 'external_io') and self.external_io:
            # Similar processing for external I/O points...
            pass
        
        if not connection_points:
            logging.warning("No valid connections to route!")
            return {}, {}
        
        # Create the MultiAgentPathfinder
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
            logging.info(f"Found paths for {len(paths)} connections")
            
            # Save the paths for later use
            self.inter_block_paths = paths
            self.inter_block_inserters = inserters
            
            # Visualize paths for debugging
            try:
                pathfinder.visualize_grid(filename="factory_obstacle_map.png")
                pathfinder.visualize_paths(filename_template="inter_block_path_{}.png")
                logging.info("Path visualizations saved to disk")
            except Exception as e:
                logging.error(f"Failed to visualize paths: {e}")
            
            return paths, inserters
            
        except Exception as e:
            logging.error(f"Error in pathfinding: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}

def manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        
def main():
    
    output_item = "electronic-circuit"
    amount = 200
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
    

    builder.visualize_factory(save_path=f"Factorys/{output_item}_factory.png")

    


def log_method_time(item, amount, method_name,assemblers_per_recipie,num_subfactories,start_time, end_time):
    execution_time = end_time - start_time
    logging.info(f"Execution time for {method_name}: {execution_time:.4f} seconds.")
    
    # Open the CSV file and append the data
    try:
        with open("execution_times_big_factory.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([item, amount, method_name,assemblers_per_recipie,num_subfactories,execution_time])
    except Exception as e:
        logging.error(f"Error logging execution time for {method_name}: {e}")
        
 
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
        
if __name__ == "__main__":
    #plot_csv_data("execution_times_big_factory.csv")
    main()
