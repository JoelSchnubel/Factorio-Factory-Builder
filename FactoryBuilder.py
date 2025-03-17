#! .venv\Scripts\python.exe

from FactorioProductionTree import FactorioProductionTree
from FactoryZ3Solver import FactoryZ3Solver
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
        
        self.images = {}
        
        self.load_modules = load_modules
        
        


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
                    "png":file_path.replace(".json", ".png")
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
        self.z3_solver = FactoryZ3Solver(self.block_data,self.output_point)
        num_factories=0
        for i, key in enumerate(self.block_data.keys()):
            

            num_factories += self.block_data[key]["num_factories"]
            
        print(f'total number of modules:{num_factories}')
            
        self.z3_solver.build_constraints()
        
        self.final_blocks,self.final_x,self.final_y = self.z3_solver.solve()
    
        print(self.final_blocks)
        print(self.final_x)
        print(self.final_y)

        
        
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
    
    
    def visualize_simple(self, cell_size=20, save_path=None):
        pygame.init()
        
        # Load item images if not already loaded
        if not hasattr(self, 'item_images'):
            self.load_item_images()
        
        # Calculate maximum coordinates needed for all blocks and gates
        max_x = 0
        max_y = 0
        for block_id, block_info in self.final_blocks.items():
            block_x = block_info['x']
            block_y = block_info['y']
            block_width = block_info['width']
            block_height = block_info['height']
            max_x = max(max_x, block_x + block_width)
            max_y = max(max_y, block_y + block_height)
            
            # Check gates
            for gate in block_info['input_points'] + block_info['output_points']:
                gate_x = block_x + gate['x']
                gate_y = block_y + gate['y']
                max_x = max(max_x, gate_x + 1)
                max_y = max(max_y, gate_y + 1)
        
        # Calculate window dimensions
        window_width = max_x * cell_size
        window_height = max_y * cell_size

        # Create Pygame window
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Factory Layout')

        # Clear the screen
        window.fill(WHITE)

        # Draw grid
        for row in range(max_x):
            for col in range(max_y):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(window, BLACK, rect, 1)

        # Draw blocks and gates
        for block_id, block_info in self.final_blocks.items():
            block_x = block_info['x']
            block_y = block_info['y']
            block_width = block_info['width']
            block_height = block_info['height']

            # Draw block
            block_rect = pygame.Rect(
                block_x * cell_size,
                block_y * cell_size,
                block_width * cell_size,
                block_height * cell_size
            )
            pygame.draw.rect(window, BLACK, block_rect, 2)

            # Draw input gates with images
            for gate in block_info['input_points']:
                gate_x = block_x + gate['x']
                gate_y = block_y + gate['y']
                if gate['item'] in self.item_images:
                    image = self.item_images[gate['item']]
                    # Scale image to cell size
                    scaled_image = pygame.transform.scale(image, (cell_size, cell_size))
                    window.blit(scaled_image, (gate_x * cell_size, gate_y * cell_size))

            # Draw output gates with images
            for gate in block_info['output_points']:
                gate_x = block_x + gate['x']
                gate_y = block_y + gate['y']
                if gate['item'] in self.item_images:
                    image = self.item_images[gate['item']]
                    # Scale image to cell size
                    scaled_image = pygame.transform.scale(image, (cell_size, cell_size))
                    window.blit(scaled_image, (gate_x * cell_size, gate_y * cell_size))

        # Update the display
        pygame.display.flip()
        
        # Save the image if a path is provided
        if save_path:
            pygame.image.save(window, save_path)
        else:
            # Wait for user to close the window
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
        
        # Quit Pygame
        pygame.quit()
        
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
    def visualize_factory(self,cell_size=20,block_size=20):
        """Draw the factory using Pygame."""
        # Initialize Pygame
        pygame.init()
        
        window_width  = self.final_x * cell_size *2
        window_height = self.final_y * cell_size *1.5

        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Factory Layout')
        # Main loop
 
            
        block_images = {}
        for block_id, block_info in self.block_data.items():
            image_path = block_info['png']  # Assuming the path is stored under 'path'
            image = pygame.image.load(image_path)
            image = pygame.transform.scale(image, (block_size * cell_size, block_size * cell_size))
            block_images[block_id] = image
            
            
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Clear the screen
            window.fill(WHITE)

            # Draw grid
            for row in range(self.final_x):
                for col in range(self.final_y):
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    pygame.draw.rect(window, BLACK, rect, 1)

            # Draw blocks
            for block_id, block_info in self.final_blocks.items():
                x = block_info['x']
                y = block_info['y']
                
                parts = block_id.split("_")  # Split by underscores
                if len(parts) >= 3:
                    block_type = "_".join(parts[1:-2])  # Remove "Block_" and "_0_0"
                else:
                    block_type = parts[1]  # Fallback if there's no coordinate suffix
                
                if block_type in block_images:
                    window.blit(block_images[block_type], ( x * cell_size, y * cell_size))
  
            # Update the display
            pygame.display.flip()
            
        # Quit Pygame
        pygame.quit()
    
        
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

    
    builder.solve_small_blocks(visualize=False)
    
    start_time = time.perf_counter()       
    print(start_time)
    builder.solve_factory()
    end_time = time.perf_counter()
    print(end_time)

    
    log_method_time(item=output_item,amount=amount,method_name="solve",assemblers_per_recipie=max_assembler_per_blueprint,num_subfactories=builder.get_num_subfactories(),start_time=start_time,end_time=end_time)
    

    #builder.visualize_factory()
    builder.visualize_simple()


def log_method_time(item, amount, method_name,assemblers_per_recipie,num_subfactories,start_time, end_time):
    execution_time = end_time - start_time
    logging.info(f"Execution time for {method_name}: {execution_time:.4f} seconds.")
    
    # Open the CSV file and append the data
    try:
        with open("execution_times_big_factory.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([item+"_copper_cable", amount, method_name,assemblers_per_recipie,num_subfactories,execution_time])
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
    sns.boxplot(
        x="steps",
        y="solve_time",
        hue="param1",  # Use param1 to differentiate if needed, or remove if not applicable
        data=df,
        palette = {5: "red", 0: "blue"}, 
        legend=False,
    )

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
   