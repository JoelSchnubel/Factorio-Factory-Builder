#! .venv\Scripts\python.exe

from FactorioProductionTree import FactorioProductionTree
from FactoryZ3Solver import FactoryZ3Solver
import pygame
import json
import os
import time

CELL_SIZE = 10


class FactoryBuilder:
    
    def __init__(self,output_item,amount,max_assembler_per_blueprint,start_width,start_height) -> None:
        
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
        
        self.block_size = 10
        self.images = {}


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
        
        # TODO Fix production data 

        factorioProductionTree = FactorioProductionTree(grid_width=self.start_width,grid_height=self.start_height)
        
        # Always include the output item
        input_items = self.get_input_items(self.output_item,list(selected_items)) + list(selected_items)

        print(input_items)

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
            
    
            
    
    
    def solve_small_blocks(self, visualize):
        
        for item in self.block_data.keys():
            self.block_data[item]["tree"].solve(self.block_data[item]["production_data"],sequential=False)
            paths, placed_inserter_information = self.block_data[item]["tree"].build_belts(max_tries=2)
            self.block_data[item]["paths"] = paths
            self.block_data[item]["placed_inserter_information"]=placed_inserter_information

            if visualize:
                self.block_data[item]["tree"].visualize_factory(paths,placed_inserter_information)

        
            
    def solve_factory(self):
        self.z3_solver = FactoryZ3Solver(self.block_data,self.output_point)
        self.z3_solver.build_constraints()
        
        self.final_blocks,self.final_x,self.final_y = self.z3_solver.solve()
    
        print(self.final_blocks)
        print(self.final_x)
        print(self.final_y)

        print(self.block_data)
        
        
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

    def visualize_factory(self):
        """Draw the factory using Pygame."""
        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode(
            (self.final_x * self.block_size, self.final_y * self.block_size)
        )
        pygame.display.set_caption('Factory Layout')

        # Load images
        self.load_images()

        # Main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Clear the screen
            screen.fill((255, 255, 255))  # White background

            # Draw each block
            for block_id, block_info in self.final_blocks.items():
                x = block_info["x"] * self.block_size
                y = block_info["y"] * self.block_size
                width = block_info["width"] * self.block_size
                height = block_info["height"] * self.block_size

                # Draw the block (image or placeholder rectangle)
                if block_id in self.images:
                    # Scale the image to fit the block size
                    image = pygame.transform.scale(self.images[block_id], (width, height))
                    screen.blit(image, (x, y))
                else:
                    # Draw a placeholder rectangle if the image is not found
                    pygame.draw.rect(screen, (0, 0, 0), (x, y, width, height))  # Black rectangle

                # Draw input and output gates
                for gate in block_info["input_points"]:
                    gate_x = x + gate["x"] * self.block_size
                    gate_y = y + gate["y"] * self.block_size
                    pygame.draw.circle(screen, (0, 255, 0), (gate_x, gate_y), 5)  # Green circle for input gates

                for gate in block_info["output_points"]:
                    gate_x = x + gate["x"] * self.block_size
                    gate_y = y + gate["y"] * self.block_size
                    pygame.draw.circle(screen, (255, 0, 0), (gate_x, gate_y), 5)  # Red circle for output gates

            # Update the display
            pygame.display.flip()

        # Quit Pygame
        pygame.quit()
    
        
def main():
    
    output_item = "electronic-circuit"
    amount = 1500
    max_assembler_per_blueprint = 5
    
    start_width = 15
    start_height = 15

    
    builder = FactoryBuilder(output_item,amount,max_assembler_per_blueprint,start_width,start_height)
    
    #num_factories, production_data = builder.eval_split()
    #print(f"Number of factories required: {num_factories}")
    
    builder.split_recipies()
    print(builder.block_data)
    builder.solve_small_blocks(visualize=False)
    
    builder.solve_factory()
    
    



def visualize_test():
    output_item = "electronic-circuit"
    amount = 1500
    max_assembler_per_blueprint = 5
    
    start_width = 15
    start_height = 15

    
    builder = FactoryBuilder(output_item,amount,max_assembler_per_blueprint,start_width,start_height)
    
    builder.final_x=15
    builder.final_y=75
    builder.final_blocks = {'Block_electronic-circuit_0_0': {'x': 0, 'y': 15, 'width': 15, 'height': 15, 'input_points': [{'id': 'electronic-circuit_input_copper-plate_0_0', 'item': 'copper-plate', 'type': 'input', 'x': -194, 'y': -195}, {'id': 'electronic-circuit_input_iron-plate_0_0', 'item': 'iron-plate', 'type': 'input', 'x': -1, 'y': 522}, {'id': 'electronic-circuit_input_electronic-circuit_0_0', 'item': 'electronic-circuit', 'type': 'input', 'x': -234, 'y': -234}], 'output_points': [{'id': 'electronic-circuit_output_copper-plate_0_0', 'item': 'copper-plate', 'type': 'output', 'x': -215, 'y': -215}, {'id': 'electronic-circuit_output_iron-plate_0_0', 'item': 'iron-plate', 'type': 'output', 'x': -115, 'y': 467}, {'id': 'electronic-circuit_output_electronic-circuit_0_0', 'item': 'electronic-circuit', 'type': 'output', 'x': -236, 'y': -290}]}, 'Block_electronic-circuit_0_1': {'x': 0, 'y': 30, 'width': 15, 'height': 15, 'input_points': [{'id': 'electronic-circuit_input_copper-plate_0_1', 'item': 'copper-plate', 'type': 'input', 'x': -204, 'y': -199}, {'id': 'electronic-circuit_input_iron-plate_0_1', 'item': 'iron-plate', 'type': 'input', 'x': 9, 'y': 669}, {'id': 'electronic-circuit_input_electronic-circuit_0_1', 'item': 'electronic-circuit', 'type': 'input', 'x': -1, 'y': -1}], 'output_points': [{'id': 'electronic-circuit_output_copper-plate_0_1', 'item': 'copper-plate', 'type': 'output', 'x': -207, 'y': -207}, {'id': 'electronic-circuit_output_iron-plate_0_1', 'item': 'iron-plate', 'type': 'output', 'x': -100, 'y': -1}, {'id': 'electronic-circuit_output_electronic-circuit_0_1', 'item': 'electronic-circuit', 'type': 'output', 'x': -505, 'y': -498}]}, 'Block_electronic-circuit_0_2': {'x': 0, 'y': 45, 'width': 15, 'height': 15, 'input_points': [{'id': 'electronic-circuit_input_copper-plate_0_2', 'item': 'copper-plate', 'type': 'input', 'x': -160, 'y': -200}, {'id': 'electronic-circuit_input_iron-plate_0_2', 'item': 'iron-plate', 'type': 'input', 'x': -99, 'y': 468}, {'id': 'electronic-circuit_input_electronic-circuit_0_2', 'item': 'electronic-circuit', 'type': 'input', 'x': -178, 'y': -178}], 'output_points': [{'id': 'electronic-circuit_output_copper-plate_0_2', 'item': 'copper-plate', 'type': 'output', 'x': -198, 'y': -198}, {'id': 'electronic-circuit_output_iron-plate_0_2', 'item': 'iron-plate', 'type': 'output', 'x': -127, 'y': 451}, {'id': 'electronic-circuit_output_electronic-circuit_0_2', 'item': 'electronic-circuit', 'type': 'output', 'x': -273, 'y': -271}]}, 'Block_electronic-circuit_0_3': {'x': 0, 'y': 0, 'width': 15, 'height': 15, 'input_points': [{'id': 'electronic-circuit_input_copper-plate_0_3', 'item': 'copper-plate', 'type': 'input', 'x': -196, 'y': -196}, {'id': 'electronic-circuit_input_iron-plate_0_3', 'item': 'iron-plate', 'type': 'input', 'x': -105, 'y': 114}, {'id': 'electronic-circuit_input_electronic-circuit_0_3', 'item': 'electronic-circuit', 'type': 'input', 'x': 117, 'y': 117}], 'output_points': [{'id': 'electronic-circuit_output_copper-plate_0_3', 'item': 'copper-plate', 'type': 'output', 'x': -466, 'y': -466}, {'id': 'electronic-circuit_output_iron-plate_0_3', 'item': 'iron-plate', 'type': 'output', 'x': 8, 'y': 647}, {'id': 'electronic-circuit_output_electronic-circuit_0_3', 'item': 'electronic-circuit', 'type': 'output', 'x': 409, 'y': -507}]}, 'Block_electronic-circuit_0_4': {'x': 0, 'y': 60, 'width': 15, 'height': 15, 'input_points': [{'id': 'electronic-circuit_input_copper-plate_0_4', 'item': 'copper-plate', 'type': 'input', 'x': 50, 'y': -478}, {'id': 'electronic-circuit_input_iron-plate_0_4', 'item': 'iron-plate', 'type': 'input', 'x': -101, 'y': 662}, {'id': 'electronic-circuit_input_electronic-circuit_0_4', 'item': 'electronic-circuit', 'type': 'input', 'x': -250, 'y': -250}], 'output_points': [{'id': 'electronic-circuit_output_copper-plate_0_4', 'item': 'copper-plate', 'type': 'output', 'x': -479, 'y': -479}, {'id': 'electronic-circuit_output_iron-plate_0_4', 'item': 'iron-plate', 'type': 'output', 'x': -104, 'y': 407}, {'id': 'electronic-circuit_output_electronic-circuit_0_4', 'item': 'electronic-circuit', 'type': 'output', 'x': -292, 'y': -291}]}}
    builder.visualize_factory()
    
        
if __name__ == "__main__":
    #main()
    visualize_test()