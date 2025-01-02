#! .venv\Scripts\python.exe

from FactorioProductionTree import FactorioProductionTree
from FactoryZ3Solver import FactoryZ3Solver
import pygame
import json





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


    def load_json(self,recipe_file):
        with open(recipe_file, "r") as file:
                recipes = json.load(file)
                return {item["id"]: item for item in recipes}
        

    # gets a list of production_data and evals each for number of assemblers
    # if number of assemblers > than limit -> split the production data in half 
    def eval_split(self,production_data,input_items):
        num_factories = 1
        factorioProductionTree = FactorioProductionTree(grid_width=self.start_width,grid_height=self.start_height)
        if self.count_assembler(production_data) > self.max_assembler_per_blueprint:
            # split in half and check again 
            production_data  = factorioProductionTree.calculate_production(self.output_item , self.amount/2,input_items) 
            production_data = factorioProductionTree.set_capacities(production_data)
            num_factories +=1 


        
        return production_data,num_factories
    
    
    # count the number of assemblers in the production data
    def count_assembler(self,production_data) -> int:
        total_assemblers = 0
        for key, value in production_data.items():
            if 'assemblers' in value:
                total_assemblers += value['assemblers']
                
        print(total_assemblers)
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
        
        production_data,num_factories = self.eval_split(production_data, list(selectable_items.keys()))
        

        
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
            
    
            
    
    
    def solve_small_blocks(self):
        
        for item in self.block_data.keys:
            self.block_data[item]["tree"].solve(self.block_data[item]["production_data"],sequential=False)
            paths, placed_inserter_information = self.block_data[item]["tree"].build_belts(max_tries=2)
            self.block_data[item]["paths"] = paths
            self.block_data[item]["placed_inserter_information"]=placed_inserter_information
    

        
            
    def solve_factory(self):
        self.z3_solver = FactoryZ3Solver(self.block_data,self.output_point)
        self.z3_solver.solve()
    

    
    def visualize_factory(self):
        pass
    
    
        
def main():
    
    output_item = "electronic-circuit"
    amount = 200
    max_assembler_per_blueprint = 10
    
    start_width = 14
    start_height = 14

    
    builder = FactoryBuilder(output_item,amount,max_assembler_per_blueprint,start_width,start_height)
    
    #num_factories, production_data = builder.eval_split()
    #print(f"Number of factories required: {num_factories}")
    
    builder.split_recipies()
    
    
    
    
if __name__ == "__main__":
    main()
