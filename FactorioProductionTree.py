#! .venv\Scripts\python.exe
import json
import logging
import pygame
import math
from z3 import And , Or
from z3Solver  import Z3Solver
from AStarPathfinder import AStarPathfinder

# Define constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
LIGHT_GREY = (211,211,211)

# Define the size of each grid cell
CELL_SIZE = 40

INSERTER_COLOR_MAP = {
    'input': (255, 165, 0),  # Orange 
    'output': (255, 20, 147) # Deep Pink
}

BELT_COLOR_MAP = {
    'start': (0, 128, 0),  # Green 
    'end': (255, 0, 0)     # Red
}


class FactorioProductionTree:
    def __init__(self) -> None:
        # Load the data from JSON
        self.setup_logging("FactorioProductionTree")
        items_data = self.load_json("recipes.json")
        
        self.machines_data = self.load_json("machine_data.json")  # Machine speeds and capacities
        
        self.grid = None
        self.grid_width = 10
        self.grid_height = 10
        
        # Create a lookup dictionary for items by their ID
        self.item_lookup = {item["id"]: item for item in items_data}


        # init after calulation of production data
        self.z3_solver = None
        self.AStar = None
    
        
        
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
                "belts": 0  
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

    def solve(self,production_data):
        # Initialize solver with grid size and production data
        z3_solver = Z3Solver(self.grid_width,self.grid_height, production_data)
        # Process the input to place assemblers
        z3_solver.solve()

        self.z3_solver = z3_solver
        
    # need to solve once before you can execute this
    def build_belts(self,max_tries):
        
        
        for i in range(max_tries):
            
            print(f'Try Number: {i}')
            obstacle_map,pairs = self.z3_solver.build_map()
            
        
            self.AStar = AStarPathfinder(obstacle_map,pairs)
            
            
            if(self.AStar.connect_belts()):
                # Successfully connected all belts, return True
                self.grid = self.AStar.grid
                self.direction_grid = self.AStar.direction_grid
                return True
            else:
                # add constraint to solver and let him solve again
                # add assembler not at same pos constraint
                model = self.z3_solver.model
                for assembler in self.z3_solver.assemblers:
                    
                    assembler_x = model.evaluate(assembler.x).as_long()
                    assembler_y = model.evaluate(assembler.y).as_long()
                    
                    constraint = Or(assembler.x != assembler_x, assembler.y != assembler_y )
                    
                    self.z3_solver.add_constraint(constraint)
                    self.z3_solver.solve()
                
        return False            
            
        
        
    def visualize_factory(self):

        pygame.init()
        
        # Set up the window size
        window_width = self.grid_width * CELL_SIZE 
        window_height = self.grid_width * CELL_SIZE
        window = pygame.display.set_mode((window_width, window_height))
        
        pygame.display.set_caption('Factory Layout Visualization')
        
        # Set up clock
        clock = pygame.time.Clock()
        
        model = self.z3_solver.model
        
        
        
        # assets
        assembler_image = pygame.image.load('assets/assembler.png')
        inserter_image = pygame.image.load('assets/inserter.png')
        belt_image = pygame.image.load('assets/conveyor.png')
        
        # Scale images
        assembler_image = pygame.transform.scale(assembler_image, (3 * CELL_SIZE, 3 * CELL_SIZE))
        inserter_image = pygame.transform.scale(inserter_image, (CELL_SIZE, CELL_SIZE))
        belt_image = pygame.transform.scale(belt_image, (CELL_SIZE, CELL_SIZE))
        
        # Define rotation angles for each direction
        direction_angle_map = {
            'right': 270,
            'down': 180,
            'left': 90,
            'up': 0
        }
        
        
        # game loop
        running = True
        while running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Fill the background with white
            window.fill(WHITE)
            
            # Draw the grid
            for x in range(0, window_width, CELL_SIZE):
                for y in range(0, window_height, CELL_SIZE):
                    pygame.draw.rect(window, BLACK, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), 1)
            
            # Draw global input belts
            for belt in self.z3_solver.global_input_belts:
                belt_x = model.evaluate(belt.x).as_long()
                belt_y = model.evaluate(belt.y).as_long()
                window.blit(belt_image, (belt_x * CELL_SIZE, belt_y * CELL_SIZE))
                
                font = pygame.font.Font(None, 20)
                text_surface = font.render(belt.id, True, BLACK)
                window.blit(text_surface, (belt_x * CELL_SIZE + 5, belt_y * CELL_SIZE + 5))
            
            # Draw global output belt
            out_belt_x = model.evaluate(self.z3_solver.global_output_belt.x).as_long()
            out_belt_y = model.evaluate(self.z3_solver.global_output_belt.y).as_long()
            window.blit(belt_image, (out_belt_x * CELL_SIZE, out_belt_y * CELL_SIZE))

            font = pygame.font.Font(None, 20)
            text_surface = font.render(self.z3_solver.global_output_belt.id, True, BLACK)
            window.blit(text_surface, (out_belt_x * CELL_SIZE + 5, out_belt_y * CELL_SIZE + 5))

            # Draw the assemblers and their inserters
            for assembler in self.z3_solver.assemblers:
                assembler_x = model.evaluate(assembler.x).as_long()
                assembler_y = model.evaluate(assembler.y).as_long()
                window.blit(assembler_image, (assembler_x * CELL_SIZE, assembler_y * CELL_SIZE))

                font = pygame.font.Font(None, 24)
                text_surface = font.render(assembler.id, True, BLACK)
                window.blit(text_surface, (assembler_x * CELL_SIZE + 5, assembler_y * CELL_SIZE + 5))

                # Draw inserters and associated belts
                for inserter in assembler.inserters:
                    inserter_x = model.evaluate(inserter.x).as_long()
                    inserter_y = model.evaluate(inserter.y).as_long()
                    
        
                    if inserter_x < assembler_x:
                        direction = 'right'
                    elif inserter_x > assembler_x:
                        direction = 'left'
                    elif inserter_y < assembler_y:
                        direction = 'down'
                    elif inserter_y > assembler_y:
                        direction = 'up'
                    else:
                        direction = 'up'  
                    
                    # based on type invert the rotation for each inserter and belt
                    swap = 0
                    if inserter.type == 'output':
                        swap = 180
                        
                    rotated_inserter_image = pygame.transform.rotate(inserter_image, direction_angle_map[direction]+swap)
                    window.blit(rotated_inserter_image, (inserter_x * CELL_SIZE, inserter_y * CELL_SIZE))

                    belt_x = model.evaluate(inserter.belt.x).as_long()
                    belt_y = model.evaluate(inserter.belt.y).as_long()
                    
                    rotated_belt_image = pygame.transform.rotate(belt_image, direction_angle_map[direction]+swap)
                    window.blit(rotated_belt_image, (belt_x * CELL_SIZE, belt_y * CELL_SIZE))

            # Draw the A* path belts (if available in the grid)
            if self.grid is not None:
                for y in range(self.grid_height):
                    for x in range(self.grid_width):
                        if self.grid[y][x] == 2:  # Belt path marked with '2'
                            direction = self.direction_grid[y][x]
                            rotated_belt_image = pygame.transform.rotate(belt_image, direction_angle_map[direction])
                            window.blit(rotated_belt_image, (x * CELL_SIZE, y * CELL_SIZE))
                        
                
            # Update display
            pygame.display.flip()
            
            clock.tick(30)

        pygame.quit()
        
    
        
def main():
    factorioProductionTree = FactorioProductionTree()
    total_requirements = factorioProductionTree.calculate_production("electronic-circuit", 10)
    #factorioProductionTree.calculate_minimal_grid_size(total_requirements)
    
    factorioProductionTree.solve(total_requirements)
    
    if(factorioProductionTree.build_belts(max_tries=20)):
        factorioProductionTree.visualize_factory()
   
if __name__ == "__main__":
    main()
