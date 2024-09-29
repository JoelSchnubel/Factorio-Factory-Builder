#! .venv\Scripts\python.exe
import json
import logging
import pygame
import math
import pprint
from z3Solver  import Z3Solver




class FactorioProductionTree:
    def __init__(self) -> None:
        # Load the data from JSON
        self.setup_logging("FactorioProductionTree")
        items_data = self.load_json("recipes.json")
        
        self.machines_data = self.load_json("machine_data.json")  # Machine speeds and capacities
        
        # Create a lookup dictionary for items by their ID
        self.item_lookup = {item["id"]: item for item in items_data}
        
        
        
    
        
        
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

    
    def calculate_production(self, item_id, items_per_minute):
        """
        Recursively calculate the production requirements, including assemblers, inserters, and belts.

        Args:
        - item_id (str): ID of the item to produce.
        - items_per_minute (float): Amount of the item to produce per minute.

        Returns:
        - dict: A dictionary with item_id as keys and the total amount required, along with the 
                number of assemblers, inserters, and belts needed for each step.
        """
        item = self.item_lookup.get(item_id)
        
        # If the item doesn't exist or there's no recipe, return the required amount as-is.
        if not item or "recipe" not in item:
            return {item_id: {"amount_per_minute": items_per_minute}}

        recipe = item["recipe"]
        time_per_unit = recipe.get("time")
        yield_per_recipe = recipe.get("yield")

        # If the recipe has no ingredients (raw resources like "iron-ore"), just return the target amount.
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

    def _calculate_assemblers(self, time_per_unit, recipe_runs_needed_per_minute):
        """
        Calculate how many assemblers are needed to produce the required amount per minute.
        """
        crafting_speed = self.machines_data["assemblers"]["crafting_speed"]
        items_per_second_per_assembler = crafting_speed / time_per_unit
        items_per_minute_per_assembler = items_per_second_per_assembler * 60
        return recipe_runs_needed_per_minute / items_per_minute_per_assembler

    def _calculate_inserters(self, recipe_runs_needed_per_minute):
        """
        Calculate how many inserters are needed to move the required amount of items per minute.
        """
        items_per_second_per_inserter = self.machines_data["inserters"]["ItemsPerSecond"]
        items_per_minute_per_inserter = items_per_second_per_inserter * 60
        return recipe_runs_needed_per_minute / items_per_minute_per_inserter

    def _calculate_belts(self, total_items_needed_per_minute):
        """
        Calculate how many belts are needed to move the required amount of items per minute.
        """
        items_per_second_per_belt = self.machines_data["belts"]["ItemsPerSecond"]
        items_per_minute_per_belt = items_per_second_per_belt * 60
        return total_items_needed_per_minute / items_per_minute_per_belt
    

    def calculate_minimal_grid_size(self,production_output):
        """
        Calculate the minimum grid size required for the production chain
        by summing the number of assemblers, inserters, and belts.
        """
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
        z3_solver = Z3Solver(20,20, production_data)
        # Process the input to place assemblers
        z3_solver.solve()

        self.z3_solver = z3_solver
        
    def display_solution(self):
        self.z3_solver.visualize_factory()
        
        
        
        
def main():
    factorioProductionTree = FactorioProductionTree()
    total_requirements = factorioProductionTree.calculate_production('electronic-circuit', 10)
    print(total_requirements)
    #factorioProductionTree.calculate_minimal_grid_size(total_requirements)
    
    factorioProductionTree.solve(total_requirements)
    factorioProductionTree.display_solution()
   
if __name__ == "__main__":
    main()
