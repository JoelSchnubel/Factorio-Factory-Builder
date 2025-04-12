#! .venv\Scripts\python.exe

from z3 import *
from z3 import Optimize 
import logging


logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for detailed information, change to INFO for less verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("factory_log.log", mode='w'),  # Specify the log file name
    ]
)


class Block:
    def __init__(self,id,width,height):
        self.width = width
        self.height = height
        
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')

    
        self.input_points = []
        self.output_points = []

        
class Gate:
    def __init__(self,id,relative_x,relative_y,item,type):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        self.relative_x = relative_x  # Relative x position within the block
        self.relative_y = relative_y  # Relative y position within the block
        self.type = type # 'start' or 'end'
        self.item = item
        self.merge = Bool(f'{id}_merge')  # Boolean to indicate if the gate is merged


        
class FactoryZ3Solver:
    def __init__(self,block_data,output_point):
        
        logging.info(f"FactoryZ3Solver")
        logging.debug(f"block data: {block_data}")
        
        self.block_data = block_data
        
        self.output_point = output_point

       
        self.solver = Optimize()
        # Create variables for assemblers, input inserters, output inserters, and belts
        
        self.blocks = []
        self.gate_connections = []
        self.obstacle_map= []
        
        self.max_x = Int("max_x")
        self.max_y = Int("max_y")
        
        logging.debug("Solver and data structures initialized")

        
    def build_constraints(self):
        logging.info("Building constraints")
        self.build_blocks()
        self.fix_position()
        self.add_bound_constraints()
        self.add_overlap_constraints()
        self.add_gate_constraints()
        self.add_gate_relative_position_constraints()
        self.minimize_unmerged_gates()
        self.minimize_map()
        
    
        
    def build_blocks(self):
        logging.info("Building blocks")
        for i, key in enumerate(self.block_data.keys()):
            
            logging.debug(f"Processing block {key}")
            
            # Retrieve block dimensions
            width = self.block_data[key]["tree"].grid_width
            height = self.block_data[key]["tree"].grid_height
            
            num_factories = self.block_data[key]["num_factories"]
            
            logging.debug(f"Block {key} dimensions: width={width}, height={height}, num_factories={num_factories}")
            
           
            # Combine input and output information into one structure
            gate_info = {}
            gate_info.update(self.block_data[key]["tree"].input_information)
            gate_info.update(self.block_data[key]["tree"].output_information)

            for factory_index in range(num_factories):
                
                # Create the block
                block = Block(f"Block_{key}_{i}_{factory_index}", width, height)
                logging.debug(f"Created Block with ID {block.id}")
                # Create gates based on type
                for item, data in gate_info.items():
                    if "input" in data:
                        input_gate = Gate(
                            id=f"{key}_input_{item}_{i}_{factory_index}",
                            relative_x=int(data["input"][0]),
                            relative_y=int(data["input"][1]),
                            item=item,
                            type="input"
                        )
                        block.input_points.append(input_gate)
                        
                        logging.debug(f"Added input gate {input_gate.id} for item {item}")
                    if "output" in data:
                        output_gate = Gate(
                            id=f"{key}_output_{item}_{i}_{factory_index}",
                            relative_x=int(data["output"][0]),
                            relative_y=int(data["output"][1]),
                            item=item,
                            type="output"
                        )
                        block.output_points.append(output_gate)
                        logging.debug(f"Added output gate {output_gate.id} for item {item}")


                # Append block to the list of blocks
                self.blocks.append(block)
                logging.debug(f"Block {block.id} added to blocks list")
    
    
    def fix_position(self):
        self.solver.add(self.blocks[0].x == 0)
        self.solver.add(self.blocks[0].y == 0)
    
    def add_bound_constraints(self):
        logging.info("Adding bound constraints")
        # Boundary and non-overlapping constraints
        
        for block in self.blocks:
            self.solver.add(block.x >= 0)
            self.solver.add(block.y >= 0)
            self.solver.add(block.x + block.width <= self.max_x)
            self.solver.add(block.y + block.height <= self.max_y)
        
        
    
    def add_overlap_constraints(self):
        logging.info("Adding overlap constraints")
        for i, block1 in enumerate(self.blocks):
                for j, block2 in enumerate(self.blocks):
                    if i >= j:
                        continue
                    self.solver.add(
                        Or(
                            block1.x + block1.width <= block2.x,
                            block2.x + block2.width <= block1.x,
                            block1.y + block1.height <= block2.y,
                            block2.y + block2.height <= block1.y
                        )
                    )
                    logging.debug(f"Added non-overlapping constraints between blocks {block1.id} and {block2.id}")
    
    
    def add_gate_relative_position_constraints(self):
        logging.info("Adding gate relative position constraints")
        for block in self.blocks:
            # Input gates
            for gate in block.input_points:
                self.solver.add(gate.x == block.x + gate.relative_x)
                self.solver.add(gate.y == block.y + gate.relative_y)
                logging.debug(f"Added relative position constraints for input gate {gate.id} in block {block.id}")
            # Output gates
            for gate in block.output_points:
                self.solver.add(gate.x == block.x + gate.relative_x)
                self.solver.add(gate.y == block.y + gate.relative_y)
                logging.debug(f"Added relative position constraints for output gate {gate.id} in block {block.id}")
                
                
    def add_gate_constraints(self):
        logging.info("Adding gate constraints")
        self.gate_connections = []  # Ensure this attribute is initialized

        for block1 in self.blocks:
            # Skip input gates of the first block
            if block1 == self.blocks[0]:
                continue
            
            for input_gate in block1.input_points:
                for block2 in self.blocks:
                    if block1 == block2:
                        continue  # Skip same block
                    for output_gate in block2.output_points:
                        if input_gate.item == output_gate.item:
                            
                            adjacency_constraint = Or(
                            And(input_gate.x == output_gate.x + 1, input_gate.y == output_gate.y),
                            And(input_gate.x == output_gate.x - 1, input_gate.y == output_gate.y),
                            And(input_gate.x == output_gate.x, input_gate.y == output_gate.y + 1),
                            And(input_gate.x == output_gate.x, input_gate.y == output_gate.y - 1)
                            )
                            
                            self.solver.add(Implies(input_gate.merge, adjacency_constraint))
                            self.gate_connections.append((input_gate, output_gate))


    def minimize_unmerged_gates(self):
        logging.info("Minimizing unmerged gates")
        # Collect all gate.merge variables (Z3 Boolean expressions)
        merged_gates = []
        for block in self.blocks:
            for gate in block.input_points + block.output_points:
                merged_gates.append(gate.merge)  # gate.merge is already a Z3 BoolRef

        # Maximize the sum of merged gates
        self.solver.maximize(Sum([If(merge, 1, 0) for merge in merged_gates]))
        logging.debug("Minimized unmerged gates constraint added")

       
    

    def minimize_map(self):
        self.solver.add(self.max_x < self.max_y)
        self.solver.minimize(self.max_x * self.max_y)
        

    def solve(self):
        if self.solver.check() == sat:
            model = self.solver.model()
            
            # Create a dictionary to store all block information
            final_blocks = {}
            
            for block in self.blocks:
                # Extract block position from the model
                x = model[block.x].as_long()
                y = model[block.y].as_long()
                
                # Store block details
                final_blocks[block.id] = {
                    "x": x,
                    "y": y,
                    "width": block.width,
                    "height": block.height,
                    "input_points": [
                        {
                            "id": gate.id,
                            "item": gate.item,
                            "type": gate.type,
                            "x": model[gate.x].as_long(),  # Relative to block position
                            "y": model[gate.y].as_long()  # Relative to block position
                        }
                        for gate in block.input_points
                    ],
                    "output_points": [
                        {
                            "id": gate.id,
                            "item": gate.item,
                            "type": gate.type,
                            "x": model[gate.x].as_long(),  # Relative to block position
                            "y": model[gate.y].as_long()   # Relative to block position
                        }
                        for gate in block.output_points
                    ]
                }
            
            # Extract the maximum dimensions of the factory
            max_x = model[Int("max_x")].as_long()
            max_y = model[Int("max_y")].as_long()
            
            return final_blocks, max_x, max_y
        else:
            return None, None, None
        
        
        
