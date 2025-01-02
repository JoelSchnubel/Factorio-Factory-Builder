from z3 import *
from z3 import Optimize , Context
import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for detailed information, change to INFO for less verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("factory_log.log", mode='w'),  # Specify the log file name
    ]
)


class Block:
    def __init__(self,id,width,height,gates=[]):
        self.width = width
        self.height = height
        
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        
        
        self.input_points = []
        self.output_points = []

        
class Gate:
    def __init__(self,id,x,y,item,type):
        self.id = id
        self.x = Int(f'{id}_{x}')
        self.y = Int(f'{id}_{y}')
        self.type = type # 'start' or 'end'
        self.item = item

       
        
        
class FactoryZ3Solver:
    def __init__(self,block_data,output_point):
    
        self.block_data = block_data
        self.output_point = output_point
       
        self.solver = Optimize()
        # Create variables for assemblers, input inserters, output inserters, and belts
        

        self.obstacle_map= []
        
        self.blocks = []
        
        
    def build_blocks(self):
        for i, key in enumerate(self.block_data.keys()):
            # Retrieve block dimensions
            width = self.block_data[key]["tree"].grid_width
            height = self.block_data[key]["tree"].grid_height

            # Create the block
            block = Block(f"Block_{key}_{i}", width, height)

            # Combine input and output information into one structure
            gate_info = {}
            gate_info.update(self.block_data[key]["tree"].input_information)
            gate_info.update(self.block_data[key]["tree"].output_information)

            # Create gates based on type
            for item, data in gate_info.items():
                if "input" in data:
                    input_gate = Gate(
                        id=f"{key}_input_{item}",
                        x=data["input"][0],
                        y=data["input"][1],
                        item=item,
                        gate_type="input"
                    )
                    block.input_points.append(input_gate)
                if "output" in data:
                    output_gate = Gate(
                        id=f"{key}_output_{item}",
                        x=data["output"][0],
                        y=data["output"][1],
                        item=item,
                        gate_type="output"
                    )
                    block.output_points.append(output_gate)

            # Append block to the list of blocks
            self.blocks.append(block)

    
    def add_constraints(self):
        max_x = Int("max_x")
        max_y = Int("max_y")

        # Boundary and non-overlapping constraints
        for block in self.blocks:
            self.solver.add(block.x >= 0)
            self.solver.add(block.y >= 0)
            self.solver.add(block.x + block.width <= max_x)
            self.solver.add(block.y + block.height <= max_y)

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


        # Add gate connection constraints
        for block1 in self.blocks:
            for input_gate in block1.input_points:
                for block2 in self.blocks:
                    for output_gate in block2.output_points:
                        if input_gate.item == output_gate.item:
                            # Connection variable: 1 if connected, 0 otherwise
                            connect_var = Int(f"connect_{input_gate.id}_{output_gate.id}")
                            self.solver.add(If(connect_var == 1, True, False))

                            # Distance variable
                            distance = Int(f"distance_{input_gate.id}_{output_gate.id}")
                            self.solver.add(
                                distance == 
                                If(connect_var == 1, 
                                   abs(input_gate.x - output_gate.x) + abs(input_gate.y - output_gate.y), 
                                   0)
                            )
                            
                            # Ensure each gate is used only once
                            self.solver.add(Sum([connect_var for _ in self.gate_connections]) <= 1)
                            self.gate_connections.append((connect_var, distance))

        # Minimize map size
        self.solver.minimize(max_x * max_y)

        # Minimize total gate connection distance
        total_distance = Sum([dist for _, dist in self.gate_connections])
        self.solver.minimize(total_distance)


    def solve(self):
        if self.solver.check() == sat:
            model = self.solver.model()
            block_positions = {
                block.id: {
                    "x": model[block.x].as_long(),
                    "y": model[block.y].as_long()
                }
                for block in self.blocks
            }
            gate_connections = {
                f"{input_gate.id}->{output_gate.id}": model[connect_var].as_long()
                for connect_var, _ in self.gate_connections
            }
            return block_positions, gate_connections
        else:
            return None, None


    def solve(self):
        if self.solver.check() == sat:
            model = self.solver.model()
            return {
                block.id: {
                    "x": model[block.x].as_long(),
                    "y": model[block.y].as_long()
                }
                for block in self.blocks
            }, model[Int("max_x")].as_long(), model[Int("max_y")].as_long()
        else:
            return None, None, None