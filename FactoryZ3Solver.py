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
    def __init__(self,id,width,height):
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
            