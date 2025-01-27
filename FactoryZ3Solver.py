#! .venv\Scripts\python.exe

from z3 import *
from z3 import Optimize , Context
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
        
        logging.info(f"FactoryZ3Solver")
        logging.debug(f"block data: {block_data}")
        
        self.block_data = block_data
        
        self.output_point = output_point

       
        self.solver = Optimize()
        # Create variables for assemblers, input inserters, output inserters, and belts
        
        self.blocks = []
        self.gate_connections = []
        self.obstacle_map= []
        
        logging.debug("Solver and data structures initialized")

        
    def build_constraints(self):
        logging.info("Building constraints")
        self.build_blocks()
        self.add_constraints()
        
        
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
                            x=int(data["input"][0]),
                            y=int(data["input"][1]),
                            item=item,
                            type="input"
                        )
                        block.input_points.append(input_gate)
                        
                        logging.debug(f"Added input gate {input_gate.id} for item {item}")
                    if "output" in data:
                        output_gate = Gate(
                            id=f"{key}_output_{item}_{i}_{factory_index}",
                            x=int(data["output"][0]),
                            y=int(data["output"][1]),
                            item=item,
                            type="output"
                        )
                        block.output_points.append(output_gate)
                        logging.debug(f"Added output gate {output_gate.id} for item {item}")


                # Append block to the list of blocks
                self.blocks.append(block)
                logging.debug(f"Block {block.id} added to blocks list")
    
    def add_constraints(self):
        
        logging.info("Adding constraints to solver")
        
        max_x = Int("max_x")
        max_y = Int("max_y")

        self.gate_connections = []  # Ensure this attribute is initialized

        # Boundary and non-overlapping constraints
        for block in self.blocks:
            self.solver.add(block.x >= 0)
            self.solver.add(block.y >= 0)
            self.solver.add(block.x + block.width <= max_x)
            self.solver.add(block.y + block.height <= max_y)
            
            logging.debug(f"Added boundary constraints for block {block.id}")

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

        # Add gate connection constraints
        for block1 in self.blocks:
            for input_gate in block1.input_points:
                for block2 in self.blocks:
                    for output_gate in block2.output_points:
                        if input_gate.item == output_gate.item:
                            # Connection variable: 1 if connected, 0 otherwise
                            
                            
                            connect_var = Int(f"connect_{input_gate.id}_{output_gate.id}")
                            self.solver.add(Or(connect_var == 0, connect_var == 1))

                            # Distance variable
                            distance = Int(f"distance_{input_gate.id}_{output_gate.id}")
                            self.solver.add(
                                distance ==
                                If(connect_var == 1,
                                z3.Abs(input_gate.x - output_gate.x) + z3.Abs(input_gate.y - output_gate.y),
                                0)
                            )

                            # Ensure each gate is used only once
                            self.solver.add(Sum([connect_var for connect_var, _ in self.gate_connections]) <= 1)

                            # Store connection variables and distances for minimization
                            self.gate_connections.append((connect_var, distance))
                            logging.debug(f"Added gate connection constraints between gates {input_gate.id} and {output_gate.id}")


        # Minimize map size
        self.solver.minimize(max_x * max_y)
        logging.debug("Minimized map size constraint added")
        # Minimize total gate connection distance
        total_distance = Sum([dist for _, dist in self.gate_connections])
        
        
        self.solver.minimize(total_distance)
        logging.debug("Minimized total gate connection distance constraint added")



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
        
        
        
