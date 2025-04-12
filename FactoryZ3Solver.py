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


class FactoryZ3Solver:
    def __init__(self, block_data, output_point):
        logging.info(f"FactoryZ3Solver")
        logging.debug(f"block data: {block_data}")
        
        self.block_data = block_data
        self.output_point = output_point
        self.solver = Optimize()
        
        self.blocks = []
        self.gates = []  # Store all gates in a flat list for easier post-processing
        self.max_x = Int("max_x")
        self.max_y = Int("max_y")
        
        logging.debug("Solver and data structures initialized")

    def build_constraints(self):
        logging.info("Building constraints")
        self.build_blocks()
        self.fix_position()
        self.add_bound_constraints()
        self.add_overlap_constraints()
        self.add_gate_relative_position_constraints()
        self.add_layout_guidance_constraints()  # Add this line
        self.add_gate_proximity_optimization()
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
                        self.gates.append(input_gate)  # Add to flat list of all gates
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
                        self.gates.append(output_gate)  # Add to flat list of all gates
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
                
    def add_gate_proximity_optimization(self):
        """
        Add optimization constraints to minimize distances between gates of the same item type
        that need to be connected between different blocks.
        """
        logging.info("Adding gate proximity optimization constraints")
        
        # Group gates by item type
        gates_by_item = {}
        for gate in self.gates:
            item = gate.item
            if item not in gates_by_item:
                gates_by_item[item] = {"input": [], "output": []}
            
            gates_by_item[item][gate.type].append(gate)
        
        # Create distance variables for each potential connection
        gate_distances = []
        
        for item, item_gates in gates_by_item.items():
            outputs = item_gates["output"]
            inputs = item_gates["input"]
            
            # Skip if no outputs or no inputs for this item
            if not outputs or not inputs:
                continue
            
            # For each output gate, create distance variables to all input gates of the same item type
            for output_gate in outputs:
                for input_gate in inputs:
                    # Skip gates within the same block (no need to connect them)
                    if self.get_block_for_gate(output_gate) == self.get_block_for_gate(input_gate):
                        continue
                    
                    # Create variables for Manhattan distance components
                    distance_x = Int(f"dist_x_{output_gate.id}_{input_gate.id}")
                    distance_y = Int(f"dist_y_{output_gate.id}_{input_gate.id}")
                    total_distance = Int(f"dist_{output_gate.id}_{input_gate.id}")
                    
                    # Add constraints to calculate absolute differences
                    self.solver.add(
                        distance_x == If(output_gate.x > input_gate.x, 
                                       output_gate.x - input_gate.x, 
                                       input_gate.x - output_gate.x)
                    )
                    self.solver.add(
                        distance_y == If(output_gate.y > input_gate.y, 
                                       output_gate.y - input_gate.y, 
                                       input_gate.y - output_gate.y)
                    )
                    
                    # Total Manhattan distance
                    self.solver.add(total_distance == distance_x + distance_y)
                    
                    # Add this distance to our collection
                    gate_distances.append(total_distance)
        
        # Create a single variable representing the sum of all gate distances
        if gate_distances:
            self.total_gate_distance = Int("total_gate_distance")
            self.solver.add(self.total_gate_distance == Sum(gate_distances))
            logging.debug(f"Created {len(gate_distances)} gate distance variables")

    def get_block_for_gate(self, gate):
        """Helper method to determine which block a gate belongs to"""
        for block in self.blocks:
            if gate in block.input_points or gate in block.output_points:
                return block
        return None

    def add_layout_guidance_constraints(self):
        """Add constraints to guide the layout toward a vertical arrangement"""
        logging.info("Adding vertical layout guidance constraints")
        
        # Group blocks by type to encourage blocks of the same type to form columns
        blocks_by_type = {}
        for block in self.blocks:
            # Extract block type from ID
            parts = block.id.split('_')
            if len(parts) >= 2:
                block_type = parts[1]
                if block_type not in blocks_by_type:
                    blocks_by_type[block_type] = []
                blocks_by_type[block_type].append(block)
        
        # For each block type with multiple instances, encourage vertical alignment
        for block_type, blocks in blocks_by_type.items():
            if len(blocks) <= 1:
                continue
            
            # Add soft constraints to encourage blocks of same type to align vertically
            for i in range(len(blocks)):
                for j in range(i+1, len(blocks)):
                    block1 = blocks[i]
                    block2 = blocks[j]
                    
                    # Create vertical alignment variable
                    align_v = Bool(f"align_v_{block1.id}_{block2.id}")
                    
                    # Add constraint for vertical alignment (same x coordinate)
                    self.solver.add(Implies(align_v, block1.x == block2.x))
                    
                    # Add soft constraint to strongly encourage vertical alignment
                    self.solver.add_soft(align_v, weight=1000)
                    
                    # Add constraint to encourage sequential vertical placement
                    # This makes block2's y coordinate equal to block1's y + block1's height
                    # when blocks are vertically aligned
                    seq_placement = Bool(f"seq_v_{block1.id}_{block2.id}")
                    self.solver.add(Implies(
                        And(align_v, seq_placement),
                        block2.y == block1.y + block1.height
                    ))
                    self.solver.add_soft(seq_placement, weight=500)
        
        # Generally encourage vertical factory shape (height > width)
        self.vertical_shape = Bool("vertical_factory_shape")
        self.solver.add(Implies(self.vertical_shape, self.max_y > self.max_x))
        self.solver.add_soft(self.vertical_shape, weight=2000)

    def minimize_map(self):
        # Prioritize minimizing width over height to encourage vertical layouts
        width_weight = 2.0  # Higher weight for width to encourage narrow layouts
        height_weight = 0.5  # Lower weight for height to allow taller layouts
        
        self.solver.minimize(width_weight * self.max_x + height_weight * self.max_y)
        
        # Minimize the total distance between connected gates
        if hasattr(self, 'total_gate_distance'):
            gate_distance_weight = 1.5  # Weight for gate distances
            self.solver.minimize(gate_distance_weight * self.total_gate_distance)
            logging.info(f"Minimizing width more than height, plus gate distances")
    
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
                            "x": model[gate.x].as_long(),
                            "y": model[gate.y].as_long()
                        }
                        for gate in block.input_points
                    ],
                    "output_points": [
                        {
                            "id": gate.id,
                            "item": gate.item,
                            "type": gate.type,
                            "x": model[gate.x].as_long(),
                            "y": model[gate.y].as_long()
                        }
                        for gate in block.output_points
                    ]
                }
            
            # Extract the maximum dimensions of the factory
            max_x = model[self.max_x].as_long()
            max_y = model[self.max_y].as_long()
            
            # After solving for block positions, determine gate connections
            gate_connections = self.determine_gate_connections(model)
            
            return final_blocks, max_x, max_y, gate_connections
        else:
            return None, None, None, None
            
    def determine_gate_connections(self, model):
        """
        Determine which gates should be connected after block positions are fixed.
        
        Args:
            model: Z3 model with solved positions
            
        Returns:
            List of gate connections (pairs of gates that should be connected)
        """
        connections = []
        
        # Group gates by item type
        gates_by_item = {}
        for gate in self.gates:
            item = gate.item
            if item not in gates_by_item:
                gates_by_item[item] = {"input": [], "output": []}
                
            actual_x = model[gate.x].as_long()
            actual_y = model[gate.y].as_long()
            
            gate_data = {
                "gate": gate,
                "id": gate.id,
                "x": actual_x,
                "y": actual_y
            }
            
            gates_by_item[item][gate.type].append(gate_data)
        
        # For each item type, connect output gates to input gates
        for item, item_gates in gates_by_item.items():
            outputs = item_gates["output"]
            inputs = item_gates["input"]
            
            # Skip if no outputs or no inputs
            if not outputs or not inputs:
                continue
                
            # Find nearest connections (greedy approach)
            remaining_inputs = inputs.copy()
            
            for output in outputs:
                if not remaining_inputs:
                    break
                
                # Find closest input gate
                best_distance = float('inf')
                best_input = None
                best_index = -1
                
                for i, input_gate in enumerate(remaining_inputs):
                    dist = manhattan_distance(
                        (output["x"], output["y"]),
                        (input_gate["x"], input_gate["y"])
                    )
                    if dist < best_distance:
                        best_distance = dist
                        best_input = input_gate
                        best_index = i
                
                if best_input:
                    connections.append((output["gate"], best_input["gate"]))
                    remaining_inputs.pop(best_index)
        
        return connections

def manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])






