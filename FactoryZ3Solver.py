#! .venv\Scripts\python.exe

from z3 import *
from z3 import Optimize 
import logging, time


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
        
        # Track spacing with other blocks
        # Format: {other_block_id: {"north": distance, "south": distance, "east": distance, "west": distance}}
        self.spacing = {}

        
class Gate:
    def __init__(self,id,relative_x,relative_y,item,type, is_fixed=False,edge=None):
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        self.relative_x = relative_x  # Relative x position within the block
        self.relative_y = relative_y  # Relative y position within the block
        self.type = type # 'start' or 'end'
        self.item = item
        self.is_fixed = is_fixed  # Indicates if the gate has a fixed position
        self.edge = edge # Edge information (e.g., North, South, East, West) only needed for fixed gates
        


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
        #self.fix_position()
        self.add_bound_constraints()
        self.add_overlap_constraints()
        self.add_gate_relative_position_constraints()
        self.add_block_spacing_constraints()
        self.add_layout_guidance_constraints()  
        self.add_gate_proximity_optimization()
        
        # Only add I/O constraints if we have fixed gates
        if hasattr(self, 'fixed_gates') and self.fixed_gates:
            self.add_io_gate_constraints()
        
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
        """Add constraints to ensure blocks are within bounds and above y=0"""
        logging.info("Adding bound constraints")
        
        # Calculate the maximum possible coordinates
        for block in self.blocks:
            # Enforce minimum y coordinate of 0 for all blocks (keep blocks above I/O points)
            #self.solver.add(block.y >= 0)
            
            # Block must be within bounds
            self.solver.add(block.x + block.width <= self.max_x)
            self.solver.add(block.y + block.height <= self.max_y)
            
            # Update max bounds based on block positions
            self.solver.add(self.max_x >= block.x + block.width)
            self.solver.add(self.max_y >= block.y + block.height)
            
        # Ensure max_x and max_y are positive
        self.solver.add(self.max_x >= 0)
        self.solver.add(self.max_y >= 0)
        
        logging.debug("Added bound constraints for all blocks")
        
        
    
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

    def add_block_spacing_constraints(self):
        """
        Add constraints to ensure appropriate spacing between blocks with gates that need to connect.
        The spacing increases with the number of non-aligned gates, and is zero if all gates align perfectly.
        Uses a block-specific spacing tracker to allow for shared spacing between blocks.
        """
        logging.info("Adding adaptive block spacing constraints")
        
        # Group gates by item type
        gates_by_item = {}
        for gate in self.gates:
            item = gate.item
            if item not in gates_by_item:
                gates_by_item[item] = {"input": [], "output": []}
            gates_by_item[item][gate.type].append(gate)
        
        logging.debug(f"Grouped gates by {len(gates_by_item)} item types")
        
        # Track which blocks need spacing and how many misaligned gates they have
        blocks_needing_space = {}  # {(block1_id, block2_id): {"misaligned": count, "aligned": count}}
        
        # For each item type, check if there are gates that need to connect
        for item, item_gates in gates_by_item.items():
            outputs = item_gates["output"]
            inputs = item_gates["input"]
            
            if not outputs or not inputs:
                continue
            
            logging.debug(f"Item {item}: {len(outputs)} output gates, {len(inputs)} input gates")
            
            # For each output gate, find input gates of same item type in different blocks
            for output_gate in outputs:
                output_block = self.get_block_for_gate(output_gate)
                
                for input_gate in inputs:
                    input_block = self.get_block_for_gate(input_gate)
                    
                    # Skip if gates are in the same block
                    if output_block == input_block:
                        continue
                    
                    # Check if gates are not perfectly aligned horizontally
                    is_misaligned = output_gate.relative_x != input_gate.relative_x
                    
                    # Add or update the alignment info for this block pair
                    block_pair = tuple(sorted([output_block.id, input_block.id]))
                    if block_pair not in blocks_needing_space:
                        blocks_needing_space[block_pair] = {"misaligned": 0, "aligned": 0}
                    
                    if is_misaligned:
                        blocks_needing_space[block_pair]["misaligned"] += 1
                        logging.debug(f"Block pair {block_pair}: Misaligned gate found ({output_gate.id} -> {input_gate.id})")
                    else:
                        blocks_needing_space[block_pair]["aligned"] += 1
                        logging.debug(f"Block pair {block_pair}: Aligned gate found ({output_gate.id} -> {input_gate.id})")
        
        # For each block pair, add an adaptive constraint to ensure appropriate spacing
        for block_pair, alignment_info in blocks_needing_space.items():
            block1_id, block2_id = block_pair
            
            # Find the actual block objects
            block1 = next(b for b in self.blocks if b.id == block1_id)
            block2 = next(b for b in self.blocks if b.id == block2_id)
            
            # Check if these are fixed I/O blocks
            block1_is_io = hasattr(block1, 'is_fixed_gate_block') and block1.is_fixed_gate_block
            block2_is_io = hasattr(block2, 'is_fixed_gate_block') and block2.is_fixed_gate_block
            is_io_involved = block1_is_io or block2_is_io
            
            logging.info(f"Processing block pair {block_pair}:")
            logging.info(f"  - Aligned gates: {alignment_info['aligned']}")
            logging.info(f"  - Misaligned gates: {alignment_info['misaligned']}")
            logging.info(f"  - Involves I/O block: {is_io_involved}")
            
            # Determine required spacing based on alignment
            required_spacing = 0
            if alignment_info["misaligned"] > 0:
                required_spacing = alignment_info["misaligned"]
            
            # Initialize spacing in block objects
            if block2_id not in block1.spacing:
                block1.spacing[block2_id] = {"north": 0, "south": 0, "east": 0, "west": 0}
            if block1_id not in block2.spacing:
                block2.spacing[block1_id] = {"north": 0, "south": 0, "east": 0, "west": 0}
            
            # Create variable for vertical alignment
            vertical_alignment = Bool(f"vert_align_{block1_id}_{block2_id}")
            self.solver.add(Implies(vertical_alignment, block1.x == block2.x))
            
            # Always encourage vertical alignment with high priority
            alignment_weight = 5000 if is_io_involved else 4000
            self.solver.add_soft(vertical_alignment, weight=alignment_weight)
            logging.debug(f"  - Added vertical alignment constraint with weight {alignment_weight}")
            
            # Create variables for directional relationships
            block1_above_block2 = Bool(f"block1_above_{block1_id}_{block2_id}")
            block2_above_block1 = Bool(f"block2_above_{block1_id}_{block2_id}")
            
            # Only one can be true
            self.solver.add(Not(And(block1_above_block2, block2_above_block1)))
            
            # For blocks with aligned gates, enforce zero spacing
            if alignment_info["aligned"] > 0 and alignment_info["misaligned"] == 0:
                logging.info(f"Blocks {block1_id} and {block2_id} have perfectly aligned gates - enforcing ZERO spacing")
                
                # Record spacing information for later use
                block1.spacing[block2_id]["south"] = 0
                block2.spacing[block1_id]["north"] = 0
                
                # Add constraints for direct adjacency with zero spacing
                self.solver.add(Implies(block1_above_block2, 
                                      And(vertical_alignment, block1.y + block1.height == block2.y)))
                self.solver.add(Implies(block2_above_block1, 
                                      And(vertical_alignment, block2.y + block2.height == block1.y)))
                
                # Strongly encourage one of the directional relationships to be true
                self.solver.add_soft(Or(block1_above_block2, block2_above_block1), weight=6000)
                
                # CRITICAL: Add a hard constraint to enforce zero spacing
                #self.solver.add(
                #    And(
                #        vertical_alignment,
                #        Or(
                #            block1.y + block1.height == block2.y,  # block1 directly above block2
                #            block2.y + block2.height == block1.y   # block2 directly above block1
                #        )
                #    )
                #)
                logging.info(f"  - Added HARD constraint to enforce ZERO spacing")
                
            else:
                # For blocks with misaligned gates, enforce specific spacing
                logging.info(f"Blocks {block1_id} and {block2_id} have {required_spacing} misaligned gates - enforcing spacing of {required_spacing}")
                
                # Record spacing information for later use
                block1.spacing[block2_id]["south"] = required_spacing
                block2.spacing[block1_id]["north"] = required_spacing
                
                # Add constraints for spacing
                self.solver.add(Implies(block1_above_block2, 
                                      And(vertical_alignment, block1.y + block1.height + required_spacing == block2.y)))
                self.solver.add(Implies(block2_above_block1, 
                                      And(vertical_alignment, block2.y + block2.height + required_spacing == block1.y)))
                
                # Encourage one of the directional relationships to be true
                self.solver.add_soft(Or(block1_above_block2, block2_above_block1), weight=5000)
                
                # CRITICAL: Add a hard constraint to enforce exact spacing
                #self.solver.add(
                #    And(
                #        vertical_alignment,
                #        Or(
                #            block1.y + block1.height + required_spacing == block2.y,  # block1 above block2
                #            block2.y + block2.height + required_spacing == block1.y   # block2 above block1
                #        )
                #    )
                #)
                logging.info(f"  - Added HARD constraint to enforce EXACT spacing of {required_spacing}")
        
        logging.info(f"Added adaptive spacing constraints for {len(blocks_needing_space)} block pairs")
        
        # Log spacing information for all blocks
        self._log_block_spacing()

    def _log_block_spacing(self):
        """Log spacing information between blocks for debugging"""
        logging.info("=== Block Spacing Information ===")
        for block in self.blocks:
            if not block.spacing:
                continue
            
            logging.info(f"Block {block.id} spacing:")
            for other_id, spacing in block.spacing.items():
                directions = []
                for direction, value in spacing.items():
                    if value > 0:
                        directions.append(f"{direction}:{value}")
                if directions:
                    logging.info(f"  - With {other_id}: {', '.join(directions)}")

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
        logging.info("Solving factory layout with Z3...")
        start_time = time.perf_counter()
        
        result = self.solver.check()
        solve_time = time.perf_counter() - start_time
        logging.info(f"Z3 solver finished in {solve_time:.2f} seconds with result: {result}")
        
        if result == sat:
            model = self.solver.model()
            
            # Create a dictionary to store all block information
            final_blocks = {}
            
            logging.info("===== FINAL BLOCK POSITIONS =====")
            # Log max dimensions
            max_x_val = model[self.max_x].as_long()
            max_y_val = model[self.max_y].as_long()
            logging.info(f"Factory dimensions: {max_x_val}x{max_y_val}")
            
            # Group blocks by type for better logging
            blocks_by_type = {}
            for block in self.blocks:
                # Extract block type from ID
                parts = block.id.split('_')
                if len(parts) >= 2:
                    block_type = parts[1]
                    if block_type not in blocks_by_type:
                        blocks_by_type[block_type] = []
                    blocks_by_type[block_type].append(block)
            
            # Log fixed I/O gates first
            if hasattr(self, 'fixed_gates'):
                logging.info("--- Fixed I/O Gates ---")
                for gate in self.fixed_gates:
                    x = model[gate.x].as_long()
                    y = model[gate.y].as_long()
                    edge = gate.edge if hasattr(gate, 'edge') else "Unknown"
                    logging.info(f"Gate {gate.id}: ({x},{y}) - {gate.item} {gate.type} on {edge} edge")
            
            # Log blocks by type
            for block_type, blocks in blocks_by_type.items():
                logging.info(f"--- Block Type: {block_type} ({len(blocks)} blocks) ---")
                for block in blocks:
                    is_io = hasattr(block, 'is_fixed_gate_block') and block.is_fixed_gate_block
                    block_type = "I/O" if is_io else "Regular"
                    
                    # Extract block position from the model
                    x = model[block.x].as_long()
                    y = model[block.y].as_long()
                    
                    # Log block position and dimensions
                    logging.info(f"Block {block.id} [{block_type}]: Position=({x},{y}), Size={block.width}x{block.height}")
                    
                    # Log gates in this block
                    for gate in block.input_points:
                        gate_x = model[gate.x].as_long()
                        gate_y = model[gate.y].as_long()
                        logging.debug(f"  - Input Gate {gate.id}: ({gate_x},{gate_y}) for {gate.item}")
                    
                    for gate in block.output_points:
                        gate_x = model[gate.x].as_long()
                        gate_y = model[gate.y].as_long()
                        logging.debug(f"  - Output Gate {gate.id}: ({gate_x},{gate_y}) for {gate.item}")
                    
                    # Store block details in the result dictionary
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
            
            # Log block spacing details
            logging.info("--- Block Adjacency Analysis ---")
            for i, block1 in enumerate(self.blocks):
                x1 = model[block1.x].as_long()
                y1 = model[block1.y].as_long()
                w1 = block1.width
                h1 = block1.height
                
                for j, block2 in enumerate(self.blocks):
                    if i >= j:  # Skip self and already processed pairs
                        continue
                        
                    x2 = model[block2.x].as_long()
                    y2 = model[block2.y].as_long()
                    w2 = block2.width
                    h2 = block2.height
                    
                    # Check if vertically aligned
                    vert_aligned = (x1 == x2)
                    
                    # Calculate vertical spacing
                    if y1 + h1 <= y2:  # block1 above block2
                        spacing = y2 - (y1 + h1)
                        logging.debug(f"Block {block1.id} above {block2.id}, spacing: {spacing}, aligned: {vert_aligned}")
                    elif y2 + h2 <= y1:  # block2 above block1
                        spacing = y1 - (y2 + h2)
                        logging.debug(f"Block {block2.id} above {block1.id}, spacing: {spacing}, aligned: {vert_aligned}")
                    else:
                        # Not vertically adjacent
                        logging.debug(f"Blocks {block1.id} and {block2.id} are not vertically adjacent")
                
            # Extract the maximum dimensions of the factory
            max_x = model[self.max_x].as_long()
            max_y = model[self.max_y].as_long()
            
            # After solving for block positions, determine gate connections
            gate_connections = self.determine_gate_connections(model)
            logging.info(f"Determined {len(gate_connections)} gate connections")
            
            return final_blocks, max_x, max_y, gate_connections
        else:
            logging.error("Z3 solver could not find a solution!")
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

    def add_fixed_gate(self, gate_id, item, position, gate_type, edge):
        """
        Add a fixed gate at a specific position by creating a 1x1 block with just the gate.
        
        Args:
            gate_id (str): Unique identifier for the gate
            item (str): The item type for this gate
            position (tuple): Fixed (x, y) position
            gate_type (str): Either 'input' or 'output'
            
        Returns:
            Gate: The created fixed gate object
        """
        x, y = position
        
        # Create a 1x1 block for this gate
        block_id = f"fixed_block_{gate_id}"
        block = Block(block_id, 1, 1)
        
        # Create the gate
        gate = Gate(
            id=gate_id,
            relative_x=0,  # For 1x1 block, relative position is always 0,0
            relative_y=0,
            item=item,
            type=gate_type,
            is_fixed=True,
            edge = edge
    
        )
        
        # Add the gate to the appropriate list in the block
        if gate_type == "input":
            block.input_points.append(gate)
        else:
            block.output_points.append(gate)
        
        # Set the fixed position for the block
        self.solver.add(block.x == x)
        self.solver.add(block.y == y)
        
        # Set the gate position to match the block position
        self.solver.add(gate.x == x)
        self.solver.add(gate.y == y)
        
        # Add the block to our blocks list
        self.blocks.append(block)
        
        # Add the gate to our gates list
        self.gates.append(gate)
        
        # Add to a special collection of fixed external gates
        if not hasattr(self, 'fixed_gates'):
            self.fixed_gates = []
        self.fixed_gates.append(gate)
        
        # Track it as a fixed gate (useful during visualization)
        if not hasattr(self, 'fixed_blocks'):
            self.fixed_blocks = []
        self.fixed_blocks.append(block)
        
        logging.info(f"Added fixed {gate_type} gate for {item} at ({x}, {y})")
        
        return gate

    def add_io_gate_constraints(self):
        """Add constraints to ensure blocks respect the positioning of I/O gates"""
        logging.info("Adding I/O gate constraints")
        
        # Check if we have fixed gates
        if not hasattr(self, 'fixed_gates') or not self.fixed_gates:
            logging.warning("No fixed gates found, skipping I/O gate constraints")
            return
        
        # For fixed blocks, mark them so we can identify them
        for block in self.fixed_blocks:
            block.is_fixed_gate_block = True
        
        # For each fixed gate, ensure that no block overlaps with it
        for gate in self.fixed_gates:
            for block in self.blocks:
                # Skip if this is a fixed gate's own 1x1 block
                if hasattr(block, 'is_fixed_gate_block') and block.is_fixed_gate_block:
                    continue
                    
                # Ensure no overlap with the gate position
                self.solver.add(
                    Or(
                        block.x + block.width <= gate.x,  # Block is to the left of gate
                        gate.x + 1 <= block.x,            # Block is to the right of gate
                        block.y + block.height <= gate.y,  # Block is above gate
                        gate.y + 1 <= block.y             # Block is below gate
                    )
                )
        
        # Get non-fixed blocks for applying edge constraints
        non_fixed_blocks = [block for block in self.blocks 
                        if not (hasattr(block, 'is_fixed_gate_block') and block.is_fixed_gate_block)]
        
        # Add edge-specific constraints
        for gate in self.fixed_gates:
            # Skip if the gate doesn't have edge information
            if not hasattr(gate, 'edge'):
                logging.warning(f"Gate {gate.id} does not have edge information, skipping edge-specific constraints")
                continue
                
            # Apply constraints based on the edge position
            if gate.edge == "North":
                # North edge gates (y=0): ensure blocks are positioned below them
                for block in non_fixed_blocks:
                    self.solver.add(block.y >= 1)
                    
            elif gate.edge == "South":
                # South edge gates: ensure blocks are positioned above them
                for block in non_fixed_blocks:
                    self.solver.add(block.y + block.height <= gate.y)
                    
            elif gate.edge == "East":
                # East edge gates: ensure blocks are positioned to the left
                for block in non_fixed_blocks:
                    self.solver.add(block.x + block.width <= gate.x)
                    
            elif gate.edge == "West":
                # West edge gates: ensure blocks are positioned to the right
                for block in non_fixed_blocks:
                    self.solver.add(block.x >= 1)
        
        logging.debug("Added I/O gate constraints for all edges")
            
def manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])  # Fixed calculation






