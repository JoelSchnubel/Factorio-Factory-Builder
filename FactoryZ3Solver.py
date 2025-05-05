#! .venv\Scripts\python.exe

from z3 import *
from z3 import Optimize 
import time , json
from logging_config import setup_logger
logger = setup_logger("FactoryZ3Solver")


class Block:
    def __init__(self,id,width,height):
        self.width = width
        self.height = height
        
        self.id = id
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')

    
        self.input_points = []
        self.output_points = []
        
        #Track gate patterns to simplify matching
        self.input_items = set()  # Items this block consumes
        self.output_items = set()  # Items this block produces
        
        #Track input/output interfaces (x positions where gates are located)
        self.input_interfaces = {}  # {item: [x_positions]}
        self.output_interfaces = {}  # {item: [x_positions]}
        
        #Connection compatibility with other blocks
        self.compatible_blocks = {}  # {block_id: {"aligned": bool, "items": [items]}}
      
        # Track spacing with other blocks
        # Format: {other_block_id: {"north": distance, "south": distance, "east": distance, "west": distance}}
        self.spacing = {}
        self.default_spacing = 0

        
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
    
        self.aligned_with = []  # List of compatible gates


class FactoryZ3Solver:
    def __init__(self, block_data, output_point):
        logger.info(f"FactoryZ3Solver")
        logger.debug(f"block data: {block_data}")
        
        self.block_data = block_data
        self.output_point = output_point
        self.solver = Optimize()
        
        self.blocks = []
        self.gates = []  
        self.max_x = Int("max_x")
        self.max_y = Int("max_y")
        
        self.item_complexity = self.calculate_item_complexity(self.load_recipe_data())
        
        logger.debug("Solver and data structures initialized")

    
    def load_recipe_data(self):
        try:
            with open('recipes.json', 'r') as file:
                data = json.load(file)
                logger.info("Loaded recipe data successfully")
                return data
        except Exception as e:
            logger.error(f"Error loading recipe data: {e}")
            return []
    
    def build_constraints(self):
        logger.info("Building constraints")
        self.build_blocks()
        #self.fix_position()
        self.add_bound_constraints()
        self.add_overlap_constraints()
        self.add_gate_relative_position_constraints()

        self.add_simplified_block_alignment_constraints()
        
        # Only add I/O constraints if we have fixed gates
        #if hasattr(self, 'fixed_gates') and self.fixed_gates:
        #    self.add_io_gate_constraints()
        
        self.minimize_map()

    def build_blocks(self):
        logger.info("Building blocks")
        for i, key in enumerate(self.block_data.keys()):
            logger.debug(f"Processing block {key}")
            
            # Retrieve block dimensions
            width = self.block_data[key]["tree"].grid_width
            height = self.block_data[key]["tree"].grid_height
            num_factories = self.block_data[key]["num_factories"]
            
            logger.debug(f"Block {key} dimensions: width={width}, height={height}, num_factories={num_factories}")
            
            # Combine input and output information into one structure
            gate_info = {}
            gate_info.update(self.block_data[key]["tree"].input_information)
            gate_info.update(self.block_data[key]["tree"].output_information)

            for factory_index in range(num_factories):
                # Create the block
                block = Block(f"Block_{key}_{i}_{factory_index}", width, height)
                logger.debug(f"Created Block with ID {block.id}")
                
                # Create gates based on type
                for item, data in gate_info.items():
                    if "input" in data:
                        
                        if item not in block.input_interfaces:
                            block.input_interfaces[item] = []
                        block.input_interfaces[item].append(int(data["input"][1]))
                        block.input_items.add(item)
                        
                        input_gate = Gate(
                            id=f"{key}_input_{item}_{i}_{factory_index}",
                            relative_x=int(data["input"][0]),
                            relative_y=int(data["input"][1]),
                            item=item,
                            type="input"
                        )
                        
                        block.input_points.append(input_gate)
                        self.gates.append(input_gate)  # Add to flat list of all gates
                        logger.debug(f"Added input gate {input_gate.id} for item {item}")
                        
                    if "output" in data:
                        
                         # Add to interface tracking
                        if item not in block.output_interfaces:
                            block.output_interfaces[item] = []
                        block.output_interfaces[item].append(int(data["output"][1]))
                        block.output_items.add(item)
                        
                        output_gate = Gate(
                            id=f"{key}_output_{item}_{i}_{factory_index}",
                            relative_x=int(data["output"][0]),
                            relative_y=int(data["output"][1]),
                            item=item,
                            type=
                            "output"
                        )
                        block.output_points.append(output_gate)
                        self.gates.append(output_gate)  # Add to flat list of all gates
                        logger.debug(f"Added output gate {output_gate.id} for item {item}")

                # Append block to the list of blocks
                self.blocks.append(block)
                logger.debug(f"Block {block.id} added to blocks list")
        self._analyze_block_compatibility()
            
    def _analyze_block_compatibility(self):
        """
        Pre-analyze which blocks should be connected based on their input/output patterns.
        This allows us to drastically reduce the number of constraints.
        """
        logger.info("Analyzing block compatibility")
        
        analyzed_pairs = set()
        
        for producer in self.blocks:
            for consumer in self.blocks:
                if producer == consumer:
                    continue
                    
                # Find items produced by 'producer' that are consumed by 'consumer'
                shared_items = producer.output_items.intersection(consumer.input_items)
                
                if not shared_items:
                    continue
                    
                # Create a unique pair identifier (sorted to ensure consistency)
                block_pair = tuple(sorted([producer.id, consumer.id]))
                
                # Skip if we've already analyzed this pair
                if block_pair in analyzed_pairs:
                    logger.debug(f"Skipping already analyzed pair: {block_pair}")
                    continue
                
                analyzed_pairs.add(block_pair)
                
                # For each shared item, check if there are aligned interfaces
                aligned_items = []
                misaligned_items = []
                
                # Count how many gates are aligned for each item
                aligned_gate_count = 0
                
                for item in shared_items:
                    # Check for each output of producer if there's a matching input on consumer
                    is_aligned = False
                    producer_outputs = producer.output_interfaces.get(item, [])
                    consumer_inputs = consumer.input_interfaces.get(item, [])

                    logger.debug(f"  Item {item}:")
                    logger.debug(f"    Producer {producer.id} output x-positions: {producer_outputs}")
                    logger.debug(f"    Consumer {consumer.id} input x-positions: {consumer_inputs}")
                    
                    item_alignments = []
                    # Compare all producer outputs with all consumer inputs to find alignments
                    for p_out_x in producer_outputs:
                        for c_in_x in consumer_inputs:
                            # Check if they have the same relative position
                            if p_out_x == c_in_x:
                                alignment_match = f"Aligned: output_x={p_out_x} == input_x={c_in_x}"
                                item_alignments.append(alignment_match)
                                aligned_gate_count += 1
                                is_aligned = True
                            else:
                                alignment_mismatch = f"Misaligned: output_x={p_out_x} != input_x={c_in_x}"
                                item_alignments.append(alignment_mismatch)
                    
                    # Log all alignment checks for this item
                    for align_detail in item_alignments:
                        logger.debug(f"    {align_detail}")
                    
                    if is_aligned:
                        aligned_items.append(item)
                        logger.debug(f"    ✓ Item {item} is ALIGNED")
                    else:
                        misaligned_items.append(item)
                        logger.debug(f"    ✗ Item {item} is MISALIGNED")
                
                # Calculate required spacing (number of misaligned items minus aligned gates)
                # but ensure it's at least 0
                spacing_needed = max(0, len(misaligned_items) - aligned_gate_count)
                
                # Create compatibility info
                if aligned_items or misaligned_items:
                    producer.compatible_blocks[consumer.id] = {
                        "aligned": len(aligned_items) > 0,
                        "perfect_alignment": len(misaligned_items) == 0 and len(aligned_items) > 0,
                        "aligned_items": aligned_items,
                        "misaligned_items": misaligned_items,
                        "spacing_needed": spacing_needed,
                        "aligned_gate_count": aligned_gate_count
                    }
                    
                    # Set the spacing value in the block's spacing dictionary
                    producer.spacing[consumer.id] = {
                        "north": spacing_needed,
                        "south": spacing_needed,
                        "east": spacing_needed,
                        "west": spacing_needed
                    }
                    
                    logger.debug(f"Block {producer.id} can connect to {consumer.id}:")
                    logger.debug(f"  - Aligned items: {aligned_items}")
                    logger.debug(f"  - Misaligned items: {misaligned_items}")
                    logger.debug(f"  - Aligned gates: {aligned_gate_count}")
                    logger.debug(f"  - Spacing needed: {spacing_needed}")
    
    def add_simplified_block_alignment_constraints(self):
        """
        Add constraints to align compatible blocks vertically with appropriate spacing.
        """
        logger.info("Adding simplified block alignment constraints")
        
        alignment_pairs = []
        
        # For each block, find compatible blocks and add alignment constraints
        for block in self.blocks:
            for other_id, compatibility in block.compatible_blocks.items():
                # Find the other block object
                other_block = next(b for b in self.blocks if b.id == other_id)
                
                # Create a unique identifier for this block pair
                block_pair = tuple(sorted([block.id, other_id]))
                
                # Skip if we've already processed this pair
                if block_pair in alignment_pairs:
                    continue
                
                alignment_pairs.append(block_pair)
                
                # Create vertical alignment variable
                vertical_align = Bool(f"v_align_{block.id}_{other_id}")
                
                # Add implication for vertical alignment
                self.solver.add(Implies(vertical_align, block.x == other_block.x))
                
                # Add strong soft constraint to encourage vertical alignment
                self.solver.add_soft(vertical_align, weight=1000)
                
                # Create variables for relative positions
                block_above = Bool(f"above_{block.id}_{other_id}")
                other_above = Bool(f"above_{other_id}_{block.id}")
                
                # Ensure only one direction is chosen
                self.solver.add(Not(And(block_above, other_above)))
                
                # Force one of them to be true (one block must be above the other)
                self.solver.add(Or(block_above, other_above))
                
         
                 # Get required spacing from the compatibility info or spacing dictionary
                required_spacing = compatibility.get("spacing_needed", 0)
                
                # Add positioning constraints based on calculated spacing
                if compatibility["perfect_alignment"]:
                    # Perfect alignment - no spacing needed
                    self.solver.add(Implies(
                        And(vertical_align, block_above),
                        block.y + block.height == other_block.y
                    ))
                    self.solver.add(Implies(
                        And(vertical_align, other_above),
                        other_block.y + other_block.height == block.y
                    ))
                    
                    # Encourage direct adjacency with high weight for perfectly aligned blocks
                    self.solver.add_soft(And(vertical_align, Or(block_above, other_above)), weight=2000)
                else:
                    # Get required spacing from the spacing dictionary or use default
                    #if other_id in block.spacing:
                    #    required_spacing = block.spacing[other_id]["south"]  # Use south spacing for vertical alignment
                    #else:
                    #    required_spacing = block.default_spacing
                        
                    # Add spacing constraints for blocks with misaligned gates
                    self.solver.add(Implies(
                        And(vertical_align, block_above),
                        block.y + block.height + required_spacing == other_block.y
                    ))
                    self.solver.add(Implies(
                        And(vertical_align, other_above),
                        other_block.y + other_block.height + required_spacing == block.y
                     ))
        
        logger.info(f"Added alignment constraints for {len(alignment_pairs)} block pairs")
    
    
    def fix_position(self):
        self.solver.add(self.blocks[0].x == 0)
        self.solver.add(self.blocks[0].y == 0)
    
    def add_bound_constraints(self):
        """Add constraints to ensure blocks are within bounds and above y=0"""
        logger.info("Adding bound constraints")
        
        # Calculate the maximum possible coordinates
        for block in self.blocks:
            # Enforce minimum y coordinate of 0 for all blocks (keep blocks above I/O points)
            #self.solver.add(block.y >= 0)
            
            self.solver.add(block.y >= 0)
            self.solver.add(block.x >= 0)
            
            # Calculate the maximum coordinates
            self.solver.add(block.x + block.width <= self.max_x)
            self.solver.add(block.y + block.height <= self.max_y)
                
            
            # Block must be within bounds
            #self.solver.add(block.x + block.width <= self.max_x)
            #self.solver.add(block.y + block.height <= self.max_y)
            
            # Update max bounds based on block positions
            #self.solver.add(self.max_x >= block.x + block.width)
            #self.solver.add(self.max_y >= block.y + block.height)
            
        # Ensure max_x and max_y are positive
        self.solver.add(self.max_x >= 0)
        self.solver.add(self.max_y >= 0)
        
        logger.debug("Added bound constraints for all blocks")
        
        
    
    def add_overlap_constraints(self):
        logger.info("Adding overlap constraints")
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
                    logger.debug(f"Added non-overlapping constraints between blocks {block1.id} and {block2.id}")
    
    
    def add_gate_relative_position_constraints(self):
        logger.info("Adding gate relative position constraints")
        for block in self.blocks:
            # Input gates
            for gate in block.input_points:
                self.solver.add(gate.x == block.x + gate.relative_x)
                self.solver.add(gate.y == block.y + gate.relative_y)
                logger.debug(f"Added relative position constraints for input gate {gate.id} in block {block.id}")
            # Output gates
            for gate in block.output_points:
                self.solver.add(gate.x == block.x + gate.relative_x)
                self.solver.add(gate.y == block.y + gate.relative_y)
                logger.debug(f"Added relative position constraints for output gate {gate.id} in block {block.id}")

    def minimize_map(self):
        # Prioritize minimizing width over height to encourage vertical layouts
        width_weight = 2.0  # Higher weight for width to encourage narrow layouts
        height_weight = 0.5  # Lower weight for height to allow taller layouts
        
        self.solver.minimize(width_weight * self.max_x + height_weight * self.max_y)

        
    def solve(self):
        logger.info("Solving factory layout with Z3 using simplified constraints...")
        start_time = time.perf_counter()
        
        result = self.solver.check()
        solve_time = time.perf_counter() - start_time
        logger.info(f"Z3 solver finished in {solve_time:.2f} seconds with result: {result}")
        
        if result == sat:
            model = self.solver.model()
            
            # Extract basic block positions
            final_blocks = self._extract_block_positions(model)
            
            # Extract max dimensions
            max_x_val = model[self.max_x].as_long()
            max_y_val = model[self.max_y].as_long()
            
            # Post-process to determine optimal gate connections
            gate_connections = self._determine_optimal_gate_connections(model, final_blocks)
            
            return final_blocks, max_x_val, max_y_val, gate_connections
        else:
            logger.error("Z3 solver could not find a solution!")
            return None, None, None, None
        
    def _extract_block_positions(self, model):
        """Extract block positions from the solved model"""
        final_blocks = {}
        
        for block in self.blocks:
            x = model[block.x].as_long()
            y = model[block.y].as_long()
            
            # Store gates with their absolute positions
            input_gates = []
            output_gates = []
            
            for gate in block.input_points:
                input_gates.append({
                    "id": gate.id,
                    "item": gate.item,
                    "type": gate.type,
                    "x": model[gate.x].as_long(),
                    "y": model[gate.y].as_long()
                })
            
            for gate in block.output_points:
                output_gates.append({
                    "id": gate.id,
                    "item": gate.item,
                    "type": gate.type,
                    "x": model[gate.x].as_long(),
                    "y": model[gate.y].as_long()
                })
            
            
            # Extract the base item name from block_id
            block_parts = block.id.split("_")
            block_type = block_parts[1] if len(block_parts) >= 3 else block.id
            
            # Get module JSON file path from block_data if available
            module_json_path = None
            if block_type in self.block_data and "json" in self.block_data[block_type]:
                module_json_path = self.block_data[block_type]["json"]
                
            # Store block details
            final_blocks[block.id] = {
                "x": x,
                "y": y,
                "width": block.width,
                "height": block.height,
                "input_points": input_gates,
                "output_points": output_gates,
                "module_json": module_json_path,  # Add the JSON file path
                "block_type": block_type  # Store the base item type for easier reference
            }
            
            logger.debug(f"Block {block.id} position: ({x}, {y}), size: {block.width}x{block.height}")
        
        return final_blocks  
        

    def _determine_optimal_gate_connections(self, model, block_positions):
        """
        Determine optimal gate connections after blocks are positioned.
        This is the post-processing step that finds the best gate-to-gate connections,
        prioritizing connections based on item complexity.
        """
        logger.info("Post-processing: Determining optimal gate connections with complexity prioritization")
        
        connections = []
        
        # Group gates by item type with their solved positions
        gates_by_item = {}
        
        for gate in self.gates:
            item = gate.item
            if item not in gates_by_item:
                gates_by_item[item] = {"input": [], "output": []}
            
            # Get solved position
            x = model[gate.x].as_long()
            y = model[gate.y].as_long()
            
            gate_info = {
                "gate": gate,
                "id": gate.id,
                "x": x,
                "y": y,
                "block": self.get_block_for_gate(gate)
            }
            
            gates_by_item[item][gate.type].append(gate_info)
        
        # Sort items by complexity (lower complexity first)
        items_sorted_by_complexity = sorted(
            gates_by_item.keys(),
            key=lambda item: self.item_complexity.get(item, float('inf'))
        )
        
        logger.info(f"Connection priority order based on complexity: {items_sorted_by_complexity}")
        
        # Process items in order of complexity (lowest first)
        for item in items_sorted_by_complexity:
            item_gates = gates_by_item[item]
            outputs = item_gates["output"]
            inputs = item_gates["input"]
            
            if not outputs or not inputs:
                continue
            
            complexity = self.item_complexity.get(item, "unknown")
            logger.debug(f"Connecting item '{item}' (complexity: {complexity})")
            
            # Sort outputs and inputs by y-coordinate (top to bottom)
            outputs.sort(key=lambda g: g["y"])
            inputs.sort(key=lambda g: g["y"])
            
            # For each output gate, find the best input gate
            for output in outputs:
                if not inputs:
                    break
                    
                # Prioritize gates that are vertically aligned
                aligned_inputs = [inp for inp in inputs if inp["x"] == output["x"]]
                
                if aligned_inputs:
                    # Find closest aligned input by y-distance
                    best_input = min(aligned_inputs, 
                                    key=lambda inp: abs(inp["y"] - output["y"]))
                    connections.append((output["gate"], best_input["gate"]))
                    inputs.remove(best_input)
                    logger.debug(f"Connected aligned gates: {output['id']} → {best_input['id']}")
                else:
                    # If no aligned inputs, find closest by Manhattan distance
                    best_input = min(inputs, 
                                    key=lambda inp: manhattan_distance(
                                        (output["x"], output["y"]), 
                                        (inp["x"], inp["y"])
                                    ))
                    connections.append((output["gate"], best_input["gate"]))
                    inputs.remove(best_input)
                    logger.debug(f"Connected nearest gates: {output['id']} → {best_input['id']}")
        
        logger.info(f"Determined {len(connections)} optimal gate connections")
        return connections
    
    
    def get_block_for_gate(self, gate):
        """Helper method to determine which block a gate belongs to"""
        for block in self.blocks:
            if gate in block.input_points or gate in block.output_points:
                return block
        return None

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
        
        logger.info(f"Added fixed {gate_type} gate for {item} at ({x}, {y})")
        
        return gate

    def add_io_gate_constraints(self):
        """Add constraints to ensure blocks respect the positioning of I/O gates"""
        logger.info("Adding I/O gate constraints with complexity-aware positioning")
        
        # Check if we have fixed gates
        if not hasattr(self, 'fixed_gates') or not self.fixed_gates:
            logger.warning("No fixed gates found, skipping I/O gate constraints")
            return
        
        # For fixed blocks, mark them so we can identify them
        for block in self.fixed_blocks:
            block.is_fixed_gate_block = True
        
        # Group fixed gates by item type
        fixed_gates_by_item = {}
        for gate in self.fixed_gates:
            item = gate.item
            if item not in fixed_gates_by_item:
                fixed_gates_by_item[item] = []
            fixed_gates_by_item[item].append(gate)
            
            # Add item complexity information
            complexity = self.item_complexity.get(item, "unknown")
            logger.info(f"Fixed gate for item '{item}' (complexity: {complexity})")
        
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
                logger.warning(f"Gate {gate.id} does not have edge information, skipping edge-specific constraints")
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
        
        logger.debug("Added I/O gate constraints for all edges with complexity awareness")
                
                
    def calculate_item_complexity(self, recipe_data):
        """
        Calculate the complexity rank of each item in the recipes.
        Complexity is determined by the maximum depth of ingredients needed.
        
        Args:
            recipe_data: List of recipe dictionaries from recipes.json
            
        Returns:
            Dictionary mapping item IDs to their complexity rank
        """
        logger.info("Calculating item complexity rankings")
        
        # Create a dictionary of recipes for easy lookup
        recipes = {item['id']: item for item in recipe_data if 'id' in item}
        
        # Cache to avoid recalculating the same items
        complexity_cache = {}
        
        def get_complexity(item_id):
            # If we've already calculated this item's complexity, return it
            if item_id in complexity_cache:
                return complexity_cache[item_id]
            
            # If item doesn't exist in recipes, default to rank 0
            if item_id not in recipes:
                return 0
                
            recipe = recipes[item_id]
            
            # Base items with no ingredients have complexity 1
            if ('recipe' not in recipe or 
                'ingredients' not in recipe['recipe'] or 
                not recipe['recipe']['ingredients']):
                complexity_cache[item_id] = 1
                return 1
                
            # Calculate complexity based on ingredients
            ingredient_complexities = []
            for ingredient in recipe['recipe']['ingredients']:
                ingredient_id = ingredient['id']
                # Avoid infinite recursion with circular dependencies
                if ingredient_id == item_id:
                    continue
                ingredient_complexities.append(get_complexity(ingredient_id))
            
            # Complexity is 1 + max complexity of ingredients
            if ingredient_complexities:
                complexity = 1 + max(ingredient_complexities)
            else:
                complexity = 1
                
            complexity_cache[item_id] = complexity
            return complexity
        
        # Calculate complexity for all items
        item_complexity = {}
        for item_id in recipes.keys():
            item_complexity[item_id] = get_complexity(item_id)
        
        logger.info(f"Calculated complexity for {len(item_complexity)} items")
        return item_complexity

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
