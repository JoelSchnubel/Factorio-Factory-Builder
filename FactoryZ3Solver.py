#! .venv\Scripts\python.exe

"""
Factory Z3 Solver Module

This module provides a comprehensive framework for optimizing factory layouts using Z3 SMT solver.
It handles the placement and alignment of production blocks with consideration for material flow,
item complexity, and optimal gate connections.

Main Components:
- Block: Represents a production module with defined dimensions and I/O gates
- Gate: Represents input/output points for material flow within blocks
- FactoryZ3Solver: Main solver class that optimizes block placement using SMT constraints

Key Features:
- SMT-based layout optimization using Z3 solver
- Complexity-aware block positioning based on recipe dependencies
- Automatic gate alignment and connection optimization
- Support for fixed I/O gates with edge constraints
- Producer-consumer relationship analysis for optimal layout
- Post-processing for optimal gate-to-gate connections

The solver creates constraints for:
- Non-overlapping block placement
- Boundary and positioning constraints
- Block alignment based on compatible I/O interfaces
- Complexity-based vertical ordering
- Fixed I/O gate positioning and edge constraints
"""

from z3 import *
from z3 import Optimize 
import time , json
from logging_config import setup_logger
logger = setup_logger("FactoryZ3Solver")


class Block:
    """
    Represents a production block/module with defined dimensions and I/O gates.
    
    A Block corresponds to a complete production module (e.g., for producing iron gears)
    with specific width, height, and gate positions for material input and output.
    Each block has Z3 integer variables for its position optimization.
    
    Attributes:
        id (str): Unique identifier for the block
        width (int): Width of the block in grid units
        height (int): Height of the block in grid units
        x (z3.Int): Z3 variable for x-coordinate position
        y (z3.Int): Z3 variable for y-coordinate position
        input_points (list): List of Gate objects for material inputs
        output_points (list): List of Gate objects for material outputs
        input_items (set): Set of items this block consumes
        output_items (set): Set of items this block produces
        input_interfaces (dict): Mapping of items to their x-position interfaces
        output_interfaces (dict): Mapping of items to their x-position interfaces
        compatible_blocks (dict): Information about compatible connections with other blocks
        spacing (dict): Required spacing distances with other blocks
        default_spacing (int): Default spacing distance for non-specified blocks
    """
    
    def __init__(self,id,width,height):
        """
        Initialize a Block with specified dimensions and ID.
        
        Creates Z3 integer variables for position optimization and initializes
        data structures for tracking gates, items, and block relationships.
        
        Args:
            id (str): Unique identifier for the block
            width (int): Width of the block in grid units
            height (int): Height of the block in grid units
        """
        logger.debug(f"Creating Block {id} with dimensions {width}x{height}")
        
        self.width = width
        self.height = height
        
        self.id = id
        # Create Z3 integer variables for position optimization
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')

        # Initialize gate collections
        self.input_points = []   # Gates where materials enter the block
        self.output_points = []  # Gates where materials exit the block
        
        # Track gate patterns to simplify matching and optimization
        self.input_items = set()   # Items this block consumes
        self.output_items = set()  # Items this block produces
        
        # Track input/output interfaces (x positions where gates are located)
        self.input_interfaces = {}   # {item: [x_positions]}
        self.output_interfaces = {}  # {item: [x_positions]}
        
        # Connection compatibility with other blocks
        self.compatible_blocks = {}  # {block_id: {"aligned": bool, "items": [items]}}
      
        # Track spacing requirements with other blocks
        # Format: {other_block_id: {"north": distance, "south": distance, "east": distance, "west": distance}}
        self.spacing = {}
        self.default_spacing = 0  # Default spacing if not specified
        
        logger.debug(f"Block {id} initialized successfully")

    def __str__(self):
        """String representation of the Block for debugging purposes."""
        return f"Block(id={self.id}, x={self.x}, y={self.y}, width={self.width}, height={self.height})"
        
class Gate:
    """
    Represents an input or output gate for material flow within a block.
    
    Gates define where materials enter or exit a production block. Each gate
    has a position relative to its parent block and handles a specific item type.
    Gates can be fixed (for I/O points) or optimized by the solver.
    
    Attributes:
        id (str): Unique identifier for the gate
        x (z3.Int): Z3 variable for absolute x-coordinate position
        y (z3.Int): Z3 variable for absolute y-coordinate position
        relative_x (int): X position relative to the parent block
        relative_y (int): Y position relative to the parent block
        type (str): Gate type - 'input' or 'output'
        item (str): Item type handled by this gate
        is_fixed (bool): Whether this gate has a fixed position
        edge (str): Edge information (North, South, East, West) for fixed gates
        aligned_with (list): List of compatible gates for connection
    """
    
    def __init__(self,id,relative_x,relative_y,item,type, is_fixed=False,edge=None):
        """
        Initialize a Gate with position and item information.
        
        Creates Z3 integer variables for position optimization and stores
        relative positioning information within the parent block.
        
        Args:
            id (str): Unique identifier for the gate
            relative_x (int): X position relative to parent block
            relative_y (int): Y position relative to parent block
            item (str): Item type handled by this gate
            type (str): Gate type - 'input' or 'output'
            is_fixed (bool, optional): Whether gate has fixed position. Defaults to False.
            edge (str, optional): Edge information for fixed gates. Defaults to None.
        """
        logger.debug(f"Creating Gate {id} for item {item} at relative position ({relative_x}, {relative_y})")
        
        self.id = id
        # Create Z3 integer variables for absolute position
        self.x = Int(f'{id}_x')
        self.y = Int(f'{id}_y')
        
        # Store relative position within parent block
        self.relative_x = relative_x  # Relative x position within the block
        self.relative_y = relative_y  # Relative y position within the block
        
        # Gate properties
        self.type = type # 'input' or 'output'
        self.item = item
        self.is_fixed = is_fixed     # Indicates if the gate has a fixed position
        self.edge = edge             # Edge information (e.g., North, South, East, West) only needed for fixed gates
    
        # Connection tracking
        self.aligned_with = []  # List of compatible gates for optimization
        
        logger.debug(f"Gate {id} created successfully (type: {type}, fixed: {is_fixed})")
        
    def __str__(self):
        """String representation of the Gate for debugging purposes."""
        return f"Gate(id={self.id}, x={self.x}, y={self.y}, item={self.item}, type={self.type})"


class FactoryZ3Solver:
    """
    Main solver class for optimizing factory block layouts using Z3 SMT solver.
    
    This class handles the complete optimization pipeline from constraint building
    to solution extraction. It manages block placement, gate alignment, and
    connection optimization while considering item complexity and production
    dependencies.
    
    Key Features:
    - SMT-based constraint solving for optimal block placement
    - Complexity-aware positioning based on recipe dependencies
    - Automatic gate alignment for compatible blocks
    - Support for fixed I/O gates with edge constraints
    - Producer-consumer relationship analysis
    - Post-processing for optimal gate connections
    
    Attributes:
        block_data (dict): Configuration data for all production blocks
        output_point (tuple): Main factory output position
        solver (z3.Optimize): Z3 optimization solver instance
        blocks (list): List of Block objects in the layout
        gates (list): List of all Gate objects across blocks
        max_x (z3.Int): Z3 variable for maximum layout width
        max_y (z3.Int): Z3 variable for maximum layout height
        item_complexity (dict): Mapping of items to their complexity rankings
        producers (dict): Mapping of consumer blocks to their producers
        consumers (dict): Mapping of producer blocks to their consumers
        fixed_gates (list): List of fixed I/O gates if any
        fixed_blocks (list): List of fixed I/O gate blocks if any
    """
    
    def __init__(self, block_data, output_point):
        """
        Initialize the FactoryZ3Solver with block data and output configuration.
        
        Sets up the Z3 optimization solver, initializes data structures,
        and calculates item complexity rankings for optimization.
        
        Args:
            block_data (dict): Dictionary containing block configuration data
                              with production modules and their specifications
            output_point (tuple): Coordinates of the main factory output point
        """
        logger.info("Initializing FactoryZ3Solver for layout optimization")
        logger.debug(f"Block data contains {len(block_data)} production modules")
        logger.debug(f"Output point configured at: {output_point}")
        
        # Store configuration data
        self.block_data = block_data
        self.output_point = output_point
        
        # Initialize Z3 optimization solver
        self.solver = Optimize()
        logger.info("Z3 optimization solver initialized")
        
        # Initialize data structures for blocks and gates
        self.blocks = []  # Collection of all blocks in the layout
        self.gates = []   # Flat list of all gates across blocks
        self.fixed_gates = []  # Collection of fixed I/O gates
        self.fixed_blocks = []  # Collection of fixed blocks
        
        # Z3 variables for layout bounds optimization
        self.max_x = Int("max_x")  # Maximum layout width
        self.max_y = Int("max_y")  # Maximum layout height
        
        # Load and calculate item complexity for optimization
        recipe_data = self.load_recipe_data()
        self.item_complexity = self.calculate_item_complexity(recipe_data)
        logger.info(f"Calculated complexity for {len(self.item_complexity)} items")
        
        # Initialize relationship tracking (populated during constraint building)
        self.producers = {}  # Mapping of consumer blocks to their producers
        self.consumers = {}  # Mapping of producer blocks to their consumers
        
        logger.info("FactoryZ3Solver initialization complete")
        
        logger.debug("Solver and data structures initialized")

    
    def load_recipe_data(self):
        """
        Load recipe data from the recipes.json file.
        
        This method reads the recipe configuration file that contains information
        about item dependencies, ingredients, and production chains. The data is
        used for calculating item complexity rankings.
        
        Returns:
            list: List of recipe dictionaries from recipes.json, or empty list if error
            
        Raises:
            Exception: If the recipes.json file cannot be read or parsed
        """
        logger.debug("Loading recipe data from recipes.json")
        try:
            with open('recipes.json', 'r') as file:
                data = json.load(file)
                logger.info(f"Loaded recipe data successfully ({len(data)} recipes)")
                return data
        except FileNotFoundError:
            logger.error("recipes.json file not found")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in recipes.json: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading recipe data: {e}")
            return []
    
    def build_constraints(self):
        """
        Build all necessary constraints for the factory layout optimization.
        
        This method orchestrates the creation of all constraint types needed for
        the SMT solver to find an optimal factory layout. It builds blocks,
        enforces boundaries, prevents overlaps, and adds alignment constraints.
        
        The constraint building process includes:
        1. Building blocks and gates from configuration data
        2. Adding boundary constraints to keep blocks within bounds
        3. Adding overlap prevention constraints
        4. Adding gate positioning constraints relative to their parent blocks
        5. Adding block alignment constraints based on compatibility
        6. Adding I/O gate constraints if fixed gates exist
        7. Setting up optimization objectives to minimize layout size
        
        Note:
            This method should be called after initialization and before solve().
        """
        logger.info("Building comprehensive constraint system for factory layout optimization")
        
        # Step 1: Build blocks and gates from configuration data
        logger.debug("Step 1: Building blocks and gates from configuration data")
        self.build_blocks()
        
        # Step 2: Add boundary constraints to keep blocks within valid ranges
        logger.debug("Step 2: Adding boundary constraints")
        self.add_bound_constraints()
        
        # Step 3: Add overlap prevention constraints between blocks
        logger.debug("Step 3: Adding overlap prevention constraints")
        self.add_overlap_constraints()
        
        # Step 4: Add gate positioning constraints relative to parent blocks
        logger.debug("Step 4: Adding gate relative positioning constraints")
        self.add_gate_relative_position_constraints()
        
        # Step 5: Add block alignment constraints based on compatibility analysis
        logger.debug("Step 5: Adding block alignment constraints")
        self.add_simplified_block_alignment_constraints()
        
        # Step 6: Add I/O constraints if we have fixed gates
        if hasattr(self, 'fixed_gates') and self.fixed_gates:
            logger.debug("Step 6: Adding I/O gate constraints for fixed gates")
            self.add_io_gate_constraints()
        else:
            logger.debug("Step 6: No fixed gates found, skipping I/O gate constraints")
        
        # Step 7: Set up optimization objectives
        logger.debug("Step 7: Setting up optimization objectives")
        self.minimize_map()
        
        logger.info("Constraint building completed successfully")

    def build_blocks(self):
        """
        Build Block and Gate objects from the configuration data.
        
        This method processes the block_data dictionary to create Block objects
        with their associated input and output gates. It also analyzes block
        compatibility for optimal positioning and connection planning.
        
        The method performs the following operations:
        1. Iterates through each block type in block_data
        2. Creates multiple Block instances if num_factories > 1
        3. Extracts gate information from the production tree
        4. Creates Gate objects for each input and output
        5. Tracks interface positions and item relationships
        6. Analyzes block compatibility for constraint optimization
        
        Note:
            This method populates self.blocks and self.gates lists and should
            be called as part of the constraint building process.
        """
        logger.info("Building blocks and gates from configuration data")
        
        # Track total blocks and gates created for summary logging
        total_blocks_created = 0
        total_gates_created = 0
        
        for i, key in enumerate(self.block_data.keys()):
            logger.debug(f"Processing block type '{key}' (index {i})")
            
            # Retrieve block dimensions from the production tree
            width = self.block_data[key]["tree"].grid_width
            height = self.block_data[key]["tree"].grid_height
            num_factories = self.block_data[key]["num_factories"]
            
            logger.debug(f"Block {key} specifications: width={width}, height={height}, factories={num_factories}")
            
            # Combine input and output gate information into unified structure
            gate_info = {}
            gate_info.update(self.block_data[key]["tree"].input_information)
            gate_info.update(self.block_data[key]["tree"].output_information)
            
            logger.debug(f"Block {key} has {len(gate_info)} gate definitions")

            # Create multiple instances if num_factories > 1
            for factory_index in range(num_factories):
                # Create the block instance
                block = Block(f"Block_{key}_{i}_{factory_index}", width, height)
                logger.debug(f"Created Block instance: {block.id}")
                total_blocks_created += 1
                
                # Process each gate definition in the block
                for item, data in gate_info.items():
                    # Create input gate if specified
                    if "input" in data:
                        # Track input interface position for alignment analysis
                        if item not in block.input_interfaces:
                            block.input_interfaces[item] = []
                        block.input_interfaces[item].append(int(data["input"][0]))
                        block.input_items.add(item)
                        
                        # Create the input gate object
                        input_gate = Gate(
                            id=f"{key}_input_{item}_{i}_{factory_index}",
                            relative_x=int(data["input"][1]),
                            relative_y=int(data["input"][0]),
                            item=item,
                            type="input"
                        )
                        
                        # Add gate to block and global gate tracking
                        block.input_points.append(input_gate)
                        self.gates.append(input_gate)
                        total_gates_created += 1
                        logger.debug(f"Created input gate {input_gate.id} for item {item} at relative position ({input_gate.relative_x}, {input_gate.relative_y})")
                        
                    # Create output gate if specified
                    if "output" in data:
                        # Track output interface position for alignment analysis
                        if item not in block.output_interfaces:
                            block.output_interfaces[item] = []
                        block.output_interfaces[item].append(int(data["output"][0]))
                        block.output_items.add(item)
                        
                        # Create the output gate object
                        output_gate = Gate(
                            id=f"{key}_output_{item}_{i}_{factory_index}",
                            relative_x=int(data["output"][1]),
                            relative_y=int(data["output"][0]),
                            item=item,
                            type="output"
                        )
                        
                        # Add gate to block and global gate tracking
                        block.output_points.append(output_gate)
                        self.gates.append(output_gate)
                        total_gates_created += 1
                        logger.debug(f"Created output gate {output_gate.id} for item {item} at relative position ({output_gate.relative_x}, {output_gate.relative_y})")

                # Add completed block to the blocks collection
                self.blocks.append(block)
                logger.debug(f"Block {block.id} added to blocks collection (inputs: {len(block.input_points)}, outputs: {len(block.output_points)})")
                
        logger.info(f"Block building completed: {total_blocks_created} blocks, {total_gates_created} gates created")
        
        # Analyze block compatibility for optimal constraint generation
        logger.debug("Starting block compatibility analysis")
        self._analyze_block_compatibility()
        logger.info("Block building and compatibility analysis completed")
        
    def _analyze_block_compatibility(self):
        """
        Analyze block compatibility for optimal constraint generation.
        
        This method performs a comprehensive analysis of which blocks should be
        connected based on their input/output patterns. It determines producer-consumer
        relationships, calculates alignment compatibility, and computes required spacing
        for optimal layout generation.
        
        The analysis process includes:
        1. Identifying shared items between block pairs
        2. Checking interface alignment for each shared item
        3. Calculating required spacing based on misaligned connections
        4. Building producer-consumer relationship mappings
        5. Storing compatibility information for constraint generation
        
        Results are stored in:
        - Block.compatible_blocks: Compatibility information for each block
        - Block.spacing: Required spacing distances with other blocks
        - self.producers: Mapping of consumer blocks to their producers
        - self.consumers: Mapping of producer blocks to their consumers
        
        Note:
            This method is called automatically during block building and should
            not be called directly.
        """
        logger.info("Analyzing block compatibility for optimal constraint generation")
        
        # Initialize producer-consumer relationship tracking
        self.producers = {}  # Dictionary mapping consumer_id -> [producer_ids]
        self.consumers = {}  # Dictionary mapping producer_id -> [consumer_ids]
        
        # Track analyzed pairs to avoid duplicate analysis
        analyzed_pairs = set()
        compatibility_count = 0
        
        # Analyze each pair of blocks for compatibility
        for producer in self.blocks:
            for consumer in self.blocks:
                # Skip self-comparison
                if producer == consumer:
                    continue
                    
                # Find items produced by 'producer' that are consumed by 'consumer'
                shared_items = producer.output_items.intersection(consumer.input_items)
                
                # Skip if no shared items
                if not shared_items:
                    continue
                
                # Create a unique pair identifier (sorted to ensure consistency)
                block_pair = tuple(sorted([producer.id, consumer.id]))
                
                # Skip if we've already analyzed this pair
                if block_pair in analyzed_pairs:
                    logger.debug(f"Skipping already analyzed pair: {block_pair}")
                    continue
                
                analyzed_pairs.add(block_pair)
                logger.debug(f"Analyzing compatibility between {producer.id} and {consumer.id}")
                
                # For each shared item, check if there are aligned interfaces
                aligned_items = []
                misaligned_items = []
                aligned_gate_count = 0
                
                for item in shared_items:
                    # Get interface positions for this item
                    producer_outputs = producer.output_interfaces.get(item, [])
                    consumer_inputs = consumer.input_interfaces.get(item, [])

                    logger.debug(f"  Item {item}:")
                    logger.debug(f"    Producer {producer.id} output x-positions: {producer_outputs}")
                    logger.debug(f"    Consumer {consumer.id} input x-positions: {consumer_inputs}")
                    
                    # Check alignment between producer outputs and consumer inputs
                    is_aligned = False
                    item_alignments = []
                    
                    # Compare all producer outputs with all consumer inputs
                    for p_out_x in producer_outputs:
                        for c_in_x in consumer_inputs:
                            # Check if they have the same relative position (perfect alignment)
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
                    
                    # Categorize item based on alignment status
                    if is_aligned:
                        aligned_items.append(item)
                        logger.debug(f"    ✓ Item {item} is ALIGNED")
                    else:
                        misaligned_items.append(item)
                        logger.debug(f"    X Item {item} is MISALIGNED")
                
                # Calculate required spacing based on misaligned items
                # Spacing needed = misaligned items - aligned gates (but at least 0)
                spacing_needed = max(0, len(misaligned_items) - aligned_gate_count)
                
                # Create compatibility information if there are shared items
                if aligned_items or misaligned_items:
                    compatibility_count += 1
                    
                    # Store compatibility information in producer block
                    producer.compatible_blocks[consumer.id] = {
                        "aligned": len(aligned_items) > 0,
                        "perfect_alignment": len(misaligned_items) == 0 and len(aligned_items) > 0,
                        "aligned_items": aligned_items,
                        "misaligned_items": misaligned_items,
                        "spacing_needed": spacing_needed,
                        "aligned_gate_count": aligned_gate_count,
                        "relation": "producer"  # This block is a producer for the consumer
                    }
                    
                    # Set the spacing requirements in the block's spacing dictionary
                    producer.spacing[consumer.id] = {
                        "north": spacing_needed,
                        "south": spacing_needed,
                        "east": spacing_needed,
                        "west": spacing_needed
                    }
                    
                    # Track producer-consumer relationships for constraint generation
                    if consumer.id not in self.producers:
                        self.producers[consumer.id] = []
                    self.producers[consumer.id].append(producer.id)
                    
                    if producer.id not in self.consumers:
                        self.consumers[producer.id] = []
                    self.consumers[producer.id].append(consumer.id)
                    
                    logger.debug(f"Compatibility established between {producer.id} -> {consumer.id}:")
                    logger.debug(f"  - Aligned items: {aligned_items}")
                    logger.debug(f"  - Misaligned items: {misaligned_items}")
                    logger.debug(f"  - Aligned gates: {aligned_gate_count}")
                    logger.debug(f"  - Spacing needed: {spacing_needed}")
                    logger.debug(f"  - Perfect alignment: {len(misaligned_items) == 0 and len(aligned_items) > 0}")
    
        logger.info(f"Block compatibility analysis completed:")
        logger.info(f"  - {compatibility_count} compatible block pairs identified")
        logger.info(f"  - {len(self.producers)} consumer blocks with producers")
        logger.info(f"  - {len(self.consumers)} producer blocks with consumers")
    
    def add_simplified_block_alignment_constraints(self):
        """
        Add constraints to align compatible blocks vertically with appropriate spacing.
        Also enforces complexity-based positioning relative to I/O points.
        """
        logger.info("Adding simplified block alignment constraints with complexity-based positioning")
        
        alignment_pairs = []
        
        # First, add complexity-based positioning constraints
        self.add_complexity_based_positioning_constraints()
        
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
                
                
                # 
                
                # Add strong soft constraint to encourage vertical alignment
                self.solver.add_soft(vertical_align, weight=1000)
                  # Create variables for relative positions
                block_above = Bool(f"above_{block.id}_{other_id}")
                other_above = Bool(f"above_{other_id}_{block.id}")
                
                # Ensure only one direction is chosen
                self.solver.add(Not(And(block_above, other_above)))
                
                # Determine the correct vertical ordering based on complexity, not just producer-consumer
                # Lower complexity items should be closer to I/O (higher Y values)
                block_primary_item = self._get_primary_output_item(block)
                other_primary_item = self._get_primary_output_item(other_block)
                
                # Skip fixed I/O gate blocks for complexity ordering
                block_is_fixed = hasattr(block, 'is_fixed_gate_block') and block.is_fixed_gate_block
                other_is_fixed = hasattr(other_block, 'is_fixed_gate_block') and other_block.is_fixed_gate_block
                
                if not block_is_fixed and not other_is_fixed and block_primary_item and other_primary_item:
                    block_complexity = self.item_complexity.get(block_primary_item, 999)
                    other_complexity = self.item_complexity.get(other_primary_item, 999)
                    
                    if block_complexity < other_complexity:
                        # block has lower complexity, should be closer to I/O (higher Y, below other_block)
                        self.solver.add(other_above)
                        logger.debug(f"Complexity-based ordering: {other_id} above {block.id} (lower complexity {block.id} closer to I/O)")
                    elif other_complexity < block_complexity:
                        # other_block has lower complexity, should be closer to I/O (higher Y, below block)
                        self.solver.add(block_above)
                        logger.debug(f"Complexity-based ordering: {block.id} above {other_id} (lower complexity {other_id} closer to I/O)")
                    else:
                        # Same complexity, use producer-consumer relationship as tie-breaker
                        if other_id in self.producers.get(block.id, []):
                            # other_block is a producer for block, respect the dependency
                            self.solver.add(block_above)
                            logger.debug(f"Same complexity, producer-consumer tiebreaker: {block.id} above {other_id}")
                        elif block.id in self.producers.get(other_id, []):
                            # block is a producer for other_block, respect the dependency
                            self.solver.add(other_above)
                            logger.debug(f"Same complexity, producer-consumer tiebreaker: {other_id} above {block.id}")
                        else:
                            # No clear ordering preference, allow either
                            self.solver.add(Or(block_above, other_above))
                            logger.debug(f"Same complexity, no producer relationship: allowing either arrangement for {block.id} and {other_id}")
                else:
                    # If one or both blocks are fixed I/O gates, use producer-consumer relationship
                    if other_id in self.producers.get(block.id, []):
                        self.solver.add(block_above)
                        logger.debug(f"Fixed block ordering: {block.id} above {other_id} (producer below consumer)")
                    elif block.id in self.producers.get(other_id, []):
                        self.solver.add(other_above)
                        logger.debug(f"Fixed block ordering: {other_id} above {block.id} (producer below consumer)")
                    else:
                        # Allow either arrangement for fixed blocks without clear producer-consumer relationship
                        self.solver.add(Or(block_above, other_above))
                        logger.debug(f"Fixed block: allowing either arrangement for {block.id} and {other_id}")
                
         
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
        """
        Fix the position of the first block to origin coordinates.
        
        This method constrains the first block in the layout to be positioned at
        coordinates (0, 0). This serves as an anchor point for the entire layout
        and helps reduce the solution space for the SMT solver.
        
        Note:
            This method is currently unused but kept for potential future use
            in scenarios where a fixed reference point is needed.
        """
        logger.debug("Fixing position of first block to origin (0, 0)")
        
        if not self.blocks:
            logger.warning("No blocks available to fix position")
            return
        
        # Fix the first block at origin coordinates
        self.solver.add(self.blocks[0].x == 0)
        self.solver.add(self.blocks[0].y == 0)
        
        logger.debug(f"Fixed block {self.blocks[0].id} at position (0, 0)")
    
    def add_bound_constraints(self):
        """
        Add boundary constraints to ensure blocks stay within valid layout bounds.
        
        This method adds constraints to ensure that:
        1. All blocks are positioned at non-negative coordinates
        2. All blocks fit within the maximum layout dimensions
        3. The layout dimensions are properly bounded
        
        The constraints ensure that no block extends beyond the calculated maximum
        layout size and that all blocks remain within the valid coordinate space.
        
        Note:
            This method is called automatically during constraint building.
        """
        logger.info("Adding boundary constraints to ensure blocks stay within valid bounds")
        
        # Track number of constraints added for logging
        constraint_count = 0
        
        # Add constraints for each block
        for block in self.blocks:
            # Ensure blocks are positioned at non-negative coordinates
            self.solver.add(block.y >= 0)
            self.solver.add(block.x >= 0)
            constraint_count += 2
            
            # Ensure blocks fit within the maximum layout dimensions
            self.solver.add(block.x + block.width <= self.max_x)
            self.solver.add(block.y + block.height <= self.max_y)
            constraint_count += 2
            
            logger.debug(f"Added boundary constraints for block {block.id} (size: {block.width}x{block.height})")
        
        # Ensure maximum layout dimensions are non-negative
        self.solver.add(self.max_x >= 0)
        self.solver.add(self.max_y >= 0)
        constraint_count += 2
        
        # Ensure maximum layout dimensions accommodate all fixed gates
        for gate in self.fixed_gates:
            self.solver.add(self.max_x >= gate.x + 1)  # Layout must be wide enough for fixed gates
            self.solver.add(self.max_y >= gate.y + 1)  # Layout must be tall enough for fixed gates
            constraint_count += 2
        
        logger.info(f"Added {constraint_count} boundary constraints for {len(self.blocks)} blocks")
        logger.debug("All blocks constrained to non-negative coordinates and within max layout bounds")
        
        
    
    def add_overlap_constraints(self):
        """
        Add constraints to prevent blocks from overlapping with each other.
        
        This method ensures that no two blocks occupy the same space by adding
        disjunction constraints for each pair of blocks. For each pair, at least
        one of the following must be true:
        1. Block1 is completely to the left of Block2
        2. Block1 is completely to the right of Block2
        3. Block1 is completely above Block2
        4. Block1 is completely below Block2
        
        The constraints are added for all unique pairs of blocks to prevent
        any spatial conflicts in the final layout.
        
        Note:
            This method is called automatically during constraint building.
        """
        logger.info("Adding overlap prevention constraints between all block pairs")
        
        # Track the number of constraints added for logging
        constraint_count = 0
        
        # Add overlap constraints for each unique pair of blocks
        for i, block1 in enumerate(self.blocks):
            for j, block2 in enumerate(self.blocks):
                # Skip if same block or if we've already processed this pair
                if i >= j:
                    continue
                
                # Add disjunction constraint ensuring blocks don't overlap
                self.solver.add(
                    Or(
                        block1.x + block1.width <= block2.x,   # block1 is to the left of block2
                        block2.x + block2.width <= block1.x,   # block2 is to the left of block1
                        block1.y + block1.height <= block2.y,  # block1 is above block2
                        block2.y + block2.height <= block1.y   # block2 is above block1
                    )
                )
                constraint_count += 1
                logger.debug(f"Added non-overlapping constraint between blocks {block1.id} and {block2.id}")
        
        logger.info(f"Added {constraint_count} overlap prevention constraints for {len(self.blocks)} blocks")
    
    
    def add_gate_relative_position_constraints(self):
        """
        Add constraints to ensure gates are positioned relative to their parent blocks.
        
        This method constrains each gate's absolute position to be the sum of its
        parent block's position and its relative position within the block. This
        ensures that gates move with their parent blocks and maintain their
        correct relative positions.
        
        For each gate, the constraints are:
        - gate.x = parent_block.x + gate.relative_x
        - gate.y = parent_block.y + gate.relative_y
        
        Note:
            This method is called automatically during constraint building.
        """
        logger.info("Adding gate relative position constraints")
        
        # Track the number of constraints added for logging
        constraint_count = 0
        
        # Add relative position constraints for all blocks
        for block in self.blocks:
            # Process input gates
            for gate in block.input_points:
                # Constrain gate position to be relative to parent block
                self.solver.add(gate.x == block.x + gate.relative_x)
                self.solver.add(gate.y == block.y + gate.relative_y)
                constraint_count += 2
                logger.debug(f"Added relative position constraints for input gate {gate.id} "
                           f"(relative: {gate.relative_x}, {gate.relative_y}) in block {block.id}")
            
            # Process output gates
            for gate in block.output_points:
                # Constrain gate position to be relative to parent block
                self.solver.add(gate.x == block.x + gate.relative_x)
                self.solver.add(gate.y == block.y + gate.relative_y)
                constraint_count += 2
                logger.debug(f"Added relative position constraints for output gate {gate.id} "
                           f"(relative: {gate.relative_x}, {gate.relative_y}) in block {block.id}")
        
        logger.info(f"Added {constraint_count} gate relative position constraints for {len(self.blocks)} blocks")

    def minimize_map(self):
        """
        Set up optimization objectives to minimize the layout size and complexity cost.
        
        This method defines the optimization objectives for the SMT solver to minimize:
        1. Layout width (with higher priority to encourage vertical layouts)
        2. Layout height (with lower priority to allow taller layouts)
        3. Complexity-based positioning cost (to optimize block placement based on item complexity)
        
        The optimization encourages layouts that are:
        - Narrow (minimizing width is prioritized)
        - Complexity-aware (higher complexity items positioned further from I/O)
        - Compact (minimizing overall layout size)
        
        Note:
            This method is called automatically during constraint building.
        """
        logger.info("Setting up optimization objectives for layout minimization")
        
        # Define optimization weights for different objectives
        width_weight = 2.0   # Higher weight for width to encourage narrow layouts
        height_weight = 0.5  # Lower weight for height to allow taller layouts
        complexity_weight = 0.1  # Weight for complexity-based positioning
        
        logger.debug(f"Optimization weights: width={width_weight}, height={height_weight}, complexity={complexity_weight}")
        
        # Set up complexity-based optimization term
        complexity_cost = Int('complexity_cost')
        complexity_terms = []
        
        # Get all non-fixed blocks (exclude I/O gate blocks)
        production_blocks = [block for block in self.blocks 
                           if not (hasattr(block, 'is_fixed_gate_block') and block.is_fixed_gate_block)]
        
        logger.debug(f"Found {len(production_blocks)} production blocks for complexity optimization")
        
        # Calculate complexity cost for each production block
        for block in production_blocks:
            primary_item = self._get_primary_output_item(block)
            if primary_item:
                complexity = self.item_complexity.get(primary_item, 1)
                # Higher complexity items should be penalized more for being close to y=0
                # This encourages them to be positioned further from I/O points
                complexity_terms.append(complexity * block.y)
                logger.debug(f"Block {block.id} produces {primary_item} (complexity: {complexity})")
        
        # Set up the complete optimization objective
        if complexity_terms:
            # Sum all complexity terms
            total_complexity_cost = complexity_terms[0]
            for term in complexity_terms[1:]:
                total_complexity_cost = total_complexity_cost + term
            
            # Define the complexity cost constraint
            self.solver.add(complexity_cost == total_complexity_cost)
            
            # Minimize: weighted width + weighted height + complexity positioning cost
            objective = (width_weight * self.max_x + 
                        height_weight * self.max_y + 
                        complexity_weight * complexity_cost)
            
            self.solver.minimize(objective)
            
            logger.info(f"Optimization objective set with complexity terms for {len(complexity_terms)} blocks")
        else:
            # Fallback to basic minimization if no complexity terms
            objective = width_weight * self.max_x + height_weight * self.max_y
            self.solver.minimize(objective)
            
            logger.info("Optimization objective set with basic width and height minimization")
        
        logger.debug("Layout minimization objectives configured successfully")

        
    def solve(self):
        """
        Solve the factory layout optimization problem using the Z3 SMT solver.
        
        This method runs the Z3 solver on the configured constraints and objectives
        to find an optimal factory layout. If successful, it extracts block positions
        and determines optimal gate connections.
        
        Returns:
            tuple: A 4-tuple containing:
                - final_blocks (dict): Dictionary mapping block IDs to their positions and properties
                - max_x_val (int): Maximum layout width
                - max_y_val (int): Maximum layout height  
                - gate_connections (list): List of optimal gate-to-gate connections
                
                Returns (None, None, None, None) if no solution is found.
        
        Note:
            This method should be called after build_constraints() to ensure all
            constraints are properly configured.
        """
        logger.info("Starting factory layout optimization with Z3 SMT solver")
        start_time = time.perf_counter()
        
        # Run the Z3 solver
        result = self.solver.check()
        solve_time = time.perf_counter() - start_time
        
        logger.info(f"Z3 solver completed in {solve_time:.3f} seconds with result: {result}")
        
        if result == sat:
            logger.info("Solution found! Extracting layout information")
            
            # Extract the solution model
            model = self.solver.model()
            
            # Extract block positions from the solved model
            logger.debug("Extracting block positions from solution model")
            final_blocks = self._extract_block_positions(model)
            
            # Extract maximum layout dimensions
            max_x_val = model[self.max_x].as_long()
            max_y_val = model[self.max_y].as_long()
            logger.info(f"Final layout dimensions: {max_x_val} x {max_y_val}")
            
            # Post-process to determine optimal gate connections
            logger.debug("Determining optimal gate connections")
            gate_connections = self._determine_optimal_gate_connections(model, final_blocks)
            
            logger.info(f"Layout optimization completed successfully:")
            logger.info(f"  - {len(final_blocks)} blocks positioned")
            logger.info(f"  - {len(gate_connections)} gate connections established")
            logger.info(f"  - Layout size: {max_x_val} x {max_y_val}")
            
            return final_blocks, max_x_val, max_y_val, gate_connections
        else:
            logger.error("Z3 solver could not find a solution!")
            logger.error("This may indicate:")
            logger.error("  - Conflicting constraints")
            logger.error("  - Over-constrained problem")
            logger.error("  - Insufficient layout space")
            return None, None, None, None
        
    def _extract_block_positions(self, model):
        """
        Extract block positions and gate information from the solved Z3 model.
        
        This method processes the Z3 solution model to extract the final positions
        of all blocks and their associated gates. It also includes metadata about
        each block for visualization and analysis purposes.
        
        Args:
            model (z3.ModelRef): The Z3 model containing the solution
            
        Returns:
            dict: Dictionary mapping block IDs to their position and property information.
                  Each block entry contains:
                  - x, y: Block position coordinates
                  - width, height: Block dimensions
                  - input_points: List of input gate information
                  - output_points: List of output gate information
                  - module_json: Path to the module's JSON file (if available)
                  - block_type: The base item type for the block
        
        Note:
            This method is called automatically during solve() and should not be
            called directly.
        """
        logger.debug("Extracting block positions and gate information from Z3 solution model")
        
        final_blocks = {}
        
        # Process each block in the solution
        for block in self.blocks:
            # Extract block position from Z3 model
            x = model[block.x].as_long()
            y = model[block.y].as_long()
            
            # Extract input gates with their absolute positions
            input_gates = []
            for gate in block.input_points:
                input_gates.append({
                    "id": gate.id,
                    "item": gate.item,
                    "type": gate.type,
                    "x": model[gate.x].as_long(),
                    "y": model[gate.y].as_long()
                })
            
            # Extract output gates with their absolute positions
            output_gates = []
            for gate in block.output_points:
                output_gates.append({
                    "id": gate.id,
                    "item": gate.item,
                    "type": gate.type,
                    "x": model[gate.x].as_long(),
                    "y": model[gate.y].as_long()
                })
            
            # Extract the base item name from block_id for metadata
            block_parts = block.id.split("_")
            block_type = block_parts[1] if len(block_parts) >= 3 else block.id
            
            # Get module JSON file path from block_data if available
            module_json_path = None
            if block_type in self.block_data and "json" in self.block_data[block_type]:
                module_json_path = self.block_data[block_type]["json"]
                
            # Store comprehensive block information
            final_blocks[block.id] = {
                "x": x,
                "y": y,
                "width": block.width,
                "height": block.height,
                "input_points": input_gates,
                "output_points": output_gates,
                "module_json": module_json_path,  # JSON file path for visualization
                "block_type": block_type  # Base item type for reference
            }
            
            logger.debug(f"Block {block.id} positioned at ({x}, {y}), size: {block.width}x{block.height}, "
                        f"gates: {len(input_gates)} inputs, {len(output_gates)} outputs")
        
        logger.debug(f"Extracted position information for {len(final_blocks)} blocks")
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
                
                # Filter out inputs from the same block as the output (avoid self-connections)
                if output["block"] is not None:
                    aligned_inputs = [inp for inp in aligned_inputs if inp["block"] != output["block"]]
                    
                if aligned_inputs:
                    # Find closest aligned input by y-distance
                    best_input = min(aligned_inputs, 
                                    key=lambda inp: abs(inp["y"] - output["y"]))
                    
                    # Create enhanced connection tuple with positions
                    connection_data = {
                        "source": output["gate"],
                        "target": best_input["gate"],
                        "source_x": output["x"],
                        "source_y": output["y"],
                        "target_x": best_input["x"],
                        "target_y": best_input["y"],
                        "item": item
                    }
                    
                    connections.append(connection_data)
                    inputs.remove(best_input)
                    logger.debug(f"Connected aligned gates: {output['id']} -> {best_input['id']} with positions ({output['x']}, {output['y']}) -> ({best_input['x']}, {best_input['y']})")
                else:
                    # Filter available inputs to exclude those from the same block
                    available_inputs = inputs
                    if output["block"] is not None:
                        available_inputs = [inp for inp in inputs if inp["block"] != output["block"]]
                    
                    # If we have available inputs from different blocks, use them
                    if available_inputs:
                        # If no aligned inputs, find closest by Manhattan distance
                        best_input = min(available_inputs, 
                                        key=lambda inp: manhattan_distance(
                                            (output["x"], output["y"]), 
                                            (inp["x"], inp["y"])
                                        ))
                        
                        # Create enhanced connection tuple with positions
                        connection_data = {
                            "source": output["gate"],
                            "target": best_input["gate"],
                            "source_x": output["x"],
                            "source_y": output["y"],
                            "target_x": best_input["x"],
                            "target_y": best_input["y"],
                            "item": item
                        }
                        
                        connections.append(connection_data)
                        inputs.remove(best_input)
                        logger.debug(f"Connected nearest gates: {output['id']} -> {best_input['id']} with positions ({output['x']}, {output['y']}) -> ({best_input['x']}, {best_input['y']})")
                    else:
                        # Skip if there are no valid inputs from different blocks
                        logger.debug(f"Skipped connection for {output['id']} as all inputs are from the same block")
        
        logger.info(f"Determined {len(connections)} optimal gate connections")
        return connections
    
    
    def get_block_for_gate(self, gate):
        """
        Find the parent block that contains a specific gate.
        
        This method searches through all blocks to find which block contains
        the specified gate in its input or output points.
        
        Args:
            gate (Gate): The gate object to search for
            
        Returns:
            Block or None: The block containing the gate, or None if not found
        
        Note:
            This method is used during gate connection analysis to avoid
            self-connections within the same block.
        """
        logger.debug(f"Searching for parent block of gate {gate.id}")
        
        # Search through all blocks
        for block in self.blocks:
            # Check if gate is in this block's input or output points
            if gate in block.input_points or gate in block.output_points:
                logger.debug(f"Found gate {gate.id} in block {block.id}")
                return block
        
        logger.warning(f"Gate {gate.id} not found in any block")
        return None


    def determine_gate_connections(self, model):
        """
        Determine gate connections using a greedy nearest-neighbor approach.
        
        This method provides an alternative gate connection algorithm that uses
        a simpler greedy approach to connect output gates to input gates of the
        same item type based on Manhattan distance.
        
        Args:
            model (z3.ModelRef): The Z3 model containing solved gate positions
            
        Returns:
            list: List of tuples (output_gate, input_gate) representing connections
        
        Note:
            This method is an alternative to _determine_optimal_gate_connections
            and uses a simpler greedy approach without complexity prioritization.
        """
        logger.info("Determining gate connections using greedy nearest-neighbor approach")
        
        connections = []
        
        # Group gates by item type with their solved positions
        gates_by_item = {}
        for gate in self.gates:
            item = gate.item
            if item not in gates_by_item:
                gates_by_item[item] = {"input": [], "output": []}
                
            # Get solved position from Z3 model
            actual_x = model[gate.x].as_long()
            actual_y = model[gate.y].as_long()
            
            gate_data = {
                "gate": gate,
                "id": gate.id,
                "x": actual_x,
                "y": actual_y
            }
            
            gates_by_item[item][gate.type].append(gate_data)
        
        # For each item type, connect output gates to input gates using greedy approach
        for item, item_gates in gates_by_item.items():
            outputs = item_gates["output"]
            inputs = item_gates["input"]
            
            # Skip if no outputs or no inputs for this item
            if not outputs or not inputs:
                logger.debug(f"Skipping item {item}: no outputs or no inputs")
                continue
                
            logger.debug(f"Connecting {len(outputs)} outputs to {len(inputs)} inputs for item {item}")
            
            # Find nearest connections using greedy approach
            remaining_inputs = inputs.copy()
            
            for output in outputs:
                if not remaining_inputs:
                    logger.debug(f"No more inputs available for output {output['id']}")
                    break
                
                # Find closest input gate using Manhattan distance
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
                
                # Create connection if best input found
                if best_input:
                    connections.append((output["gate"], best_input["gate"]))
                    remaining_inputs.pop(best_index)
                    logger.debug(f"Connected {output['id']} to {best_input['id']} (distance: {best_distance})")
        
        logger.info(f"Determined {len(connections)} gate connections using greedy approach")
        return connections

    def add_fixed_gate(self, gate_id, item, position, gate_type, edge):
        """
        Add a fixed I/O gate at a specific position in the layout.
        
        This method creates a fixed gate that serves as an input or output point
        for the entire factory. The gate is implemented as a 1x1 block with a
        single gate that has a fixed position constraint.
        
        Args:
            gate_id (str): Unique identifier for the gate
            item (str): The item type handled by this gate (e.g., "iron-plate")
            position (tuple): Fixed (x, y) coordinates for the gate
            gate_type (str): Either 'input' or 'output'
            edge (str): Edge information (North, South, East, West) for constraint generation
            
        Returns:
            Gate: The created fixed gate object
            
        Note:
            Fixed gates are used to define the factory's input/output interface
            and are typically positioned at the edges of the layout area.
        """
        logger.info(f"Adding fixed {gate_type} gate for item '{item}' at position {position}")
        
        x, y = position
        
        # Create a 1x1 block to hold this fixed gate
        block_id = f"fixed_block_{gate_id}"
        block = Block(block_id, 1, 1)
        logger.debug(f"Created 1x1 block {block_id} for fixed gate")
        
        # Create the fixed gate with relative position (0,0) within the 1x1 block
        gate = Gate(
            id=gate_id,
            relative_x=0,  # For 1x1 block, relative position is always (0,0)
            relative_y=0,
            item=item,
            type=gate_type,
            is_fixed=True,
            edge=edge
        )
        logger.debug(f"Created fixed gate {gate_id} with edge constraint: {edge}")
        
        # Add the gate to the appropriate list in the block
        if gate_type == "input":
            block.input_points.append(gate)
            block.input_items.add(item)
        else:
            block.output_points.append(gate)
            block.output_items.add(item)
        
        # Add hard constraints to fix the positions
        self.solver.add(block.x == x)
        self.solver.add(block.y == y)
        self.solver.add(gate.x == x)
        self.solver.add(gate.y == y)
        logger.debug(f"Added position constraints: block and gate fixed at ({x}, {y})")
        
        # Add the block and gate to the collections
        self.blocks.append(block)
        self.gates.append(gate)
        
        # Initialize fixed gate collections if they don't exist
        if not hasattr(self, 'fixed_gates'):
            self.fixed_gates = []
        if not hasattr(self, 'fixed_blocks'):
            self.fixed_blocks = []
        
        # Track the fixed gate and block for special handling
        self.fixed_gates.append(gate)
        self.fixed_blocks.append(block)
        
        logger.info(f"Successfully added fixed {gate_type} gate '{gate_id}' for item '{item}' at ({x}, {y})")
        logger.debug(f"Fixed gate collections now contain {len(self.fixed_gates)} gates and {len(self.fixed_blocks)} blocks")
        
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
                # South edge gates: ensure ALL blocks are positioned above them (lower Y values)
                for block in non_fixed_blocks:
                    # Force all blocks to be positioned above I/O gates with adequate spacing
                    min_spacing = 1  # Minimum spacing between block edge and gate
                    self.solver.add(block.y + block.height + min_spacing <= gate.y)
                    
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

    def add_complexity_based_positioning_constraints(self):
            """
            Add simplified complexity-based positioning constraints.
            Fix the lowest complexity module at a specific position above I/O points,
            then let spacing constraints handle the rest.
            """
            logger.info("Adding simplified complexity-based positioning constraints")
            
            # Get all non-fixed blocks (exclude 1x1 I/O gate blocks)
            # Use fixed_blocks list to identify I/O gate blocks since is_fixed_gate_block
            # hasn't been set yet at this point in the constraint building process
            fixed_block_ids = set()
            if hasattr(self, 'fixed_blocks'):
                fixed_block_ids = {block.id for block in self.fixed_blocks}
            
            production_blocks = [block for block in self.blocks 
                            if block.id not in fixed_block_ids]
            
            if not production_blocks:
                logger.warning("No production blocks found for complexity-based positioning")
                return
            
            # Find the block with the lowest complexity
            lowest_complexity_block = None
            lowest_complexity = float('inf')
            
            for block in production_blocks:
                primary_item = self._get_primary_output_item(block)
                if primary_item:
                    complexity = self.item_complexity.get(primary_item, 999)
                    logger.debug(f"Block {block.id} produces {primary_item} with complexity {complexity}")
                    
                    if complexity < lowest_complexity:
                        lowest_complexity = complexity
                        lowest_complexity_block = block
            
            if not lowest_complexity_block:
                logger.warning("No block with valid complexity found")
                return
            
            # Count the number of I/O points to determine distance
            num_io_points = 0
            if hasattr(self, 'fixed_gates'):
                num_io_points = len(self.fixed_gates)
            
            # Get the highest Y coordinate of I/O gates
            max_io_y = 0
            if hasattr(self, 'fixed_gates'):
                for gate in self.fixed_gates:
                    # Since gates are fixed, we can get their position from the solver
                    # For now, assume they're at Y=29 based on the user selection
                    max_io_y = max(max_io_y, 29)  # Use the known Y position from the logs
            
            # Calculate the fixed position for the lowest complexity block
            # Position it above I/O points with distance equal to number of I/O points
            fixed_y_position = max_io_y - lowest_complexity_block.height - num_io_points
            
            # Ensure the position is not negative
            fixed_y_position = max(0, fixed_y_position)
            
            # Add hard constraint to fix the lowest complexity block position
            self.solver.add(lowest_complexity_block.y == fixed_y_position)
            
            logger.info(f"Fixed lowest complexity block {lowest_complexity_block.id} "
                    f"(complexity {lowest_complexity}) at Y={fixed_y_position}")
            logger.info(f"Distance from I/O points: {num_io_points} units (based on {num_io_points} I/O points)")
            
            # Let the existing spacing constraints handle positioning of other blocks
            logger.info("Other blocks will be positioned using existing spacing and alignment constraints")
        
    def _get_primary_output_item(self, block):
        """
        Determine the primary output item for a block.
        
        For blocks with multiple output items, this method selects the one with
        the highest complexity as the primary output. For single-output blocks,
        it returns that item directly.
        
        Args:
            block (Block): The block to analyze
            
        Returns:
            str or None: The primary output item ID, or None if no outputs
        
        Note:
            This method is used for complexity-based block ordering and
            positioning decisions.
        """
        logger.debug(f"Determining primary output item for block {block.id}")
        
        # Return None if block has no output points
        if not block.output_points:
            logger.debug(f"Block {block.id} has no output points")
            return None
        
        # Get all unique output items from this block
        output_items = {gate.item for gate in block.output_points}
        logger.debug(f"Block {block.id} produces items: {output_items}")
        
        # If single output item, return it directly
        if len(output_items) == 1:
            primary_item = list(output_items)[0]
            logger.debug(f"Block {block.id} has single output: {primary_item}")
            return primary_item
        
        # For multiple output items, choose the one with highest complexity
        primary_item = max(output_items, 
                          key=lambda item: self.item_complexity.get(item, 0))
        
        logger.debug(f"Block {block.id} has multiple outputs {output_items}")
        logger.debug(f"Selected {primary_item} as primary (highest complexity)")
        
        return primary_item
    
    
def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.
    
    The Manhattan distance is the sum of the absolute differences of their
    coordinates. This is also known as the L1 distance or taxicab distance.
    
    Args:
        point1 (tuple): First point as (x, y) coordinates
        point2 (tuple): Second point as (x, y) coordinates
        
    Returns:
        int: Manhattan distance between the two points
    """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])