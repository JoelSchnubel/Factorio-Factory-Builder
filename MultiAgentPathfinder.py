#! .venv\Scripts\python.exe

"""
Multi-Agent Pathfinding Module

This module provides a sophisticated multi-agent pathfinding system designed for factory
layout optimization in Factorio-style games. It handles complex pathfinding scenarios
including underground belt routing, splitter management, and optimal path selection for
multiple item types with different constraints.

Main Components:
- Splitter: Represents belt splitters with input/output routing capabilities
- MultiAgentPathfinder: Main pathfinding engine with A* algorithm and extensions

Key Features:
- Multi-agent pathfinding for multiple item types simultaneously
- Underground belt support with configurable lengths
- Belt splitter integration for complex routing scenarios
- Fluid item handling with expanded positioning options
- Optimal path selection with distance and complexity optimization
- Conflict resolution between different item paths
- Inserter placement and management
- Grid-based obstacle avoidance

Pathfinding Capabilities:
- A* algorithm implementation with heuristic optimization
- Underground path detection and routing
- Splitter-aware path planning
- Output item prioritization for material flow
- Dynamic grid modification for path conflicts
- Multiple start/destination point handling

The pathfinding system is designed to work with factory layouts where:
- Items need to be transported from production modules to consumers
- Underground belts can cross obstacles with length limitations
- Splitters can route items to multiple destinations
- Fluid items require pipe networks with different constraints
- Optimal routing minimizes belt usage and path conflicts

"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from logging_config import setup_logger
logger = setup_logger("Pathfinder")



class Splitter:
    """
    Represents a belt splitter in the factory layout system.
    
    A Splitter is a factory component that can route a single input item stream
    to multiple output destinations. It has a main position on the belt grid
    and a secondary position for proper belt connectivity.
    
    Attributes:
        item (str): The item type this splitter handles
        position (tuple): Primary position (x, y) - the main belt connection point
        next_position (tuple): Secondary position for belt orientation
        direction (str): Orientation direction of the splitter
        inputs (list): List of input connection points
        outputs (list): List of output connection points
    """
    
    def __init__(self, item, position, direction):
        """
        Initialize a Splitter with item type, position, and orientation.
        
        Args:
            item (str): The item type this splitter will handle
            position (tuple): Primary position (x, y) coordinates
            direction (str): Orientation direction for the splitter
        """
        logger.debug(f"Creating Splitter for item '{item}' at position {position} with direction '{direction}'")
        
        self.item = item
        self.position = position  # Belt point/anchor point
        self.next_position = None  # Point next to the belt point depending on orientation
        self.direction = direction
        self.inputs = []   # List of input connection points
        self.outputs = []  # List of output connection points
        
        logger.debug(f"Splitter created successfully: {self}")

    def __str__(self):
        """String representation of the Splitter for debugging purposes."""
        return f"Splitter(item={self.item}, position={self.position}, direction={self.direction})"
    
    def to_dict(self):
        """
        Convert the Splitter to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation containing all splitter properties
        """
        logger.debug(f"Converting Splitter {self} to dictionary")
        
        splitter_dict = {
            'item': self.item,
            'position': self.position,
            'next_position': self.next_position,
            'direction': self.direction,
            'inputs': self.inputs,
            'outputs': self.outputs
        }
        
        logger.debug(f"Splitter dictionary created with {len(self.inputs)} inputs and {len(self.outputs)} outputs")
        return splitter_dict

    @classmethod
    def from_dict(cls, data):
        """
        Create a Splitter instance from a dictionary representation.
        
        Args:
            data (dict): Dictionary containing splitter configuration data
            
        Returns:
            Splitter: New Splitter instance with properties from the dictionary
        """
        logger.debug(f"Creating Splitter from dictionary: {data}")
        
        # Create the basic splitter instance
        splitter = cls(
            item=data['item'],
            position=data['position'],
            direction=data['direction']
        )
        
        # Set optional properties with defaults
        splitter.next_position = data.get('next_position')
        splitter.inputs = data.get('inputs', [])
        splitter.outputs = data.get('outputs', [])
        
        logger.debug(f"Splitter created from dictionary: {splitter}")
        logger.debug(f"Loaded {len(splitter.inputs)} inputs and {len(splitter.outputs)} outputs")
        
        return splitter
    

class MultiAgentPathfinder:
    """
    Advanced multi-agent pathfinding system for factory layout optimization.
    
    This class implements a sophisticated pathfinding algorithm designed for complex
    factory scenarios where multiple item types need to be routed simultaneously
    from various production points to consumption points while avoiding obstacles
    and optimizing for minimal path conflicts.
    
    Key Features:
    - Multi-agent A* pathfinding with conflict resolution
    - Underground belt support for crossing obstacles
    - Belt splitter integration for complex routing
    - Fluid item handling with expanded connection options
    - Output item prioritization for material flow optimization
    - Dynamic grid modification for path optimization
    - Inserter placement and management
    
    Pathfinding Modes:
    - Standard pathfinding: Basic A* with obstacle avoidance
    - Underground pathfinding: Extended A* with underground belt segments
    - Splitter-aware pathfinding: Routes through belt splitters
    - Optimal pathfinding: Finds shortest paths when enabled
    
    Grid Management:
    - Obstacle map: Static obstacles that cannot be crossed
    - Working grid: Dynamic grid modified during pathfinding
    - Path marking: Marks successful paths as obstacles for future paths
    
    Attributes:
        obstacle_map (list): Original obstacle grid (read-only)
        working_grid (list): Modified grid used during pathfinding
        points (dict): Dictionary of item routing information
        paths (dict): Dictionary storing found paths for each item
        inserters (dict): Dictionary storing inserter placements
        allow_underground (bool): Whether underground belts are permitted
        allow_splitters (bool): Whether belt splitters are enabled
        find_optimal_paths (bool): Whether to find shortest paths
        output_item (str): Priority item for output routing
        width (int): Grid width
        height (int): Grid height
        underground_length (int): Maximum underground belt length
        pipe_underground_length (int): Maximum underground pipe length
        splitters (dict): Dictionary of available splitters by item type
        directions (list): Valid movement directions for pathfinding
    """
    
    def __init__(self, obstacle_map, points, allow_underground=False, underground_length=3, 
                 allow_splitters=False, splitters={}, find_optimal_paths=False, 
                 output_item=None, pipe_underground_length=8):
        """
        Initialize the MultiAgentPathfinder with configuration and constraints.
        
        Sets up the pathfinding environment including obstacle processing, grid
        initialization, and configuration of various pathfinding features.
        
        Args:
            obstacle_map (list): 2D grid where non-zero values represent obstacles
            points (dict): Dictionary mapping item keys to routing information:
                          - item: item type name
                          - start_points: list of starting positions
                          - destination: list of destination positions
                          - is_fluid: whether item is fluid (optional)
                          - inserter_mapping: mapping of positions to inserters (optional)
            allow_underground (bool, optional): Enable underground belt pathfinding. Defaults to False.
            underground_length (int, optional): Maximum underground belt length. Defaults to 3.
            allow_splitters (bool, optional): Enable belt splitter routing. Defaults to False.
            splitters (dict, optional): Dictionary of splitters by item type. Defaults to {}.
            find_optimal_paths (bool, optional): Find shortest paths instead of first valid. Defaults to False.
            output_item (str, optional): Priority item for output routing. Defaults to None.
            pipe_underground_length (int, optional): Maximum underground pipe length. Defaults to 8.
        """
        logger.info("Initializing MultiAgentPathfinder")
        logger.debug(f"Grid dimensions: {len(obstacle_map[0]) if obstacle_map else 0}x{len(obstacle_map)}")
        logger.debug(f"Number of items to route: {len(points)}")
        logger.debug(f"Underground enabled: {allow_underground}, Splitters enabled: {allow_splitters}")
        logger.debug(f"Optimal paths: {find_optimal_paths}, Output item: {output_item}")
        
        # Process obstacle map - convert non-zero values to 1 (obstacle) and zero to 0 (free)
        self.obstacle_map = [[1 if cell > 0 else 0 for cell in row] for row in obstacle_map]
        
        # Store grid dimensions
        self.height = len(obstacle_map)
        self.width = len(obstacle_map[0]) if self.height > 0 else 0
        logger.debug(f"Processed obstacle map: {self.width}x{self.height}")
        
        # Store routing configuration
        self.points = points
        
        # Underground belt configuration
        self.allow_underground = allow_underground
        self.underground_length = underground_length
        self.pipe_underground_length = pipe_underground_length
        logger.debug(f"Underground configuration: belt={underground_length}, pipe={pipe_underground_length}")
        
        # Splitter configuration
        self.allow_splitters = allow_splitters
        self.splitters = splitters
        if allow_splitters:
            logger.debug(f"Splitters available for {len(splitters)} item types")
        
        # Output prioritization
        self.output_item = output_item
        if output_item:
            logger.info(f"Output item prioritization enabled for: {output_item}")
        
        # Pathfinding optimization setting
        self.find_optimal_paths = find_optimal_paths
        if find_optimal_paths:
            logger.info("Optimal pathfinding enabled - will find shortest paths")
        
        # Initialize working grid as a copy of the obstacle map
        self.working_grid = [row[:] for row in obstacle_map]
        logger.debug("Working grid initialized as copy of obstacle map")
        
        # Initialize result storage
        self.paths = {}      # Store final paths for each item
        self.inserters = {}  # Store inserter placements
        
        # Define movement directions for pathfinding (right, down, left, up)
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        logger.info("MultiAgentPathfinder initialization completed successfully")
        
        
    def is_valid_position(self, position):
        """
        Check if a position is valid for pathfinding.
        
        A position is considered valid if it's within the grid bounds and
        not occupied by an obstacle in the current working grid.
        
        Args:
            position (tuple): Position coordinates (x, y) to check
            
        Returns:
            bool: True if position is valid for pathfinding, False otherwise
        """
        x, y = position
        
        # Check bounds and obstacle status
        is_valid = (0 <= x < self.width and 
                   0 <= y < self.height and 
                   self.working_grid[y][x] == 0)
        
        logger.debug(f"Position {position} validity check: {is_valid}")
        return is_valid
    
    def heuristic(self, a, b):
        """
        Calculate Manhattan distance heuristic between two points.
        
        The Manhattan distance is used as the heuristic function for A* pathfinding.
        It provides an optimistic estimate of the cost to reach the goal from a given position.
        
        Args:
            a (tuple): Starting position (x, y)
            b (tuple): Goal position (x, y)
            
        Returns:
            int: Manhattan distance between the two points
        """
        distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
        logger.debug(f"Heuristic distance from {a} to {b}: {distance}")
        return distance
    
    
    def expand_fluid_positions(self, positions):
        """
        Expand valid starting positions for fluid items to include adjacent tiles.
        
        For fluid items (like oil or water), pipe connections can be made from any
        adjacent tile to the specified position. This method expands the list of
        valid starting positions to include all adjacent non-obstacle tiles.
        
        Args:
            positions (list): List of original position tuples (x, y)
            
        Returns:
            list: Expanded list of positions including all valid adjacent tiles
        """
        logger.info(f"Expanding fluid positions from {len(positions)} original positions")
        logger.debug(f"Original positions: {positions}")
        
        expanded_positions = []
        
        # Process each original position
        for pos in positions:
            logger.debug(f"Processing position {pos} for fluid expansion")
            
            # Check all four adjacent directions
            for dx, dy in self.directions:
                adjacent_pos = (pos[0] + dx, pos[1] + dy)
                
                # Validate the adjacent position
                if (0 <= adjacent_pos[0] < self.width and 
                    0 <= adjacent_pos[1] < self.height and 
                    self.working_grid[adjacent_pos[1]][adjacent_pos[0]] == 0 and
                    adjacent_pos not in expanded_positions):
                    
                    expanded_positions.append(adjacent_pos)
                    logger.debug(f"Added adjacent position {adjacent_pos} for fluid item")
                else:
                    logger.debug(f"Rejected adjacent position {adjacent_pos} (out of bounds, obstacle, or duplicate)")
        
        logger.info(f"Fluid position expansion completed: {len(positions)} -> {len(expanded_positions)} positions")
        return expanded_positions

    def find_path(self, start, goal, is_fluid=False):
        """
        Find an optimal path from start to goal using A* algorithm with underground support.
        
        This method implements an enhanced A* pathfinding algorithm that can handle
        both standard surface paths and underground belt/pipe segments. The algorithm
        automatically chooses the best routing method based on obstacles and configuration.
        
        Args:
            start (tuple): Starting position (x, y)
            goal (tuple): Goal position (x, y)
            is_fluid (bool, optional): Whether this is a fluid item requiring pipes. Defaults to False.
            
        Returns:
            tuple: A 2-tuple containing:
                - path (list): List of positions [(x, y), ...] forming the path, or None if no path exists
                - underground_segments (dict): Dictionary of underground segments with format:
                  {'segment_0': {'start': (x, y), 'end': (x, y)}, ...} or empty if none used
        """
        logger.info(f"Finding path from {start} to {goal} (fluid: {is_fluid})")
        
        # If underground paths are not allowed, use standard A* algorithm
        if not self.allow_underground:
            logger.debug("Underground disabled, using standard A* pathfinding")
            path = self.find_path_astar(start, goal)
            if path:
                logger.info(f"Standard A* found path with length {len(path)}")
            else:
                logger.warning("Standard A* failed to find path")
            return path, {}
        
        logger.debug("Using enhanced A* with underground support")
        
        # Priority queue for A* algorithm
        # Format: (f_score, position, underground_segments_list)
        open_set = []
        heapq.heappush(open_set, (0, start, []))
        
        # Pathfinding data structures
        came_from = {}                    # Track parent nodes for path reconstruction
        g_score = {start: 0}             # Cost from start to each node
        f_score = {start: self.heuristic(start, goal)}  # Estimated total cost via each node
        underground_segments = {start: []}  # Track underground segments for each node
        
        logger.debug(f"A* search initialized with start={start}, goal={goal}")
        
        # Main A* search loop
        
        while open_set:
            # Get the node with lowest f_score
            _, current, current_segments = heapq.heappop(open_set)
            
            # If we reached the goal, reconstruct and return the path
            if current == goal:
                path = [current]
                node = current
                while node in came_from:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
                
                # Validate the underground segments - ensure they connect to the path
                valid_segments = []
                for segment in current_segments:
                    segment_start = segment['start']
                    segment_end = segment['end']
                    
                    # Check if segment start and end are in the path or connect to another segment
                    start_valid = segment_start in path
                    end_valid = segment_end in path
                    
                    # If either endpoint isn't in the path, check if it connects to another segment
                    if not start_valid:
                        for other_segment in current_segments:
                            if other_segment != segment and other_segment['end'] == segment_start:
                                start_valid = True
                                break
                    
                    if not end_valid:
                        for other_segment in current_segments:
                            if other_segment != segment and other_segment['start'] == segment_end:
                                end_valid = True
                                break
                    
                    # Only include segments where both endpoints are valid
                    if start_valid and end_valid:
                        valid_segments.append(segment)
                
                # Convert segments list to dictionary for easier processing
                segment_dict = {}
                for i, segment in enumerate(valid_segments):
                    segment_dict[f"segment_{i}"] = segment
                    
                return path, segment_dict
            
            # Check all adjacent nodes (normal moves)
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip invalid positions
                if not self.is_valid_position(neighbor):
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + 1  # Cost of 1 for each step
                
                # If we found a better path to this neighbor, update our records
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    underground_segments[neighbor] = current_segments.copy()
                    
                    # Add to open set if not already there
                    if not any(pos == neighbor for _, pos, _ in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor, current_segments.copy()))
            
            # Try underground paths if allowed
            if self.allow_underground:
                # Try in all four directions
                for dx, dy in self.directions:
                    # Check if we can go underground in this direction
                    can_go_underground, entry, exit, following = self.can_build_underground(current, dx, dy, goal,is_fluid)
                    
                    if can_go_underground:
                        # Underground paths have a higher cost
                        underground_cost = 5  # Higher cost for using underground
                        tentative_g_score = g_score[current] + underground_cost
                        
                        # Check if this gives us a better path
                        if exit not in g_score or tentative_g_score < g_score[exit]:
                            came_from[exit] = current
                            g_score[exit] = tentative_g_score
                            f_score[exit] = tentative_g_score + self.heuristic(exit, goal)
                            
                            # Create a new segments list with this underground segment added
                            new_segments = current_segments.copy()
                            new_segments.append({'start': entry, 'end': exit})
                            underground_segments[exit] = new_segments
                            
                            # Add to open set
                            if not any(pos == exit for _, pos, _ in open_set):
                                heapq.heappush(open_set, (f_score[exit], exit, new_segments))

                            # We also need to ensure the following position gets added to the path
                            # if it's not the goal itself
                            if following != goal and (following not in g_score or tentative_g_score + 1 < g_score[following]):
                                # Add the position after the exit with an additional step cost
                                came_from[following] = exit
                                g_score[following] = tentative_g_score + 1  # +1 for the extra step
                                f_score[following] = tentative_g_score + 1 + self.heuristic(following, goal)
                                
                                # Add to open set
                                if not any(pos == following for _, pos, _ in open_set):
                                    # No need to add underground segments for this position
                                    heapq.heappush(open_set, (f_score[following], following, underground_segments[exit]))
        # No path found
        return None, {}

    def can_build_underground(self, position, dx, dy, goal, is_fluid):
        """
        Check if we can build an underground path from the given position in the specified direction.
        
        This method determines whether an underground belt or pipe segment can be constructed
        to bypass obstacles. It validates the path length, checks for obstacles to cross,
        ensures proper spacing between segments, and confirms the exit position is valid.
        
        Underground Path Requirements:
        - Must have at least one obstacle to cross (otherwise surface path is preferred)
        - Exit position must be free and within grid bounds
        - Following position after exit must be free (for belt continuation)
        - Cannot create segments that connect directly to other segment endpoints
        - Must move the path closer to the goal to avoid backtracking
        
        Args:
            position (tuple): Starting position (x, y) for the underground segment
            dx (int): X direction vector (-1, 0, 1) for underground direction
            dy (int): Y direction vector (-1, 0, 1) for underground direction
            goal (tuple): The ultimate goal position for pathfinding
            is_fluid (bool): Whether this is a fluid item (affects underground length)
            
        Returns:
            tuple: A 4-tuple containing:
                - can_build (bool): Whether underground path is possible
                - entry_position (tuple): Entry point of underground segment (same as position)
                - exit_position (tuple): Exit point of underground segment
                - following_position (tuple): Position after exit for path continuation
        """
        logger.debug(f"Checking underground path from {position} in direction ({dx}, {dy})")
        
        # Determine underground length based on item type
        underground_length = self.pipe_underground_length if is_fluid else self.underground_length
        logger.debug(f"Underground length limit: {underground_length} ({'fluid' if is_fluid else 'belt'})")
        
        x, y = position
        entry = (x, y)  # The entry position is the current position
        
        # Don't allow diagonal movement for undergrounds
        if dx != 0 and dy != 0:
            logger.debug("Diagonal underground movement not allowed")
            return False, None, None, None
        
        underground_length = self.pipe_underground_length if is_fluid else self.underground_length
        
        x, y = position
        entry = (x, y)  # The entry position is the current position
        
        # Don't allow diagonal movement for undergrounds
        if dx != 0 and dy != 0:
            return False, None, None, None
        
        # Check if the current position was just used as an underground exit in the same direction
        # This avoids having underground segments directly connected to each other
        logger.debug("Checking for conflicting underground segments")
        
        # Check previous segments in existing paths
        for segment_dict in self.paths.values():
            for path_data in segment_dict:
                if 'underground_segments' in path_data and path_data['underground_segments']:
                    for _, segment in path_data['underground_segments'].items():
                        # If this position was an exit of a segment in the same direction
                        if segment['end'] == position:
                            # Get the direction vector of the previous segment
                            prev_dx = segment['end'][0] - segment['start'][0]
                            prev_dy = segment['end'][1] - segment['start'][1]
                            
                            # Normalize the direction vector
                            if prev_dx != 0:
                                prev_dx = prev_dx // abs(prev_dx)
                            if prev_dy != 0:
                                prev_dy = prev_dy // abs(prev_dy)
                            
                            # If the direction is the same, don't allow another underground
                            if prev_dx == dx and prev_dy == dy:
                                logger.debug(f"Underground blocked: position {position} already used as exit in same direction")
                                return False, None, None, None
        
        # Track segment ends to avoid placing segments right next to each other
        # This prevents cases where one segment ends at position A and another starts at position A
        logger.debug("Collecting existing segment endpoints")
        segment_ends = []
        for segment_dict in self.paths.values():
            for path_data in segment_dict:
                if 'underground_segments' in path_data and path_data['underground_segments']:
                    for _, segment in path_data['underground_segments'].items():
                        segment_ends.append(segment['end'])
                        
        # Don't start a new underground from a position that's already an end point
        if position in segment_ends:
            logger.debug(f"Underground blocked: position {position} is already an endpoint of another segment")
            return False, None, None, None
        
        # Check if there's at least one obstacle in the path that needs to be crossed
        has_obstacle = False
        logger.debug("Searching for valid underground exit points")
        
        # Look for a valid exit point within the underground length
        # Try increasing lengths from minimum (2) to maximum allowed
        for length in range(2, underground_length + 1):
            logger.debug(f"Trying underground length {length}")
            
            # Calculate potential exit position
            exit_x = x + dx * length
            exit_y = y + dy * length
            exit_pos = (exit_x, exit_y)
            
            # Calculate the position after the exit (needed for belt continuation)
            following_x = exit_x + dx
            following_y = exit_y + dy
            following_pos = (following_x, following_y)
            
            # Check if exit is within grid bounds
            if not (0 <= exit_x < self.width and 0 <= exit_y < self.height):
                logger.debug(f"Exit position {exit_pos} out of bounds")
                continue
                
            # Special case: If the exit is the goal itself, we don't need to check the following position
            if exit_pos == goal:
                logger.debug(f"Exit position {exit_pos} is the goal itself")
                
                # We still need to check for obstacles between current position and goal
                for i in range(1, length):
                    check_x = x + dx * i
                    check_y = y + dy * i
                    
                    # Check if this position is an obstacle
                    if 0 <= check_x < self.width and 0 <= check_y < self.height:
                        cell_value = self.working_grid[check_y][check_x]
                        if cell_value != 0:
                            has_obstacle = True
                            logger.debug(f"Found obstacle at ({check_x}, {check_y}) - underground justified")
                            break
                
                # If there's at least one obstacle, this is a valid underground to the goal
                if has_obstacle:
                    logger.debug(f"Valid underground path directly to goal from {entry} to {exit_pos}")
                    return True, entry, exit_pos, goal
                else:
                    logger.debug("No obstacles found - surface path would be better")
                continue
            
            # Normal case: Check the position following the exit
            # The following position is needed for belt continuation after underground exit
            if not (0 <= following_x < self.width and 0 <= following_y < self.height):
                logger.debug(f"Following position {following_pos} out of bounds")
                continue
            
            # Check if exit position is free
            if self.working_grid[exit_y][exit_x] != 0:
                logger.debug(f"Exit position {exit_pos} is occupied")
                continue
                
            # Check if following position is free
            if self.working_grid[following_y][following_x] != 0:
                logger.debug(f"Following position {following_pos} is occupied")
                continue
            
            # Check if there's at least one obstacle between the current position and the exit
            # that would require an underground path
            has_obstacle = False
            for i in range(1, length):
                check_x = x + dx * i
                check_y = y + dy * i
                
                # Check if this position is an obstacle
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    cell_value = self.working_grid[check_y][check_x]
                    if cell_value != 0:
                        has_obstacle = True
                        logger.debug(f"Found obstacle at ({check_x}, {check_y}) - underground justified")
                        break
            
            # If there's no obstacle to cross, don't use an underground
            if not has_obstacle:
                logger.debug("No obstacles found - surface path would be better")
                continue
            
            # Check if this exit is closer to the goal (to avoid backtracking)
            current_distance = self.heuristic(position, goal)
            exit_distance = self.heuristic(exit_pos, goal)
            
            if exit_distance < current_distance:
                logger.debug(f"Exit {exit_pos} is closer to goal (distance: {exit_distance} vs {current_distance})")
                
                # Ensure exit point isn't already an entry point for another segment
                is_valid_exit = True
                for segment_dict in self.paths.values():
                    for path_data in segment_dict:
                        if 'underground_segments' in path_data and path_data['underground_segments']:
                            for _, segment in path_data['underground_segments'].items():
                                if segment['start'] == exit_pos:
                                    is_valid_exit = False
                                    logger.debug(f"Exit {exit_pos} is already an entry point for another segment")
                                    break
                
                if is_valid_exit:
                    # This is a valid exit that's closer to the goal
                    logger.debug(f"Valid underground path found from {entry} to {exit_pos}")
                    return True, entry, exit_pos, following_pos
                else:
                    logger.debug(f"Exit {exit_pos} rejected because it's already an entry point")
            else:
                logger.debug(f"Exit {exit_pos} rejected because it doesn't get closer to goal")
        
        # No valid exit found
        logger.debug("No valid underground exit found")
        return False, None, None, None

    
    
    
    
    
    # This is a simpler version without underground handling for use when undergrounds are disabled
    def find_path_astar(self, start, goal):
        """
        Standard A* pathfinding algorithm without underground belt support.
        
        This method implements the classic A* pathfinding algorithm for finding
        the shortest path between two points on a grid. It's used as a fallback
        when underground paths are disabled or as a component of the enhanced
        pathfinding system.
        
        Algorithm Details:
        - Uses Manhattan distance as the heuristic function
        - Explores nodes in order of lowest f-score (g + h)
        - Reconstructs path by backtracking through parent nodes
        - Only considers orthogonal movement (no diagonal)
        
        Args:
            start (tuple): Starting position (x, y)
            goal (tuple): Goal position (x, y)
            
        Returns:
            list: List of positions [(x, y), ...] forming the path, or None if no path exists
        """
        logger.debug(f"Starting standard A* pathfinding from {start} to {goal}")
        
        # Priority queue for A* - stores (f_score, position)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Track parent nodes for path reconstruction
        came_from = {}
        
        # Track actual cost from start to each node
        g_score = {start: 0}
        
        # Track estimated total cost from start to goal via each node
        f_score = {start: self.heuristic(start, goal)}
        
        logger.debug(f"Initial heuristic distance: {f_score[start]}")
        
        # Main A* search loop
        while open_set:
            # Get the node with lowest f_score
            current_f, current = heapq.heappop(open_set)
            
            # If we reached the goal, reconstruct and return the path
            if current == goal:
                logger.debug("Goal reached, reconstructing path")
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                logger.info(f"Standard A* found path with length {len(path)}")
                return path
            
            # Explore all adjacent nodes
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip invalid positions (out of bounds or obstacles)
                if not self.is_valid_position(neighbor):
                    continue
                
                # Calculate tentative g_score (cost from start to neighbor via current)
                tentative_g_score = g_score[current] + 1  # Cost of 1 for each step
                
                # If we found a better path to this neighbor, update our records
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if not any(pos == neighbor for _, pos in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        logger.warning(f"Standard A* failed to find path from {start} to {goal}")
        return None
    def mark_path_on_grid(self, path, value=1, underground_segments=None, start_splitter=None, dest_splitter=None, inserter=None):
        """
        Mark a path and its associated components on the working grid with the specified value.
        
        This method updates the working grid to mark a completed path as occupied,
        preventing future paths from using the same positions. It handles all
        components of a path including the surface route, underground segments,
        splitter positions, and inserter locations.
        
        Grid Marking Strategy:
        - Surface path: All positions marked as obstacles
        - Underground segments: Entry/exit points and intermediate positions marked
        - Splitter positions: Both main and secondary positions marked
        - Inserter positions: Marked to prevent conflicts
        
        Args:
            path (list): List of (x, y) positions in the surface path
            value (int, optional): Value to mark on the grid. Defaults to 1 (obstacle).
            underground_segments (dict, optional): Dictionary of underground segments to mark.
                                                  Format: {'segment_0': {'start': (x, y), 'end': (x, y)}}
            start_splitter (Splitter, optional): Starting splitter object to mark
            dest_splitter (Splitter, optional): Destination splitter object to mark
            inserter (tuple, optional): Inserter position (x, y) to mark
        """
        logger.debug(f"Marking path on grid with value {value}")
        
        # Handle empty path
        if path is None:
            logger.warning("Cannot mark empty path on grid")
            return
        
        # Mark the entire surface path
        logger.debug(f"Marking {len(path)} surface path positions")
        for i, (x, y) in enumerate(path):
            if 0 <= x < self.width and 0 <= y < self.height:
                prev_value = self.working_grid[y][x]
                self.working_grid[y][x] = value
                logger.debug(f"Marked position ({x}, {y}) with value {value} (was {prev_value})")
        
        # Mark underground segments
        if underground_segments:
            logger.debug(f"Marking {len(underground_segments)} underground segments")
            for segment_id, segment in underground_segments.items():
                start_x, start_y = segment['start']
                end_x, end_y = segment['end']
                
                logger.debug(f"Marking underground segment {segment_id} from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                
                # Mark the entrance and exit points
                self.working_grid[start_y][start_x] = value
                self.working_grid[end_y][end_x] = value
                
                # Mark all intermediate positions along the underground route
                dir_x = 0 if start_x == end_x else (end_x - start_x) // abs(end_x - start_x)
                dir_y = 0 if start_y == end_y else (end_y - start_y) // abs(end_y - start_y)
                
                curr_x, curr_y = start_x, start_y
                while (curr_x, curr_y) != (end_x, end_y):
                    curr_x += dir_x
                    curr_y += dir_y
                    if (curr_x, curr_y) != (end_x, end_y):  # Don't mark the end point twice
                        self.working_grid[curr_y][curr_x] = value
                        logger.debug(f"Marked underground intermediate position ({curr_x}, {curr_y})")
        
        # Mark splitter positions
        if start_splitter:
            logger.debug(f"Marking start splitter at {start_splitter.position}")
            pos_x, pos_y = start_splitter.position
            self.working_grid[pos_y][pos_x] = value
            
            if start_splitter.next_position:
                next_x, next_y = start_splitter.next_position
                self.working_grid[next_y][next_x] = value
                logger.debug(f"Marked start splitter secondary position at ({next_x}, {next_y})")
        
        if dest_splitter:
            logger.debug(f"Marking destination splitter at {dest_splitter.position}")
            pos_x, pos_y = dest_splitter.position
            self.working_grid[pos_y][pos_x] = value
            
            if dest_splitter.next_position:
                next_x, next_y = dest_splitter.next_position
                self.working_grid[next_y][next_x] = value
                logger.debug(f"Marked destination splitter secondary position at ({next_x}, {next_y})")
        
        # Mark inserter position
        if inserter:
            inserter_x, inserter_y = inserter
            if 0 <= inserter_x < self.width and 0 <= inserter_y < self.height:
                prev_value = self.working_grid[inserter_y][inserter_x]
                self.working_grid[inserter_y][inserter_x] = value
                logger.debug(f"Marked inserter at ({inserter_x}, {inserter_y}) with value {value} (was {prev_value})")
        
        logger.debug("Path marking completed successfully")
    
                            
    def place_output_inserter(self):
        """
        Place inserters at the output item locations for priority routing.
        
        This method handles the placement of inserters for output items, which are
        items that need to be extracted from the factory system. It manages cases
        where multiple output items might share inserter positions and ensures
        proper inserter placement validation.
        
        Output Item Processing:
        - Identifies all items matching the designated output item type
        - Validates inserter positions are within bounds and available
        - Handles shared inserter positions between multiple outputs
        - Updates the working grid to mark inserter positions as occupied
        - Prioritizes start points based on inserter availability
        
        Inserter Placement Strategy:
        - Prefer unused inserter positions over shared ones
        - Validate inserter positions are not occupied by obstacles
        - Update item start points to prioritize valid inserter placements
        - Store inserter mappings for later use in path generation
        
        Returns:
            set: Set of used inserter position strings, or None if no output item defined
        """
        if not self.output_item:
            logger.info("No output item defined, skipping output inserter placement")
            return None
        
        logger.info(f"Placing output inserters for item type: {self.output_item}")
        
        # Track which inserter positions have already been used
        used_inserters = set()
        
        # Process each item that matches the output item type
        for item_key, item_data in self.points.items():
            if item_data.get('item') != self.output_item:
                logger.debug(f"Skipping {item_key} - not an output item (type: {item_data.get('item')})")
                continue
                
            logger.info(f"Processing output item: {item_key}")
            
            # Extract item configuration
            start_points = item_data.get('start_points', [])
            destinations = item_data.get('destination', [])
            inserter_mapping = item_data.get('inserter_mapping', {})
            
            logger.debug(f"Start points for {item_key}: {start_points}")
            logger.debug(f"Destination points for {item_key}: {destinations}")
            logger.debug(f"Inserter mapping entries: {len(inserter_mapping)}")
            
            # Validate item configuration
            if not start_points:
                logger.warning(f"No start points defined for {item_key}, cannot place output inserter")
                continue
                
            if not destinations:
                logger.warning(f"No destination points defined for {item_key}, cannot place output inserter")
                continue
            
            if not inserter_mapping:
                logger.warning(f"No inserter mapping defined for {item_key}, cannot place output inserter")
                continue
            
            # Generate all valid start points with their inserters
            logger.debug(f"Evaluating inserter placements for {len(start_points)} start points")
            valid_starts = []
            
            for start_pos in start_points:
                start_pos_str = str(start_pos)
                
                if start_pos_str not in inserter_mapping:
                    logger.debug(f"No inserter mapping for start position {start_pos}, skipping")
                    continue
                    
                inserter_pos = inserter_mapping[start_pos_str]
                inserter_pos_str = str(inserter_pos)
                
                # Check if inserter position is within grid bounds
                ix, iy = inserter_pos
                if not (0 <= ix < self.width and 0 <= iy < self.height):
                    logger.warning(f"Inserter position {inserter_pos} is out of bounds, skipping")
                    continue
                    logger.warning(f"Inserter position {inserter_pos} is out of bounds, skipping")
                    continue
                
                # Check if this inserter has already been used
                if inserter_pos_str in used_inserters:
                    # This inserter is already placed - it's still valid, just mark it
                    valid_starts.append((start_pos, inserter_pos, True))
                    logger.debug(f"Inserter already exists at {inserter_pos} for {start_pos}")
                else:
                    # Check if the position is free in the working grid
                    if self.working_grid[iy][ix] == 0:
                        valid_starts.append((start_pos, inserter_pos, False))
                        logger.debug(f"Found valid inserter placement at {inserter_pos} for start {start_pos}")
                    else:
                        logger.warning(f"Inserter position {inserter_pos} is occupied (value: {self.working_grid[iy][ix]})")
            
            # Sort valid starts: unused inserters first, then used ones
            # This prioritizes fresh inserter placements over shared ones
            valid_starts.sort(key=lambda x: x[2])  # False (unused) comes first
            logger.debug(f"Found {len(valid_starts)} valid inserter placements for {item_key}")
            
            if not valid_starts:
                logger.warning(f"No valid inserter placements found for {item_key}")
                continue
            
            # Use the first valid start (prioritizing unused inserters)
            start_pos, inserter_pos, already_used = valid_starts[0]
            logger.info(f"Selected inserter placement: {inserter_pos} for start {start_pos} (already used: {already_used})")
            
            # Place the inserter if it hasn't been placed already
            if not already_used:
                ix, iy = inserter_pos
                prev_value = self.working_grid[iy][ix]
                self.working_grid[iy][ix] = 1
                logger.info(f"Placed inserter at {inserter_pos} (previous grid value: {prev_value})")
                used_inserters.add(str(inserter_pos))
            else:
                logger.debug(f"Inserter at {inserter_pos} already placed, reusing")
            
            # Update the start points to prioritize the selected one
            old_start_points = self.points[item_key]['start_points']
            self.points[item_key]['start_points'] = [start_pos]
            logger.info(f"Updated start_points for {item_key}: {old_start_points} -> {[start_pos]}")
            
            # Store the inserter in our collection
            if item_key not in self.inserters:
                self.inserters[item_key] = {}
                
            self.inserters[item_key][str(start_pos)] = inserter_pos
            logger.debug(f"Stored inserter mapping: {item_key}[{start_pos}] -> {inserter_pos}")
        
        # Return results
        if not used_inserters:
            logger.warning("Failed to place any output inserters")
            return None
            
        logger.info(f"Successfully placed {len(used_inserters)} unique inserters")
        return used_inserters
      
    
    
    def find_paths_for_all_items(self, IO_paths=False):
        """
        Find optimal paths for all items in the factory layout.
        
        This is the main pathfinding method that coordinates the routing of all
        items from their production points to consumption points. It handles
        prioritization, conflict resolution, and optimization across multiple
        item types while respecting various constraints.
        
        Processing Strategy:
        1. Sort items with output items processed first (priority routing)
        2. Place output inserters for priority items
        3. Process splitters to determine I/O points
        4. Find paths for each item in priority order
        5. Handle fluid item expansion for pipe connections
        6. Manage splitter routing for complex belt networks
        7. Resolve conflicts between overlapping paths
        
        Path Selection Algorithm:
        - Evaluates all possible start-destination pairs
        - Sorts by heuristic distance for efficiency
        - Attempts paths in order of increasing complexity
        - Marks successful paths to prevent conflicts
        - Supports optimal path finding when enabled
        
        Args:
            IO_paths (bool, optional): Whether to create I/O specific paths. Defaults to False.
            
        Returns:
            tuple: A 2-tuple containing:
                - paths (dict): Dictionary mapping item keys to their path data
                - inserters (dict): Dictionary mapping item keys to their inserter positions
        """
        logger.info("Starting pathfinding for all items")
        logger.debug(f"Input/Output paths mode: {IO_paths}")
        
        # Separate output items from other items for priority processing
        output_items = {}
        other_items = {}
        
        for item_key, item_data in self.points.items():
            if item_data.get('item') == self.output_item:
                output_items[item_key] = item_data
            else:
                other_items[item_key] = item_data
    
        # Create processing order - output items first for priority routing
        sorted_keys = list(output_items.keys()) + list(other_items.keys())
        logger.info(f"Processing {len(sorted_keys)} items in priority order")
        logger.debug(f"Output items: {len(output_items)}, Other items: {len(other_items)}")
        
        # Place output inserters if output item is defined
        if self.output_item:
            logger.info("Placing output inserters for priority routing")
            self.place_output_inserter()
        
        # Process all splitters first to determine positions and I/O points
        if self.allow_splitters:
            logger.info("Processing splitters for belt routing")
            self.process_splitters()
        
        # Initialize tracking variables
        all_previous_paths = {}   # Track paths for merging with same item types
        used_splitters = {}       # Track which splitters are used for I/O
        
        # Main pathfinding loop for each item
        for item_key in sorted_keys:
            item_data = self.points[item_key]
            
            logger.info(f"=== Processing item: {item_key} ===")
            logger.debug(f"Item data: {item_data}")
            
            # Check if this is a fluid item (affects pathfinding behavior)
            is_fluid = item_data.get('is_fluid', False)
            logger.debug(f"Item {item_key} is fluid: {is_fluid}")
            
            # Extract basic item information
            start_points = item_data['start_points'].copy()
            destinations = item_data['destination'].copy()
            inserter_mapping = item_data.get('inserter_mapping', None)
            item_name = item_data['item']
            
            logger.debug(f"Initial start points: {start_points}")
            logger.debug(f"Initial destinations: {destinations}")
            logger.debug(f"Has inserter mapping: {inserter_mapping is not None}")
            
            # Handle fluid item expansion (pipes can connect from adjacent tiles)
            if is_fluid and not IO_paths:
                logger.info(f"Expanding fluid connection points for {item_key}")
                original_count = len(start_points)
                start_points = self.expand_fluid_positions(start_points)
                logger.info(f"Fluid expansion: {original_count} -> {len(start_points)} start points")
            
            # Handle splitter routing for belt networks (non-fluid items only)
            if self.allow_splitters and item_name in self.splitters and not is_fluid:
                logger.info(f"Processing splitter routing for {item_key}")
                
                # Find splitters that are relevant to our start/destination points
                relevant_start_splitters = []
                relevant_dest_splitters = []
                
                # Check each splitter to see if it's positioned at one of our start/destination points
                for splitter in self.splitters[item_name]:
                    # A splitter is relevant to start points if its position matches a start point
                    for start_point in item_data['start_points']:
                        if splitter.position == start_point:
                            relevant_start_splitters.append(splitter)
                            logger.debug(f"Found splitter at start point {start_point}")
                    
                    # A splitter is relevant to destination points if its position matches a destination
                    for dest_point in item_data['destination']:
                        if splitter.position == dest_point:
                            relevant_dest_splitters.append(splitter)
                            logger.debug(f"Found splitter at destination point {dest_point}")
                
                logger.debug(f"Found {len(relevant_start_splitters)} start splitters and {len(relevant_dest_splitters)} destination splitters")
                
                # If we have start splitters, replace start points with their outputs
                if len(relevant_start_splitters) > 0:
                    logger.info("Replacing start points with splitter outputs")
                    start_points = []
                
                # If we have destination splitters, replace destinations with their inputs
                if len(relevant_dest_splitters) > 0:
                    logger.info("Replacing destinations with splitter inputs")
                    destinations = []
                
                # Add output points from start splitters as start points
                for splitter in relevant_start_splitters:
                    for output_point in splitter.outputs:
                        if self.is_valid_position(output_point):
                            start_points.append(output_point)
                            logger.debug(f"Added splitter output {output_point} as start point")
                
                # Add input points from destination splitters as destinations
                for splitter in relevant_dest_splitters:
                    for input_point in splitter.inputs:
                        if self.is_valid_position(input_point):
                            destinations.append(input_point)
                            logger.debug(f"Added splitter input {input_point} as destination")
            
            
            
            # Handle output item merging (merge existing paths of same item type as destinations)
            if self.output_item and item_name == self.output_item:
                logger.info(f"Processing output item {item_key} - merging existing paths as destinations")
                
                for other_item_key, other_paths in self.paths.items():
                    for other_path_data in other_paths:  # Iterate through each path data in the list
                        if other_item_key != item_key and other_path_data['item'] == item_name:
                            logger.debug(f"Merging path from {other_item_key} with same item type")
                            
                            # Get underground segments to exclude their endpoints
                            underground_segments = other_path_data.get('underground_segments', {})
                            underground_entries = []
                            underground_exits = []
                            
                            # Collect all underground entry/exit points
                            for segment in underground_segments.values():
                                underground_entries.append(segment['start'])
                                underground_exits.append(segment['end'])
                            
                            # Only add points that are not part of underground segments
                            filtered_path = []
                            for point in other_path_data['path']:
                                if point not in underground_entries and point not in underground_exits:
                                    filtered_path.append(point)
                            
                            destinations.extend(filtered_path)
                            logger.debug(f"Added {len(filtered_path)} destination points from {other_item_key} "
                                       f"(filtered out {len(other_path_data['path']) - len(filtered_path)} underground points)")
                            
            # Handle non-output item merging (merge destinations of same item type as start points)
            elif item_name != self.output_item:
                logger.debug(f"Processing non-output item {item_key} - merging destinations as start points")
                
                for other_item_key, other_paths in self.paths.items():
                    for other_path_data in other_paths:  # Iterate through each path data in the list
                        if other_item_key != item_key and other_path_data['item'] == item_name:
                            logger.debug(f"Checking path from {other_item_key} for destination merging")
                            
                            # Check if destination is not part of an underground segment
                            underground_segments = other_path_data.get('underground_segments', {})
                            destination_is_underground = False
                            
                            for segment in underground_segments.values():
                                if (segment['end'] == other_path_data['destination'] or 
                                    segment['start'] == other_path_data['destination']):
                                    destination_is_underground = True
                                    break
                            
                            if not destination_is_underground:
                                start_points.extend([other_path_data['destination']])
                                logger.debug(f"Added destination from {other_item_key} as start point")
                                
            # Check if start points and destination points have the same coordinates
            # If so, skip pathfinding and just handle inserter placement
            finished = False
            for point in start_points:
                if point in destinations:
                    finished = True
                    logger.info(f"Start point {point} is also a destination - no pathfinding needed")
                    
                    # Check if we need to place an inserter at the shared point
                    if inserter_mapping and str(point) in inserter_mapping:
                        inserter = inserter_mapping[str(point)]
                        logger.info(f"Placing inserter at {inserter} for shared start/destination point {point}")
                        
                        # Check if inserter position is within bounds
                        ix, iy = inserter
                        if 0 <= ix < self.width and 0 <= iy < self.height:
                            # Mark inserter position as obstacle
                            prev_value = self.working_grid[iy][ix]
                            self.working_grid[iy][ix] = 1
                            
                            # Store inserter reference
                            if item_key not in self.inserters:
                                self.inserters[item_key] = {}
                            self.inserters[item_key][str(point)] = inserter
                            
                            logger.info(f"Marked inserter at {inserter} as obstacle (previous value: {prev_value})")
                        else:
                            logger.warning(f"Inserter position {inserter} is out of bounds")
                    
                    break
            
            # Skip pathfinding if start and destination are the same
            if finished:
                logger.info(f"Skipping pathfinding for {item_key} - start and destination are the same")
                continue
            
            # Ensure start and destination points are valid on the working grid
            logger.debug("Temporarily marking start and destination points as valid")
            for start in start_points:
                x, y = start
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.working_grid[y][x] = 0  # Mark as valid
            
            for dest in destinations:
                x, y = dest
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.working_grid[y][x] = 0  # Mark as valid
            
            # Create a list of all possible start-destination pairs with their heuristic distance
            logger.debug(f"Generating start-destination pairs: {len(start_points)} starts × {len(destinations)} destinations")
            all_pairs = []
            for start in start_points:
                # Skip out of bounds start points
                if not (0 <= start[0] < self.width and 0 <= start[1] < self.height):
                    continue
                    
                for dest in destinations:
                    # Skip out of bounds destinations
                    if not (0 <= dest[0] < self.width and 0 <= dest[1] < self.height):
                        continue
           
                    # Check if this is a splitter start or destination
                    start_splitter = None
                    dest_splitter = None
                    
                    # Check if start point is from a splitter's outputs
                    if self.allow_splitters and item_name in self.splitters:
                        for splitter in relevant_start_splitters:
                            if start in splitter.outputs:
                                start_splitter = splitter
                                logger.info(f"Start point {start} is an output of splitter at {splitter.position}")
                                break

                        # Check if destination point is from a splitter's inputs
                        for splitter in relevant_dest_splitters:
                            if dest in splitter.inputs:
                                dest_splitter = splitter
                                logger.info(f"Destination point {dest} is an input of splitter at {splitter.position}")
                                break
                    
                    # Calculate heuristic distance
                    h_distance = self.heuristic(start, dest)
                    
                    
                    all_pairs.append((h_distance, start, dest, start_splitter, dest_splitter))
                    
            #if not self.find_optimal_paths:
                # Sort pairs by heuristic distance (ascending)
            all_pairs.sort()
            
            # Try to find a path for each pair in order of increasing heuristic distance
            path_found = False
            best_path = None
            best_start = None
            best_dest = None
            best_inserter = None
            best_underground_segments = None
            best_start_splitter = None
            best_dest_splitter = None
            best_path_length = float('inf')  # Track the best path length
            
            logger.debug(f"All pairs sorted by heuristic distance: {all_pairs}")
            
            # Completely revise the way we handle marking and restoring splitter positions
            for h_distance, start, dest, start_splitter, dest_splitter in all_pairs:
                # Skip pairs with heuristic distance greater than our current best path
                if self.find_optimal_paths and path_found and h_distance >= best_path_length:
                    continue
                
                # Make a deep copy of the working grid for this iteration
                # This ensures we start fresh for each path attempt
                iteration_grid = [row[:] for row in self.working_grid]
                
                # Check if we need to place an inserter at this start point
                inserter = None
                if inserter_mapping and str(start) in inserter_mapping and not start_splitter:
                    inserter = inserter_mapping[str(start)]
                    # Temporarily mark inserter position as obstacle
                    ix, iy = inserter
                    iteration_grid[iy][ix] = 1
                
                # Mark all splitters for the current item as obstacles EXCEPT
                # the specific I/O points we're trying to use
                if self.allow_splitters and item_name in self.splitters:
                    for splitter in self.splitters[item_name]:
                        if splitter.position and splitter.next_position:
                            pos_x, pos_y = splitter.position
                            next_x, next_y = splitter.next_position
                            
                            # Mark main position as obstacle unless it's our current start/dest
                            if (pos_x, pos_y) != start and (pos_x, pos_y) != dest:
                                iteration_grid[pos_y][pos_x] = 1
                            
                            # Mark next position as obstacle unless it's our current start/dest
                            if (next_x, next_y) != start and (next_x, next_y) != dest:
                                iteration_grid[next_y][next_x] = 1
                
                # Always ensure the start and destination points are valid
                if 0 <= start[0] < self.width and 0 <= start[1] < self.height:
                    iteration_grid[start[1]][start[0]] = 0
                
                if 0 <= dest[0] < self.width and 0 <= dest[1] < self.height:
                    iteration_grid[dest[1]][dest[0]] = 0
                
                # Temporarily replace the working grid with our iteration grid
                original_working_grid = self.working_grid
                self.working_grid = iteration_grid
                
                # Try to find a path with the modified grid
                path, underground_segments = self.find_path(start, dest,is_fluid=is_fluid)
                
                logger.debug(f"Path found: {path} with underground segments: {underground_segments}")
                
                # Restore the original working grid
                self.working_grid = original_working_grid
                
                # If we found a path, check if it's better than our current best
                if path:
                    # Calculate actual path length
                    path_length = len(path)
                    
                    
                    # Update best path if this one is better (or if it's the first valid path)
                    if not path_found or path_length < best_path_length:
                        best_path = path
                        best_start = start
                        best_dest = dest
                        best_inserter = inserter
                        best_underground_segments = underground_segments
                        best_start_splitter = start_splitter
                        best_dest_splitter = dest_splitter
                        best_path_length = path_length
                        path_found = True
                        
                        logger.info(f"Found {'better ' if path_found else ''}path for {item_key} "
                                    f"from {start} to {dest} with length {path_length}")
                    
                    # If we're not finding optimal paths, stop after the first valid path
                    if not self.find_optimal_paths:
                        break
                    
                logger.debug(f"path_found{path_found} for {item_key} from {start} to {dest}")
                # If we found a path, add it to our results
            if path_found:
                    
                    logger.debug(f"Adding path for {item_key} from {best_start} to {best_dest}")
                    
                    # Add the path to our results
                    if item_key not in self.paths:
                        self.paths[item_key] = []
                    
                    # Track if this path uses splitters at start or destination
                    uses_splitter_start = best_start_splitter is not None
                    uses_splitter_dest = best_dest_splitter is not None
                    
                    # Add to used_splitters if splitters are used
                    if uses_splitter_start or uses_splitter_dest:
                        if item_name not in used_splitters:
                            used_splitters[item_name] = []
                        
                        if uses_splitter_start and best_start_splitter not in used_splitters[item_name]:
                            used_splitters[item_name].append(best_start_splitter)
                        
                        if uses_splitter_dest and best_dest_splitter not in used_splitters[item_name]:
                            used_splitters[item_name].append(best_dest_splitter)
                    
                    self.paths[item_key].append({
                        'item': item_name,
                        'path': best_path,
                        'start': best_start,
                        'destination': best_dest,
                        'underground_segments': best_underground_segments,
                        'start_splitter': best_start_splitter,
                        'dest_splitter': best_dest_splitter
                    })
                    
                    # Add inserter information if applicable
                    if best_inserter:
                        if item_key not in self.inserters:
                            self.inserters[item_key] = {}
                        self.inserters[item_key][str(best_start)] = best_inserter
                    
                    logger.debug("======================================================")
                    logger.debug(f"Marking path on grid for {item_key} from {best_start} to {best_dest}")
                    logger.debug(f"BEFORE :Working grid:\n{np.array(self.working_grid)}")
                    # Mark the path on the working grid along with underground segments and splitters
                    self.mark_path_on_grid(best_path, value=1, underground_segments=best_underground_segments, 
                       start_splitter=best_start_splitter, dest_splitter=best_dest_splitter,inserter=best_inserter)
                    
                    logger.debug(f"AFTER: Working grid:\n{np.array(self.working_grid)}")
                    logger.debug("======================================================")
                    # Add this path to all_previous_paths for future items to merge with
                    if item_key not in all_previous_paths:
                        all_previous_paths[item_key] = []
                    all_previous_paths[item_key].append({'path': best_path})
                        
            else:
                    # No path found for this item
                    logger.warning(f"No path found for {item_key} with any start-destination pair")
            
        return self.paths, self.inserters
        
    def visualize_grid(self, filename='grid.png'):
        """
        Visualize the current state of the working grid.
        
        Creates a visual representation of the working grid showing obstacles,
        paths, and other marked positions. This is useful for debugging and
        understanding the pathfinding results.
        
        Visualization Features:
        - Color-coded cell types using viridis colormap
        - Grid lines for position reference
        - Obstacle positions clearly marked
        - Path positions shown as marked cells
        - Colorbar legend for cell type interpretation
        
        Args:
            filename (str, optional): Output filename for the visualization. Defaults to 'grid.png'.
            
        Returns:
            str: The filename of the saved visualization
        """
        logger.info(f"Creating grid visualization: {filename}")
        plt.figure(figsize=(10, 10))
        
        # Create a colormap (updated to avoid deprecation warning)
        import matplotlib as mpl
        cmap = mpl.colormaps['viridis'].resampled(15)
        
        # Plot the grid
        plt.imshow(self.working_grid, cmap=cmap, interpolation='nearest')
        
        # Add a colorbar
        plt.colorbar(label='Cell Type')
        
        # Add grid lines for position reference
        plt.grid(True, color='black', linewidth=0.5, alpha=0.3)
        
        # Adjust the grid to match the cell centers
        plt.gca().set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add title and labels
        plt.title('Factory Layout Grid')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Save the figure
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Grid visualization saved as {filename}")
        return filename

    def visualize_paths(self, filename_template='path_{}.png'):
        """Visualize each item's path separately with splitters."""
        for item_key, item_paths in self.paths.items():
            plt.figure(figsize=(10, 10))
            
            # Create a base grid with just obstacles
            base_grid = np.array(self.obstacle_map)
            
            # Create a colormap (updated to avoid deprecation warning)
            import matplotlib as mpl
            cmap = mpl.colormaps['viridis'].resampled(15)
            
            # Plot the grid
            plt.imshow(base_grid, cmap=cmap, interpolation='nearest')
            
            # Add grid lines
            plt.grid(True, color='black', linewidth=0.5, alpha=0.3)
            plt.gca().set_xticks(np.arange(-0.5, self.width, 1), minor=True)
            plt.gca().set_yticks(np.arange(-0.5, self.height, 1), minor=True)
            plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Extract item name (without index)
            item_name = item_key.split('_')[0] if '_' in item_key else item_key
            
            # Plot splitters for this item if any
            if self.allow_splitters and item_name in self.splitters:
                for splitter in self.splitters[item_name]:
                    # Draw the splitter as a rectangle
                    pos = splitter.position
                    next_pos = splitter.next_position
                    
                    if next_pos:
                        # Plot splitter main body as a rectangle
                        min_x = min(pos[0], next_pos[0])
                        max_x = max(pos[0], next_pos[0])
                        min_y = min(pos[1], next_pos[1])
                        max_y = max(pos[1], next_pos[1])
                        
                        # Create a rectangle patch
                        from matplotlib.patches import Rectangle
                        rect = Rectangle((min_x - 0.5, min_y - 0.5), max_x - min_x + 1, max_y - min_y + 1, 
                                        linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.5)
                        plt.gca().add_patch(rect)
                        
                        # Draw an arrow to show direction
                        dx = splitter.direction[0] * 0.4
                        dy = splitter.direction[1] * 0.4
                        center_x = (pos[0] + next_pos[0]) / 2
                        center_y = (pos[1] + next_pos[1]) / 2
                        plt.arrow(center_x, center_y, dx, dy, head_width=0.2, head_length=0.3, 
                                fc='black', ec='black', alpha=0.8)
                        
                        # Mark input points
                        for input_pos in splitter.inputs:
                            plt.plot(input_pos[0], input_pos[1], 'r>', markersize=8)  # Input as red right arrow
                        
                        # Mark output points
                        for output_pos in splitter.outputs:
                            plt.plot(output_pos[0], output_pos[1], 'g<', markersize=8)  # Output as green left arrow
                    else:
                        # If next_pos is not defined, just mark the position
                        plt.plot(pos[0], pos[1], 'ys', markersize=10)  # Yellow square
            
            # Plot each path
            for path_data in item_paths:
                path = path_data['path']
                path_x = [pos[0] for pos in path]
                path_y = [pos[1] for pos in path]
                
                # Plot the path with arrows to show direction
                plt.plot(path_x, path_y, 'r-', linewidth=2)
                
                # Add arrows to show direction
                for i in range(len(path) - 1):
                    dx = path[i+1][0] - path[i][0]
                    dy = path[i+1][1] - path[i][1]
                    plt.arrow(path[i][0], path[i][1], dx * 0.6, dy * 0.6, 
                            head_width=0.2, head_length=0.3, fc='blue', ec='blue')
                
                # Mark start and destination
                plt.plot(path[0][0], path[0][1], 'go', markersize=10)  # Start
                plt.plot(path[-1][0], path[-1][1], 'mo', markersize=10)  # Destination
                
                # Highlight splitter connections
                if path_data.get('start_splitter'):
                    start_pos = path[0]
                    plt.plot(start_pos[0], start_pos[1], 'yo', markersize=12, alpha=0.7)  # Yellow circle
                    plt.text(start_pos[0], start_pos[1], 'S', ha='center', va='center', color='black', fontweight='bold')
                
                if path_data.get('dest_splitter'):
                    dest_pos = path[-1]
                    plt.plot(dest_pos[0], dest_pos[1], 'yo', markersize=12, alpha=0.7)  # Yellow circle
                    plt.text(dest_pos[0], dest_pos[1], 'D', ha='center', va='center', color='black', fontweight='bold')
                
                # Add inserter if applicable
                item_inserters = self.inserters.get(item_key, {})
                if str(path[0]) in item_inserters:
                    ins_x, ins_y = item_inserters[str(path[0])]
                    plt.plot(ins_x, ins_y, 'ys', markersize=10)  # Inserter
                
                # Plot underground segments if any
                if 'underground_segments' in path_data and path_data['underground_segments']:
                    for segment_key, segment in path_data['underground_segments'].items():
                        start_pos = segment['start']
                        end_pos = segment['end']
                        
                        # Plot thick yellow line for underground segments
                        plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                                'y-', linewidth=4, alpha=0.6)
                        
                        # Mark entry and exit points with special markers
                        plt.plot(start_pos[0], start_pos[1], 'kv', markersize=8)  # Entry (down arrow)
                        plt.plot(end_pos[0], end_pos[1], 'k^', markersize=8)  # Exit (up arrow)
            
            # Add a legend
            handles = [
                plt.Line2D([0], [0], color='r', linewidth=2, label='Path'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Start'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=10, label='Destination'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='y', markersize=10, label='Splitter'),
                plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='r', markersize=8, label='Splitter Input'),
                plt.Line2D([0], [0], marker='<', color='w', markerfacecolor='g', markersize=8, label='Splitter Output'),
                plt.Line2D([0], [0], color='y', linewidth=4, alpha=0.6, label='Underground'),
            ]
            plt.legend(handles=handles, loc='upper right')
            
            # Add a title
            plt.title(f"Paths for {item_key}")
            
            # Save the figure
            plt.savefig(filename_template.format(item_key.replace('/', '_')))
            plt.close()

    def process_splitters(self):
        """
        Process all splitters to determine valid placements and I/O points.
        For each splitter, determine if it can be positioned correctly with at least
        one valid I/O point.
        
        Returns:
            dict: Dictionary of valid splitters by item type
        """
        
        logger.info("Processing all splitters")
        
        # Track valid splitter positions to avoid overlaps
        occupied_positions = set()
        valid_splitters = {}
        
        # First pass: Determine valid splitter placements
        for item, splitter_list in self.splitters.items():
            valid_splitters[item] = []
            
            for splitter in splitter_list:
                pos = splitter.position  # Anchor position (where belt connects)
                direction = splitter.direction  # Direction the splitter faces
                
                # Determine the next position based on direction
                # For a splitter that's 2x1 with long side perpendicular to direction:
                next_positions = []
                if direction[0] != 0:  # Horizontal direction (left/right)
                    # For horizontal splitters, check positions above and below
                    next_positions = [(pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]
                else:  # Vertical direction (up/down)
                    # For vertical splitters, check positions to left and right
                    next_positions = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1])]
                
                # Check each potential next position
                for next_pos in next_positions:
                    # Skip if next position is out of bounds
                    if not (0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height):
                        continue
                    
                    # Skip if next position is already occupied
                    if next_pos in occupied_positions or self.obstacle_map[next_pos[1]][next_pos[0]] != 0:
                        continue
                    
                    # At this point, both anchor and next positions are valid
                    # Let's compute potential input/output points
                    
                    # Compute input points (opposite of direction)
                    input_dx = -direction[0]
                    input_dy = -direction[1]
                    
                    input_points = []
                    anchor_input = (pos[0] + input_dx, pos[1] + input_dy)
                    next_input = (next_pos[0] + input_dx, next_pos[1] + input_dy)
                    
                    # Check if input points are valid
                    if (0 <= anchor_input[0] < self.width and 
                        0 <= anchor_input[1] < self.height and
                        self.obstacle_map[anchor_input[1]][anchor_input[0]] == 0):
                        input_points.append(anchor_input)
                        
                    if (0 <= next_input[0] < self.width and 
                        0 <= next_input[1] < self.height and
                        self.obstacle_map[next_input[1]][next_input[0]] == 0):
                        input_points.append(next_input)
                    
                    # Compute output points (same as direction)
                    output_dx = direction[0]
                    output_dy = direction[1]
                    
                    output_points = []
                    anchor_output = (pos[0] + output_dx, pos[1] + output_dy)
                    next_output = (next_pos[0] + output_dx, next_pos[1] + output_dy)
                    
                    # Check if output points are valid
                    if (0 <= anchor_output[0] < self.width and 
                        0 <= anchor_output[1] < self.height and
                        self.obstacle_map[anchor_output[1]][anchor_output[0]] == 0):
                        output_points.append(anchor_output)
                        
                    if (0 <= next_output[0] < self.width and 
                        0 <= next_output[1] < self.height and
                        self.obstacle_map[next_output[1]][next_output[0]] == 0):
                        output_points.append(next_output)
                    
                    # Only consider splitter valid if there's at least one valid input and output
                    if input_points and output_points:
                        # Create a new splitter object with this configuration
                        new_splitter = Splitter(item=item, position=pos, direction=direction)
                        new_splitter.next_position = next_pos
                        new_splitter.inputs = input_points
                        new_splitter.outputs = output_points
                        
                        # Mark positions as occupied
                        occupied_positions.add(pos)
                        occupied_positions.add(next_pos)
                        
                        # Add to valid splitters
                        valid_splitters[item].append(new_splitter)
                        
                        logger.info(f"Valid splitter at {pos} with next position {next_pos}, facing {direction}")
                        logger.info(f"  Inputs: {input_points}")
                        logger.info(f"  Outputs: {output_points}")
                        
                        # We've found a valid next position for this splitter, no need to check others
                        #break
        
        # Update splitters dictionary with only valid splitters
        self.splitters = valid_splitters
        return valid_splitters

    def visualize_used_splitters(self, filename='used_splitters.png'):
        """Visualize only the splitters used in generated paths."""
        plt.figure(figsize=(10, 10))
        
        # Create a base grid with just obstacles
        base_grid = np.array(self.obstacle_map)
        
        # Create a colormap
        import matplotlib as mpl
        cmap = mpl.colormaps['viridis'].resampled(15)
        
        # Plot the grid
        plt.imshow(base_grid, cmap=cmap, interpolation='nearest')
        
        # Add grid lines
        plt.grid(True, color='black', linewidth=0.5, alpha=0.3)
        plt.gca().set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Find splitters used in paths
        used_splitters = {}
        
        for item_key, item_paths in self.paths.items():
            item_name = item_key.split('_')[0] if '_' in item_key else item_key
            
            if item_name not in used_splitters:
                used_splitters[item_name] = []
            
            for path_data in item_paths:
                # Check for splitters used at start or destination
                if path_data.get('start_splitter'):
                    if path_data['start_splitter'] not in used_splitters[item_name]:
                        used_splitters[item_name].append(path_data['start_splitter'])
                
                if path_data.get('dest_splitter'):
                    if path_data['dest_splitter'] not in used_splitters[item_name]:
                        used_splitters[item_name].append(path_data['dest_splitter'])
        
        # Plot only the splitters used in paths
        from matplotlib.patches import Rectangle
        
        colors = ['yellow', 'orange', 'lime', 'cyan', 'magenta']  # Different colors for different items
        item_count = 0
        
        # Track legend items
        legend_items = []
        
        for item, splitter_list in used_splitters.items():
            color = colors[item_count % len(colors)]
            item_count += 1
            
            legend_items.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=color, markersize=10, label=item))
            
            for splitter in splitter_list:
                pos = splitter.position
                next_pos = splitter.next_position
                
                if next_pos:
                    # Plot splitter main body as a rectangle
                    min_x = min(pos[0], next_pos[0])
                    max_x = max(pos[0], next_pos[0])
                    min_y = min(pos[1], next_pos[1])
                    max_y = max(pos[1], next_pos[1])
                    
                    # Create a rectangle patch
                    rect = Rectangle((min_x - 0.5, min_y - 0.5), max_x - min_x + 1, max_y - min_y + 1, 
                                    linewidth=2, edgecolor='black', facecolor=color, alpha=0.5)
                    plt.gca().add_patch(rect)
                    
                    # Draw an arrow to show direction
                    dx = splitter.direction[0] * 0.4
                    dy = splitter.direction[1] * 0.4
                    center_x = (pos[0] + next_pos[0]) / 2
                    center_y = (pos[1] + next_pos[1]) / 2
                    plt.arrow(center_x, center_y, dx, dy, head_width=0.2, head_length=0.3, 
                            fc='black', ec='black', alpha=0.8)
                    
                    # Mark input points
                    for input_pos in splitter.inputs:
                        plt.plot(input_pos[0], input_pos[1], 'r>', markersize=8)
                    
                    # Mark output points
                    for output_pos in splitter.outputs:
                        plt.plot(output_pos[0], output_pos[1], 'g<', markersize=8)
                    
                    # Add text labels
                    plt.text(center_x, center_y, item[:3], ha='center', va='center', 
                            color='black', fontweight='bold', fontsize=8)
                else:
                    # If next_pos is not defined, just mark the position
                    plt.plot(pos[0], pos[1], 's', color=color, markersize=10)
        
        # Also plot the paths to show how they connect to splitters
        for item_key, item_paths in self.paths.items():
            for path_data in item_paths:
                path = path_data['path']
                path_x = [pos[0] for pos in path]
                path_y = [pos[1] for pos in path]
                
                # Plot the path with a thin line
                plt.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=0.6)
                
                # Mark start and destination
                plt.plot(path[0][0], path[0][1], 'go', markersize=8)  # Start
                plt.plot(path[-1][0], path[-1][1], 'mo', markersize=8)  # Destination
        
        # Add additional legend items
        legend_items.append(plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='r', markersize=8, label='Input'))
        legend_items.append(plt.Line2D([0], [0], marker='<', color='w', markerfacecolor='g', markersize=8, label='Output'))
        legend_items.append(plt.Line2D([0], [0], color='b', linewidth=1.5, alpha=0.6, label='Path'))
        legend_items.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Start'))
        legend_items.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=8, label='Destination'))
        
        # Add the legend
        plt.legend(handles=legend_items, loc='upper right')
        
        # Add a title
        plt.title("Splitters Used in Paths")
        
        # Save the figure
        plt.savefig(filename)
        plt.close()
        
        return filename