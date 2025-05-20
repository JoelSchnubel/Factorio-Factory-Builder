#! .venv\Scripts\python.exe
import heapq
import numpy as np
import matplotlib.pyplot as plt
from logging_config import setup_logger
logger = setup_logger("Pathfinder")



class Splitter:
    def __init__(self, item ,position, direction):
        self.item = item
        self.position = position # belt point/ anchor point
        self.next_position = None # point next to the belt point depending on orientation
        self.direction = direction
        self.inputs = []
        self.outputs = []

        
    def __str__(self):
        return f"Splitter(item={self.item}, position={self.position}, direction={self.direction})"
    
    def to_dict(self):
        return {
            'item': self.item,
            'position': self.position,
            'next_position': self.next_position,
            'direction': self.direction,
            'inputs': self.inputs,
            'outputs': self.outputs
        }

    @classmethod
    def from_dict(cls, data):
      
        splitter = cls(
            item=data['item'],
            position=data['position'],
            direction=data['direction']
        )
        splitter.next_position = data.get('next_position')
        splitter.inputs = data.get('inputs', [])
        splitter.outputs = data.get('outputs', [])
        return splitter
    

class MultiAgentPathfinder:
    """
    A multi-agent pathfinding algorithm that finds paths for multiple items
    from their respective start points to their destinations.
    """
    
    def __init__(self, obstacle_map, points, allow_underground=False, underground_length=3,allow_splitters=False,splitters={},find_optimal_paths=False,output_item=None,pipe_underground_length=8):
        """
        Initialize the pathfinder with an obstacle map and points to connect.
        
        Args:
            obstacle_map (list): 2D grid where non-zero values represent obstacles
            points (dict): Dictionary with item information including start points and destinations
        """
        self.obstacle_map = [[1 if cell > 0 else 0 for cell in row] for row in obstacle_map]
        self.height = len(obstacle_map)
        self.width = len(obstacle_map[0]) if self.height > 0 else 0
        self.points = points
        self.allow_underground = allow_underground
        self.underground_length = underground_length
        self.pipe_underground_length = pipe_underground_length  #
        
        self.allow_splitters = allow_splitters
        self.splitters = splitters
        self.output_item = output_item
        
        self.find_optimal_paths = find_optimal_paths
        
        # Working grid - initially a copy of the obstacle map
        self.working_grid = [row[:] for row in obstacle_map]
        
        # Store the final paths for each item
        self.paths = {}
        
        # Store the inserters that need to be placed
        self.inserters = {}
        
        # Directions for adjacent moves (right, down, left, up)
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        
    def is_valid_position(self, position):
        """Check if a position is within bounds and not an obstacle."""
        x, y = position
        
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.working_grid[y][x] == 0)
    
    def heuristic(self, a, b):
        """Calculate Manhattan distance heuristic between points a and b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    
    def expand_fluid_positions(self, positions):
        """
        For fluid items, expand the valid starting to be all adjacent tiles.
        
        Args:
            positions (list): List of position tuples (x, y)
            
        Returns:
            list: Expanded list of positions including adjacent tiles
        """
        expanded_positions = []
        
        for pos in positions:
            # Check all four adjacent directions
            for dx, dy in self.directions:
                adjacent_pos = (pos[0] + dx, pos[1] + dy)
                # Make sure position is in bounds and not an obstacle
                if (0 <= adjacent_pos[0] < self.width and 
                    0 <= adjacent_pos[1] < self.height and 
                    self.working_grid[adjacent_pos[1]][adjacent_pos[0]] == 0 and
                    adjacent_pos not in expanded_positions):
                    expanded_positions.append(adjacent_pos)
                    logger.info(f"Added adjacent position {adjacent_pos} for fluid item")
        
        return expanded_positions

    def find_path(self, start, goal,is_fluid=False):
        """
        Find a path from start to goal using A* algorithm with support for underground paths.
        
        Args:
            start (tuple): Starting position (x, y)
            goal (tuple): Goal position (x, y)
            
        Returns:
            list: List of positions [(x, y), ...] forming the path, or None if no path exists
            dict: Dictionary of underground segments {'start': (x, y), 'end': (x, y)} or empty if none used
        """
        
        
        # If underground paths are not allowed, just use the standard A*
        if not self.allow_underground:
            path = self.find_path_astar(start, goal)
            return path, {}
        
        # Priority queue for A*
        open_set = []
        heapq.heappush(open_set, (0, start, []))  # (f_score, position, underground_segments)
        
        # Dictionary to track where we came from
        came_from = {}
        
        # Dictionary to track the cost from start to each node
        g_score = {start: 0}
        
        # Dictionary to track the estimated total cost from start to goal via each node
        f_score = {start: self.heuristic(start, goal)}
        
        # Dictionary to track underground segments for each node
        underground_segments = {start: []}
        
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

    def can_build_underground(self, position, dx, dy, goal,is_fluid):
        """
        Check if we can build an underground path from the given position in the specified direction.
        
        Args:
            position (tuple): Starting position (x, y)
            dx (int): X direction (-1, 0, 1)
            dy (int): Y direction (-1, 0, 1)
            goal (tuple): The ultimate goal position
            
        Returns:
            tuple: (can_build, entry_position, exit_position) - whether underground path is possible and positions
        """
        
        underground_length = self.pipe_underground_length if is_fluid else self.underground_length
        
        x, y = position
        entry = (x, y)  # The entry position is the current position
        
        # Don't allow diagonal movement for undergrounds
        if dx != 0 and dy != 0:
            return False, None, None, None
        
        # Check if the current position was just used as an underground exit in the same direction
        # This avoids having underground segments directly connected to each other
        # Check previous segments in our 'pipe_underground_max_length'
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
                                return False, None, None, None
        
        # Track segment ends to avoid placing segments right next to each other
        # This avoids case where segment_1 ends at position A and segment_2 starts at position A
        segment_ends = []
        for segment_dict in self.paths.values():
            for path_data in segment_dict:
                if 'underground_segments' in path_data and path_data['underground_segments']:
                    for _, segment in path_data['underground_segments'].items():
                        segment_ends.append(segment['end'])
                        
        # Don't start a new underground from a position that's already an end point
        if position in segment_ends:
            return False, None, None, None
        
        # Check if there's at least one obstacle in the path that needs to be crossed
        has_obstacle = False
        
        # Look for a valid exit point within the underground length
        #for length in range(underground_length+1, 2, -1):
        for length in range(2, underground_length+1, 1):   
            
            exit_x = x + dx * length
            exit_y = y + dy * length
            exit_pos = (exit_x, exit_y)
            
            following_x = exit_x + dx
            following_y = exit_y + dy
            following_pos = (following_x, following_y)
            
            logger.debug(f"Checking underground path from {entry} to {exit_pos} with length {length-1}")
            
            # Check if exit is in bounds
            if not (0 <= exit_x < self.width and 0 <= exit_y < self.height):
                continue
                
                
            # Special case: If the exit is the goal itself, we don't need to check the following position
            if exit_pos == goal:
                # We still need to check for obstacles between current and goal
                for i in range(1, length):
                    check_x = x + dx * i
                    check_y = y + dy * i
                    
                    # Check if this position is an obstacle
                    if 0 <= check_x < self.width and 0 <= check_y < self.height:
                        cell_value = self.working_grid[check_y][check_x]
                        if cell_value != 0:
                            has_obstacle = True
                            break
                
                # If there's at least one obstacle, this is a valid underground to the goal
                if has_obstacle:
                    logger.debug(f"Found valid underground path directly to goal from {entry} to {exit_pos}")
                    return True, entry, exit_pos, goal
                continue
            
            # Normal case: Check the position following the exit
            # Check if following position is in bounds (needed for underground belt)
            if not (0 <= following_x < self.width and 0 <= following_y < self.height):
                continue
            
            # Check if exit position is free
            if self.working_grid[exit_y][exit_x] != 0:
                continue
                
            # Check if following position is free
            if self.working_grid[following_y][following_x] != 0:
                continue
            
            # Check if there's at least one obstacle between the current position and the exit
            # that would require an underground path
            for i in range(1, length):
                check_x = x + dx * i
                check_y = y + dy * i
                
                # Check if this position is an obstacle
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    cell_value = self.working_grid[check_y][check_x]
                    if cell_value != 0:
                        has_obstacle = True
                        break
            
            # If there's no obstacle to cross, don't use an underground
            if not has_obstacle:
                continue
            
            # Check if this exit is closer to the goal (to avoid backtracking)
            current_distance = self.heuristic(position, goal)
            exit_distance = self.heuristic(exit_pos, goal)
            
            if exit_distance < current_distance:
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
                    logger.debug(f"Found valid underground path from {entry} to {exit_pos} with obstacle? {has_obstacle}")
                    return True, entry, exit_pos, following_pos
                else:
                    logger.debug(f"Exit {exit_pos} rejected because it's already an entry point")
            else:
                logger.debug(f"Exit {exit_pos} rejected because it doesn't get closer to goal")
        
        # No valid exit found
        return False, None, None, None

    
    
    
    
    
    # This is a simpler version without underground handling for use when undergrounds are disabled
    def find_path_astar(self, start, goal):
        """Basic A* algorithm without underground support."""
        # Priority queue for A*
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Dictionary to track where we came from
        came_from = {}
        
        # Dictionary to track the cost from start to each node
        g_score = {start: 0}
        
        # Dictionary to track the estimated total cost from start to goal via each node
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            # Get the node with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # If we reached the goal, reconstruct and return the path
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            # Check all adjacent nodes
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
                    
                    # Add to open set if not already there
                    if not any(pos == neighbor for _, pos in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None
    def mark_path_on_grid(self, path, value=1, underground_segments=None, start_splitter=None, dest_splitter=None,inserter=None):
        """
        Mark a path and its associated components on the working grid with the specified value.
        
        Args:
            path (list): List of (x, y) positions in the path
            value (int): Value to mark on the grid (default: 1 for obstacle)
            underground_segments (dict): Dictionary of underground segments to mark
            start_splitter (Splitter): Starting splitter object if any
            dest_splitter (Splitter): Destination splitter object if any
        """
        if path is None:
            return
        
        # Mark the entire path
        for x, y in path:
            self.working_grid[y][x] = value
        
        # Mark underground segments
        if underground_segments:
            for _, segment in underground_segments.items():
                start_x, start_y = segment['start']
                end_x, end_y = segment['end']
                
                # Mark the entrance and exit
                self.working_grid[start_y][start_x] = value
                self.working_grid[end_y][end_x] = value
                
                # Mark the path between entrance and exit
                dir_x = 0 if start_x == end_x else (end_x - start_x) // abs(end_x - start_x)
                dir_y = 0 if start_y == end_y else (end_y - start_y) // abs(end_y - start_y)
                
                curr_x, curr_y = start_x, start_y
                while (curr_x, curr_y) != (end_x, end_y):
                    curr_x += dir_x
                    curr_y += dir_y
                    if (curr_x, curr_y) != (end_x, end_y):  # Don't mark the end point twice
                        self.working_grid[curr_y][curr_x] = value
        
        # Mark splitter positions
        if start_splitter:
            pos_x, pos_y = start_splitter.position
            self.working_grid[pos_y][pos_x] = value
            
            if start_splitter.next_position:
                next_x, next_y = start_splitter.next_position
                self.working_grid[next_y][next_x] = value
        
        if dest_splitter:
            pos_x, pos_y = dest_splitter.position
            self.working_grid[pos_y][pos_x] = value
            
            if dest_splitter.next_position:
                next_x, next_y = dest_splitter.next_position
                self.working_grid[next_y][next_x] = value
                
        if inserter:
            inserter_x, inserter_y = inserter
            self.working_grid[inserter_y][inserter_x] = value
    
                            
    def place_output_inserter(self):
        """
        Place inserters at the output item locations if defined.
        Handles multiple output items that might share inserter positions.
        """
        if not self.output_item:
            logger.info("No output item defined, skipping output inserter placement")
            return None
        
        logger.info(f"Placing output inserter for item {self.output_item}")
        
        # Track which inserter positions have already been used
        used_inserters = set()
        
        # Process each output item
        for item_key, item_data in self.points.items():
            if item_data.get('item') != self.output_item:
                logger.debug(f"Skipping {item_key} as it's not the output item {self.output_item}")
                continue
                
            logger.info(f"Found output item match: {item_key}")
            
            # Get start points, destinations, and inserter mapping
            start_points = item_data.get('start_points', [])
            destinations = item_data.get('destination', [])
            inserter_mapping = item_data.get('inserter_mapping', {})
            
            logger.debug(f"Start points for {item_key}: {start_points}")
            logger.debug(f"Destination points for {item_key}: {destinations}")
            logger.debug(f"Inserter mapping for {item_key}: {inserter_mapping}")
            
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
            valid_starts = []
            
            for start_pos in start_points:
                start_pos_str = str(start_pos)
                
                if start_pos_str not in inserter_mapping:
                    logger.debug(f"No inserter mapping for start position {start_pos}, skipping")
                    continue
                    
                inserter_pos = inserter_mapping[start_pos_str]
                inserter_pos_str = str(inserter_pos)
                
                # Check if inserter position is in bounds
                ix, iy = inserter_pos
                if not (0 <= ix < self.width and 0 <= iy < self.height):
                    logger.warning(f"Inserter position {inserter_pos} is out of bounds, skipping")
                    continue
                
                # Check if this inserter has already been used
                if inserter_pos_str in used_inserters:
                    # This inserter is already placed - it's still valid, just mark it
                    valid_starts.append((start_pos, inserter_pos, True))
                    logger.info(f"Inserter already exists at {inserter_pos} for {start_pos}")
                else:
                    # Check if the position is free in the working grid
                    if self.working_grid[iy][ix] == 0:
                        valid_starts.append((start_pos, inserter_pos, False))
                        logger.info(f"Found valid inserter placement at {inserter_pos} for start {start_pos}")
                    else:
                        logger.warning(f"Inserter position {inserter_pos} is not valid (value: {self.working_grid[iy][ix]}), skipping")
            
            # Sort valid starts: unused inserters first, then used ones
            valid_starts.sort(key=lambda x: x[2])  # False (unused) comes first
            
            if not valid_starts:
                logger.warning(f"No valid inserter placements found for {item_key}")
                continue
            
            # Use the first valid start
            start_pos, inserter_pos, already_used = valid_starts[0]
            
            # Place the inserter if it hasn't been placed already
            if not already_used:
                ix, iy = inserter_pos
                prev_value = self.working_grid[iy][ix]
                self.working_grid[iy][ix] = 1
                logger.info(f"Marked inserter at {inserter_pos} as obstacle (previous value: {prev_value})")
                used_inserters.add(str(inserter_pos))
            
            # Update the start points to prioritize this one
            old_start_points = self.points[item_key]['start_points']
            self.points[item_key]['start_points'] = [start_pos]
            logger.info(f"Updated start_points for {item_key} from {old_start_points} to {[start_pos]}")
            
            # Store the inserter in our collection
            if item_key not in self.inserters:
                self.inserters[item_key] = {}
                
            self.inserters[item_key][str(start_pos)] = inserter_pos
            logger.info(f"Stored inserter {inserter_pos} for {item_key} at start position {start_pos}")
        
        if not used_inserters:
            logger.warning("Failed to place any output inserters")
            return None
            
        logger.info(f"Successfully placed {len(used_inserters)} inserters")
        return used_inserters
      
    
    
    def find_paths_for_all_items(self,IO_paths=False):
        """
        Find paths for all items in the points dictionary.
        
        For each item, finds the path from the start point to destination with the best heuristic value.
        If that fails, tries other pairs in order of increasing heuristic distance.
        
        Returns:
            tuple: (paths, inserters, used_splitters) where:
                - paths: dictionary of paths for each item
                - inserters: dictionary of inserters that need to be placed
                - used_splitters: dictionary of splitters used as I/O points
        """
        
        # Sort the points -> output items first
        # Separate output items from other items
        output_items = {}
        other_items = {}
        
        for item_key, item_data in self.points.items():
            if item_data.get('item') == self.output_item:
                output_items[item_key] = item_data
            else:
                other_items[item_key] = item_data
    
        # Create a sorted processing order - output items first
        sorted_keys = list(output_items.keys()) + list(other_items.keys())
        logger.info(f"Processing items in order: {sorted_keys}")
    
        
        
        # place output inserter if output item is defined
        if self.output_item:
            self.place_output_inserter()
        
    
        # Process all splitters first to determine positions and I/O points
        if self.allow_splitters:
            self.process_splitters()
        

        all_previous_paths = {}
        
        # Track which splitters are used for I/O
        used_splitters = {}
        
        # Process items in order (could use a priority system in the future)
       
       
        
        for item_key in sorted_keys:
            
            
            item_data = self.points[item_key] 
            
            logger.info(f"Finding path for {item_key}")
            logger.debug(f"Working grid:\n{np.array(self.working_grid)}")
            
            is_fluid = item_data.get('is_fluid', False)
            
          
            
            # Extract item information
            start_points = item_data['start_points'].copy()
            destinations = item_data['destination'].copy()
            inserter_mapping = item_data.get('inserter_mapping', None)
            item_name = item_data['item']
    
            if is_fluid and not IO_paths:
                logger.info(f"Processing fluid item {item_key}, expanding start positions")
                original_starts = start_points.copy()
                start_points = self.expand_fluid_positions(start_points)
                logger.info(f"Expanded start positions for {item_key} from {len(original_starts)} to {len(start_points)}")
            
            

            
            # if item is output item, add all other paths of this item to the destination points
            if self.output_item and item_name == self.output_item:
                    for other_item_key, other_paths in self.paths.items():
                        for other_path_data in other_paths:  # Iterate through each path data in the list
                            if other_item_key != item_key and other_path_data['item'] == item_name:
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
                                logger.info(f"Added {len(filtered_path)} destination points from other item {other_item_key} (filtered out {len(other_path_data['path']) - len(filtered_path)} underground points)")
                                        
                            
            # if the item is not the output item, add the destination points of all other paths with the same item to the start points
            if item_name != self.output_item:
                    for other_item_key, other_paths in self.paths.items():
                        
                        for other_path_data in other_paths:  # Iterate through each path data in the list
                            if other_item_key != item_key and other_path_data['item'] == item_name:
                                # point is not allowed to be a undground segment
                                underground_segments = other_path_data.get('underground_segments', {})

                                is_underground = True
                                for segment in underground_segments.values():
                                    if segment['end'] == other_path_data['destination'] or segment['start'] == other_path_data['destination']:
                                       is_underground = False
                                       break
        
                                if is_underground:
                                    start_points.extend([other_path_data['destination']])
                                    logger.info(f"Added start points from other item {other_item_key}")
                    
            
            
            if self.allow_splitters and item_name in self.splitters and not is_fluid:
                # Find splitters that are relevant to our start/destination points
                relevant_start_splitters = []
                relevant_dest_splitters = []
                
                
                
                # Check each splitter to see if it's positioned at one of our start/destination points
                for splitter in self.splitters[item_name]:
                    # A splitter is relevant to start points if its position matches a start point
                    for start_point in item_data['start_points']:
                        if splitter.position == start_point:
                            relevant_start_splitters.append(splitter)
                            logger.info(f"Found splitter at start point {start_point}")
                    
                    # A splitter is relevant to destination points if its position matches a destination
                    for dest_point in item_data['destination']:
                        if splitter.position == dest_point:
                            relevant_dest_splitters.append(splitter)
                            logger.info(f"Found splitter at destination point {dest_point}")
                
                if len(relevant_start_splitters) > 0:
                    start_points = []
                
                if len(relevant_dest_splitters) > 0:
                    destinations = []
                
                
       
                                
                # Add output points from start splitters as start points
                for splitter in relevant_start_splitters:
                    for output_point in splitter.outputs:
                        if self.is_valid_position(output_point):
                            start_points.append(output_point)
                            logger.info(f"Added splitter output {output_point} as start point")
                
                # Add input points from destination splitters as destinations
                for splitter in relevant_dest_splitters:
                    for input_point in splitter.inputs:
                        if self.is_valid_position(input_point):
                            destinations.append(input_point)
                            logger.info(f"Added splitter input {input_point} as destination")
            
                                
            # check if start points and destination points have the same coordinates
            # -> skip then
            finished = False
            for point in start_points:
                if point in destinations:
                    finished = True
                    logger.debug(f"Start point {point} is also a destination, skipping pathfinding for this item")
                    
                    # check if we need to place an inserter aswell
                    if inserter_mapping and str(point) in inserter_mapping:
                        inserter = inserter_mapping[str(point)]
                        logger.info(f"Found inserter at {inserter} for start/destination point {point}")
                        
                        
                        # Check if inserter position is in bounds
                        ix, iy = inserter
                        if 0 <= ix < self.width and 0 <= iy < self.height:
                            # Mark inserter position as obstacle
                            prev_value = self.working_grid[iy][ix]
                            self.working_grid[iy][ix] = 1
                            if item_key not in self.inserters:
                                self.inserters[item_key] = {}
                            
                            self.inserters[item_key][str(point)] = inserter
                            logger.info(f"Marked inserter at {inserter} as obstacle (previous value: {prev_value})")
                    
                    break
            
            
            if finished:
                continue
            
            
            # Always mark start and destination points as valid (set to 0) temporarily
            for start in start_points:
                x, y = start
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.working_grid[y][x] = 0  # Mark as valid
            
            for dest in destinations:
                x, y = dest
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.working_grid[y][x] = 0  # Mark as valid
            
            # Create a list of all possible start-destination pairs with their heuristic distance
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
        """Visualize the current state of the grid."""
        plt.figure(figsize=(10, 10))
        
        # Create a colormap (updated to avoid deprecation warning)
        import matplotlib as mpl
        
        cmap = mpl.colormaps['viridis'].resampled(15)
        
        # Plot the grid
        plt.imshow(self.working_grid, cmap=cmap, interpolation='nearest')
        
        # Add a colorbar
        plt.colorbar(label='Cell Type')
        
        # Add grid lines
        plt.grid(True, color='black', linewidth=0.5, alpha=0.3)
        
        # Adjust the grid to match the cell centers
        plt.gca().set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Save the figure
        plt.savefig(filename)
        plt.close()
        
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

