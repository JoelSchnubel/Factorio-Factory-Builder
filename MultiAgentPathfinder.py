import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt


class Splitter:
    def __init__(self, item ,position, direction):
        self.item = item
        self.position = position
        self.direction = direction
        self.inputs = []
        self.outputs = []



class MultiAgentPathfinder:
    """
    A multi-agent pathfinding algorithm that finds paths for multiple items
    from their respective start points to their destinations.
    """
    
    def __init__(self, obstacle_map, points, allow_underground=False, underground_length=3,allow_splitters=False,splitters={}):
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
        
        self.allow_splitters = allow_splitters
        self.splitters = splitters
        
        # Working grid - initially a copy of the obstacle map
        self.working_grid = [row[:] for row in obstacle_map]
        
        # Store the final paths for each item
        self.paths = {}
        
        # Store the inserters that need to be placed
        self.inserters = {}
        
        # Directions for adjacent moves (right, down, left, up)
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler("multi_agent_pathfinding.log", mode='w')]
        )
    
    def is_valid_position(self, position):
        """Check if a position is within bounds and not an obstacle."""
        x, y = position
        
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.working_grid[y][x] == 0)
    
    def heuristic(self, a, b):
        """Calculate Manhattan distance heuristic between points a and b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start, goal):
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
                
                # Convert segments list to dictionary for easier processing
                segment_dict = {}
                for i, segment in enumerate(current_segments):
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
                    can_go_underground, entry, exit = self.can_build_underground(current, dx, dy, goal)
                    
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
        
        # No path found
        return None, {}

    def can_build_underground(self, position, dx, dy, goal):
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
        x, y = position
        entry = (x, y)  # The entry position is the current position
        
        # Don't allow diagonal movement for undergrounds
        if dx != 0 and dy != 0:
            return False, None, None
        
        # Check if the current position was just used as an underground exit in the same direction
        # This avoids having underground segments directly connected to each other
        # Check previous segments in our path
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
                                return False, None, None
        
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
            return False, None, None
        
        # Check if there's at least one obstacle in the path that needs to be crossed
        has_obstacle = False
        
        # Look for a valid exit point within the underground length
        for length in range(2, self.underground_length + 1):
            exit_x = x + dx * length
            exit_y = y + dy * length
            exit_pos = (exit_x, exit_y)
            
            following_x = exit_x + dx
            following_y = exit_y + dy
            
            # Check if exit is in bounds
            if not (0 <= exit_x < self.width and 0 <= exit_y < self.height):
                continue
                
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
                    if self.working_grid[check_y][check_x] != 0:
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
                                    break
                
                if is_valid_exit:
                    # This is a valid exit that's closer to the goal
                    return True, entry, exit_pos
        
        # No valid exit found
        return False, None, None

    
    
    
    
    
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
    
    def mark_path_on_grid(self, path, value=1):
        """Mark a path on the working grid with the specified value."""
        if path is None:
            return
            
        for x, y in path:
            self.working_grid[y][x] = value
    
    
    def check_for_splitters(self, item_key):
        """
        Check if the item needs splitters at start or destination points.
        Updates the working grid with splitter placements.
        
        Args:
            item_key (str): The item key
            item_data (dict): The item data containing start points, destinations
        """
        if not self.allow_splitters:
            return
            
        # If splitters are in item_data, use those
        item_splitters = []


        # Also check for splitters in the class-level splitters dictionary
        if self.splitters and item_key in self.splitters:
            item_splitters.extend(self.splitters[item_key])
        
        # If no splitters for this item, nothing to do
        if not item_splitters:
            return
        
        # Store splitter information for this item
        if item_key not in self.splitters:
            self.splitters[item_key] = []
        
        # Place splitters on the grid and add them to the item's splitter list
        for splitter in item_splitters:
            pos = splitter['pos']
            direction = splitter['direction']
            
            # Check if position is valid
            if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
                logging.warning(f"Splitter position {pos} is out of bounds for {item_key}")
                continue
            
            # Calculate the positions of the splitter (2x1 structure)
            positions = self.get_splitter_positions(pos, direction)
            
            # Check if all positions are valid
            valid = True
            for p in positions:
                if not (0 <= p[0] < self.width and 0 <= p[1] < self.height):
                    valid = False
                    logging.warning(f"Splitter position {p} is out of bounds")
                    break
            
            if not valid:
                logging.warning(f"Splitter at {pos} with direction {direction} is not valid for {item_key}")
                continue
            
            # Add splitter with positions to item's splitters if not already there
            splitter_with_positions = {
                'pos': pos,
                'direction': direction,
                'positions': positions
            }
            
            # Check if this splitter is already added
            already_added = False
            for existing in self.splitters[item_key]:
                if existing['pos'] == pos and existing['direction'] == direction:
                    already_added = True
                    break
                    
            if not already_added:
                self.splitters[item_key].append(splitter_with_positions)
            
            # Mark splitter positions as temporarily free for pathfinding
            for p in positions:
                if 0 <= p[0] < self.width and 0 <= p[1] < self.height:
                    # Mark as 0 (free) during pathfinding to allow paths through splitter
                    self.working_grid[p[1]][p[0]] = 0
                    

    def get_splitter_positions(self, pos, direction):
        """
        Get all positions occupied by a splitter.
        
        Args:
            pos (tuple): Base position of the splitter (x, y)
            direction (tuple): Direction the splitter faces (dx, dy)
        
        Returns:
            list: List of positions occupied by the splitter
        """
        x, y = pos
        dx, dy = direction
        
        # Determine perpendicular direction for the width of the splitter
        if dx != 0:  # Horizontal splitter
            # Width is along y-axis
            return [(x, y), (x, y+1)]
        else:  # Vertical splitter
            # Width is along x-axis
            return [(x, y), (x+1, y)]

    def get_splitter_inputs(self, splitter):
        """
        Get the input positions for a splitter.
        
        Args:
            splitter (dict): Splitter information including position and direction
        
        Returns:
            list: List of input positions
        """
        pos = splitter['pos']
        direction = splitter['direction']
        positions = splitter['positions']
        
        # Input is in the opposite direction from the splitter's direction
        input_dx = -direction[0]
        input_dy = -direction[1]
        
        inputs = []
        for p in positions:
            input_pos = (p[0] + input_dx, p[1] + input_dy)
            inputs.append(input_pos)
        
        return inputs

    def get_splitter_outputs(self, splitter):
        """
        Get the output positions for a splitter.
        
        Args:
            splitter (dict): Splitter information including position and direction
        
        Returns:
            list: List of output positions
        """
        pos = splitter['pos']
        direction = splitter['direction']
        positions = splitter['positions']
        
        # Output is in the direction the splitter faces
        output_dx = direction[0]
        output_dy = direction[1]
        
        outputs = []
        for p in positions:
            output_pos = (p[0] + output_dx, p[1] + output_dy)
            outputs.append(output_pos)
        
        return outputs

    def find_paths_for_all_items(self):
        """
        Find paths for all items in the points dictionary.
        
        For each item, finds the path from the start point to destination with the best heuristic value.
        If that fails, tries other pairs in order of increasing heuristic distance.
        
        Returns:
            tuple: (paths, inserters) where paths is a dictionary of paths for each item
                  and inserters is a dictionary of inserters that need to be placed
        """
        
        all_previous_paths = {}
        # Process items in order (could use a priority system in the future)
        for item_key, item_data in self.points.items():
            logging.info(f"Finding path for {item_key}")
            
            # Check for and process splitters for this item
            if self.allow_splitters:
                self.check_for_splitters(item_data)
            
            # Extract item information
            start_points = item_data['start_points'].copy()
            destinations = item_data['destination'].copy()
            inserter_mapping = item_data.get('inserter_mapping', None)
            
            # If splitters are enabled, add splitter inputs/outputs as valid start/destination points
            if self.allow_splitters and item_key in self.splitters:
                for splitter in self.splitters[item_key]:
                    # Add splitter inputs as potential destinations (for merging)
                    inputs = self.get_splitter_inputs(splitter)
                    for input_pos in inputs:
                        if input_pos not in destinations and self.is_valid_position(input_pos):
                            destinations.append(input_pos)
                    
                    # Add splitter outputs as potential start points (for branching)
                    outputs = self.get_splitter_outputs(splitter)
                    for output_pos in outputs:
                        if output_pos not in start_points and self.is_valid_position(output_pos):
                            start_points.append(output_pos)
            
            # Add previously created paths as potential destinations for merging
            for prev_item, prev_paths in all_previous_paths.items():
                # Check if this is the same item type
                if prev_item.split('_')[0] == item_key.split('_')[0]:
                    # Add every point of the previous paths as a potential destination
                    for prev_path in prev_paths:
                        destinations.extend(prev_path['path'])
                            
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
                    is_splitter_start = start not in item_data['start_points']
                    is_splitter_dest = dest not in item_data['destination']
                    
                    # Calculate heuristic distance (give bonus to splitter paths)
                    h_distance = self.heuristic(start, dest)
                    
                    # Slightly prefer paths that use splitters
                    if is_splitter_start or is_splitter_dest:
                        h_distance = h_distance * 0.95  # 5% bonus for splitter paths
                    
                    all_pairs.append((h_distance, start, dest, is_splitter_start, is_splitter_dest))
            
            # Sort pairs by heuristic distance (ascending)
            all_pairs.sort()
            
            # Try to find a path for each pair in order of increasing heuristic distance
            path_found = False
            best_path = None
            best_start = None
            best_dest = None
            best_inserter = None
            best_underground_segments = None
            uses_splitter_start = False
            uses_splitter_dest = False
            
            for h_distance, start, dest, is_splitter_start, is_splitter_dest in all_pairs:
                # Check if we need to place an inserter at this start point
                inserter = None
                if inserter_mapping and str(start) in inserter_mapping and not is_splitter_start:
                    inserter = inserter_mapping[str(start)]
                    # Temporarily mark inserter position as obstacle
                    ix, iy = inserter
                    original_value = self.working_grid[iy][ix]
                    self.working_grid[iy][ix] = 1
                
                # Try to find a path
                path, underground_segments = self.find_path(start, dest)
                
                # If we found a path, use it and stop looking
                if path:
                    best_path = path
                    best_start = start
                    best_dest = dest
                    best_inserter = inserter
                    best_underground_segments = underground_segments
                    uses_splitter_start = is_splitter_start
                    uses_splitter_dest = is_splitter_dest
                    path_found = True
                    
                    logging.info(f"Found path for {item_key} from {start} to {dest} with heuristic distance {h_distance}")
                    break
                else:
                    # If the path failed, restore inserter position if needed
                    if inserter:
                        self.working_grid[iy][ix] = original_value
            
            # If we found a path, add it to our results
            if path_found:
                # Add the path to our results
                if item_key not in self.paths:
                    self.paths[item_key] = []
                self.paths[item_key].append({
                    'path': best_path,
                    'start': best_start,
                    'destination': best_dest,
                    'underground_segments': best_underground_segments,
                    'uses_splitter_start': uses_splitter_start,
                    'uses_splitter_dest': uses_splitter_dest
                })
                
                # Add inserter information if applicable
                if best_inserter:
                    if item_key not in self.inserters:
                        self.inserters[item_key] = {}
                    self.inserters[item_key][str(best_start)] = best_inserter
                
                # Mark the path on the working grid
                self.mark_path_on_grid(best_path)
                
                # Add this path to all_previous_paths for future items to merge with
                if item_key not in all_previous_paths:
                    all_previous_paths[item_key] = []
                all_previous_paths[item_key].append({'path': best_path})
                    
            else:
                # No path found for this item
                logging.warning(f"No path found for {item_key} with any start-destination pair")
        
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
        """Visualize each item's path separately."""
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
            
            # Add a title
            plt.title(f"Paths for {item_key}")
            
            # Save the figure
            plt.savefig(filename_template.format(item_key.replace('/', '_')))
            plt.close()