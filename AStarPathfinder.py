import heapq
import logging
import os
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("AStar.log", mode='w'),
    ]
)

# Directions for moving in a grid (up, down, left, right)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
DIRECTION_MAP = {(0, 1): 'right', (1, 0): 'down', (0, -1): 'left', (-1, 0): 'up'}

class AStarPathFinder:
    def __init__(self, grid, points, invert_paths, underground_length=3, allow_jump=True, use_splitters=False):
        self.base_grid = grid
        self.underground_length = underground_length
        self.allow_jump = allow_jump
        self.points = points
        self.paths = {}
        self.invert_paths = invert_paths
        self.splitters = {}  # Dictionary to store splitter information
        self.use_splitters = use_splitters  # Flag to control splitter usage
  
    def heuristic(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def is_valid(self, node, grid):
        x, y = node
        return 0 <= x < len(grid[0]) and 0 <= y < len(grid) and grid[y][x] == 0
    
    def is_valid_jump(self, start, end, direction, grid):
        """Check if a unidirectional underground belt jump is valid."""
        sx, sy = start
        ex, ey = end
        dx, dy = direction
        
        # Check if start, end, and positions before/after are valid
        if not self.is_valid(end, grid):
            return False
        
        # Check position behind entrance
        behind_x, behind_y = sx - dx, sy - dy
        if not (0 <= behind_x < len(grid[0]) and 0 <= behind_y < len(grid) and 
                (grid[behind_y][behind_x] == 0 or grid[behind_y][behind_x] == 9)):
            return False
        
        # Check position after exit
        ahead_x, ahead_y = ex + dx, ey + dy
        if not (0 <= ahead_x < len(grid[0]) and 0 <= ahead_y < len(grid) and 
                (grid[ahead_y][ahead_x] == 0 or grid[ahead_y][ahead_x] == 9)):
            return False
        
        # Check distance constraints
        distance = abs(ex - sx) + abs(ey - sy)
        if distance < 2 or distance > self.underground_length + 1:
            return False
        
        # Ensure the jump is in the same direction as specified (unidirectional)
        actual_dx = 0 if ex == sx else (1 if ex > sx else -1)
        actual_dy = 0 if ey == sy else (1 if ey > sy else -1)
        if dx != actual_dx or dy != actual_dy:
            return False
        
        
        return True
    
    def astar(self, grid, start, goal):
        """
        Find the shortest path from start to goal using A* algorithm.
        
        Supports optional splitter mechanics based on the use_splitters flag.
        """
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_costs = {start: 0}
        f_costs = {start: self.heuristic(start, goal)}
        direction_grid = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]
        
        logging.info(f"Starting A* from {start} to {goal}")
        
        while open_list:
            _, current = heapq.heappop(open_list)
            logging.debug(f"Processing node {current}")
            
            if current == goal:
                logging.info(f"Goal reached at {current}")
                return self.reconstruct_path(came_from, current), direction_grid
            
            for direction in DIRECTIONS:
                dx, dy = direction
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if we can move normally to the neighbor
                if self.is_valid(neighbor, grid):
                    tentative_g_cost = g_costs[current] + 1
                    
                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        came_from[neighbor] = current
                        g_costs[neighbor] = tentative_g_cost
                        f_costs[neighbor] = tentative_g_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_costs[neighbor], neighbor))
                        
                        direction_grid[neighbor[1]][neighbor[0]] = direction
                        logging.debug(f"Moving to {neighbor} with direction {DIRECTION_MAP[direction]}")
                
                # Check for splitter entrance if splitters are enabled
                if self.use_splitters:
                    cx, cy = current
                    for splitter_id, splitter_info in self.splitters.items():
                        splitter_pos = splitter_info['position']
                        splitter_dir = splitter_info['direction']
                        entrance_positions = self.get_splitter_entrances(splitter_pos, splitter_dir)
                        exit_position = self.get_splitter_exit(splitter_pos, splitter_dir)
                        
                        # If current position is an entrance to this splitter
                        if (cx, cy) in entrance_positions:
                            # Check if the exit is valid
                            if self.is_valid(exit_position, grid):
                                tentative_g_cost = g_costs[current] + 2  # Penalty for using splitter
                                
                                if exit_position not in g_costs or tentative_g_cost < g_costs[exit_position]:
                                    came_from[exit_position] = current
                                    g_costs[exit_position] = tentative_g_cost
                                    f_costs[exit_position] = tentative_g_cost + self.heuristic(exit_position, goal)
                                    heapq.heappush(open_list, (f_costs[exit_position], exit_position))
                                    
                                    # Set direction to match the splitter's output direction
                                    direction_grid[exit_position[1]][exit_position[0]] = splitter_dir
                                    logging.debug(f"Using splitter from {current} to {exit_position}")
                
                # Underground belt logic - unidirectional
                elif self.allow_jump:
                    # Try all valid jump distances in this direction
                    for jump_distance in range(2, self.underground_length + 2):  
                        jump_target = (current[0] + dx * jump_distance, current[1] + dy * jump_distance)
                        
                        # Check if the jump is valid (including entrance/exit space requirements)
                        if self.is_valid_jump(current, jump_target, direction, grid):
                            # Underground belts have a higher cost to prefer regular belts when possible
                            tentative_g_cost = g_costs[current] + jump_distance + 1.5  # Penalty for underground
                            
                            if jump_target not in g_costs or tentative_g_cost < g_costs[jump_target]:
                                came_from[jump_target] = current
                                g_costs[jump_target] = tentative_g_cost
                                f_costs[jump_target] = tentative_g_cost + self.heuristic(jump_target, goal)
                                heapq.heappush(open_list, (f_costs[jump_target], jump_target))
                                direction_grid[jump_target[1]][jump_target[0]] = direction
                                logging.debug(f"Underground belt from {current} to {jump_target} in direction {DIRECTION_MAP[direction]}")
                            
                            # Also check the position behind the entrance
                                behind_x = current[0] - dx
                                behind_y = current[1] - dy
                                behind_pos = (behind_x, behind_y)
                                # Calculate cost to reach this position
                                behind_g_cost = g_costs[current] + 1
                                if behind_pos not in g_costs or behind_g_cost < g_costs[behind_pos]:
                                        # Determine the direction from behind position to current
                                        behind_direction = (-dx, -dy)
                                        
                                        came_from[behind_pos] = current
                                        g_costs[behind_pos] = behind_g_cost
                                        f_costs[behind_pos] = behind_g_cost + self.heuristic(behind_pos, goal)
                                        heapq.heappush(open_list, (f_costs[behind_pos], behind_pos))
                                        
                                        if 0 <= behind_y < len(direction_grid) and 0 <= behind_x < len(direction_grid[0]):
                                            direction_grid[behind_y][behind_x] = behind_direction
                                        
                                        logging.debug(f"Added position behind entrance: {behind_pos}")

                            break
        
        logging.info("No path found")
        return None, direction_grid

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        logging.info(f"Reconstructed path: {path}")
        return path

    def get_jump_markers(self, path):
        jump_markers = []  # List to store the jump markers (entrance, exit, direction)
        
        # Iterate through the path to check for jumps
        for i in range(1, len(path)):
            start = path[i - 1]
            end = path[i]
            
            # Calculate the Manhattan distance between the points
            distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
            
            # If the distance is greater than 1, it's an underground belt
            if distance > 1:
                # Calculate the direction vector
                dx = 1 if end[0] > start[0] else (-1 if end[0] < start[0] else 0)
                dy = 1 if end[1] > start[1] else (-1 if end[1] < start[1] else 0)
                
                # Create underground entrance and exit markers with direction
                jump_markers.append((start, end, (dx, dy)))
        
        return jump_markers
    
    def get_splitter_entrances(self, position, direction):
        """Get the entrance positions for a splitter."""
        x, y = position
        dx, dy = direction
        
        # Splitters are 2x1 structures
        # The entrances are opposite to the output direction
        if dx == 0:  # Vertical splitter (up or down)
            # Two entrance positions side by side horizontally
            return [(x, y - dy), (x + 1, y - dy)]
        else:  # Horizontal splitter (left or right)
            # Two entrance positions side by side vertically
            return [(x - dx, y), (x - dx, y + 1)]
    
    def get_splitter_exit(self, position, direction):
        """Get the exit position for a splitter."""
        x, y = position
        dx, dy = direction
        
        # The exit is in the direction of the splitter
        return (x + dx, y + dy)
    
    def add_splitter(self, splitter_id, position, direction, item_type):
        """Add a new splitter to the pathfinder."""
        self.splitters[splitter_id] = {
            'position': position,
            'direction': direction,
            'item_type': item_type,
            'entrances': self.get_splitter_entrances(position, direction),
            'exit': self.get_splitter_exit(position, direction)
        }
        
        # Mark the splitter position and entrances on the grid
        x, y = position
        if 0 <= y < len(self.base_grid) and 0 <= x < len(self.base_grid[0]):
            self.base_grid[y][x] = 13  # Use 13 to represent splitter
        
        # Mark the second tile of the splitter (it's 2x1)
        if direction[0] == 0:  # Vertical
            if 0 <= y < len(self.base_grid) and 0 <= x+1 < len(self.base_grid[0]):
                self.base_grid[y][x+1] = 13
        else:  # Horizontal
            if 0 <= y+1 < len(self.base_grid) and 0 <= x < len(self.base_grid[0]):
                self.base_grid[y+1][x] = 13
        
        logging.info(f"Added splitter {splitter_id} at {position} facing {DIRECTION_MAP[direction]} for {item_type}")
        return self.splitters[splitter_id]
    
    def find_path_for_item(self):
        placed_inserter_information = []
        paths = {}
        grid = [row[:] for row in self.base_grid]  # Deep copy of the grid for each function call
        
        # Place splitters only if they are enabled
        if self.use_splitters:
            # First place optimal splitters at destination points
            for item_type in set(points['item'] for points in self.points.values()):
                # Find potential splitter locations for this item type, prioritizing destinations
                potential_locations = self.find_splitter_locations(item_type)
                
                # Place splitters at high-priority locations (near destinations)
                splitters_added = 0
                for location in potential_locations:
                    if location.get('priority') == 'high' and splitters_added < 3:  # Limit to 3 splitters per item type
                        self.add_splitter(
                            f"splitter_{item_type}_{splitters_added}", 
                            location['position'], 
                            location['direction'], 
                            item_type
                        )
                        splitters_added += 1
        
        # Clear starting and destination points on the grid
        for item, points in self.points.items():
            for x, y in points['start_points'] + points['destination']:
                grid[y][x] = 0
            found_path = False
            
            # Calculate all possible pairs of start and destination points, sorted by distance
            pairs = [
                (start, dest, self.heuristic(start, dest))
                for start in points['start_points']
                for dest in points['destination']
            ]
            pairs.sort(key=lambda x: x[2])  # Sort by distance (smallest first)
            
            for start, dest, _ in pairs:
                logging.info(f"Attempting path for {item} from {start} to {dest}")
                print(f"Attempting path for {item} from {start} to {dest}")

                # Check if we need to add an inserter before pathfinding
                inserter_position = None
                if points.get("inserter_mapping") is not None:
                    inserter_position = points["inserter_mapping"].get(str(start))
                    if inserter_position:
                        ix, iy = inserter_position
                        # Mark inserter position as obstacle in the pathfinding grid
                        grid[iy][ix] = 12
                
                # Check if this item can use existing splitters (only if splitters are enabled)
                if self.use_splitters:
                    item_type = points.get('item')
                    
                    # First, prioritize finding a path to a splitter entrance that's already connected
                    for splitter_id, splitter_info in self.splitters.items():
                        if splitter_info['item_type'] == item_type:
                            # If we're starting near a splitter entrance, we can use it
                            for entrance in splitter_info['entrances']:
                                if self.heuristic(start, entrance) <= 5:  # Increased search radius
                                    # Check if this splitter's exit is already connected to a destination
                                    exit_pos = splitter_info['exit']
                                    is_connected = False
                                    
                                    # Look through existing paths to see if any go from the splitter exit to a destination
                                    for existing_item, path_info in paths.items():
                                        if self.points[existing_item]['item'] == item_type and path_info.get('used_splitter') == splitter_id:
                                            existing_path = path_info['path']
                                            # Find where the splitter exit is in the path
                                            try:
                                                exit_index = existing_path.index(exit_pos)
                                                # Get the path from the splitter exit to the destination
                                                exit_to_dest = existing_path[exit_index:]
                                                is_connected = True
                                                break
                                            except ValueError:
                                                continue
                                    
                                    # Add a path to this entrance
                                    entrance_path, entrance_dir_grid = self.astar(grid, start, entrance)
                                    if entrance_path:
                                        if is_connected:
                                            # Use the existing path from exit to destination
                                            combined_path = entrance_path + exit_to_dest[1:]  # Skip the first element (splitter exit)
                                        else:
                                            # Find a new path from the splitter exit to the destination
                                            exit_path, exit_dir_grid = self.astar(grid, exit_pos, dest)
                                            if not exit_path:
                                                continue  # Try next entrance if no path to destination
                                            
                                            combined_path = entrance_path + exit_path[1:]  # Skip the first element (splitter exit)
                                        
                                        # Add splitter usage to the path information
                                        paths[item] = {
                                            "path": combined_path,
                                            "direction_grid": entrance_dir_grid,  # Use the entrance grid for now
                                            "jump_markers": self.get_jump_markers(combined_path),
                                            "used_splitter": splitter_id
                                        }
                                        found_path = True
                                        
                                        # Mark the path on the grid
                                        for px, py in combined_path:
                                            self.base_grid[py][px] = 9
                                            grid[py][px] = 9
                                        
                                        break  # Found a path using this splitter
                            
                            if found_path:
                                break  # Found a path using some splitter
                
                # If we haven't found a path using splitters (or if splitters are disabled), try normal pathfinding
                if not found_path:
                    path, direction_grid = self.astar(grid, start, dest)
                    
                    if path:
                        # Record the inserter information if we placed one
                        if inserter_position:
                            ix, iy = inserter_position
                            placed_inserter_information.append([points['item'], ix, iy])
                            self.base_grid[iy][ix] = 12  # Also update the base grid
                        
                        jump_markers = self.get_jump_markers(path)
                        
                        # Mark the path on both grids to prevent overlaps
                        for px, py in path:
                            self.base_grid[py][px] = 9
                            grid[py][px] = 9  # Update the working grid too
                        
                        # Add point information to the retrieval points
                        for updated_item, updated_points in self.points.items():
                            if updated_item != item and updated_points['item'] == points['item']:
                                updated_points['destination'].extend(path)
                        
                        paths[item] = {
                            "path": path,
                            "direction_grid": direction_grid,
                            "jump_markers": jump_markers
                        }
                        found_path = True
                        break
                    
                    # If path not found and inserter was placed, remove it from the grid
                    elif inserter_position:
                        ix, iy = inserter_position
                        grid[iy][ix] = 0  # Reset the inserter position
            
            if found_path:
                continue
            
        return paths, placed_inserter_information
    
    def find_splitter_locations(self, item_type):
        """
        Find potential locations to place splitters for a given item type.
        Prioritizes locations near destination points to merge incoming belts.
        """
        potential_locations = []
        
        # First, collect all destination points for this item type
        destination_points = []
        for item, points in self.points.items():
            if points.get('item') == item_type:
                destination_points.extend(points['destination'])
        
        # Look for opportunities to place splitters near destination points
        for dest_x, dest_y in destination_points:
            # Check all directions for potential splitter placement
            for direction in DIRECTIONS:
                dx, dy = direction
                
                # The splitter should be placed so the destination is at one of its entrances
                # So we need to go in the opposite direction from the destination
                splitter_x = dest_x + dx  # Position where splitter might be placed
                splitter_y = dest_y + dy
                
                # Check if this position is valid for a splitter
                if self.can_place_splitter((splitter_x, splitter_y), (-dx, -dy), item_type, allow_path_overlap=True):
                    potential_locations.append({
                        'position': (splitter_x, splitter_y),
                        'direction': (-dx, -dy),  # Splitter faces opposite direction to feed into destination
                        'priority': 'high',  # High priority as it's near a destination
                        'destination': (dest_x, dest_y)
                    })
        
        # Also look at all existing paths for potential merger points
        for item, path_info in self.paths.items():
            if self.points[item]['item'] == item_type:
                path = path_info['path']
                
                # Find locations where paths run parallel or intersect
                for i in range(1, len(path) - 1):
                    x, y = path[i]
                    
                    # Check the surrounding area for other paths of the same type
                    for dx, dy in DIRECTIONS:
                        nx, ny = x + dx, y + dy
                        
                        # Skip if out of bounds
                        if not (0 <= nx < len(self.base_grid[0]) and 0 <= ny < len(self.base_grid)):
                            continue
                        
                        # If we found another path point for the same item type
                        if self.base_grid[ny][nx] == 9:
                            # Check all possible splitter directions
                            for direction in DIRECTIONS:
                                if self.can_place_splitter((x, y), direction, item_type, allow_path_overlap=True):
                                    potential_locations.append({
                                        'position': (x, y),
                                        'direction': direction,
                                        'priority': 'medium',  # Medium priority as it merges paths but not at destination
                                    })
        
        # Sort locations by priority (high, then medium)
        potential_locations.sort(key=lambda loc: 0 if loc.get('priority') == 'high' else 1)
        
        return potential_locations

    def can_place_splitter(self, position, direction, item_type, allow_path_overlap=True):
        """
        Check if a splitter can be placed at the given position and direction.
        """
        x, y = position
        dx, dy = direction
        
        # Check if the main position is within bounds
        if not (0 <= x < len(self.base_grid[0]) and 0 <= y < len(self.base_grid)):
            return False
        
        # Check the second tile of the splitter (2x1 structure)
        second_x, second_y = x, y
        if dx == 0:  # Vertical splitter
            second_x = x + 1
        else:  # Horizontal splitter
            second_y = y + 1
        
        # Check if the second position is within bounds
        if not (0 <= second_x < len(self.base_grid[0]) and 0 <= second_y < len(self.base_grid)):
            return False
        
        # Check if both tiles are either empty or paths (9)
        main_tile_valid = self.base_grid[y][x] == 0 or (allow_path_overlap and self.base_grid[y][x] == 9)
        second_tile_valid = self.base_grid[second_y][second_x] == 0 or (allow_path_overlap and self.base_grid[second_y][second_x] == 9)
        
        if not (main_tile_valid and second_tile_valid):
            return False
        
        # Check entrance positions (opposite to the direction)
        entrances = self.get_splitter_entrances(position, direction)
        for ex, ey in entrances:
            if not (0 <= ex < len(self.base_grid[0]) and 0 <= ey < len(self.base_grid)):
                return False
            
            # Entrances should be either empty or paths
            if self.base_grid[ey][ex] != 0 and (not allow_path_overlap or self.base_grid[ey][ex] != 9):
                return False
        
        # Check exit position (in the direction)
        exit_x, exit_y = self.get_splitter_exit(position, direction)
        if not (0 <= exit_x < len(self.base_grid[0]) and 0 <= exit_y < len(self.base_grid)):
            return False
        
        # Exit should be empty or a path
        if self.base_grid[exit_y][exit_x] != 0 and (not allow_path_overlap or self.base_grid[exit_y][exit_x] != 9):
            return False
        
        return True

    def is_path_for_item(self, x, y, item_type):
        """Check if the position is part of a path for the given item type."""
        # Check all paths to see if this position is on a path for the right item
        for item, path_info in self.paths.items():
            if self.points[item]['item'] == item_type:
                if (x, y) in path_info['path']:
                    return True
        return False

    def is_destination_for_item(self, x, y, item_type):
        """Check if the position is a destination point for the given item type."""
        for item, points in self.points.items():
            if points.get('item') == item_type and (x, y) in points['destination']:
                return True
        return False


    def visualize_grid(self, output_path="grid_visualization.png", cell_size=32):
        """
        Create a visual representation of the grid with paths, underground belts, and splitters.
        
        Args:
            output_path (str): Path to save the visualization image
            cell_size (int): Size of each grid cell in pixels
        """
        try:
            # Load assets
            assets_folder = "assets"
            conveyor_img = Image.open(os.path.join(assets_folder, "conveyor.png")).convert("RGBA").resize((cell_size, cell_size))
            underground_img = Image.open(os.path.join(assets_folder, "underground.png")).convert("RGBA").resize((cell_size, cell_size))
            splitter_img_raw = Image.open(os.path.join(assets_folder, "splitter.png")).convert("RGBA")
            
            # Create horizontal and vertical splitter images (2x1 or 1x2)
            horizontal_splitter = Image.new("RGBA", (cell_size*2, cell_size), (0, 0, 0, 0))
            horizontal_splitter.paste(splitter_img_raw.resize((cell_size*2, cell_size)), (0, 0))
            
            vertical_splitter = Image.new("RGBA", (cell_size, cell_size*2), (0, 0, 0, 0))
            vertical_splitter.paste(splitter_img_raw.resize((cell_size, cell_size*2)).transpose(Image.ROTATE_90), (0, 0))
            
            # Determine grid dimensions
            grid_height = len(self.base_grid)
            grid_width = len(self.base_grid[0]) if grid_height > 0 else 0
            
            # Create an image for the grid
            img_width = grid_width * cell_size
            img_height = grid_height * cell_size
            grid_img = Image.new("RGBA", (img_width, img_height), (50, 50, 50, 255))  # Dark gray background
            draw = ImageDraw.Draw(grid_img)
            
            # Draw grid cells
            for y in range(grid_height):
                for x in range(grid_width):
                    cell_value = self.base_grid[y][x]
                    cell_pos = (x * cell_size, y * cell_size)
                    
                    # Draw different cell types
                    if cell_value == 0:  # Empty cell
                        draw.rectangle((cell_pos[0], cell_pos[1], 
                                    cell_pos[0] + cell_size, cell_pos[1] + cell_size), 
                                    fill=(30, 30, 30))
                    elif cell_value == 1:  # Obstacle
                        draw.rectangle((cell_pos[0], cell_pos[1], 
                                    cell_pos[0] + cell_size, cell_pos[1] + cell_size), 
                                    fill=(200, 50, 50))  # Red
                    elif cell_value == 9:  # Path
                        grid_img.paste(conveyor_img, cell_pos, conveyor_img)
                    elif cell_value == 12:  # Inserter
                        draw.rectangle((cell_pos[0], cell_pos[1], 
                                    cell_pos[0] + cell_size, cell_pos[1] + cell_size), 
                                    fill=(50, 150, 250))  # Blue
                    elif cell_value == 13:  # Splitter
                        # Don't draw splitter here - will handle separately
                        pass
            
            # Draw underground belt jumps
            for item, path_info in self.paths.items():
                jumps = path_info.get('jump_markers', [])
                for entrance, exit, direction in jumps:
                    # Draw underground entrance
                    entrance_x, entrance_y = entrance
                    entrance_pos = (entrance_x * cell_size, entrance_y * cell_size)
                    rotated_underground = self.rotate_image(underground_img, direction)
                    grid_img.paste(rotated_underground, entrance_pos, rotated_underground)
                    
                    # Draw underground exit
                    exit_x, exit_y = exit
                    exit_pos = (exit_x * cell_size, exit_y * cell_size)
                    # Rotate exit in opposite direction
                    opposite_dir = (-direction[0], -direction[1])
                    rotated_exit = self.rotate_image(underground_img, opposite_dir)
                    grid_img.paste(rotated_exit, exit_pos, rotated_exit)
            
            # Draw splitters
            for splitter_id, splitter_info in self.splitters.items():
                pos = splitter_info['position']
                direction = splitter_info['direction']
                x, y = pos
                
                # Choose horizontal or vertical splitter based on direction
                if direction[0] == 0:  # Vertical (up or down)
                    splitter = vertical_splitter
                    pos = (x * cell_size, y * cell_size)
                    # Rotate based on direction
                    if direction[1] > 0:  # Down
                        splitter = self.rotate_image(vertical_splitter, (0, 1))
                    else:  # Up
                        splitter = self.rotate_image(vertical_splitter, (0, -1))
                else:  # Horizontal (left or right)
                    splitter = horizontal_splitter
                    pos = (x * cell_size, y * cell_size)
                    # Rotate based on direction
                    if direction[0] > 0:  # Right
                        splitter = self.rotate_image(horizontal_splitter, (1, 0))
                    else:  # Left
                        splitter = self.rotate_image(horizontal_splitter, (-1, 0))
                
                grid_img.paste(splitter, pos, splitter)
            
            # Draw start and destination points
            for item, points in self.points.items():
                # Draw start points
                for x, y in points.get('start_points', []):
                    pos = (x * cell_size, y * cell_size)
                    draw.ellipse((pos[0] + cell_size//4, pos[1] + cell_size//4,
                                pos[0] + 3*cell_size//4, pos[1] + 3*cell_size//4),
                                fill=(0, 255, 0))  # Green circle
                    
                # Draw destination points
                for x, y in points.get('destination', []):
                    pos = (x * cell_size, y * cell_size)
                    draw.ellipse((pos[0] + cell_size//4, pos[1] + cell_size//4,
                                pos[0] + 3*cell_size//4, pos[1] + 3*cell_size//4),
                                fill=(255, 150, 0))  # Orange circle
            
            # Save the image
            grid_img.save(output_path)
            print(f"Grid visualization saved to {output_path}")
            
            return True
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return False

    def rotate_image(self, img, direction):
        """Rotate an image based on direction vector."""
        if direction == (1, 0):  # Right
            return img
        elif direction == (-1, 0):  # Left
            return img.transpose(Image.ROTATE_180)
        elif direction == (0, 1):  # Down
            return img.transpose(Image.ROTATE_270)
        elif direction == (0, -1):  # Up
            return img.transpose(Image.ROTATE_90)
        return img

    def visualize_path(self, item, output_path=None, cell_size=32):
        """
        Create a visual representation of a specific path.
        
        Args:
            item (str): The item identifier whose path to visualize
            output_path (str): Path to save the visualization image (default: item_path.png)
            cell_size (int): Size of each grid cell in pixels
        """
        if output_path is None:
            output_path = f"{item}_path.png"
        
        if item not in self.paths:
            print(f"No path found for item {item}")
            return False
        
        path_info = self.paths[item]
        path = path_info['path']
        jumps = path_info.get('jump_markers', [])
        used_splitter = path_info.get('used_splitter')
        
        # First create the base grid visualization
        if not self.visualize_grid(output_path, cell_size):
            return False
        
        # Load the base image and draw the specific path on top
        try:
            img = Image.open(output_path).convert("RGBA")
            draw = ImageDraw.Draw(img)
            
            # Highlight the specific path
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                
                # Draw a line connecting the points
                draw.line(
                    (x1 * cell_size + cell_size//2, y1 * cell_size + cell_size//2,
                    x2 * cell_size + cell_size//2, y2 * cell_size + cell_size//2),
                    fill=(255, 255, 0, 180), width=3  # Yellow semi-transparent line
                )
            
            # Highlight the points along the path
            for x, y in path:
                draw.ellipse(
                    (x * cell_size + cell_size//3, y * cell_size + cell_size//3,
                    x * cell_size + 2*cell_size//3, y * cell_size + 2*cell_size//3),
                    outline=(255, 255, 0), width=2  # Yellow outline
                )
            
            # Highlight splitter used if any
            if used_splitter:
                splitter_info = self.splitters.get(used_splitter)
                if splitter_info:
                    x, y = splitter_info['position']
                    draw.rectangle(
                        (x * cell_size, y * cell_size,
                        x * cell_size + cell_size, y * cell_size + cell_size),
                        outline=(0, 255, 255), width=3  # Cyan outline
                    )
            
            # Save the enhanced image
            img.save(output_path)
            print(f"Path visualization for {item} saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error creating path visualization: {e}")
            return False