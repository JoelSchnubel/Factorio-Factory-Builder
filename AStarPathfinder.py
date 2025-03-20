import heapq
import logging



logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for detailed information, change to INFO for less verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("AStar.log", mode='w'),  # Specify the log file name
    ]
)
# Directions for moving in a grid (up, down, left, right)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
DIRECTION_MAP = {(0, 1): 'right', (1, 0): 'down', (0, -1): 'left', (-1, 0): 'up'}

class AStarPathFinder:
    def __init__(self, grid,points,underground_length=3,allow_jump = True):
        self.base_grid  = grid
        self.underground_length = underground_length
        self.allow_jump = allow_jump
        self.points = points
        self.paths = {}
  
        
        
    def heuristic(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def is_valid(self, node, grid):
        x, y = node
        return 0 <= x < len(grid[0]) and 0 <= y < len(grid) and grid[y][x] == 0
    
    def can_jump(self, current, direction, grid):
        x, y = current
        dx, dy = direction
        logging.debug(f"Checking jump possibility from {current} in direction {direction}")
        
        # Check if the underground entrance point is valid
        if not (0 <= x < len(grid[0]) and 0 <= y < len(grid)):
            logging.debug(f"Underground entrance out of bounds: ({x}, {y})")
            return False
        
        if grid[y][x] != 0:
            logging.debug(f"Underground entrance position not available: ({x}, {y})")
            return False
            
        # Calculate underground exit point
        exit_x = x + dx * (self.underground_length + 1)
        exit_y = y + dy * (self.underground_length + 1)
        
        # Check if the exit point is valid
        if not (0 <= exit_x < len(grid[0]) and 0 <= exit_y < len(grid)):
            logging.debug(f"Underground exit out of bounds: ({exit_x}, {exit_y})")
            return False
            
        if grid[exit_y][exit_x] != 0:
            logging.debug(f"Underground exit position not available: ({exit_x}, {exit_y})")
            return False
        
        # Check that all intermediate positions can be tunneled under
        for step in range(1, self.underground_length + 1):
            nx, ny = x + dx * step, y + dy * step
            
            # Check if position is within bounds
            if not (0 <= nx < len(grid[0]) and 0 <= ny < len(grid)):
                logging.debug(f"Underground path out of bounds at: ({nx}, {ny})")
                return False
        
        logging.debug(f"Underground path possible from ({x}, {y}) to ({exit_x}, {exit_y})")
        return True
        
    
    def astar(self, grid, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_costs = {start: 0}
        f_costs = {start: self.heuristic(start, goal)}
        direction_grid = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]
        jump_markers = []
        
        logging.info(f"Starting A* from {start} to {goal}")
        
        while open_list:
            _, current = heapq.heappop(open_list)
            logging.info(f"Processing node {current}")
            
            if current == goal:
                logging.info(f"Goal reached at {current}")
                return self.reconstruct_path(came_from, current), direction_grid#, jump_markers
            
            for direction in DIRECTIONS:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                
                if self.is_valid(neighbor, grid):
                    tentative_g_cost = g_costs[current] + 1
                    
                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        came_from[neighbor] = current
                        g_costs[neighbor] = tentative_g_cost
                        f_costs[neighbor] = tentative_g_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_costs[neighbor], neighbor))
                        
                        direction_grid[neighbor[1]][neighbor[0]] = direction
                        logging.info(f"Moving to {neighbor} with direction {DIRECTION_MAP[direction]}")
                        
                elif self.allow_jump:
                    for length in range(1, self.underground_length + 1):
                        jump_target = (current[0] + direction[0] * length, current[1] + direction[1] * length)
        
                        if self.is_valid(jump_target, grid):
                            logging.debug(f"Checking jump to {jump_target} from {current} with length {length}")
                            
                            tentative_g_cost = g_costs[current] + self.underground_length
                            
                            if jump_target not in g_costs or tentative_g_cost < g_costs[jump_target]:
                                came_from[jump_target] = current
                                g_costs[jump_target] = tentative_g_cost
                                f_costs[jump_target] = tentative_g_cost + self.heuristic(jump_target, goal)
                                heapq.heappush(open_list, (f_costs[jump_target], jump_target))
                                #jump_markers.append((current, jump_target))
                                direction_grid[jump_target[1]][jump_target[0]] = direction
                                logging.info(f"Jumping to {jump_target} from {current} with direction {DIRECTION_MAP[direction]}")
                            
        logging.info("No path found")
        return None, direction_grid #jump_markers

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        logging.info(f"Reconstructed path: {path}")
        return path

    def get_jump_markers(self,path):
        jump_markers = []  # List to store the jump markers

        # Iterate through the path to check for jumps
        for i in range(1, len(path)):
            start = path[i - 1]
            end = path[i]
            
            # Calculate the Manhattan distance between the points
            distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
            
            # If the distance is greater than 1, it's a jump
            if distance > 1:
                jump_markers.append((start, end))
        
        return jump_markers
    
    def find_path_for_item(self):
        placed_inserter_information = []
        paths = {}
        grid = [row[:] for row in self.base_grid]  # Deep copy of the grid for each function call
        
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

                path, direction_grid, underground_paths = self.astar(grid, start, dest)
                
                if path:
                    # check if we need to add an inserter:
                    if points.get("inserter_mapping") is not None:
                        (ix, iy) = points["inserter_mapping"][str(start)]
                        placed_inserter_information.append([points['item'], ix, iy])
                        self.base_grid[ix][iy] = 12
                    
                    # Mark the path on the grid to prevent overlaps
                    for px, py in path:
                        self.base_grid[py][px] = 9
                    
                    # Mark underground belt entrances and exits
                    for entrance, exit_point, direction in underground_paths:
                        # Mark entrance (10 could be the code for underground belt entrance)
                        self.base_grid[entrance[1]][entrance[0]] = 10
                        
                        # Mark exit (11 could be the code for underground belt exit)
                        self.base_grid[exit_point[1]][exit_point[0]] = 11
                    
                    # Add point information to the retrieval points
                    for updated_item, updated_points in self.points.items():
                        if updated_item != item and updated_points['item'] == points['item']:
                            self.points[updated_item]['destination'].extend(path)
                    
                    paths[item] = {
                        "path": path,
                        "direction_grid": direction_grid,
                        "underground_paths": underground_paths
                    }
                    found_path = True
                    break
                
            if found_path:
                continue
            
        return paths, placed_inserter_information
    
    
    
