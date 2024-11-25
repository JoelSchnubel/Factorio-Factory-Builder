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
    def __init__(self, grid,underground_length=3,allow_jump = True):
        self.base_grid  = grid
        self.underground_length = underground_length
        self.allow_jump = allow_jump
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
        
        # Go through each step up to the jump distance
        
        for step in range(1, self.underground_length + 1):
            nx, ny = x + dx * step, y + dy * step
            
            
            # Log the current step coordinates
            logging.debug(f"Step {step}: Checking node ({nx}, {ny})")
            
            # Check if the target position is within bounds
            if not (0 <= nx < len(grid[0]) and 0 <= ny < len(grid)):
                logging.debug(f"Out of bounds: ({nx}, {ny})")
                return False
            
            # Check for obstacles only if we have not reached the final step of the jump
            if grid[ny][nx] != 0:
                if step < self.underground_length:
                    logging.debug(f"Obstacle found at ({nx}, {ny}), blocking the jump")
                    return False
                else:
                    logging.debug(f"Obstacle found at ({nx}, {ny}), but this is the final jump position")

        # If we successfully checked all steps and found no obstacles until the final jump
        logging.debug(f"Jump possible to ({nx}, {ny})")
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
    
    def find_path_for_item(self, start_dest_sets):
        paths = {}
        grid = [row[:] for row in self.base_grid]  # Deep copy of the grid for each function call
        # Clear starting and destination points on the grid
        for item, points in start_dest_sets.items():
            for x, y in points['start_points'] + points['destination']:
                grid[y][x] = 0
            found_path = False
            for start in points['start_points']:
                for dest in points['destination']:
                    logging.info(f"Attempting path for {item} from {start} to {dest}")
                    path, direction_grid = self.astar(grid, start, dest)
                    
                    jump_markers = self.get_jump_markers(path)
                    
                    if path:
                        paths[item] = {
                            "path": path,
                            "direction_grid": direction_grid,
                            "jump_markers": jump_markers
                        }
                        found_path = True
                        break
                if found_path:
                    break
        return paths
    
    
def main():
    
    grid = [
        [99, 44, 33, 33, 33, 0, 0, 0, 0, 1],
        [1, 0, 33, 33, 33, 0, 0, 0, 0, 1],
        [1, 0, 33, 33, 33, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 99, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 44, 0, 0, 0, 0, 1],
        [1, 99, 44, 33, 33, 33, 0, 0, 0, 1],
        [1, 0, 0, 33, 33, 33, 0, 0, 0, 1],
        [1, 0, 0, 33, 33, 33, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 22, 22, 22, 22, 22]
    ]
        
    start_dest_sets = {
    'electronic-circuit': {
        'destination': [(6, 9), (7, 9), (8, 9), (9, 9), (5, 9)],
        'start_points': [(3, 4), (5, 4), (1, 7), (1, 8), (8, 6), (8, 7), (8, 8)]
    }
    }
    '''
    grid = [
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [1,  1,  1,  1,  1],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0]
    ]

    start_dest_sets = {
        'simple-item': {
            'start_points': [(0, 0)],
            'destination': [(4, 4)]
        }
    }
    '''

    
    astar_pathfinder = AStarPathFinder(grid)
    paths = astar_pathfinder.find_path_for_item(start_dest_sets)
    
    print("Path results:")
    for item, path_data in paths.items():
        print(f"Item: {item}")
        print("Path:", path_data["path"])
        print("Direction Grid" ,path_data["direction_grid"])
        print("Jump Markers:", path_data["jump_markers"])
        
    
   
if __name__ == "__main__":
    main()