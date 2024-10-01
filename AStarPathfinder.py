import heapq

# Directions for moving in a grid (up, down, left, right)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
class AStarPathfinder:
    def __init__(self, grid, pairs):
        self.grid = grid
        self.pairs = self.remove_redundant_pairs(pairs)
        self.width = len(grid[0])
        self.height = len(grid)

    # Removes redundant pairs of start and goal points
    def remove_redundant_pairs(self, pairs):
        unique_pairs = []
        for pair in pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
                
        self.adapt_grid(unique_pairs)
        return unique_pairs
    
    # Modifies the grid based on the pairs -> make start and goal points not obstacle
    def adapt_grid(self,pairs):
        for start, goal in pairs:
            sx, sy = start
            gx, gy = goal
            # Set the start and goal points in the grid to 0
            self.grid[sy][sx] = 0  
            self.grid[gy][gx] = 0
    
    # Manhattan distance 
    def heuristic(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    # Check if the node is within bounds and not an obstacle
    def is_valid(self, node):
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == 0

    # search algorithm to find the shortest path from start to goal
    def astar(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_costs = {start: 0}
        f_costs = {start: self.heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for direction in DIRECTIONS:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if self.is_valid(neighbor):
                    tentative_g_cost = g_costs[current] + 1

                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        came_from[neighbor] = current
                        g_costs[neighbor] = tentative_g_cost
                        f_costs[neighbor] = tentative_g_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_costs[neighbor], neighbor))

        # No path found
        return None

    # Reconstruct the path from start to goal
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    # Define a function to connect all belt start and end points using A*
    # if all pairs can be connnected return True else False
    def connect_belts(self):
        self.direction_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for points in self.pairs:
            # Connect each consecutive point in the pair
            belt_start = points[0]
            belt_end = points[1]
            path = self.astar(belt_start, belt_end)

            if path:
                print(f"Path from {belt_start} to {belt_end}")
                # Mark the path on the grid, excluding start and goal points
                
                
                for i in range(1, len(path) - 1):  # Skip start and end points
                    x, y = path[i]
                    next_x, next_y = path[i + 1]
                    dx, dy = next_x - x, next_y - y
                    
                    # Mark direction based on the relative position of the next cell
                    if dx == 1 and dy == 0:
                        direction = 'right'
                    elif dx == -1 and dy == 0:
                        direction = 'left'
                    elif dx == 0 and dy == 1:
                        direction = 'down'
                    elif dx == 0 and dy == -1:
                        direction = 'up'
                    else:
                        direction = None  
                    
                    self.direction_grid[y][x] = direction
                
                for x, y in path[1:-1]:  # Skip start and end points
                    self.grid[y][x] = 2  # Belt path marked with '2'
            else:
                print(f"No valid path found from {belt_start} to {belt_end}")
                return False
            
        return True
    
    
    




def main():

    return
    astar = AStarPathfinder(grid,pairs)

    grid = astar.connect_belts()
    
    #paths = astar.connect_pairs()
    
    for row in grid:
        print(row)
if __name__ == "__main__":
    main()