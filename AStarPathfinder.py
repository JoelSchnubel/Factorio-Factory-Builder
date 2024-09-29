import heapq

# Directions for moving in a grid (up, down, left, right)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Can extend with diagonal directions

class AStarPathfinder:
    def __init__(self, grid, grid_width, grid_height):
        self.grid = grid
        self.width = grid_width
        self.height = grid_height

    def heuristic(self, current, goal):
        """Manhattan distance heuristic for A*."""
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def is_valid(self, node):
        """Check if the node is within bounds and not an obstacle (assembler or inserter)."""
        x, y = node
        if 0 <= x < self.width and 0 <= y < self.height and self.grid[x][y] == 0:
            return True
        return False

    def astar(self, start, goal):
        """A* search algorithm to find the shortest path from start to goal."""
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

    def reconstruct_path(self, came_from, current):
        """Reconstruct the path from start to goal using the came_from dictionary."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path
    

# Example grid: 0 = free, 1 = obstacle (assembler or inserter)
# Create the grid based on assembler and inserter positions
def create_grid(assemblers, inserters, grid_width, grid_height):
    grid = [[0 for _ in range(grid_height)] for _ in range(grid_width)]
    
    # Mark assemblers on the grid
    for assembler in assemblers:
        x, y = assembler
        for i in range(3):
            for j in range(3):
                grid[x + i][y + j] = 1  # Occupied by an assembler

    # Mark inserters on the grid
    for inserter in inserters:
        x, y = inserter
        grid[x][y] = 1  # Occupied by an inserter
    
    return grid

# Define a function to connect all belt start and end points using A*
def connect_belts(belt_pairs, grid, grid_width, grid_height):
    pathfinder = AStarPathfinder(grid, grid_width, grid_height)
    
    for belt_start, belt_end in belt_pairs:
        path = pathfinder.astar(belt_start, belt_end)
        if path:
            print(f"Path from {belt_start} to {belt_end}: {path}")
            # Mark the path on the grid
            for x, y in path:
                grid[x][y] = 2  # Belt path marked with '2'
        else:
            print(f"No valid path found from {belt_start} to {belt_end}")
    
    return grid

# Example usage:

# Define some assemblers and inserters (assembler is 3x3, inserter is 1x1)
assemblers = [(1, 1), (5, 1)]  # Assemblers placed at these positions
inserters = [(4, 2), (8, 2)]  # Inserters placed next to assemblers

# Define belt start and end positions
belt_pairs = [((4, 2), (8, 2))]  # Example: belt starts at (4, 2) and ends at (8, 2)

# Define the grid size
grid_width = 10
grid_height = 5

# Create the grid with obstacles
grid = create_grid(assemblers, inserters, grid_width, grid_height)

# Connect belts using A*
connected_grid = connect_belts(belt_pairs, grid, grid_width, grid_height)

# Display the final grid with paths
for row in connected_grid:
    print(' '.join(map(str, row)))