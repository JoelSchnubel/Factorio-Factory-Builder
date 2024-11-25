#! .venv\Scripts\python.exe

class TestEnvironment:
    def __init__(self, width, height):
        # Setting up width, height, and obstacle maps for the test
        self.width = width
        self.height = height
        self.obstacle_maps = []

    def set_obstacle_map(self,obstacle_map):
        self.obstacle_maps.append(obstacle_map)
    
    def setup_obstacle(self, x, y, value=1):
        """Set up an obstacle at specified coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacle_maps[-1][y][x] = value

    def check_space_around_path(self, belt_list, item_type):
        obstacle_map = self.obstacle_maps[-1]
        height = len(obstacle_map)  # Number of rows
        width = len(obstacle_map[0]) if height > 0 else 0  # Number of columns

        available_positions = {}  # To store results with belt as key

        for belt in belt_list:
            if belt.item == item_type:
                belt_x = belt.int_x
                belt_y = belt.int_y

                # Store possible positions
                directions = {}

                # Helper function to check if a 3x3 area is free
                def is_3x3_space_clear(start_x, start_y):
                    if not (0 <= start_x < width and 0 <= start_y < height):
                        return False  # Out of bounds
                    for dx in range(3):
                        for dy in range(3):
                            x, y = start_x + dx, start_y + dy
                            if x >= width or y >= height or obstacle_map[y][x] != 0:
                                return False
                    return True

                # Check left - three possible 3x3 positions aligned vertically
                if belt_x > 0 and obstacle_map[belt_y][belt_x - 1] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        left_x = belt_x - 4  # Place structure left of the belt
                        left_y = belt_y + offset
                        if is_3x3_space_clear(left_x, left_y):
                            inserter_pos = (belt_x - 1, belt_y)
                            directions['left'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((left_x, left_y))

                # Check right - three possible 3x3 positions aligned vertically
                if belt_x < width - 1 and obstacle_map[belt_y][belt_x + 1] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        right_x = belt_x + 2
                        right_y = belt_y + offset
                        if is_3x3_space_clear(right_x, right_y):
                            inserter_pos = (belt_x + 1, belt_y)
                            directions['right'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((right_x, right_y))

                # Check up - three possible 3x3 positions aligned horizontally
                if belt_y > 0 and obstacle_map[belt_y - 1][belt_x] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        up_x = belt_x + offset
                        up_y = belt_y - 4
                        if is_3x3_space_clear(up_x, up_y):
                            inserter_pos = (belt_x, belt_y - 1)
                            directions['up'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((up_x, up_y))

                # Check down - three possible 3x3 positions aligned horizontally
                if belt_y < height - 1 and obstacle_map[belt_y + 1][belt_x] == 0:
                    assembler_positions = []
                    for offset in range(-2, 1):  # Offset -2, -1, 0
                        down_x = belt_x + offset
                        down_y = belt_y + 2
                        if is_3x3_space_clear(down_x, down_y):
                            inserter_pos = (belt_x, belt_y + 1)
                            directions['down'] = {
                                'inserter_pos': inserter_pos,
                                'assembler_pos': assembler_positions
                            }
                            assembler_positions.append((down_x, down_y))

                # Store the directions in the available positions dictionary
                if directions:
                    available_positions[belt] = directions

        return available_positions
    

          

    def run_test(self):
        # Set up a sample obstacle map
        #self.setup_obstacle(3, 1)  # Example: place an obstacle at (3, 1)

        # Define belts for testing with x, y positions
        belt_list = [
            Belt(5, 0, "IronPlate"),
            Belt(5, 1, "IronPlate"),
            Belt(5, 2, "IronPlate"),
            Belt(5, 3, "IronPlate"),
        ]

        # Run the test for each item type
        results = self.check_space_around_path(belt_list, "IronPlate")
        print("Available positions for IronPlate:", results)

        results = self.check_space_around_path(belt_list, "CopperPlate")
        print("Available positions for CopperPlate:", results)

# Belt class for testing
class Belt:
    def __init__(self, x, y, item):
        self.int_x = x
        self.int_y = y
        self.item = item

# Create and run the test environment
test_env = TestEnvironment(10, 10)  # For a 10x10 grid
test_env.set_obstacle_map([
            [0, 0, 0, 0, 0, 0, 0, 22, 22, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 22, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
test_env.run_test()
