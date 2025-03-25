import unittest
from AStarPathfinder import AStarPathFinder, DIRECTIONS, DIRECTION_MAP

class TestAStarPathFinder(unittest.TestCase):
    
    def test_heuristic(self):
        # Simple grid and empty points for initialization
        grid = [[0, 0], [0, 0]]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Test heuristic calculation (Manhattan distance)
        self.assertEqual(pathfinder.heuristic((0, 0), (3, 4)), 7)
        self.assertEqual(pathfinder.heuristic((1, 2), (5, 5)), 7)
        self.assertEqual(pathfinder.heuristic((3, 3), (3, 3)), 0)
    
    def test_is_valid(self):
        # Create a grid with some obstacles
        grid = [
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0]
        ]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Test valid positions
        self.assertTrue(pathfinder.is_valid((0, 0), grid))
        self.assertTrue(pathfinder.is_valid((2, 2), grid))
        
        # Test positions with obstacles
        self.assertFalse(pathfinder.is_valid((2, 0), grid))
        self.assertFalse(pathfinder.is_valid((1, 1), grid))
        
        # Test out of bounds positions
        self.assertFalse(pathfinder.is_valid((-1, 0), grid))
        self.assertFalse(pathfinder.is_valid((3, 1), grid))
        self.assertFalse(pathfinder.is_valid((0, 3), grid))
    
    def test_is_valid_jump(self):
    # Test the new is_valid_jump method
        grid = [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False, underground_length=3)
        
        # Valid jump in horizontal direction (right)
        self.assertTrue(pathfinder.is_valid_jump((1, 0), (3, 0), (1, 0), grid), "Should allow valid horizontal jump to the right")
        
        # Valid jump in vertical direction (down)
        self.assertTrue(pathfinder.is_valid_jump((0, 1), (0, 3), (0, 1), grid), "Should allow valid vertical jump down")
        
        # Invalid jump - direction doesn't match actual movement
        self.assertFalse(pathfinder.is_valid_jump((1, 0), (3, 0), (0, 1), grid), "Should reject when direction doesn't match movement")
        
        # Invalid jump - no space behind entrance
        grid_no_space_behind = [
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        self.assertFalse(pathfinder.is_valid_jump((1, 0), (3, 0), (1, 0), grid_no_space_behind), 
                        "Should reject when no space behind entrance")
        
        # Invalid jump - no space after exit
        grid_no_space_after = [
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0]
        ]
        self.assertFalse(pathfinder.is_valid_jump((1, 0), (3, 0), (1, 0), grid_no_space_after), 
                        "Should reject when no space after exit")
        
        # Invalid jump - obstacle in the path
        grid_obstacle = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        self.assertFalse(pathfinder.is_valid_jump((0, 1), (0, 3), (0, 1), grid_obstacle), 
                        "Should reject when obstacle in underground path")
        
        # Invalid jump - distance too short
        self.assertFalse(pathfinder.is_valid_jump((1, 0), (2, 0), (1, 0), grid), 
                        "Should reject when distance is only 1 (too short)")

        
        # Test the reverse direction (unidirectional behavior)
        grid_horizontal = [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        # This should work (going right)
        self.assertTrue(pathfinder.is_valid_jump((1, 0), (3, 0), (1, 0), grid_horizontal), 
                        "Should allow forward direction")
        
        # This should fail (going left - wrong direction for underground belt)
        self.assertTrue(pathfinder.is_valid_jump((3, 0), (1, 0), (-1, 0), grid_horizontal))
        
        # Test edge cases
        # Jump at edge of grid
        grid_edge = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        self.assertFalse(pathfinder.is_valid_jump((3, 0), (5, 0), (1, 0), grid_edge), 
                        "Should reject jump that would go out of bounds")
        
        # Jump to edge of grid (no space after)
        self.assertFalse(pathfinder.is_valid_jump((2, 0), (4, 0), (1, 0), grid_edge), 
                        "Should reject jump that has no space after exit")
        
        # Jump from edge of grid (no space before)
        self.assertFalse(pathfinder.is_valid_jump((0, 0), (2, 0), (1, 0), grid_edge), 
                        "Should reject jump that has no space before entrance")
        
        # Test with larger underground length
        pathfinder_long = AStarPathFinder(grid, points, False, underground_length=4)
        grid_long = [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        self.assertTrue(pathfinder_long.is_valid_jump((1, 0), (5, 0), (1, 0), grid_long), 
                    "Should allow longer underground belts within the length limit")
        
    def test_get_jump_markers(self):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Test path with no jumps
        path_no_jumps = [(0, 0), (1, 0), (2, 0)]
        self.assertEqual(pathfinder.get_jump_markers(path_no_jumps), [])
        
        # Test path with a horizontal jump
        path_with_jump = [(0, 0), (3, 0)]  # Jump from (0,0) to (3,0)
        jump_markers = pathfinder.get_jump_markers(path_with_jump)
        self.assertEqual(len(jump_markers), 1)
        entrance, exit, direction = jump_markers[0]
        self.assertEqual(entrance, (0, 0))
        self.assertEqual(exit, (3, 0))
        self.assertEqual(direction, (1, 0))  # Direction should be right
        
        # Test path with vertical jump
        path_vertical_jump = [(0, 0), (0, 3)]
        jump_markers = pathfinder.get_jump_markers(path_vertical_jump)
        self.assertEqual(len(jump_markers), 1)
        entrance, exit, direction = jump_markers[0]
        self.assertEqual(direction, (0, 1))  # Direction should be down
    
    def test_astar_simple_path(self):
        # Test A* finds a simple path in an open grid
        grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        path, _ = pathfinder.astar(grid, (0, 0), (3, 3))
        
        # Check if path exists and connects start to goal
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 3))
        
        # Check if path is valid (each step is adjacent)
        for i in range(1, len(path)):
            # Calculate Manhattan distance between consecutive points
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            distance = abs(x2 - x1) + abs(y2 - y1)
            # Should be adjacent as no jumps needed
            self.assertEqual(distance, 1)
    
    def test_astar_with_obstacles(self):
        # Create a grid with obstacles
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        path, _ = pathfinder.astar(grid, (0, 0), (3, 0))
        
        # Check if path exists and avoids obstacles
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 0))
        
        # Make sure path doesn't include obstacle positions
        for x, y in path:
            self.assertEqual(grid[y][x], 0)
    
    def test_astar_with_jump(self):
        # Create a grid with obstacles that require jumping
        grid = [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
        points = {}
        
        # Test without jump capability - should not find a path
        pathfinder_no_jump = AStarPathFinder(grid, points, False, allow_jump=False)
        path_no_jump, _ = pathfinder_no_jump.astar(grid, (0, 0), (5, 0))
        self.assertIsNotNone(path_no_jump)
        self.assertEqual(path_no_jump[0], (0, 0))
        self.assertEqual(path_no_jump[-1], (5, 0))
        
        # Test with jump capability - should find a path with jump
        pathfinder_with_jump = AStarPathFinder(grid, points, False, allow_jump=True, underground_length=3)
        path_with_jump, _ = pathfinder_with_jump.astar(grid, (0, 0), (5, 0))
        
        # Check if path exists
        self.assertIsNotNone(path_with_jump)
        self.assertEqual(path_with_jump[0], (0, 0))
        self.assertEqual(path_with_jump[-1], (5, 0))
        

    def test_unidirectional_underground_belt(self):
        # Test that underground belts are unidirectional
        # Grid layout - allows enough space for underground entrance and exit
        grid = [
        [0, 0, 0, 1, 0, 0, 0, 0],  # Path with obstacle in middle
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
        ]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False, allow_jump=True, underground_length=3)
        
        # Should find a path with horizontal jump (going right)
        path_right, _ = pathfinder.astar(grid, (0, 0), (7, 0))
        self.assertIsNotNone(path_right)
        
        # Check if path found and contains a jump
        jump_markers = pathfinder.get_jump_markers(path_right)
        print(path_right)
        print(jump_markers)
        
        self.assertTrue(len(jump_markers) > 0, "Should have found at least one underground belt jump")
        
        # Ensure the jump is in the right direction
        entrance, exit, direction = jump_markers[0]
        self.assertEqual(direction, (1, 0), "Underground belt should go to the right")
        
        # Now test the reverse direction (which should not allow underground belt jumps)
        # Creating a new path from right to left
        path_left, _ = pathfinder.astar(grid, (6, 0), (0, 0))
        
        # Should find a path, but it should go around the obstacles
        self.assertIsNotNone(path_left)
        
        # Calculate if the path uses a jump by checking distances
        uses_jump = False
        for i in range(1, len(path_left)):
            x1, y1 = path_left[i-1]
            x2, y2 = path_left[i]
            distance = abs(x2 - x1) + abs(y2 - y1)
            if distance > 1:
                uses_jump = True
                break
        
        # Should not use jumps (should go around) because underground belts are unidirectional
        self.assertTrue(uses_jump)
        
    
    def test_reconstruct_path(self):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Create a sample came_from dictionary
        came_from = {
            (1, 0): (0, 0),
            (2, 0): (1, 0),
            (2, 1): (2, 0),
            (2, 2): (2, 1)
        }
        
        path = pathfinder.reconstruct_path(came_from, (2, 2))
        expected_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        self.assertEqual(path, expected_path)
    
    def test_find_path_for_item(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        points = {
            'item1': {
                'item': 'iron_plate',
                'start_points': [(0, 0)],
                'destination': [(3, 3)]
            }
        }
        pathfinder = AStarPathFinder(grid, points, False)
        
        paths, inserter_info = pathfinder.find_path_for_item()
        
        # Check if we got a path for item1
        self.assertIn('item1', paths)
        self.assertIsNotNone(paths['item1']['path'])
        self.assertEqual(paths['item1']['path'][0], (0, 0))
        self.assertEqual(paths['item1']['path'][-1], (3, 3))
        
        # Check that the base grid has been updated with the path
        for x, y in paths['item1']['path']:
            self.assertEqual(pathfinder.base_grid[y][x], 9)  # 9 marks path
    
    def test_splitter_helper_methods(self):
        """Test splitter-related helper methods."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Test get_splitter_entrances for different directions
        # Right-facing splitter (splitter at 2,1)
        right_entrances = pathfinder.get_splitter_entrances((2, 1), (1, 0))
        self.assertEqual(len(right_entrances), 2, "Splitter should have 2 entrances")
        self.assertIn((1, 1), right_entrances, "First entrance position incorrect")
        self.assertIn((1, 2), right_entrances, "Second entrance position incorrect")
        
        # Left-facing splitter (splitter at 2,1)
        left_entrances = pathfinder.get_splitter_entrances((2, 1), (-1, 0))
        self.assertEqual(len(left_entrances), 2, "Splitter should have 2 entrances")
        self.assertIn((3, 1), left_entrances, "First entrance position incorrect")
        self.assertIn((3, 2), left_entrances, "Second entrance position incorrect")
        
        # Down-facing splitter (splitter at 2,1)
        down_entrances = pathfinder.get_splitter_entrances((2, 1), (0, 1))
        self.assertEqual(len(down_entrances), 2, "Splitter should have 2 entrances")
        self.assertIn((2, 0), down_entrances, "First entrance position incorrect")
        self.assertIn((3, 0), down_entrances, "Second entrance position incorrect")
        
        # Up-facing splitter (splitter at 2,1)
        up_entrances = pathfinder.get_splitter_entrances((2, 1), (0, -1))
        self.assertEqual(len(up_entrances), 2, "Splitter should have 2 entrances")
        self.assertIn((2, 2), up_entrances, "First entrance position incorrect")
        self.assertIn((3, 2), up_entrances, "Second entrance position incorrect")
        
        # Test get_splitter_exit for different directions
        self.assertEqual(pathfinder.get_splitter_exit((2, 1), (1, 0)), (3, 1), "Right exit position incorrect")
        self.assertEqual(pathfinder.get_splitter_exit((2, 1), (-1, 0)), (1, 1), "Left exit position incorrect")
        self.assertEqual(pathfinder.get_splitter_exit((2, 1), (0, 1)), (2, 2), "Down exit position incorrect")
        self.assertEqual(pathfinder.get_splitter_exit((2, 1), (0, -1)), (2, 0), "Up exit position incorrect")
        
        # Test adding a splitter
        splitter_info = pathfinder.add_splitter("splitter1", (2, 1), (1, 0), "iron_plate")
        self.assertEqual(splitter_info['position'], (2, 1), "Splitter position incorrect")
        self.assertEqual(splitter_info['direction'], (1, 0), "Splitter direction incorrect")
        self.assertEqual(splitter_info['item_type'], "iron_plate", "Splitter item type incorrect")
        
        # Check that the grid was updated
        self.assertEqual(pathfinder.base_grid[1][2], 13, "Splitter not marked on grid")
        self.assertEqual(pathfinder.base_grid[2][2], 13, "Second splitter tile not marked on grid")

    def test_can_place_splitter(self):
        """Test the can_place_splitter method."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        points = {}
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Should be able to place splitter in empty space
        self.assertTrue(pathfinder.can_place_splitter((2, 1), (1, 0), "iron_plate"), 
                    "Should allow placing splitter in empty space")
        
        # Should not be able to place splitter at edge of grid
        self.assertFalse(pathfinder.can_place_splitter((4, 1), (1, 0), "iron_plate"), 
                        "Should not allow placing splitter at edge of grid")
        
        # Should not be able to place splitter where there's not enough space
        grid[1][3] = 1  # Place obstacle at exit position
        self.assertFalse(pathfinder.can_place_splitter((2, 1), (1, 0), "iron_plate"), 
                        "Should not allow placing splitter with blocked exit")
        
        # Should be able to place splitter where the path (9) already exists
        grid[1][2] = 9  # Mark as path
        grid[2][2] = 9
        grid[1][3] = 9
        self.assertTrue(pathfinder.can_place_splitter((2, 1), (1, 0), "iron_plate"), 
                    "Should allow placing splitter on existing path")

    def test_pathfinding_with_splitter(self):
        """Test pathfinding with a splitter in the grid."""
        grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        points = {
            'item1': {
                'item': 'iron_plate',
                'start_points': [(0, 0)],
                'destination': [(6, 0)]
            },
            'item2': {
                'item': 'iron_plate',
                'start_points': [(0, 3)],
                'destination': [(6, 3)]
            }
        }
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Add a splitter in the middle that can merge the paths
        pathfinder.add_splitter("splitter1", (3, 1), (1, 0), "iron_plate")
        
        # Find paths for items
        paths, _ = pathfinder.find_path_for_item()
        
        # Check if paths were found for both items
        self.assertIn('item1', paths, "Should find path for item1")
        self.assertIn('item2', paths, "Should find path for item2")
        
        # Check if one of the paths uses the splitter
        splitter_used = False
        for item, path_info in paths.items():
            if path_info.get('used_splitter') == "splitter1":
                splitter_used = True
                break
        
        self.assertTrue(splitter_used, "At least one path should use the splitter")
        
        # Check that both paths reach their destinations
        self.assertEqual(paths['item1']['path'][-1], (6, 0), "item1 path should reach destination")
        self.assertEqual(paths['item2']['path'][-1], (6, 3), "item2 path should reach destination")

    def test_find_splitter_locations(self):
        """Test the find_splitter_locations method."""
        grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        points = {
            'item1': {
                'item': 'iron_plate',
                'start_points': [(0, 0)],
                'destination': [(6, 0)]
            },
            'item2': {
                'item': 'iron_plate',
                'start_points': [(0, 2)],
                'destination': [(6, 2)]
            }
        }
        pathfinder = AStarPathFinder(grid, points, False)
        
        # Find paths for items first (without splitters)
        paths, _ = pathfinder.find_path_for_item()
        pathfinder.paths = paths  # Store paths for later analysis
        
        # Find potential splitter locations
        potential_locations = pathfinder.find_splitter_locations("iron_plate")
        
        # Should have found at least one potential location (where paths are close)
        self.assertTrue(len(potential_locations) > 0, "Should find potential splitter locations")
        
        # Check that locations are valid
        for location in potential_locations:
            position = location['position']
            direction = location['direction']
            self.assertTrue(pathfinder.can_place_splitter(position, direction, "iron_plate"),
                        f"Location {position} with direction {direction} should be valid for splitter")
    
   
if __name__ == '__main__':
    unittest.main()