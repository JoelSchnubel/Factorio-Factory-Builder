import unittest
from MultiAgentPathfinder import MultiAgentPathfinder , Splitter


class TestMultiAgentPathfinder(unittest.TestCase):

    def test_simple(self):
        # Example usage:
        return
        grid = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 33, 33, 33, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 33, 33, 33, 0, 0, 1, 0, 0, 22],
            [1, 0, 33, 33, 33, 44, 33, 33, 33, 0, 0, 1, 0, 0, 22],
            [1, 0, 33, 33, 33, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [99, 44, 33, 33, 33, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22]
        ]

        points = {
            'electronic-circuit_0': {
                'item': 'electronic-circuit',
                'destination': [(14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8),
                                (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 0)],
                'start_points': [(6, 2), (7, 2), (6, 8), (7, 8), (8, 8), (4, 4), (4, 5), (10, 4), (10, 5), (10, 6)],
                'inserter_mapping': {
                    '(6, 2)': (6, 3), '(7, 2)': (7, 3), '(6, 8)': (6, 7), '(7, 8)': (7, 7), 
                    '(8, 8)': (8, 7), '(4, 4)': (5, 4), '(4, 5)': (5, 5), '(10, 4)': (9, 4), 
                    '(10, 5)': (9, 5), '(10, 6)': (9, 6)
                }
            },
            'iron-plate_0': {
                'item': 'iron-plate',
                'destination': [(11, 0), (11, 14), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
                                (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13)],
                'start_points': [(8, 2)],
                'inserter_mapping': None
            }
        }

        # Create the pathfinder
        pathfinder = MultiAgentPathfinder(grid, points,allow_underground=True,underground_length=3)

        # Find paths for all items
        paths, inserters = pathfinder.find_paths_for_all_items()
        
        print(f"paths: {paths}")
        print(f"inserters: {inserters}")

        # Visualize results
        pathfinder.visualize_grid('final_grid.png')
        pathfinder.visualize_paths('path_{}.png')
        
    def test_underground(self):
        return
        grid = [[0,0,0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,1,0,0,0,0]]
        
        points = {
            'iron-plate_0': {
                'item': 'iron-plate',
                'destination': [(11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
                                (11, 6)],
                'start_points': [(2, 5)],
                'inserter_mapping': None
            }
        }
         
        # Create the pathfinder
        pathfinder = MultiAgentPathfinder(grid, points,allow_underground=True,underground_length=3)

        # Find paths for all items
        paths, inserters = pathfinder.find_paths_for_all_items()
        
        #print(f"paths: {paths}")
        #print(f"inserters: {inserters}")

        # Visualize results
        #pathfinder.visualize_grid('final_grid.png')
        #pathfinder.visualize_paths('path_{}.png')

        
    def test_merging(self):
        return
        grid = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 33, 33, 33, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 33, 33, 33, 0, 0, 1, 0, 0, 22],
            [1, 0, 33, 33, 33, 44, 33, 33, 33, 0, 0, 1, 0, 0, 22],
            [1, 0, 33, 33, 33, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [99, 44, 33, 33, 33, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22]
        ]

        points = {
            'iron-plate_0': {
                'item': 'iron-plate',
                'destination': [(11, 0), (11, 14), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
                                (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13)],
                'start_points': [(2, 10)],
                'inserter_mapping': None
            },
              'iron-plate_1': {
                'item': 'iron-plate',
                'destination': [(11, 0), (11, 14), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
                                (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13)],
                'start_points': [(2, 11)],
                'inserter_mapping': None
            }
        }
        
         # Create the pathfinder
        pathfinder = MultiAgentPathfinder(grid, points,allow_underground=True,underground_length=3)

        # Find paths for all items
        paths, inserters = pathfinder.find_paths_for_all_items()
        
        #print(f"paths: {paths}")
        #print(f"inserters: {inserters}")

        # Visualize results
        #pathfinder.visualize_grid('final_grid.png')
        #pathfinder.visualize_paths('path_{}.png')


    def test_splitter(self):
    
        input_info = {'copper-plate': {'input': (14, 0), 'output': (2, 0), 'paths': {'copper-plate': [{'path': [(0, 14), (0, 13), (0, 12), (0, 11), (0, 10), (0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2)], 'start': (0, 14), 'destination': (0, 2), 'underground_segments': {}, 'uses_splitter_start': False, 'uses_splitter_dest': False}]}}, 'iron-plate': {'input': (13, 7), 'output': (2, 7), 'paths': {'iron-plate': [{'path': [(7, 13), (7, 12), (7, 11), (7, 10), (7, 9), (7, 8), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2)], 'start': (7, 13), 'destination': (7, 2), 'underground_segments': {}, 'uses_splitter_start': False, 'uses_splitter_dest': False}]}}}
        output_info= {'electronic-circuit': {'input': (0, 14), 'output': (13, 14), 'paths': {'electronic-circuit': [{'path': [(14, 0), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13)], 'start': (14, 0), 'destination': (14, 13), 'underground_segments': {}, 'uses_splitter_start': False, 'uses_splitter_dest': False}]}}}
    
        splitters = self.prepare_splitter_information(input_info, output_info)

        
        grid = [[1,0,0,0,0,0,1,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,1,0,0,1],
                [1,0,0,0,0,0,1,0,1,0,0,1],
                [1,0,0,0,0,0,1,0,1,0,0,1],
                [1,0,0,0,0,0,1,0,1,0,0,1],
                [1,0,0,0,0,0,1,0,1,0,0,1]]
        
        points = {
            'iron-plate_0': {
                'item': 'iron-plate',
                'destination': [(11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
                                (11, 6)],
                'start_points': [(0, 3)],
                'inserter_mapping': None
            }
        }
         
        # Create the pathfinder
        pathfinder = MultiAgentPathfinder(grid, 
                                          points,
                                          allow_underground=True,
                                          underground_length=3,
                                          allow_splitters=True,
                                          splitters=splitters)

        # Find paths for all items
        paths, inserters = pathfinder.find_paths_for_all_items()
        
        #print(f"paths: {paths}")
        #print(f"inserters: {inserters}")

        # Visualize results
        #pathfinder.visualize_grid('final_grid.png')
        #pathfinder.visualize_paths('path_{}.png')
        
    def prepare_splitter_information(self, input_information, output_information):
        """
        Prepare splitter information and add orientation data to paths.
        Creates a splitter object for every point in the path.
        
        Returns:
            dict: Dictionary mapping each item to a list of Splitter objects
        """
        splitters = {}
        
        # Process input information
        for item, data in input_information.items():
            # Skip if no paths are available
            if data['paths'] is None or item not in data['paths']:
                continue
                
            # Initialize splitter list for this item if needed
            if item not in splitters:
                splitters[item] = []
            
            # Add orientation information to each path
            for path_data in data['paths'][item]:
                path = path_data['path']
                # Initialize or reset orientation data
                path_data['orientation'] = {}
                
                # Calculate orientation for each segment of the path
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    
                    # Calculate direction vector
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    # Store orientation
                    path_data['orientation'][current] = (dx, dy)
                
                # For the last point, use the same direction as the previous segment
                if len(path) > 1:
                    last = path[-1]
                    second_last = path[-2]
                    dx = last[0] - second_last[0]
                    dy = last[1] - second_last[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    path_data['orientation'][last] = (dx, dy)
                
                # Create a Splitter object for EVERY point in the path
                for point in path:
                    # Get the orientation for this point
                    orientation = path_data['orientation'].get(point, (0, 0))  # Default to (0, 0) if not found
                    
                    # Add as potential splitter using the Splitter class
                    splitters[item].append(Splitter(
                        item=item,
                        position=point,
                        direction=orientation
                    ))
        
        # Process output information using the same approach
        for item, data in output_information.items():
            # Skip if no paths are available
            if data['paths'] is None or item not in data['paths']:
                continue
                
            # Initialize splitter list for this item if needed
            if item not in splitters:
                splitters[item] = []
            
            # Add orientation information to each path
            for path_data in data['paths'][item]:
                path = path_data['path']
                # Initialize or reset orientation data
                path_data['orientation'] = {}
                
                # Calculate orientation for each segment of the path
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    
                    # Calculate direction vector
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    # Store orientation
                    path_data['orientation'][current] = (dx, dy)
                
                # For the last point, use the same direction as the previous segment
                if len(path) > 1:
                    last = path[-1]
                    second_last = path[-2]
                    dx = last[0] - second_last[0]
                    dy = last[1] - second_last[1]
                    
                    # Normalize
                    if dx != 0:
                        dx = dx // abs(dx)
                    if dy != 0:
                        dy = dy // abs(dy)
                    
                    path_data['orientation'][last] = (dx, dy)
                
                # Create a Splitter object for EVERY point in the path
                for point in path:
                    # Get the orientation for this point
                    orientation = path_data['orientation'].get(point, (0, 0))  # Default to (0, 0) if not found
                    
                    # Add as potential splitter using the Splitter class
                    splitters[item].append(Splitter(
                        item=item,
                        position=point,
                        direction=orientation
                    ))
        
        return splitters

    def test_orientation_info(self):
        """Test that orientation information is properly added to path data and splitters are created."""
        input_info = {
            'copper-plate': {
                'input': (14, 0), 
                'output': (2, 0), 
                'paths': {
                    'copper-plate': [{
                        'path': [(0, 14), (0, 13), (0, 12), (0, 11), (0, 10), (0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2)],
                        'start': (0, 14),
                        'destination': (0, 2),
                        'underground_segments': {},
                        'uses_splitter_start': False,
                        'uses_splitter_dest': False
                    }]
                }
            }
        }
        
        # Process the input info to add orientation data
        splitters = self.prepare_splitter_information(input_info, {})
        
        # Check that splitters dictionary contains the expected item
        self.assertIn('copper-plate', splitters, "Splitters should be generated for copper-plate")
        
        # Check that there are 13 splitter objects for the item (one for each path point)
        path_length = len(input_info['copper-plate']['paths']['copper-plate'][0]['path'])
        self.assertEqual(len(splitters['copper-plate']), path_length, 
                        f"Should have {path_length} splitter positions for copper-plate (one for each path point)")
        
        # Check that they are Splitter objects
        for splitter in splitters['copper-plate']:
            self.assertIsInstance(splitter, Splitter, "Should be a Splitter object")
        
        # Print all splitters for inspection
        print(f"Created {len(splitters['copper-plate'])} splitters for copper-plate")
        for i, splitter in enumerate(splitters['copper-plate']):
            print(f"Splitter {i}: pos={splitter.position}, dir={splitter.direction}")
        
      
       

if __name__ == '__main__':
    unittest.main()