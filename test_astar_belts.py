import os
import json
import logging
import pygame
from FactorioProductionTree import FactorioProductionTree
from AStarPathFinder import AStarPathFinder

def test_rebuild_belts(module_file):
    """
    Test the belt rebuilding functionality with the newer AStarPathFinder
    using data loaded directly from a saved module
    
    Args:
        module_file: Path to a stored production tree module JSON file
    """
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("belt_rebuild_test.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Create a FactorioProductionTree instance
    factorio_tree = FactorioProductionTree()
    
    # Load the saved production tree data
    logging.info(f"Loading production tree from {module_file}")
    factorio_tree.load_data(module_file)
    
    # Create retrieval points from loaded data
    retrieval_points = factorio_tree.retrieval_points
    
    # Create obstacle map from loaded data
    obstacle_map = factorio_tree.obstacle_map
    
    logging.debug(f"Retrieval points: {retrieval_points}")
    logging.debug(f"Obstacle map: {obstacle_map}")
    
    obstacle_map = [
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            99,
            44,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            99,
            44,
            33,
            33,
            33,
            44,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            33,
            33,
            33,
            44,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            99,
            44,
            33,
            33,
            33,
            44,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            33,
            33,
            33,
            0,
            0,
            0,
            44,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            99,
            44,
            33,
            33,
            33,
            0,
            0,
            9,
            9,
            12,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            0,
            0,
            44,
            12,
            0,
            9,
            9,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            0,
            33,
            33,
            33,
            44,
            0,
            9,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            0,
            33,
            33,
            33,
            0,
            9,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            0,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            0,
            44,
            44,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            99,
            44,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            99,
            44,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            1,
            0,
            33,
            33,
            33,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]
    ]
    
    retrieval_points = {
        "electronic-circuit_0": {
            "item": "electronic-circuit",
            "destination": [
                [
                    14,
                    1
                ],
                [
                    14,
                    2
                ],
                [
                    14,
                    3
                ],
                [
                    14,
                    4
                ],
                [
                    14,
                    5
                ],
                [
                    14,
                    6
                ],
                [
                    14,
                    7
                ],
                [
                    14,
                    8
                ],
                [
                    14,
                    9
                ],
                [
                    14,
                    10
                ],
                [
                    14,
                    11
                ],
                [
                    14,
                    12
                ],
                [
                    14,
                    13
                ],
                [
                    14,
                    0
                ],
                [
                    14,
                    14
                ],
                [
                    7,
                    9
                ],
                [
                    8,
                    9
                ],
                [
                    9,
                    9
                ],
                [
                    10,
                    9
                ],
                [
                    13,
                    9
                ],
                [
                    14,
                    9
                ],
                [
                    7,
                    6
                ],
                [
                    7,
                    7
                ],
                [
                    7,
                    9
                ]
            ],
            "start_points": [
                [
                    5,
                    6
                ],
                [
                    5,
                    12
                ],
                [
                    1,
                    8
                ],
                [
                    1,
                    9
                ],
                [
                    1,
                    10
                ],
                [
                    7,
                    9
                ],
                [
                    7,
                    10
                ]
            ],
            "inserter_mapping": {
                "(5, 6)": [
                    5,
                    7
                ],
                "(5, 12)": [
                    5,
                    11
                ],
                "(1, 8)": [
                    2,
                    8
                ],
                "(1, 9)": [
                    2,
                    9
                ],
                "(1, 10)": [
                    2,
                    10
                ],
                "(7, 9)": [
                    6,
                    9
                ],
                "(7, 10)": [
                    6,
                    10
                ]
            }
        },
        "electronic-circuit_1": {
            "item": "electronic-circuit",
            "destination": [
                [
                    14,
                    1
                ],
                [
                    14,
                    2
                ],
                [
                    14,
                    3
                ],
                [
                    14,
                    4
                ],
                [
                    14,
                    5
                ],
                [
                    14,
                    6
                ],
                [
                    14,
                    7
                ],
                [
                    14,
                    8
                ],
                [
                    14,
                    9
                ],
                [
                    14,
                    10
                ],
                [
                    14,
                    11
                ],
                [
                    14,
                    12
                ],
                [
                    14,
                    13
                ],
                [
                    14,
                    0
                ],
                [
                    14,
                    14
                ],
                [
                    7,
                    9
                ],
                [
                    8,
                    9
                ],
                [
                    9,
                    9
                ],
                [
                    10,
                    9
                ],
                [
                    13,
                    9
                ],
                [
                    14,
                    9
                ],
                [
                    7,
                    6
                ],
                [
                    7,
                    7
                ],
                [
                    7,
                    9
                ]
            ],
            "start_points": [
                [
                    6,
                    0
                ],
                [
                    7,
                    0
                ],
                [
                    8,
                    0
                ],
                [
                    6,
                    6
                ],
                [
                    7,
                    6
                ],
                [
                    10,
                    2
                ],
                [
                    10,
                    3
                ],
                [
                    10,
                    4
                ]
            ],
            "inserter_mapping": {
                "(6, 0)": [
                    6,
                    1
                ],
                "(7, 0)": [
                    7,
                    1
                ],
                "(8, 0)": [
                    8,
                    1
                ],
                "(6, 6)": [
                    6,
                    5
                ],
                "(7, 6)": [
                    7,
                    5
                ],
                "(10, 2)": [
                    9,
                    2
                ],
                "(10, 3)": [
                    9,
                    3
                ],
                "(10, 4)": [
                    9,
                    4
                ]
            }
        },
        "iron-plate_0": {
            "item": "iron-plate",
            "destination": [
                [
                    11,
                    14
                ],
                [
                    11,
                    0
                ],
                [
                    11,
                    1
                ],
                [
                    11,
                    2
                ],
                [
                    11,
                    3
                ],
                [
                    11,
                    4
                ],
                [
                    11,
                    5
                ],
                [
                    11,
                    6
                ],
                [
                    11,
                    7
                ],
                [
                    11,
                    8
                ],
                [
                    11,
                    9
                ],
                [
                    11,
                    10
                ],
                [
                    11,
                    11
                ],
                [
                    11,
                    12
                ],
                [
                    11,
                    13
                ],
                [
                    8,
                    6
                ],
                [
                    8,
                    7
                ],
                [
                    8,
                    8
                ]
            ],
            "start_points": [
                [
                    7,
                    8
                ]
            ],
            "inserter_mapping": None
        },
        "iron-plate_1": {
            "item": "iron-plate",
            "destination": [
                [
                    11,
                    14
                ],
                [
                    11,
                    0
                ],
                [
                    11,
                    1
                ],
                [
                    11,
                    2
                ],
                [
                    11,
                    3
                ],
                [
                    11,
                    4
                ],
                [
                    11,
                    5
                ],
                [
                    11,
                    6
                ],
                [
                    11,
                    7
                ],
                [
                    11,
                    8
                ],
                [
                    11,
                    9
                ],
                [
                    11,
                    10
                ],
                [
                    11,
                    11
                ],
                [
                    11,
                    12
                ],
                [
                    11,
                    13
                ],
                [
                    7,
                    8
                ],
                [
                    8,
                    8
                ],
                [
                    9,
                    8
                ],
                [
                    10,
                    8
                ],
                [
                    11,
                    8
                ]
            ],
            "start_points": [
                [
                    8,
                    6
                ]
            ],
            "inserter_mapping": None
        }
    }
    
    
    # Create AStarPathFinder with improved underground belt handling
    astar_pathfinder = AStarPathFinder(
        obstacle_map, 
        retrieval_points,
        underground_length=3,  # Typical underground belt length
        allow_jump=True
    )
    
    paths, placed_inserter_info = astar_pathfinder.find_path_for_item()
    
    
    
    # Save the rebuilt data
    rebuilt_file = f"rebuilt_{os.path.basename(module_file)}"
    with open(rebuilt_file, "w") as f:
        json.dump({
            "paths": paths,
            "placed_inserter_information": placed_inserter_info
        }, f, indent=2)
    
    logging.info(f"Rebuilt production tree saved to {rebuilt_file}")
    
    return paths, placed_inserter_info

if __name__ == "__main__":
    # Check for existing modules in the Modules directory
    module_dir = "Modules"
    if not os.path.exists(module_dir):
        print(f"Error: {module_dir} directory not found")
        exit(1)
    
    module_files = [f for f in os.listdir(module_dir) if f.endswith('.json')]
    
    if not module_files:
        print(f"Error: No module files found in {module_dir}")
        exit(1)
    
    # Print available modules
    print("Available modules:")
    for i, module in enumerate(module_files):
        print(f"{i+1}. {module}")
    
    # Let user select a module
    selection = input("\nEnter the number of the module to test (or press Enter for the first one): ")
    
    try:
        if selection.strip() == "":
            selected_index = 0
        else:
            selected_index = int(selection) - 1
            
        if selected_index < 0 or selected_index >= len(module_files):
            raise ValueError("Invalid selection")
            
        selected_module = os.path.join(module_dir, module_files[selected_index])
        print(f"\nTesting with module: {selected_module}")
        
        # Test the belt rebuilding
        paths, inserters = test_rebuild_belts(selected_module)
        
        # Display some stats about the rebuilt paths
        print("\nRebuild Stats:")
        print(f"Total paths: {len(paths)}")
        print(f"Total new inserters: {len(inserters)}")
        
        # Show paths with underground belts
        underground_paths = 0
        for item, path_data in paths.items():
            if 'underground_paths' in path_data and path_data['underground_paths']:
                underground_paths += 1
        
        print(f"Paths with underground belts: {underground_paths}")
        
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        exit(1)