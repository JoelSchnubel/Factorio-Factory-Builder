# Factorio Factory Layout Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688.svg)](https://fastapi.tiangolo.com/)

A comprehensive framework for optimizing Factorio factory layouts using SMT solvers and advanced pathfinding algorithms. This project applies formal methods to solve the complex problem of factory design automation, achieving up to 97x performance improvements over previous approaches.

## 🎯 Overview

This project tackles the challenging problem of automated factory layout optimization in Factorio, with applications extending to real-world logistics and manufacturing systems. The solution combines:

- **SMT Solver Integration**: Uses Z3, CVC5, and Yices for constraint satisfaction
- **Advanced Pathfinding**: Multi-agent pathfinding for belt and pipe routing
- **Interactive Web Interface**: FastAPI-based web application for easy use
- **Blueprint Generation**: Automatic Factorio blueprint creation
- **Performance Analysis**: Comprehensive solver evaluation and benchmarking

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Functionality

- **Production Tree Calculation**: Recursive analysis of recipe dependencies
- **Layout Optimization**: SMT-based placement optimization for assemblers and infrastructure
- **Multi-Solver Support**: Z3, CVC5, and Yices solver backends
- **Pathfinding**: Advanced A\* pathfinding for belt and pipe networks
- **Blueprint Export**: Direct integration with Factorio blueprint format

### Advanced Features

- **Interactive GUI**: Manual input/output point configuration
- **Multi-Agent Pathfinding**: Efficient routing for complex factory layouts
- **Power Pole Optimization**: Automatic electrical infrastructure placement
- **Fluid Handling**: Support for both belt-based and pipe-based transport
- **Performance Benchmarking**: Comprehensive solver evaluation framework

### Web Interface

- **Production Tree Generator**: Single module optimization
- **Factory Builder**: Multi-module factory construction
- **Solver Evaluation**: Performance comparison tools
- **JSON Editor**: Configuration file management

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Required SMT Solvers

- **Z3**: Included with python package
- **CVC5**: Download from [CVC5 releases](https://github.com/cvc5/cvc5/releases)
- **Yices**: Download from [Yices website](https://yices.csl.sri.com/)

### Setup

1. **Clone the repository**:

```bash
git clone https://github.com/JoelSchnubel/Bachelor-Thesis.git
cd Bachelor-Thesis
```

2. **Create and activate virtual environment**:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Configure solver paths** in `solver_eval.py`:

```python
CVC5_PATH = "path/to/cvc5.exe"
YICES_PATH = "path/to/yices-smt2.exe"
```

5. **Run the application**:

```bash
python main.py
```

The web interface will be available at `http://localhost:8000`

## 🎮 Usage

### Web Interface

1. **Access the application**: Open `http://localhost:8000` in your browser
2. **Production Tree Generator**: Create single module blueprints
3. **Factory Builder**: Build complex multi-module factories
4. **Solver Evaluation**: Compare solver performance

### Command Line Interface

```python
from FactorioProductionTree import FactorioProductionTree

# Create a factory for producing electronic circuits
factory = FactorioProductionTree(width=20, height=15)
production_data = factory.calculate_production("electronic-circuit", 60)

# Set up manual I/O points
factory.manual_Input()
factory.manual_Output()

# Optimize layout
factory.solve(production_data, solver_type="z3")

# Generate blueprint
paths, inserters = factory.build_belts()
factory.create_blueprint("output.json", "blueprint.txt")
```

### Solver Evaluation

```python
import solver_eval

# Run comprehensive solver evaluation
results = solver_eval.evaluate_solvers()
solver_eval.save_results_to_csv(results)
```

## 🏗️ Architecture

### Core Components

```
├── FactorioProductionTree.py     # Main production planning and optimization
├── SMT_Solver.py                 # Z3 SMT solver integration
├── MultiAgentPathfinder.py       # A* pathfinding for belt routing
├── FactoryBuilder.py             # Multi-module factory construction
├── solver_eval.py                # Solver evaluation framework
└── main.py                       # FastAPI web interface
```

### Key Classes

- **FactorioProductionTree**: Main factory planning and optimization
- **SMTSolver**: Z3-based layout optimization
- **MultiAgentPathfinder**: Pathfinding for material transport
- **FactoryBuilder**: Large-scale factory construction

## 📊 Performance

Our optimization approach achieves significant performance improvements:

- **Up to 97x faster** than previous constraint-based approaches
- **Scalable to large factories** with hundreds of assemblers
- **Efficient pathfinding** for complex belt networks
- **Multi-solver support** for optimal performance

### Benchmarking Results

The framework includes comprehensive benchmarking capabilities:

```bash
python solver_eval.py
```

Results are saved to `results/solver_evaluation.csv` with detailed performance metrics.

## 🔧 Configuration

### Main Configuration (`config.json`)

```json
{
  "grid": {
    "default_width": 16,
    "default_height": 10
  },
  "machines": {
    "default_assembler": "assembling-machine-2",
    "default_furnace": "electric-furnace"
  },
  "pathfinding": {
    "allow_underground": true,
    "allow_splitters": true,
    "max_tries": 3
  }
}
```

### Recipe Data (`recipes.json`)

Contains all Factorio recipes with ingredients, crafting times, and machine requirements.

Custome recipes can be added to the file.

### Machine Data (`machine_data.json`)

Defines machine capabilities, crafting speeds, and transport specifications.

Can be adapted to include new machines with specialized I/O points.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Remark 

If you encounter an error while creating a module, it's most likely due to a missing asset for the item or machine you're trying to use. Only a selection of item assets and the most essential machine, inserter, and belt assets are preloaded.

To add new assets, simply copy the image from the Factorio Wiki and place it in the assets folder. Make sure all filenames are in lowercase and use hyphens `-` instead of underscores `_`.


## 🎓 Academic Usage

This project is part of a Bachelor's thesis: "Optimization of Factorio Factory Layouts: An SMT-Solver Approach" by Joel Schnubel at the University of Saarland.

### Citation

If you use this software in your research, please cite using GitHubs cite feature.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **University of Saarland** for academic support
- **Factorio community** for inspiration and game mechanics
- **SMT solver developers** (Z3, CVC5, Yices teams)
- **Open source contributors** for foundational libraries


