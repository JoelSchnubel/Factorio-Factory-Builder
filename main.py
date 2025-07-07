"""
Factorio Blueprint Generator - Main Application Module

This module provides a FastAPI web interface for generating Factorio blueprints
and managing complex factory production systems. It serves as the main entry point
for the application and handles web requests for:

- Single module production tree generation
- Multi-module factory building and optimization  
- Solver evaluation and performance analysis
- Configuration management through web interface

The application integrates multiple solver backends (Z3, Gurobi) and provides
both API endpoints and web interfaces for factory optimization.

Author: [Your Name]
Date: [Current Date]
Version: 1.0.0
"""

#! .venv\Scripts\python.exe

import os
import json
import sys
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

# Import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FactorioProductionTree import FactorioProductionTree, log_method_time
from FactoryBuilder import FactoryBuilder
import solver_eval
from logging_config import setup_logger

# Setup logger for main application
logger = setup_logger("MainApp")
logger.info("Starting Factorio Blueprint Generator application")

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Factorio Blueprint Generator",
    description="A web interface for generating Factorio blueprints and factories",
    version="1.0.0"
)
logger.info("FastAPI application initialized")

# Create required directories if they don't exist
templates_dir = Path('templates')
templates_dir.mkdir(exist_ok=True)
logger.info(f"Templates directory ready: {templates_dir}")

static_dir = Path('static')
static_dir.mkdir(exist_ok=True)
logger.info(f"Static files directory ready: {static_dir}")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
logger.info("Template engine and static file serving configured")

# Data models for API requests
class ProductionTreeRequest(BaseModel):
    """
    Request model for single production tree generation.
    
    Attributes:
        item (str): Item ID to produce (e.g., "electronic-circuit")
        amount (int): Target production amount per minute
        width (int): Factory grid width (default: 16)
        height (int): Factory grid height (default: 10)
    """
    item: str
    amount: int
    width: int = 16
    height: int = 10
    input_items: str = ""

class FactoryBuilderRequest(BaseModel):
    """
    Request model for factory building and optimization.
    
    Attributes:
        output_item (str): Item ID to produce (e.g., "iron-plate")
        amount (int): Target production amount per minute
        max_assembler_per_blueprint (int): Max assemblers per blueprint (default: 5)
        start_width (int): Initial factory grid width (default: 15)
        start_height (int): Initial factory grid height (default: 15)
    """
    output_item: str
    amount: int
    max_assembler_per_blueprint: int = 5
    start_width: int = 15
    start_height: int = 15

class SolverEvalRequest(BaseModel):
    """
    Request model for SMT solver evaluation.
    
    Attributes:
        solvers (List[str]): List of solver backends to use (default: ["z3"])
    """
    solvers: List[str] = ["z3"]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home page route.

    Displays the main interface for the Factorio Blueprint Generator.
    """
    return templates.TemplateResponse("home.html", {"request": request, "title": "Factorio Blueprint Generator"})

@app.get("/production-tree", response_class=HTMLResponse)
async def production_tree_page(request: Request):
    """
    Production tree page route.

    Displays the production tree generation interface.
    """
    return templates.TemplateResponse(
        "production_tree.html", 
        {"request": request, "title": "Production Tree Generator"}
    )

@app.post("/run-production-tree", response_class=HTMLResponse)
async def run_production_tree(
    request: Request, 
    item: str = Form(...), 
    amount: int = Form(...),
    width: int = Form(16),
    height: int = Form(10),
    input_items: str = Form("")
):
    """
    Run production tree generation.

    Processes the production tree generation request and returns the result.

    Args:
        request (Request): The HTTP request object.
        item (str): Item ID to produce.
        amount (int): Target production amount per minute.
        width (int): Factory grid width.
        height (int): Factory grid height.
        input_items (str): Comma-separated list of input items.

    Returns:
        HTMLResponse: The response containing the result of the operation.
    """
    try:
        solver_type = "z3"
        # Parse input items
        input_items_list = [item.strip() for item in input_items.split(",")] if input_items else []
        
        # Record start time
        start_time = time.time()
        
        # Create the production tree
        factory = FactorioProductionTree(width, height)
        factory.amount = amount
        production_data  = factory.calculate_production(item,amount,input_items=input_items_list)
        factory.production_data = production_data
        production_data = factory.set_capacities(production_data)


        factory.manual_Input()
        factory.manual_Output()

        factory.add_manual_IO_constraints(production_data,solver_type=solver_type)
    
        assembler_counts = factory.count_assemblers(production_data)
        
        execution_time=0

        start_time = time.perf_counter()
        factory.solve(production_data,solver_type=solver_type)
        end_time = time.perf_counter()
        log_method_time(item, 1, "solve", assembler_counts, start_time, end_time,solver_type)

        execution_time += round(end_time - start_time, 2)

        start_time = time.perf_counter()
        paths, placed_inserter_information = factory.build_belts(max_tries=10)
        end_time = time.perf_counter()
        log_method_time(item, 1, "build_belts", assembler_counts, start_time, end_time,solver_type)

        execution_time += round(end_time - start_time, 2)

        factory.store_data(f'Modules/{item}_{amount}_{input_items}_module',paths,placed_inserter_information)
    
        factory.place_power_poles(f'Modules/{item}_{amount}_{input_items}_module.json')
        
        factory.create_blueprint(f'Modules/{item}_{amount}_{input_items}_module.json',f'Blueprints/{item}_{amount}_{input_items}_module.txt')
        
        factory.visualize_factory(paths,placed_inserter_information,store=True,file_path=f'Modules/{item}_{amount}_{input_items}_module.png')
        
        
        
        
        return templates.TemplateResponse(
            "production_tree.html", 
            {
                "request": request, 
                "title": "Production Tree Generator",
                "result": "Blueprint created successfully!",
                "success": True,
                "blueprint_path": f'Modules/{item}_{amount}_{input_items}_module.png',
                "blueprint_string_path": f'Blueprints/{item}_{amount}_{input_items}_module.txt',
                "execution_time": execution_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error in production tree generation: {str(e)}")
        return templates.TemplateResponse(
            "production_tree.html", 
            {
                "request": request, 
                "title": "Production Tree Generator",
                "result": f"Error: {str(e)}",
                "success": False
            }
        )


@app.get("/factory-builder", response_class=HTMLResponse)
async def factory_builder_page(request: Request):
    """
    Factory builder page route.

    Displays the factory building and optimization interface.
    """
    return templates.TemplateResponse(
        "factory_builder.html", 
        {"request": request, "title": "Factory Builder"}
    )

@app.post("/run-factory-builder", response_class=HTMLResponse)
async def run_factory_builder(
    request: Request,
    output_item: str = Form(...),
    amount: int = Form(...),
    max_assembler_per_blueprint: int = Form(5),
    #start_width: int = Form(15),
    #start_height: int = Form(15)
):
    """
    Run factory building and optimization.

    Processes the factory building request and returns the result.

    Args:
        request (Request): The HTTP request object.
        output_item (str): Item ID to produce.
        amount (int): Target production amount per minute.
        max_assembler_per_blueprint (int): Max assemblers per blueprint.

    Returns:
        HTMLResponse: The response containing the result of the operation.
    """
    try:
        # Record start time
        
        
        # Create the factory builder
        builder = FactoryBuilder(
            output_item, 
            amount, 
            max_assembler_per_blueprint, 
            1, 
            1, 
            load_modules=True
        )
        
        # Split recipes
        builder.split_recipies()
        
        start_time = time.time() 
        # Solve the factory
        builder.solve_factory()
        
        # Calculate execution time
        execution_time = round(time.time() - start_time, 2)
        
        # Check if the factory image exists
        factory_path = f"Factorys/factory_{output_item}_{amount}.png"
        image_path = factory_path if os.path.exists(factory_path) else None
        
        return templates.TemplateResponse(
            "factory_builder.html", 
            {
                "request": request, 
                "title": "Factory Builder",
                "result": "Factory created successfully!",
                "success": True,
                "factory_path": image_path,
                "execution_time": execution_time
            }
        )
    except Exception as e:
        logger.error(f"Error in factory building: {str(e)}")
        return templates.TemplateResponse(
            "factory_builder.html", 
            {
                "request": request, 
                "title": "Factory Builder",
                "result": f"Error: {str(e)}",
                "success": False
            }
        )

@app.get("/solver-eval", response_class=HTMLResponse)
async def solver_eval_page(request: Request):
    """
    Solver evaluation page route.

    Displays the SMT solver evaluation interface.
    """
    return templates.TemplateResponse(
        "solver_eval.html", 
        {"request": request, "title": "SMT Solver Evaluation"}
    )

@app.post("/run-solver-eval", response_class=HTMLResponse)
async def run_solver_eval(request: Request, solvers: List[str] = Form(None)):
    """
    Run SMT solver evaluation.

    Processes the solver evaluation request and returns the result.

    Args:
        request (Request): The HTTP request object.
        solvers (List[str]): List of solver backends to use.

    Returns:
        HTMLResponse: The response containing the result of the operation.
    """
    if not solvers:
        solvers = ["z3"]  # Default to z3 if none selected
        
    try:
        # Temporarily set the SOLVERS in solver_eval to match user selection
        original_solvers = solver_eval.SOLVERS
        solver_eval.SOLVERS = solvers
        
        # Capture output using string IO
        import io
        import sys
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        # Run the evaluation
        solver_eval.evaluate_solvers()
        
        # Get the output
        output = new_stdout.getvalue()
        
        # Reset stdout
        sys.stdout = old_stdout
        
        # Reset solvers
        solver_eval.SOLVERS = original_solvers
        
        return templates.TemplateResponse(
            "solver_eval.html", 
            {
                "request": request, 
                "title": "SMT Solver Evaluation",
                "result": output
            }
        )
    except Exception as e:
        logger.error(f"Error in solver evaluation: {str(e)}")
        return templates.TemplateResponse(
            "solver_eval.html", 
            {
                "request": request, 
                "title": "SMT Solver Evaluation",
                "result": f"Error: {str(e)}"
            }
        )

@app.get("/json-editor", response_class=HTMLResponse)
async def json_editor_page(request: Request):
    """
    JSON editor page route.

    Displays the JSON configuration editor interface.
    """
    return templates.TemplateResponse(
        "json_editor.html", 
        {"request": request, "title": "JSON Editor"}
    )

@app.get("/api/json/{file_name}")
async def get_json_file(file_name: str):
    """
    API endpoint to get a JSON configuration file.

    Args:
        file_name (str): The name of the file to retrieve.

    Returns:
        JSONResponse: The requested JSON file content.
    """
    allowed_files = ["machine_data.json", "recipes.json", "config.json"]
    if file_name not in allowed_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(file_name, 'r') as file:
            # Parse and then re-serialize with proper formatting
            data = json.load(file)
            # Return formatted JSON as text response for better editor handling
            return JSONResponse(
                content=data,
                headers={"Content-Type": "application/json"},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/json/{file_name}", response_class=JSONResponse)
async def save_json_file(file_name: str, request: Request):
    """
    API endpoint to save a JSON configuration file.

    Args:
        file_name (str): The name of the file to save.
        request (Request): The HTTP request object containing the JSON body.

    Returns:
        JSONResponse: The result of the save operation.
    """
    allowed_files = ["machine_data.json", "recipes.json", "config.json"]
    if file_name not in allowed_files:
        return JSONResponse({"success": False, "message": "File not allowed"})
    
    try:
        # Get raw request body
        content = await request.body()
        content_str = content.decode('utf-8')
        
        # Validate JSON
        json.loads(content_str)
        
        # Create backup
        backup_name = f"{file_name}.bak"
        if os.path.exists(file_name):
            with open(backup_name, 'w') as backup_file:
                with open(file_name, 'r') as original_file:
                    backup_file.write(original_file.read())
        
        # Write new content
        with open(file_name, 'w') as file:
            file.write(content_str)
        
        return JSONResponse({"success": True})
    except json.JSONDecodeError as e:
        return JSONResponse({"success": False, "message": f"Invalid JSON: {str(e)}"})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    # Create necessary directories
    create_directory_if_not_exists('Modules')
    create_directory_if_not_exists('Blueprints')
    create_directory_if_not_exists('Factorys')
    
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
