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

# Setup logger
logger = setup_logger("MainApp")

# Initialize FastAPI app
app = FastAPI(
    title="Factorio Blueprint Generator",
    description="A web interface for generating Factorio blueprints and factories",
    version="1.0.0"
)

# Create a templates directory if it doesn't exist
templates_dir = Path('templates')
templates_dir.mkdir(exist_ok=True)

# Create a static directory if it doesn't exist
static_dir = Path('static')
static_dir.mkdir(exist_ok=True)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data models
class ProductionTreeRequest(BaseModel):
    item: str
    amount: int
    width: int = 16
    height: int = 10
    input_items: str = ""

class FactoryBuilderRequest(BaseModel):
    output_item: str
    amount: int
    max_assembler_per_blueprint: int = 5
    start_width: int = 15
    start_height: int = 15

class SolverEvalRequest(BaseModel):
    solvers: List[str] = ["z3"]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "title": "Factorio Blueprint Generator"})

@app.get("/production-tree", response_class=HTMLResponse)
async def production_tree_page(request: Request):
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
    return templates.TemplateResponse(
        "solver_eval.html", 
        {"request": request, "title": "SMT Solver Evaluation"}
    )

@app.post("/run-solver-eval", response_class=HTMLResponse)
async def run_solver_eval(request: Request, solvers: List[str] = Form(None)):
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
    return templates.TemplateResponse(
        "json_editor.html", 
        {"request": request, "title": "JSON Editor"}
    )

@app.get("/api/json/{file_name}")
async def get_json_file(file_name: str):
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
