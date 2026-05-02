import os
import sys
import json
import base64
import tempfile
import subprocess
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_image_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.post("/execute")
async def execute_code(
    image: UploadFile = File(...),
    code: str = Form(...)
):
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded image as 'input.png' in the isolated temp directory
        image_path = os.path.join(temp_dir, "input.png")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # Save user Python code
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        # Execute user code. We use sys.executable to run with the same virtualenv.
        process = subprocess.run(
            [sys.executable, "script.py"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120  # increased timeout for potentially heavy algorithms
        )
        
        # Look for results.json convention
        results_json_path = os.path.join(temp_dir, "results.json")
        metrics = {}
        if os.path.exists(results_json_path):
            with open(results_json_path, "r", encoding="utf-8") as f:
                try:
                    metrics = json.load(f)
                except Exception as e:
                    metrics = {"json_parse_error": str(e)}
        
        # Scrape dynamically generated images
        output_images = []
        for filename in os.listdir(temp_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename != "input.png":
                filepath = os.path.join(temp_dir, filename)
                img_b64 = encode_image_base64(filepath)
                # assuming png output for base64 mime header
                mime_type = "image/jpeg" if filename.lower().endswith(('.jpg', '.jpeg')) else "image/png"
                output_images.append({
                    "name": filename,
                    "data": f"data:{mime_type};base64,{img_b64}"
                })
                
        return {
            "status": "success" if process.returncode == 0 else "error",
            "metrics": metrics,
            "images": output_images,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exit_code": process.returncode
        }
    except subprocess.TimeoutExpired as e:
        return {
            "status": "error",
            "error": "Execution timed out after 120 seconds. Is there an infinite loop?",
            "stdout": e.stdout.decode("utf-8") if e.stdout else "",
            "stderr": e.stderr.decode("utf-8") if e.stderr else ""
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        # Cleanup isolated temp dir immediately
        shutil.rmtree(temp_dir, ignore_errors=True)
