# server.py

import asyncio
import os
import uuid
import traceback

from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from starlette.responses import FileResponse

# Import your existing pipeline code.
# Adjust imports as needed to point to your actual pipeline's location.
from src.main import main as run_pipeline  # This is hypothetical; adjust to how your pipeline is actually invoked.

app = FastAPI()

# Configure CORS so that your React dev server (e.g. http://localhost:3000) can call these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set this to your specific frontend URL, e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We'll store progress and results in global state for simplicity.
# In a production system, you'd likely store these in a database or distributed cache.
class PipelineStatus:
    def __init__(self):
        self.stage: str = "idle"            # "idle", "running", "done", "error"
        self.message: Optional[str] = None  # Additional info
        self.progress: float = 0.0          # Percentage or fraction of progress
        self.error: Optional[str] = None    # Error text if encountered
        self.output_file: Optional[str] = None  # Path to pipeline result if applicable

pipeline_status = PipelineStatus()

# We define a background task that runs the pipeline:
async def pipeline_task():
    # Update global pipeline_status as we proceed
    try:
        pipeline_status.stage = "running"
        pipeline_status.progress = 0.0
        pipeline_status.message = "Starting pipeline..."
        pipeline_status.error = None

        # ---------------------------------------------------------------------
        # Example of how you might run your pipeline in steps and update progress:
        # If your `src/main.py` just has a `main()` function that does everything,
        # you'll simply call `run_pipeline()` and then set progress to 100%. 
        # This sample is purely illustrative of stage-by-stage updates.
        # ---------------------------------------------------------------------

        # 1) Start the pipeline
        pipeline_status.message = "Pipeline in progress..."
        # For demonstration, we'll pretend we have multiple steps:

        # (A) Step 1
        pipeline_status.progress = 0.1
        await asyncio.sleep(2)  # Simulate some async delay
        # Possibly call some partial pipeline function or start your real pipeline

        # (B) Step 2
        pipeline_status.progress = 0.5
        pipeline_status.message = "Processing..."
        await asyncio.sleep(2)  # Simulate more time

        # Actually call your pipeline
        # This is blocking if your pipeline is synchronous. 
        # For demonstration, let's do a short sleep instead.
        # If your pipeline is sync, run it in a thread via `asyncio.to_thread(run_pipeline)`.
        # If your pipeline is async, just `await run_pipeline()`.
        pipeline_status.message = "Running src/main.py pipeline now..."
        await asyncio.sleep(2)

        # Example: run it in a thread if it's synchronous
        await asyncio.to_thread(run_pipeline)
        # Or if you can do: await run_pipeline() if it's truly async

        # (C) Step 3
        pipeline_status.progress = 0.9
        pipeline_status.message = "Finishing..."
        await asyncio.sleep(1)

        # Let's pretend the pipeline wrote a JSON file to data/output/final_results.json
        # We'll store that path in pipeline_status for future download
        output_file_path = "data/output/final_results.json"  # Adjust to your actual output
        pipeline_status.output_file = output_file_path

        # Mark success
        pipeline_status.progress = 1.0
        pipeline_status.stage = "done"
        pipeline_status.message = "Pipeline completed successfully."
        pipeline_status.error = None

    except Exception as ex:
        pipeline_status.stage = "error"
        pipeline_status.error = f"{type(ex).__name__}: {str(ex)}"
        pipeline_status.message = traceback.format_exc()

# --------------------------------------------------------------------------------
#  Endpoint to start the pipeline
# --------------------------------------------------------------------------------
@app.post("/start-analysis")
async def start_analysis(background_tasks: BackgroundTasks):
    if pipeline_status.stage == "running":
        return {"detail": "Analysis is already running. Please wait for it to finish."}
    # Reset status
    pipeline_status.stage = "idle"
    pipeline_status.message = ""
    pipeline_status.progress = 0.0
    pipeline_status.error = None
    pipeline_status.output_file = None

    # Kick off background task
    background_tasks.add_task(pipeline_task)
    return {"detail": "Analysis started."}

# --------------------------------------------------------------------------------
#  Endpoint to get current progress
# --------------------------------------------------------------------------------
@app.get("/progress")
def get_progress():
    return {
        "stage": pipeline_status.stage,
        "message": pipeline_status.message,
        "progress": pipeline_status.progress,
        "error": pipeline_status.error
    }

# --------------------------------------------------------------------------------
#  Endpoint to download results
# --------------------------------------------------------------------------------
@app.get("/download-results")
def download_results():
    if pipeline_status.stage != "done" or not pipeline_status.output_file:
        return {"detail": "Results are not yet available for download."}
    # Return file download
    file_path = pipeline_status.output_file
    if not os.path.exists(file_path):
        return {"detail": f"Output file not found at path: {file_path}"}

    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type='application/octet-stream', filename=filename)


# --------------------------------------------------------------------------------
# You can run this file directly:
# --------------------------------------------------------------------------------
#   uvicorn server:app --host 0.0.0.0 --port 8000
# --------------------------------------------------------------------------------

