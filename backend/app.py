import subprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "FastAPI up and running"}

@app.get("/run-script")
def run_script():
    """
    Endpoint that runs the script in src/main.py
    For example, you might run a function or sub-process.
    """
    try:
        # (1) Option A: Directly run a function from main.py
        # from src.main import some_function
        # result = some_function()

        # (2) Option B: Use a subprocess call if main.py is meant to be run as a script
        completed_process = subprocess.run(["python", "src/main.py"], capture_output=True, text=True)

        return {
            "status": "success",
            "stdout": completed_process.stdout,
            "stderr": completed_process.stderr
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
