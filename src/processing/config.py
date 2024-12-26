# processing/config.py
import os

# --------------------------------------------------------------------
# Configuration Management
# --------------------------------------------------------------------
# Place all hard-coded paths and values (or environment variables) here
# to centralize their management.

INFO_PATH = os.getenv("INFO_PATH", "data/input/info.json")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
ST_WEIGHT = float(os.getenv("ST_WEIGHT", "0.5"))

DEFAULT_RESEARCH_OBJECTIVES = "No specific research objectives provided."
DEFAULT_THEORETICAL_FRAMEWORK = {}

# You can extend this file as needed for additional configuration parameters
