# utilities/__init__.py

"""
This package initializes all configuration and utility functions for the application.

It acts as a central point for loading settings from config.ini, prompts from
prompts.yaml, detailed schema information from a CSV file, and provides
a human-readable logging facility for the pipeline's operation.

By loading configuration and static data once at startup and storing it in
global variables, we ensure efficient, DRY (Don't Repeat Yourself) access
across all modules of the application.
"""

import configparser
import os
import sys
import yaml
import pandas as pd # Import pandas for CSV handling
import logging # For human-readable logging
from datetime import datetime # For timestamping logs

# --- Configuration Setup ---
config = configparser.ConfigParser()

# Standard boilerplate to find the root directory and load config.ini
def is_root_dir():
    """Checks if the current working directory is the project's root."""
    current_dir = os.getcwd()
    # A more robust check for the root might be looking for a specific file or folder
    return os.path.exists(os.path.join(current_dir, "config.ini"))

if is_root_dir():
    root_dir = os.getcwd()
else:
    # This handles cases where scripts might be run from a subdirectory
    # (e.g., if you had a /tests folder in the future)
    # It assumes the script is one level down from the root.
    # For a deeper structure, this might need to be more robust.
    potential_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if os.path.exists(os.path.join(potential_root, "config.ini")):
        root_dir = potential_root
    else:
        # Fallback if we can't easily determine root (e.g., complex module imports)
        # This might require setting PYTHONPATH or having a more explicit root marker.
        print("Warning: Could not auto-determine project root. Assuming current directory.")
        root_dir = os.getcwd()


config_path = os.path.join(root_dir, 'config.ini')
if os.path.exists(config_path):
    config.read(config_path)
else:
    raise FileNotFoundError(f"FATAL: config.ini not found at the expected path: {config_path}")

print(f"current dir:  {os.getcwd()}")
print(f"root_dir set to: {root_dir}")


# --- Utility Functions ---
def format_prompt(context_prompt: str, **kwargs: str) -> str:
    """Formats a prompt string by replacing placeholders."""
    return context_prompt.format(**kwargs)

def load_yaml(file_path: str) -> dict:
    """Loads a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL: YAML file not found at {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"FATAL: Error parsing YAML file {file_path}: {e}")

def format_schema_for_prompt(df: pd.DataFrame) -> str:
    """
    Formats the schema DataFrame into a concise, Markdown-formatted string
    that is highly optimized for an LLM prompt.
    """
    if df.empty:
        return "No detailed schema description available. The agent will rely on its general knowledge and basic table/column names."
    
    formatted_string = "### Full Database Schema Reference\n"
    # Group by table to make the output structured and easy for the LLM to read
    for table_name, group in df.groupby('table_name'):
        formatted_string += f"\n**Table: `{table_name}`**\n"
        for _, row in group.iterrows():
            # Create a concise line for each column
            col_name = row['column_name']
            col_desc = row['column_description']
            
            # Handle type information if available
            col_type = row.get('type')
            if pd.notna(col_type) and col_type:
                type_text = f" (`{col_type}`)"
            else:
                type_text = ""
            
            # Handle relevant_columns if they exist
            relevant_cols = row.get('relevant_columns')
            if pd.notna(relevant_cols) and relevant_cols:
                relevant_text = f" (Related: {relevant_cols})"
            else:
                relevant_text = ""
            
            formatted_string += (
                f"- **`{col_name}`**{type_text}: {col_desc}{relevant_text}\n"
            )
    return formatted_string


# --- NEW: Human-Readable Logging Setup ---
pipeline_logger = logging.getLogger('PipelineNarrativeLogger')
pipeline_logger.setLevel(logging.INFO)

# Create a handler to output logs to a file.
# The log file will be named with the current date.
log_filename = f"pipeline_narrative_log_{datetime.now().strftime('%Y%m%d')}.txt"
log_filepath = os.path.join(root_dir, log_filename) # Place log in root directory
log_file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
log_file_handler.setLevel(logging.INFO)

# Create a simple formatter: timestamp - message
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_file_handler.setFormatter(formatter)

# Add the handler to the logger, but only if it doesn't have one already.
# This prevents duplicate log entries if this module is reloaded (e.g., in Jupyter).
if not pipeline_logger.handlers:
    pipeline_logger.addHandler(log_file_handler)
    print(f"Pipeline narrative logging configured. Log file: {log_filepath}")

def log_pipeline_event(session_id: str, event_description: str, details: str = ""):
    """Logs a human-readable event in the pipeline's story."""
    log_message = f"SESSION [{session_id}] | {event_description}"
    if details:
        # Ensure details are formatted nicely, especially if they are multi-line JSON
        if details.strip().startswith(("{","[")): # Basic check for JSON
            try:
                parsed_json = json.loads(details)
                details_formatted = json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                details_formatted = details # Keep as is if not valid JSON
        else:
            details_formatted = details

        log_message += f"\nDETAILS:\n{details_formatted}\n---------------------------------"
    pipeline_logger.info(log_message)


# --- Global Variables Initialized at Startup ---

# [CONFIG]
EMBEDDING_MODEL = config.get('CONFIG', 'embedding_model', fallback='vertex')
VECTOR_STORE = config.get('CONFIG', 'vector_store', fallback='bigquery-vector')
LOGGING = config.getboolean('CONFIG','logging', fallback=True)
EXAMPLES = config.getboolean('CONFIG', 'kgq_examples', fallback=True)
USE_SESSION_HISTORY = config.getboolean('CONFIG', 'use_session_history', fallback=True)
USE_COLUMN_SAMPLES = config.getboolean('CONFIG','use_column_samples', fallback=False)

#[GCP]
PROJECT_ID =  config.get('GCP', 'project_id', fallback=None)
if not PROJECT_ID:
    raise ValueError("FATAL: `project_id` is not set in config.ini under [GCP]")

#[BIGQUERY]
BQ_REGION = config.get('BIGQUERY', 'bq_dataset_region', fallback='us-central1')
BQ_DATASET_NAME = config.get('BIGQUERY', 'bq_dataset_name', fallback='your_dataset')
BQ_LOG_TABLE_NAME = config.get('BIGQUERY', 'bq_log_table_name', fallback='audit_log_table')
BQ_BLUEWHALE_TABLE = config.get('BIGQUERY', 'investors_table_name', fallback='investors_table')
BQ_COMPANY_TABLE = config.get('BIGQUERY', 'companies_table_name', fallback='companies_table')

#[PROMPTS]
PROMPTS = load_yaml(os.path.join(root_dir, 'prompts.yaml'))

# --- Public Exports ---
# Defines what can be imported from this module with `from utilities import ...`
__all__ = [
    # Config variables
    "EMBEDDING_MODEL", "DESCRIPTION_MODEL", "VECTOR_STORE", "LOGGING", "EXAMPLES",
    "USE_SESSION_HISTORY", "USE_COLUMN_SAMPLES", "PROJECT_ID", "BQ_REGION",
    "BQ_DATASET_NAME" "BQ_LOG_TABLE_NAME",
    # Prompt and Schema objects
    "PROMPTS",
    # Logging function
    "log_pipeline_event",
    # Other utilities
    "root_dir",
]