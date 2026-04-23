# dbconnectors/__init__.py

# This module acts as a centralized hub for all database connectors.
# It provides a single import point for other parts of the application.

# Import individual connector classes
from .BQConnector import BQConnector

# Import shared utilities and config from the utilities module
from utilities import (
    PROJECT_ID, BQ_REGION, BQ_DATASET_NAME,
    BQ_LOG_TABLE_NAME
)

# --- BigQuery Connector Instance ---
# This is the primary database connector for the application.
# It's initialized once and can be imported throughout the codebase.
try:
    bqconnector = BQConnector(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_NAME,
        region=BQ_REGION
    )
    print("BigQuery client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize BigQuery client: {e}")
    bqconnector = None

# --- Public Exports ---
# Define what can be imported from this module with `from dbconnectors import ...`
__all__ = [
    # Connector classes
    "BQConnector",
    # Connector instances
    "bqconnector",
]