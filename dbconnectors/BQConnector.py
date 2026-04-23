# dbconnectors/BQConnector.py

"""
A dedicated connector for Google BigQuery.

This class encapsulates all direct interactions with BigQuery, including
establishing a connection, executing queries, and performing validation
checks (dry runs). It is a critical dependency for the SQLTool.
"""

from abc import ABC
from datetime import datetime

import google.auth
import pandas as pd
from google.cloud import bigquery
from utilities import PROJECT_ID, BQ_DATASET_NAME, format_schema_for_prompt
from .core import DBConnector

def get_auth_user() -> str:
    """Helper function to identify the authenticated user or service account."""
    try:
        credentials, project_id = google.auth.default()
        if hasattr(credentials, 'service_account_email'):
            return credentials.service_account_email
        # For user credentials, the principal email might be available
        elif hasattr(credentials, 'id_token'):
             # This is a bit of a guess, but often works for user accounts
            return "User Account (email not available)"
    except Exception:
        pass
    return "Not Determined"


def bq_specific_data_types() -> str:
    """
    Returns a string containing a summary of BigQuery data types.
    This can be injected into prompts to provide context to the LLM if needed.
    """
    return '''
    -   **Numeric:** INTEGER (INT64), FLOAT (FLOAT64), NUMERIC, BIGNUMERIC.
    -   **String/Bytes:** STRING, BYTES.
    -   **Boolean:** BOOL.
    -   **Date/Time:** DATE, TIME, DATETIME, TIMESTAMP.
    -   **Geospatial:** GEOGRAPHY.
    -   **Structured:** ARRAY, STRUCT.
    '''


class BQConnector(DBConnector, ABC):
    """

    Manages connections and query execution for Google BigQuery.
    """
    def __init__(self, project_id: str, region: str, dataset_id: str):
        """
        Initializes the BigQuery connector.

        Args:
            project_id: The Google Cloud Project ID.
            region: The default region for BigQuery jobs.
            dataset_id: The name of the dataset for logging/system tables.
            audit_log_table_name: The name of the table for audit logs.
        """
        self.project_id = project_id
        self.region = region
        self.dataset_id = dataset_id
        self.client = self.getconn()

    def getconn(self) -> bigquery.Client:
        """Establishes and returns a BigQuery client instance."""
        try:
            client = bigquery.Client(project=self.project_id)
            print("BigQuery client initialized successfully.")
            return client
        except Exception as e:
            print(f"[!!!] CRITICAL: Failed to initialize BigQuery client. Error: {e}")
            # In a real app, you might exit or handle this more gracefully.
            return None

    def retrieve_df(self, query: str) -> pd.DataFrame:
        """
        Executes a BigQuery query and returns the results as a Pandas DataFrame.

        Args:
            query: The SQL query string to execute.

        Returns:
            A pandas DataFrame containing the query results.
        
        Raises:
            Exception: Propagates exceptions from the BigQuery client.
        """
        if not self.client:
            raise ConnectionError("BigQuery client is not initialized.")
        
        # The `query_and_wait` method handles job creation, execution, and result fetching.
        return self.client.query_and_wait(query).to_dataframe()

    def test_sql_plan_execution(self, generated_sql: str) -> tuple[bool, str]:
        """
        Performs a dry run of a SQL query to validate syntax and references.
        This does not execute the query or incur processing costs.

        Args:
            generated_sql: The SQL query to validate.

        Returns:
            A tuple containing:
            - A boolean indicating if the query is valid (True) or not (False).
            - A string with a success message or the error details.
        """
        if not self.client:
            return False, "BigQuery client is not initialized."
            
        try:
            # Configure the query job for a dry run.
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            
            # Initiate the dry run.
            query_job = self.client.query(generated_sql, job_config=job_config)

            # If no exception is raised, the syntax is valid.
            success_message = f"Query is valid and will process {query_job.total_bytes_processed} bytes."
            return True, success_message
        except Exception as e:
            # An exception indicates an error in the SQL.
            return False, str(e)
        
    def get_schema_context(self, dataframe) -> str:
        return format_schema_for_prompt(dataframe)
    
    def get_table_vector_match(self, question_embeddings) -> str:
        print(f"    -> Embedder: Searching tables through vector matching.")
        sql_query = f'''
            SELECT base.table_name, base.table_description, distance
            FROM vector_search(
                (SELECT * FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.table_embeddings`), 
                "embedding", 
                (SELECT {question_embeddings} as qe), 
                top_k=> 2,
                distance_type=>"COSINE"
            ) where 1-distance > 0.49
        '''
        try:
            tables_df = self.retrieve_df(sql_query)
            for index, row in tables_df.iterrows():
                print(f"    -> Embedder: Table Found: {row['table_name']} Table Match Score: {round(1 - row['distance'], 2)}")
            return tables_df
        except Exception as e:
            print(f"Error in table vector matching: {e}")
    
    def get_header_vector_match(self, quesiton_embeddings, table) -> str:
        print(f"    -> Embedder: Searching headers through vector matching.")
        sql_query = f'''
            SELECT base.table_name, base.column_name, base.column_description, base.relevant_columns, base.type, distance
            FROM vector_search(
                (SELECT * FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.header_embeddings` WHERE table_name = '{table}' AND column_name NOT IN ('name', 'company_name', 'investor_name', 'reported_date', 'folder_date', 'reported_date_date', 'published_date', 'reported_date_raw')), 
                "embedding", 
                (SELECT {quesiton_embeddings} as qe), 
                top_k=> 3,
                distance_type=>"COSINE"
            ) where 1-distance > 0.45

            UNION ALL

            SELECT table_name, column_name, column_description, relevant_columns, type, NULL AS distance
            FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.header_embeddings`
            WHERE table_name = '{table}' AND column_name IN ('name', 'company_name', 'investor_name', 'reported_date', 'folder_date', 'reported_date_date', 'published_date', 'reported_date_raw');
        '''
        try:
            headers_df = self.retrieve_df(sql_query)
            for index, row in headers_df.iterrows():
                print(f"    -> Embedder: Header Found: {row['column_name']} Header Match Score: {round(1 - row['distance'], 2)}")
            return headers_df
        except Exception as e:
            print(f"Error in header vector matching: {e}")
            return pd.DataFrame()

    def get_kgq_vector_match(self, question_embeddings) -> str:
        sql_query = f'''
            SELECT base.query, distance, base.question
            FROM vector_search(
                (SELECT * FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.kgq_embeddings`), 
                "embedding", 
                (SELECT {question_embeddings} as qe), 
                top_k=> 1,
                distance_type=>"COSINE"
            ) where 1-distance > 0.59
        '''
        try:
            response_df = self.retrieve_df(sql_query)
            if response_df.empty:
                return None
            else:
                retrieved_query = response_df.iloc[0, 0]
                print(f"    -> Embedder: KGQ Match Question: {response_df.iloc[0, 2]}")
                print(f"    -> Embedder: KGQ Match Score: {round(1 - response_df.iloc[0, 1], 2)}")
                return retrieved_query
        except Exception as e:
            print(f"Error in KGQ vector matching: {e}")

    def expand_schema_with_relevant_columns(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expands the schema context by including relevant columns when they exist.
        
        Args:
            base_df: DataFrame with matched headers containing relevant_columns
            
        Returns:
            Expanded DataFrame with additional relevant columns included
        """
        if base_df.empty:
            return base_df
            
        print(f"    -> Embedder: Expanding schema with relevant columns...")
        
        # Collect all relevant columns to fetch
        relevant_columns_to_fetch = []
        tables_with_relevant_cols = set()
        
        for _, row in base_df.iterrows():
            if pd.notna(row['relevant_columns']) and row['relevant_columns']:
                table_name = row['table_name']
                # Parse relevant columns (assuming they're comma-separated)
                rel_cols = [col.strip() for col in row['relevant_columns'].split(',')]
                for col in rel_cols:
                    if col:  # Skip empty strings
                        relevant_columns_to_fetch.append((table_name, col))
                        tables_with_relevant_cols.add(table_name)
        
        if not relevant_columns_to_fetch:
            print(f"       - No relevant columns found to expand")
            return base_df
            
        # Build SQL to fetch relevant columns
        conditions = []
        for table_name, col_name in relevant_columns_to_fetch:
            conditions.append(f"(table_name = '{table_name}' AND column_name = '{col_name}')")
        
        if conditions:
            fetch_sql = f"""
            SELECT table_name, column_name, column_description, relevant_columns, type
            FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.header_embeddings`
            WHERE ({' OR '.join(conditions)})
            """
            
            try:
                additional_df = self.retrieve_df(fetch_sql)
                print(f"       - Found {len(additional_df)} additional relevant columns")
                
                # Combine base results with additional relevant columns, removing duplicates
                combined_df = pd.concat([base_df, additional_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['table_name', 'column_name'], keep='first')
                
                print(f"       - Total columns in expanded schema: {len(combined_df)}")
                return combined_df
                
            except Exception as e:
                print(f"       - Error fetching relevant columns: {e}")
                return base_df
        
        return base_df