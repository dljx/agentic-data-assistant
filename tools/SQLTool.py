# tools/SQLTool.py

"""
Provides the `execute_sql_query` function as a callable tool for the
google.generativeai model.

This module defines a simple, stateless function that performs one job perfectly:
it receives a BigQuery SQL string, executes it safely, and returns the
result. The complex task of *generating* the SQL is now handled by the
ReasoningAgent.
"""

import pandas as pd
import json
import re
from dbconnectors import bqconnector
import sqlglot
from sqlglot import exp

def execute_sql_query(sql_query: str) -> str:
    """
    Executes a BigQuery SQL query and returns the result as a JSON string.
    This tool should be used to answer any questions about quantitative data
    from the database, such as transactions, ownership, amounts, or dates.
    The model calling this tool is responsible for constructing the valid
    BigQuery SQL based on the provided schema.

    Args:
        sql_query: A syntactically correct BigQuery SQL query string.

    Returns:
        A string containing the query results in JSON format, or a
        detailed error message if the query fails.
    """
    print(f"\n    -> [SQLTool] Received request to execute query:\n       {sql_query}\n")
    
    # SMART QUERY OPTIMIZATION: Add reasonable limits to prevent massive data retrieval
    optimized_query = _optimize_query_for_size(sql_query)
    if optimized_query != sql_query:
        print(f"    -> [SQLTool] Query optimized to prevent large data retrieval")
        sql_query = optimized_query
    
    try:
        is_correct, result_message = bqconnector.test_sql_plan_execution(sql_query)
        if not is_correct:
            error_msg = f"SQL Validation Error: The provided query is invalid. BigQuery's feedback: {result_message}"
            print(f"    <- [SQLTool] Responding with validation error: {error_msg}")
            return error_msg
        
        print(f"    -> [SQLTool] Query validation successful. {result_message}")

    except Exception as e:
        error_msg = f"An unexpected error occurred during the SQL validation (dry run) phase: {e}"
        print(f"    <- [SQLTool] Responding with unexpected validation error: {error_msg}")
        return error_msg

    try:
        result_df = bqconnector.retrieve_df(sql_query)
        
        if result_df.empty:
            response = "Query executed successfully but returned no results."
            print(f"    <- [SQLTool] Responding with: {response}")
            return response
        
        # SMART DATA LIMITING: Prevent token overflow
        response = _limit_and_format_results(result_df)
        
        print(f"    <- [SQLTool] Responding with processed results (first 100 chars): {response[:2]}...")

        result = {}
        tables, list = extract_tables_and_columns_sqlparse(sql_query)
        result_data = []
        for item_dict in json.loads(response)['data']:
            list_of_pairs = [[key, value] for key, value in item_dict.items()]
            result_data.append(list_of_pairs)
        result[tables[0]] = {'columns': list, 'data': result_data[0]}
        return response, result

    except Exception as e:
        error_msg = f"An unexpected error occurred during the final SQL execution phase: {e}"
        print(f"    <- [SQLTool] Responding with execution error: {error_msg}")
        return error_msg


def _optimize_query_for_size(sql_query: str, max_rows: int = 1000) -> str:
    """
    Optimizes SQL queries to prevent massive data retrieval by adding reasonable limits.
    
    Args:
        sql_query: The original SQL query
        max_rows: Maximum number of rows to allow
        
    Returns:
        Optimized SQL query with size constraints
    """
    # Normalize whitespace and make case-insensitive
    normalized_query = re.sub(r'\s+', ' ', sql_query.strip())
    
    # Check if query already has a LIMIT clause
    if re.search(r'\bLIMIT\s+\d+\b', normalized_query, re.IGNORECASE):
        return sql_query  # Already has a limit, don't modify
    
    # Check if it's an aggregation query (likely to return small results)
    aggregation_keywords = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP BY']
    if any(keyword in normalized_query.upper() for keyword in aggregation_keywords):
        return sql_query  # Aggregation queries typically return small results
    
    # For SELECT queries without LIMIT, add a reasonable limit
    if normalized_query.upper().startswith('SELECT'):
        # Add LIMIT before ORDER BY if it exists
        if re.search(r'\bORDER\s+BY\b', normalized_query, re.IGNORECASE):
            optimized = re.sub(
                r'(\bORDER\s+BY\b.*?)$', 
                f'\\1 LIMIT {max_rows}', 
                sql_query, 
                flags=re.IGNORECASE
            )
        else:
            # Add LIMIT at the end
            optimized = sql_query.rstrip(';') + f' LIMIT {max_rows}'
        
        return optimized
    
    return sql_query  # Don't modify non-SELECT queries


def _limit_and_format_results(df: pd.DataFrame, max_rows: int = 100, max_chars: int = 50000) -> str:
    """
    Intelligently limits and formats query results to prevent token overflow.
    
    Args:
        df: The pandas DataFrame with query results
        max_rows: Maximum number of rows to include
        max_chars: Maximum character count for the response
        
    Returns:
        A formatted JSON string with results and metadata
    """
    total_rows = len(df)
    
    # If dataset is large, provide summary statistics
    if total_rows > max_rows:
        print(f"    -> [SQLTool] Large dataset detected ({total_rows} rows). Applying smart limiting...")
        
        # Take first N rows
        limited_df = df.head(max_rows)
        
        # Create response with metadata
        response_data = {
            "data": limited_df.to_dict('records'),
            "metadata": {
                "total_rows_in_query": total_rows,
                "rows_returned": len(limited_df),
                "note": f"Showing first {max_rows} rows of {total_rows} total results to prevent token overflow.",
                "columns": list(df.columns)
            }
        }
        
        # Add summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            summary_stats = {}
            for col in numeric_columns:
                summary_stats[col] = {
                    "sum": float(df[col].sum()) if not pd.isna(df[col].sum()) else None,
                    "avg": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "count": int(df[col].count())
                }
            response_data["summary_statistics"] = summary_stats
    else:
        # Small dataset - return all data
        response_data = {
            "data": df.to_dict('records'),
            "metadata": {
                "total_rows": total_rows,
                "note": "Complete dataset returned."
            }
        }
    
    # Convert to JSON and check size
    json_response = json.dumps(response_data, default=str, indent=None)
    
    # If still too large, further truncate
    if len(json_response) > max_chars:
        print(f"    -> [SQLTool] Response still too large ({len(json_response)} chars). Further truncating...")
        
        # Reduce to even fewer rows
        smaller_limit = min(20, len(df))
        smaller_df = df.head(smaller_limit)
        
        response_data = {
            "data": smaller_df.to_dict('records'),
            "metadata": {
                "total_rows_in_query": total_rows,
                "rows_returned": smaller_limit,
                "note": f"Heavily truncated to {smaller_limit} rows due to size constraints. Original query returned {total_rows} rows.",
                "columns": list(df.columns)
            }
        }
        
        # Add just basic summary for the most important numeric column
        if len(numeric_columns) > 0:
            main_col = numeric_columns[0]  # Assume first numeric column is most important
            response_data["key_summary"] = {
                main_col: {
                    "total_sum": float(df[main_col].sum()) if not pd.isna(df[main_col].sum()) else None,
                    "record_count": total_rows
                }
            }
        
        json_response = json.dumps(response_data, default=str, indent=None)
    
    return json_response

def extract_tables_and_columns_sqlparse(sql_query, dialect='bigquery'):
    tables = set()
    columns = set()

    try:
        # Parse the query. Specify the dialect for accurate parsing.
        parsed_expression = sqlglot.parse_one(sql_query, read=dialect)

        # 1. Extract all explicit table references (including those in subqueries/CTEs)
        for table_exp in parsed_expression.find_all(exp.Table):
            tables.add(table_exp.name) # Add the base name

        # 2. Comprehensive Column Extraction:
        # Iterate through all Column expressions found anywhere in the AST
        for col_exp in parsed_expression.find_all(exp.Column):
            columns.add(col_exp.name) # Add the column name

        # 3. Handle SELECT * specifically if you want it explicitly
        # While find_all(exp.Column) might catch some * if it's within a qualified path
        # it's good to explicitly check for top-level SELECT *
        for select_exp in parsed_expression.find_all(exp.Select):
            for expression in select_exp.expressions:
                if isinstance(expression, exp.Star):
                    columns.add("*")

        # You might also want to explicitly handle aliases in SELECT for clarity,
        # although find_all(exp.Column) will get the original column if it's there.
        # For 'col AS alias', find_all(exp.Column) will get 'col'.
        # If you specifically want the alias, you'd add:
        for select_exp in parsed_expression.find_all(exp.Select):
            for expression in select_exp.expressions:
                if isinstance(expression, exp.Alias):
                    # If you want the alias to be considered a "column" in your output
                    columns.add(expression.alias_or_name)

    except Exception as e:
        print(f"Error parsing query '{sql_query}': {e}")
        return set(), set()

    return list(tables), list(columns)