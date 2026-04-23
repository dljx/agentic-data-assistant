# tools/__init__.py

"""
This package initializes the tools that the ReasoningAgent can execute.

Tools are specialized, reusable components that perform a single, well-defined task.
In the new architecture, these are simple, decorated functions that the Gemini
model can call directly.
"""

from .SQLTool import execute_sql_query

# This list defines what becomes available when a module executes
# `from tools import *`, and it clearly documents the public API of this package.
__all__ = [
    "execute_sql_query",
]