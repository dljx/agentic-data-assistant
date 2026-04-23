# agents/core.py

"""
Provides the base class for all agents, using the low-level `genai.Client`.

This core class handles the initialization of the client that will be used
to make direct API calls to the Google Generative AI service, correctly configured
for a Vertex AI environment.
"""

from abc import ABC
from typing import Dict, Any
from google import genai

# Import the project and region details from our central config
from utilities import PROJECT_ID, BQ_REGION


class Agent(ABC):
    """
    The abstract base class for all agents using the genai.Client,
    configured for Vertex AI.
    """
    agentType: str = "Agent"

    def __init__(self, **kwargs):
        """
        Initializes the agent by creating a genai.Client instance configured
        to use the Vertex AI backend, as per the working example.
        """
        try:
            # This is the definitive, correct initialization pattern.
            self.client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=BQ_REGION
            )
            # Initialize token usage tracking
            self.token_usage = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize genai.Client for Vertex AI. "
                             f"Please ensure your environment is authenticated correctly and "
                             f"the correct IAM roles (e.g., Vertex AI User) are assigned. Error: {e}")

    def _extract_token_usage(self, response) -> Dict[str, int]:
        """
        Extracts token usage information from a response and updates the agent's token usage.
        
        Args:
            response: The response object from generate_content
            
        Returns:
            Dictionary containing token usage information
        """
        token_info = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_info['prompt_tokens'] = getattr(usage, 'prompt_token_count', 0)
                token_info['completion_tokens'] = getattr(usage, 'candidates_token_count', 0)
                token_info['total_tokens'] = getattr(usage, 'total_token_count', 0)
                
                # Update the agent's cumulative token usage
                self.token_usage['prompt_tokens'] += token_info['prompt_tokens']
                self.token_usage['completion_tokens'] += token_info['completion_tokens']
                self.token_usage['total_tokens'] += token_info['total_tokens']
                
                print(f"    -> {self.agentType} token usage: {token_info}")
        except Exception as e:
            print(f"    -> Warning: Could not extract token usage from response: {e}")
        
        return token_info

    def get_token_usage(self) -> Dict[str, int]:
        """
        Returns the cumulative token usage for this agent.
        
        Returns:
            Dictionary containing cumulative token usage information
        """
        return self.token_usage.copy()

    def reset_token_usage(self):
        """
        Resets the token usage counters for this agent.
        """
        self.token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }