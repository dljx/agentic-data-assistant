# agents/SearchAgent.py

"""
The SearchAgent is a specialist agent responsible for interacting with the web.

Its sole purpose is to receive a specific, qualitative question, translate it
into an effective search query, execute it using the native GoogleSearch tool,
and return a concise summary of the findings.
"""

from typing import Any, Dict
import json
import traceback

from google import genai
from google.genai.types import (
    Tool, GoogleSearch, GenerateContentConfig
)

from agents.core import Agent
from utilities import PROMPTS, format_prompt


class SearchAgent(Agent):
    """
    An agent that specializes in answering questions by searching the web.
    """
    agentType: str = "SearchAgent"

    def __init__(self, model_id: str = 'gemini-2.5-flash'):
        """
        Initializes the SearchAgent.
        """
        super().__init__()
        self.model_id = model_id
        self.generation_config = GenerateContentConfig(temperature=0.2)

    def run(self, question: str) -> Dict[str, Any]:
        """
        Executes a tool-calling loop to answer a specific research question.

        Args:
            question: A clear, specific question that requires web research.

        Returns:
            A dictionary containing the summarized search results.
        """
        # 1. Define the agent's single tool: GoogleSearch.
        google_search_tool = Tool(google_search=GoogleSearch())
        tool_config = GenerateContentConfig(tools=[google_search_tool], temperature=0)

        # 2. Prepare the prompt.
        try:
            prompt_template = PROMPTS['search_agent_prompt']
        except KeyError:
            return {"error": "Could not find 'search_agent_prompt' in prompts.yaml."}

        initial_prompt = format_prompt(
            prompt_template,
            question=question
        )

        # 3. Set up and run the tool-calling loop.
        #    Even though it might only be one call, using the loop structure is robust.
        conversation_history: list = [genai.types.Content(parts=[genai.types.Part(text=initial_prompt)], role="user")]
        
        print(f"    -> SearchAgent starting loop for question: \"{question}\"")
        try:
            # For a search agent, we typically don't need a multi-turn conversation.
            # A single `generate_content` call with the tool is usually sufficient.
            # The model will use the tool and then synthesize the results in one go.
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=conversation_history,
                config=tool_config,
            )
            
            # Extract token usage from the response
            self._extract_token_usage(response)
            
            # The model's response should be the summarized text.
            final_text = response.candidates[0].content.parts[0].text
            print(f"    -> SearchAgent finished with a text response.")
            return {"summary": final_text}

        except Exception as e:
            print(f"    [!!!] An error occurred in the SearchAgent: {e}")
            traceback.print_exc()
            return {"error": f"An unexpected error occurred during web search: {e}"}