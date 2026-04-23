# agents/PlanningAgent.py

"""
The PlanningAgent is the strategic "project manager" of the Open_Data_QnA pipeline.

After the EntityResolutionAgent has deconstructed the user's question, the
PlanningAgent takes this structured information and devises a multi-step
plan. This plan specifies which specialist agent to call for each step
(e.g., QueryAgent for database tasks, SearchAgent for web research) to
acquire the necessary data to answer the user's question.

This version includes a toggle to enable or disable the use of the SearchAgent,
allowing for control over web grounding.
"""

import json
from typing import Any, Dict, List

from agents.core import Agent
from utilities import PROMPTS, format_prompt
from google.genai.types import GenerateContentConfig


class PlanningAgent(Agent):
    """
    An agent for creating a multi-step, multi-agent execution plan.
    It respects a flag to enable or disable web grounding.
    """

    agentType: str = "PlanningAgent"

    def __init__(self, model_id: str = 'gemini-2.5-flash'):
        """
        Initializes the PlanningAgent.

        A powerful model is recommended as creating a logical, multi-step plan
        requires strong reasoning and instruction-following capabilities.
        """
        super().__init__()
        self.model_id = model_id
        # The planner needs to be deterministic to produce consistent plans.
        self.generation_config = GenerateContentConfig(temperature=0.0)

    def _clean_llm_response(self, llm_response: str) -> str:
        """
        Cleans the raw output from the LLM to isolate the JSON plan.
        Handles markdown code blocks and other extraneous text.
        """
        cleaned_response = llm_response.strip().replace("```json", "").replace("```", "")
        start_index = cleaned_response.find('[')
        end_index = cleaned_response.rfind(']')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return cleaned_response[start_index : end_index + 1]
        start_index = cleaned_response.find('{')
        end_index = cleaned_response.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return f"[{cleaned_response[start_index : end_index + 1]}]"
        return cleaned_response

    def create_plan(self, structured_request: Dict[str, Any], enable_grounding: bool, schema_context: str = "", conversation_history: str = None) -> List[Dict[str, Any]]:
        """
        Generates a step-by-step plan to answer the user's query.

        Args:
            structured_request: The structured JSON object from EntityResolutionAgent.
            enable_grounding: A boolean flag. If False, the planner is forbidden
                              from including the SearchAgent in its plan.
            schema_context: The detailed BigQuery schema context to help make better plans.
            conversation_history: Previous conversation context.
        Returns:
            A list of dictionaries, where each dictionary represents a
            step in the execution plan. Returns an empty list on failure.
        """
        try:
            prompt_template = PROMPTS['planning_prompt']
        except KeyError:
            print("[!] PlanningAgent Error: Could not find 'planning_prompt' in prompts.yaml")
            return []

        # THIS IS THE CHANGE: Dynamically add a restriction to the prompt if grounding is disabled.
        # We use the placeholder {grounding_restriction} in the prompt.
        if enable_grounding:
            # No restriction if grounding is enabled.
            restriction_text = (
                "\n\n**CRITICAL RESTRICTION:** Grounding is enabled for this query. "
                "You MUST use the `SearchAgent` in your plan. "
            )
        else:
            restriction_text = (
                "\n\n**CRITICAL RESTRICTION:** Grounding is disabled for this query. "
                "You are FORBIDDEN from using the `SearchAgent` in your plan. "
                "You must create a plan that uses ONLY the `QueryAgent`."
            )

        # Convert the structured request dictionary to a JSON string for the prompt.
        request_str = json.dumps(structured_request, indent=2)

        prompt = format_prompt(
            prompt_template,
            structured_request=request_str,
            grounding_restriction=restriction_text, # Fill in the placeholder
            schema_context=schema_context,  # Add the missing schema_context parameter
            conversation_history=conversation_history or "This is the start of a new conversation."
        )

        # Call the LLM to generate the plan.
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt],
                config=self.generation_config # Use the deterministic config
            )
            llm_response = response.text
        except Exception as e:
            print(f"[!] PlanningAgent Error: API call failed. {e}")
            return []

        # Clean and parse the LLM's JSON response.
        cleaned_response = self._clean_llm_response(llm_response)

        try:
            plan = json.loads(cleaned_response)
            if isinstance(plan, list):
                return plan
            else:
                print(f"[!] PlanningAgent Warning: Expected a list but got {type(plan)}. Returning empty plan.")
                return []

        except json.JSONDecodeError:
            print(f"[!] PlanningAgent Error: Failed to decode JSON plan from LLM.")
            print(f"    Raw LLM response was:\n{llm_response}")
            print(f"    Cleaned response was:\n{cleaned_response}")
            return []