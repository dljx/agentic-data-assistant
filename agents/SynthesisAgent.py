# agents/SynthesisAgent.py

"""
The SynthesisAgent is the final agent in the multi-agent pipeline.

Its sole responsibility is to receive the user's original question along with
all the structured evidence collected by the specialist agents (QueryAgent,
SearchAgent). It then synthesizes this information into a final, cohesive,
and easy-to-understand natural language response for the end-user. It is
responsible for identifying and explaining any discrepancies in the evidence.
"""

from typing import Any, Dict
import json
import traceback

from agents.core import Agent
from utilities import PROMPTS, format_prompt
from google.genai.types import GenerateContentConfig


class SynthesisAgent(Agent):
    """
    An agent that specializes in creating a final answer from collected evidence.
    """
    agentType: str = "SynthesisAgent"

    def __init__(self, model_id: str = 'gemini-2.5-pro'):
        """
        Initializes the SynthesisAgent with a powerful model for nuanced
        language generation and reasoning.
        """
        super().__init__()
        self.model_id = model_id
        # This agent does not use tools, so its config is simple.
        self.generation_config = GenerateContentConfig(temperature=0.2) # A little creativity is okay for the final text.


    def run(self, original_question: str, evidence: Dict[str, Any], conversation_history: str = None) -> Dict[str, Any]:
        """
        Generates the final natural language response for the user.

        Args:
            original_question: The original, unmodified question from the user.
            evidence: The dictionary containing all data gathered by the
                      QueryAgent and SearchAgent during the execution phase.
            conversation_history: Previous conversation context.

        Returns:
            A dictionary containing the final answer and token usage information.
        """
        # 1. Prepare the evidence for the prompt.
        try:
            # Use json.dumps for a clean, universally readable format for the evidence.
            evidence_str = json.dumps(evidence, indent=2, default=str)
        except TypeError as e:
            evidence_str = f"Could not serialize evidence. Error: {e}"

        # 2. Prepare the prompt.
        try:
            prompt_template = PROMPTS['synthesis_prompt']
        except KeyError:
            return {
                "answer": "I'm sorry, an internal error occurred (could not find synthesis_prompt).",
                "token_usage": self.get_token_usage()
            }

        prompt = format_prompt(
            prompt_template,
            original_question=original_question,
            evidence=evidence_str,
            conversation_history=conversation_history or "This is the start of a new conversation."
        )
        
        # 3. Call the LLM to generate the final, synthesized response.
        print("    -> SynthesisAgent generating final answer...")
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt],
                config=self.generation_config
            )
            
            # Extract token usage from the response
            token_info = self._extract_token_usage(response)
            
            final_answer = response.text
            print("    -> Synthesis complete.")
            
            return {
                "answer": final_answer.strip(),
                "token_usage": self.get_token_usage(),
                "last_call_tokens": token_info
            }

        except Exception as e:
            print(f"    [!!!] An error occurred in the SynthesisAgent: {e}")
            traceback.print_exc()
            return {
                "answer": f"I'm sorry, I encountered an error while trying to formulate the final answer. (Details: {e})",
                "token_usage": self.get_token_usage()
            }