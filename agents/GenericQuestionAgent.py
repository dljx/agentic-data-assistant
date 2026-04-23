# agents/GenericQuestionAgent.py

import json
from typing import Any, Dict, Optional
from google.genai.types import GenerateContentConfig
from agents.core import Agent
from utilities import PROMPTS, format_prompt


class GenericQuestionAgent(Agent):
    """
    An agent that detects if a question is generic and handles it appropriately.
    If generic, it provides a helpful assistant response.
    If specific, it indicates the question should go through entity resolution.
    """
    agentType: str = "GenericQuestionAgent"

    def __init__(self, model_id: str = 'gemini-2.5-flash'):
        super().__init__()
        self.model_id = model_id
        # Set temperature to 0 for deterministic classification
        self.generation_config = GenerateContentConfig(temperature=0.0)

    def generate_llm_response(self, prompt: str) -> str:
        """
        Uses the agent's client to make a simple, one-shot call to the LLM.
        """
        try:
            print("    -> GenericQuestionAgent: Analyzing question...")
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt],
                config=self.generation_config
            )
            
            # Extract token usage from the response
            self._extract_token_usage(response)
            
            print(f"    -> GenericQuestionAgent: Received classification response.")
            return response.text
        except Exception as e:
            print(f"    -> Error during GenericQuestionAgent LLM call: {e}")
            return f"Error: {e}"

    def detect_generic_question(self, user_question: str, conversation_history: str = None) -> Dict[str, Any]:
        """
        Determines if a question is generic or specific to financial entities.
        
        Args:
            user_question: The current user question
            conversation_history: Previous conversation context
        
        Returns:
            Dict containing is_generic flag, confidence, and reasoning
        """
        try:
            prompt_template = PROMPTS['generic_question_detection_prompt']
        except KeyError:
            return {"error": "Could not find 'generic_question_detection_prompt' in prompts.yaml"}

        prompt = format_prompt(
            prompt_template,
            user_question=user_question,
            conversation_history=conversation_history or "This is the start of a new conversation."
        )
        
        llm_response_str = self.generate_llm_response(prompt)

        try:
            clean_response = llm_response_str.strip().replace("```json", "").replace("```", "")
            parsed_json = json.loads(clean_response)
            # print(f"    -> GenericQuestionAgent: Classification result: Generic: {parsed_json.get("is_generic", False)}, Confidence: {parsed_json.get("confidence", 0.0)}, Reasoning: {parsed_json.get("reasoning", "")}")
            
            return {
                "is_generic": parsed_json.get("is_generic", False),
                "confidence": parsed_json.get("confidence", 0.0),
                "reasoning": parsed_json.get("reasoning", ""),
                "original_question": user_question
            }
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"    -> [!] GenericQuestionAgent failed to parse classification response. Error: {e}")
            print(f"        Raw LLM response was: {llm_response_str}")
            return {
                "is_generic": False,  # Default to specific to be safe
                "confidence": 0.0,
                "reasoning": "Failed to parse classification response",
                "original_question": user_question,
                "error": "Failed to parse classification response from the language model."
            }

    def handle_generic_question(self, user_question: str, conversation_history: str = None) -> str:
        """
        Provides a helpful assistant response for generic questions.
        
        Args:
            user_question: The current user question
            conversation_history: Previous conversation context
        
        Returns:
            String containing the assistant's response
        """
        try:
            prompt_template = PROMPTS['generic_assistant_prompt']
        except KeyError:
            return "I'm here to help! However, I'm having trouble accessing my response templates. Could you please rephrase your question?"

        prompt = format_prompt(
            prompt_template,
            user_question=user_question,
            conversation_history=conversation_history or "This is the start of a new conversation."
        )
        
        # Use higher temperature for more natural responses
        config = GenerateContentConfig(temperature=0.7)
        
        try:
            print("    -> GenericQuestionAgent: Generating helpful response...")
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt],
                config=config
            )
            
            # Extract token usage from the response
            self._extract_token_usage(response)
            
            print(f"    -> GenericQuestionAgent: Generated response for generic question.")
            return response.text
            
        except Exception as e:
            print(f"    -> Error during GenericQuestionAgent response generation: {e}")
            return f"I'm here to help! However, I encountered an error while processing your question. Please try again or rephrase your question."

    def run(self, user_question: str, conversation_history: str = None) -> Dict[str, Any]:
        """
        Main method that detects if a question is generic and handles it appropriately.
        
        Args:
            user_question: The current user question
            conversation_history: Previous conversation context
        
        Returns:
            Dict containing either:
            - For generic questions: {"is_generic": True, "answer": "response"}
            - For specific questions: {"is_generic": False, "should_continue": True}
        """
        print("    -> GenericQuestionAgent: Starting question analysis...")
        
        # First, detect if the question is generic
        classification = self.detect_generic_question(user_question, conversation_history)
        
        if classification.get("error"):
            return classification
        
        if classification.get("is_generic", False):
            print(f"    -> GenericQuestionAgent: Question classified as GENERIC (confidence: {classification.get('confidence', 0.0):.2f})")
            print(f"    -> Reasoning: {classification.get('reasoning', '')}")
            
            # Handle the generic question
            response = self.handle_generic_question(user_question, conversation_history)
            
            return {
                "is_generic": True,
                "answer": response,
                "confidence": classification.get("confidence", 0.0),
                "reasoning": classification.get("reasoning", ""),
                "original_question": user_question
            }
        else:
            print(f"    -> GenericQuestionAgent: Question classified as SPECIFIC (confidence: {classification.get('confidence', 0.0):.2f})")
            print(f"    -> Reasoning: {classification.get('reasoning', '')}")
            print("    -> Proceeding with entity resolution pipeline...")
            
            return {
                "is_generic": False,
                "should_continue": True,
                "confidence": classification.get("confidence", 0.0),
                "reasoning": classification.get("reasoning", ""),
                "original_question": user_question
            } 