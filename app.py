# app.py

"""
The main application entry point and orchestrator for the advanced, multi-agent
Open_Data_QnA chatbot.

This script orchestrates the entire workflow:
1.  **Parse:** The `EntityResolutionAgent` deconstructs the user's question.
2.  **Plan:** The `PlanningAgent` creates a dynamic, multi-step plan involving
    specialist agents.
3.  **Execute:** A central loop iterates through the plan, calling the correct
    specialist (`QueryAgent` or `SearchAgent`) for each step and collecting evidence.
4.  **Synthesize:** The `SynthesisAgent` receives all collected evidence and
    formulates the final, user-facing answer.

This version includes a toggle to enable/disable web grounding and a detailed,
human-readable logging system to trace the pipeline's execution for each query.
"""

import json
import re
import pandas as pd
import uuid # For generating unique session IDs for logging
from typing import Any, Dict, List, Tuple

# --- Agent and Connector Imports ---
from agents import (
    Embedder,
    EntityResolutionAgent,
    GenericQuestionAgent,
    PlanningAgent,
    QueryAgent,
    SearchAgent,
    SynthesisAgent
)
from dbconnectors import bqconnector
from tools.SQLTool import extract_tables_and_columns_sqlparse
# Import the new logging function from utilities
from utilities import PROJECT_ID, BQ_DATASET_NAME, BQ_BLUEWHALE_TABLE, BQ_COMPANY_TABLE, log_pipeline_event


class ChatbotPipeline:
    """
    Encapsulates the entire state and logic of the multi-agent chatbot pipeline.
    """
    def __init__(self):
        """Initializes all agents and pre-loads necessary data."""
        print("--- System Initializing ---")
        # Initialize all the specialist agents
        self.embedder = Embedder.EmbedderAgent()
        self.generic_agent = GenericQuestionAgent()
        self.entity_agent = EntityResolutionAgent()
        self.planning_agent = PlanningAgent()
        self.query_agent = QueryAgent()
        self.search_agent = SearchAgent()
        self.synthesis_agent = SynthesisAgent()
        
        # Pre-fetch and cache entity names from BigQuery for faster resolution.
        self.company_names, self.investor_names = self._load_entities()
        print("--- System Ready ---\n")

    def _load_entities(self) -> Tuple[List[str], List[str]]:
        """
        Loads distinct company and investor names from the BigQuery tables.
        """
        print("Loading and caching entities from BigQuery...")
        try:
            companies_df = bqconnector.retrieve_df(f"SELECT DISTINCT name FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.{BQ_COMPANY_TABLE}` WHERE name IS NOT NULL")
            investors_df = bqconnector.retrieve_df(f"SELECT DISTINCT name FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.{BQ_BLUEWHALE_TABLE}` WHERE name IS NOT NULL")
            all_company_names = companies_df['name'].dropna().tolist()
            all_investor_names = investors_df['name'].dropna().tolist()
            print(f"-> Loaded {len(all_company_names)} company names and {len(all_investor_names)} investor names.")
            return all_company_names, all_investor_names
        except Exception as e:
            print(f"[!!!] CRITICAL ERROR: Could not fetch entity lists from BigQuery: {e}")
            return [], []

    def _substitute_placeholders(self, text: str, evidence: Dict) -> str:
        """
        Finds all placeholders like `{{step_1_output.key.nested_key}}` and replaces
        them with the actual values from the evidence dictionary.
        This version handles simple list indexing like `{{step_1_output.result[0].date}}`.
        """
        placeholders = re.findall(r"\{\{(.*?)\}\}", text)
        
        for placeholder in placeholders:
            keys = placeholder.split('.')
            current_value = evidence
            placeholder_found = True
            try:
                for key_part in keys:
                    # Handle list indexing, e.g., result[0]
                    if '[' in key_part and key_part.endswith(']'):
                        list_key, index_str = key_part[:-1].split('[')
                        index = int(index_str)
                        if list_key: # If there's a key before the index
                            current_value = current_value[list_key][index]
                        else: # If it's just an indexed list, e.g., {{my_list[0]}}
                            current_value = current_value[index]
                    else:
                        current_value = current_value[key_part]
                
                # Replace the placeholder with the found value
                text = text.replace(f"{{{{{placeholder}}}}}", str(current_value))
            except (KeyError, IndexError, TypeError) as e:
                print(f"[WARNING] Orchestrator: Could not resolve placeholder: '{placeholder}'. Error: {e}. Replacing with 'DATA_NOT_FOUND'.")
                text = text.replace(f"{{{{{placeholder}}}}}", "DATA_NOT_FOUND")
                placeholder_found = False
                
        return text

    def run(self, user_question: str, enable_grounding: bool = True, conversation_history: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Executes the full Parse -> Plan -> Execute -> Synthesize pipeline.
        
        Args:
            user_question: The user's current question
            enable_grounding: Whether to enable web search
            conversation_history: Formatted conversation history from previous exchanges
            session_id: Session identifier for logging (optional)
        
        Returns:
            A dictionary containing the final answer and comprehensive token usage information.
        """
        # Use provided session_id or generate a unique ID for this specific run/session for logging
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        # Ensure we only use the first 8 characters for logging
        log_session_id = session_id[:8] if session_id else str(uuid.uuid4())[:8]

        # Reset token usage for all agents before starting
        self.generic_agent.reset_token_usage()
        self.entity_agent.reset_token_usage()
        self.planning_agent.reset_token_usage()
        self.query_agent.reset_token_usage()
        self.search_agent.reset_token_usage()
        self.synthesis_agent.reset_token_usage()

        log_pipeline_event(log_session_id, "PIPELINE STARTED")
        log_pipeline_event(log_session_id, f"User asked: \"{user_question}\"")
        log_pipeline_event(log_session_id, f"Grounding (web search) for this question: {'ENABLED' if enable_grounding else 'DISABLED'}")
        
        # Log conversation history if available
        if conversation_history and conversation_history != "This is the start of a new conversation.":
            log_pipeline_event(log_session_id, "Conversation history available for context")

        # STAGE 0: CHECK IF QUESTION IS GENERIC
        log_pipeline_event(log_session_id, "Stage 0: Checking if Question is Generic (GenericQuestionAgent)")
        generic_result = self.generic_agent.run(user_question, conversation_history)
        
        if generic_result.get("error"):
            error_msg = f"Could not classify question: {generic_result['error']}"
            log_pipeline_event(log_session_id, f"PIPELINE FAILED at Stage 0: {error_msg}")
            return {
                "answer": f"I had trouble processing your request: {generic_result['error']}",
                "token_usage": self._get_total_token_usage(),
                "agent_token_breakdown": self._get_agent_token_breakdown()
            }
        
        if generic_result.get("is_generic", False):
            log_pipeline_event(log_session_id, f"Question classified as GENERIC (confidence: {generic_result.get('confidence', 0.0):.2f})")
            log_pipeline_event(log_session_id, f"Reasoning: {generic_result.get('reasoning', '')}")
            log_pipeline_event(log_session_id, "PIPELINE COMPLETED - Generic Question Handled")
            return {
                "answer": generic_result.get("answer", "I'm here to help!"),
                "token_usage": self._get_total_token_usage(),
                "agent_token_breakdown": self._get_agent_token_breakdown(),
                "is_generic": True
            }
        
        log_pipeline_event(log_session_id, f"Question classified as SPECIFIC (confidence: {generic_result.get('confidence', 0.0):.2f})")
        log_pipeline_event(log_session_id, f"Reasoning: {generic_result.get('reasoning', '')}")
        log_pipeline_event(log_session_id, "Proceeding with entity resolution pipeline...")

        # STAGE 1: PARSE & UNDERSTAND
        log_pipeline_event(log_session_id, "Stage 1: Understanding the Question (EntityResolutionAgent)")
        entityagent_response = self.entity_agent.resolve(
            user_question, self.company_names, self.investor_names, conversation_history
        )
        if len(entityagent_response) == 2:
            structured_request, company = entityagent_response
            ticker_df = bqconnector.retrieve_df(f"SELECT ticker FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.{BQ_COMPANY_TABLE}` WHERE name = '{company}'")
        else:
            structured_request = entityagent_response
            ticker_df = pd.DataFrame()
        log_pipeline_event(log_session_id, "EntityResolutionAgent finished.", 
                           f"Identified Entities: Company='{structured_request.get('resolved_company_name')}', "
                           f"Investor='{structured_request.get('resolved_investor_name')}'.\n"
                           f"Key Metrics/Focus: {structured_request.get('metrics')}\n"
                           f"Time Constraints: {structured_request.get('time_constraints')}")

        if structured_request.get("error"):
            error_msg = f"Could not understand the question: {structured_request['error']}"
            log_pipeline_event(log_session_id, f"PIPELINE FAILED at Stage 1: {error_msg}")
            return {
                "answer": f"I had trouble understanding your request: {structured_request['error']}",
                "token_usage": self._get_total_token_usage(),
                "agent_token_breakdown": self._get_agent_token_breakdown()
            }

        # STAGE 2: GET QUESTIION EMBEDDINGS AND VECTOR SEARCH THE DATABASE
        log_pipeline_event(log_session_id, "Stage 2: Understanding Database Context")
        # Creating question embedding
        question_embedding = self.embedder.create(user_question)
        # Check for kgq
        kgq = bqconnector.get_kgq_vector_match(question_embedding)

        schema_context_df = pd.DataFrame(columns=['table_name', 'column_name', 'column_description', 'relevant_columns', 'type'])
        if kgq:
            table_list, header_list = extract_tables_and_columns_sqlparse(kgq)
            table_list = [f"'{PROJECT_ID}.{BQ_DATASET_NAME}.{table}'" for table in table_list]
            tables = ", ".join(table_list)
            print(f"    -> Embedder: Table match found: {tables}")
            header_list = [f"'{header}'" for header in header_list]
            headers = ", ".join(header_list)
            kgq_df = bqconnector.retrieve_df(f'SELECT * FROM `{PROJECT_ID}.{BQ_DATASET_NAME}.schema_header_master` WHERE table_name IN ({tables}) AND column_name IN ({headers})')
            # Expand schema with relevant columns if they exist
            expanded_header_df = bqconnector.expand_schema_with_relevant_columns(kgq_df)
            headers_found = kgq_df['column_name'].str.cat(sep=', ')
            print(f"    -> Embedder: Header match found: {headers_found}")
            schema_context_df = pd.concat([schema_context_df, kgq_df], ignore_index=True)
            schema_context_df = pd.concat([schema_context_df, expanded_header_df], ignore_index=True)
        else:
            relevant_tables_df = bqconnector.get_table_vector_match(question_embedding)
            # Check if no matches were found
            if relevant_tables_df.empty:
                print(f"    -> Embedder: No matches found.")
            else:
                # Match relevant columns based on tables found
                for index, row in relevant_tables_df.iterrows():
                    print(f"    -> Embedder: Table match found: {row['table_name']}")
                    header_df = bqconnector.get_header_vector_match(question_embedding, row['table_name'])
                    if not header_df.empty:
                        # Expand schema with relevant columns if they exist
                        expanded_header_df = bqconnector.expand_schema_with_relevant_columns(header_df)
                    
                        schema_context_df = pd.concat([schema_context_df, expanded_header_df], ignore_index=True)
                        headers_found = expanded_header_df['column_name'].str.cat(sep=', ')
                        print(f"    -> Embedder: Header match found: {headers_found}")
                    
                        # Log if relevant columns were expanded
                        if len(expanded_header_df) > len(header_df):
                            additional_count = len(expanded_header_df) - len(header_df)
                            print(f"    -> Embedder: Added {additional_count} additional relevant columns")
                    else:
                        print(f"    -> Embedder: Header match not found.")
        
        # Remove any duplicate columns that might have been added
        schema_context_df = schema_context_df.drop_duplicates(subset=['table_name', 'column_name'], keep='first')
        log_pipeline_event(log_session_id, "Database schema context loaded for planning.")
        
        # STAGE 3: PLAN WITH CONTEXT
        log_pipeline_event(log_session_id, "Stage 3: Creating a Plan with Database Context (PlanningAgent)")
        plan = self.planning_agent.create_plan(
            structured_request=structured_request,
            enable_grounding=enable_grounding,
            schema_context=bqconnector.get_schema_context(dataframe=schema_context_df),  # Pass the schema context
            conversation_history=conversation_history
        )
        log_pipeline_event(log_session_id, "PlanningAgent finished.", 
                           f"Generated Plan:\n{json.dumps(plan, indent=2)}")
        if not plan:
            error_msg = "Could not create a plan to answer the question."
            log_pipeline_event(log_session_id, f"PIPELINE FAILED at Stage 3: {error_msg}")
            return {
                "answer": "I'm sorry, I was unable to formulate a plan to answer your question.",
                "token_usage": self._get_total_token_usage(),
                "agent_token_breakdown": self._get_agent_token_breakdown()
            }

        # STAGE 4: EXECUTE (Orchestration Loop)
        log_pipeline_event(log_session_id, "Stage 4: Executing the Plan (Calling Specialist Agents)")
        evidence = {}
        wms_tables = {}
        for step_details in plan:
            step_num = step_details['step']
            agent_name = step_details['agent_to_call']
            question_for_agent = step_details['question']
            storage_key = step_details['store_as']

            log_pipeline_event(log_session_id, f"Executing Plan Step {step_num}: Calling {agent_name}")

            # Substitute placeholders with data from previous steps
            question_for_agent = self._substitute_placeholders(question_for_agent, evidence)
            log_pipeline_event(log_session_id, f"Task for {agent_name}: \"{question_for_agent}\"")

            # Query Agent debugging step
            with open("query_agent_result.json", "a") as f:
                json.dump(step_details, f, indent=4)

            step_result = {}
            if agent_name == "QueryAgent":
                query_agent_response = self.query_agent.run(dataframe=schema_context_df, reference_query=kgq, question=question_for_agent)
                if len(query_agent_response) == 2:
                    step_result, query_response = query_agent_response
                    wms_tables.update(query_response)
                else:
                    print("      -> QueryAgent: failed falling back to search agent")
                    step_result = self.search_agent.run(user_question)
            elif agent_name == "SearchAgent":
                if enable_grounding:
                    step_result = self.search_agent.run(question_for_agent)
                else:
                    log_pipeline_event(log_session_id, "Grounding disabled by user. Skipping SearchAgent call.")
                    step_result = {"summary": "Web search was disabled for this query."}
            else:
                error_msg = f"Unknown agent '{agent_name}' specified in plan for step {step_num}."
                log_pipeline_event(log_session_id, f"PIPELINE FAILED at Stage 3: {error_msg}")
                step_result = {"error": error_msg}
            
            evidence[storage_key] = step_result
            log_pipeline_event(log_session_id, f"{agent_name} finished Step {step_num}. Result stored as '{storage_key}'.",
                               f"Data found snippet: {str(step_result)[:500]}...")
            
            # Query Agent debugging step
            with open("query_agent_result.json", "a") as f:
                json.dump(step_result, f, indent=4)

            f.close()

        # STAGE 5: SYNTHESIZE
        log_pipeline_event(log_session_id, "Stage 5: Combining Evidence for Final Answer (SynthesisAgent)")
        synthesis_result = self.synthesis_agent.run(
            original_question=user_question,
            evidence=evidence,
            conversation_history=conversation_history
        )
        
        # Handle the new dictionary return format from SynthesisAgent
        if isinstance(synthesis_result, dict):
            final_answer = synthesis_result.get("answer", "No answer provided")
        else:
            # Fallback for backward compatibility
            final_answer = synthesis_result
            
        log_pipeline_event(log_session_id, "SynthesisAgent finished.", f"Final Answer: {final_answer}")
        log_pipeline_event(log_session_id, "PIPELINE COMPLETED SUCCESSFULLY")
        
        # Collect comprehensive token usage information
        total_tokens = self._get_total_token_usage()
        agent_breakdown = self._get_agent_token_breakdown()
        
        if ticker_df.empty:
            ticker = ''
        else:
            ticker = ticker_df.iloc[0, 0]

        response_body = {
                "response_body": final_answer,
                "company_id": ticker,
                "wms_tables": wms_tables
            }
        
        result_body = {
            "answer": final_answer,
            "token_usage": total_tokens,
            "agent_token_breakdown": agent_breakdown
        }

        return result_body, response_body

    def _get_total_token_usage(self) -> Dict[str, int]:
        """
        Aggregates token usage from all agents.
        
        Returns:
            Dictionary with total token usage across all agents.
        """
        total_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        for agent in [self.generic_agent, self.entity_agent, self.planning_agent, 
                      self.query_agent, self.search_agent, self.synthesis_agent]:
            agent_usage = agent.get_token_usage()
            total_usage['prompt_tokens'] += agent_usage['prompt_tokens']
            total_usage['completion_tokens'] += agent_usage['completion_tokens']
            total_usage['total_tokens'] += agent_usage['total_tokens']
        
        return total_usage

    def _get_agent_token_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Gets detailed token usage breakdown by agent.
        
        Returns:
            Dictionary with token usage for each agent.
        """
        return {
            'GenericQuestionAgent': self.generic_agent.get_token_usage(),
            'EntityResolutionAgent': self.entity_agent.get_token_usage(),
            'PlanningAgent': self.planning_agent.get_token_usage(),
            'QueryAgent': self.query_agent.get_token_usage(),
            'SearchAgent': self.search_agent.get_token_usage(),
            'SynthesisAgent': self.synthesis_agent.get_token_usage()
        }


if __name__ == "__main__":
    chatbot = ChatbotPipeline()
    
    # Example of a question that benefits from grounding
    q1 = "For Ormat Technologies, did Blackrock's most recent purchase coincide with any significant news?"
    print("\n--- Running Q1 with Grounding ENABLED ---")
    chatbot.run(q1, enable_grounding=True)
    
    print("\n\n" + "*"*80 + "\n")