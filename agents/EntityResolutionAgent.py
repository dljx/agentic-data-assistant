# agents/EntityResolutionAgent.py

import json
import re
from typing import Any, Dict, List, Optional, Set
from google.genai.types import GenerateContentConfig # Ensure this is imported if used for temp
from agents.core import Agent
from utilities import PROMPTS, format_prompt


class EntityResolutionAgent(Agent):
    """
    An agent for parsing user questions and resolving entities against a known list.
    This version uses the LLM to resolve ambiguity and then uses multiple similarity
    algorithms to ground the LLM's suggestion against the database.
    Includes enhanced logging and improved text normalization.
    """
    agentType: str = "EntityResolutionAgent"

    def __init__(self, model_id: str = 'gemini-2.5-flash'):
        super().__init__()
        self.model_id = model_id
        # Set temperature to 0 for deterministic JSON output
        self.generation_config = GenerateContentConfig(temperature=0.0)
        
        # Corporate suffixes for normalization
        self.corporate_suffixes = [
            'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 
            'llc', 'llp', 'lp', 'co', 'company', 'group', 'holdings', 
            'plc', 'sa', 'nv', 'gmbh', 'ag', 'se', 'spa'
        ]

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity names for better matching by handling common variations.
        """
        if not name:
            return ""
            
        # Convert to lowercase and strip whitespace
        normalized = name.lower().strip()
        
        # Remove common punctuation and special characters
        normalized = re.sub(r'[,\.;:\-\(\)\[\]"\'`]', ' ', normalized)
        
        # Remove possessive markers
        normalized = re.sub(r"['']s\b", '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Handle corporate suffixes - normalize variations
        for suffix in self.corporate_suffixes:
            # Handle suffix with period (e.g., "Inc." -> "Inc")
            pattern = rf'\b{re.escape(suffix)}\.?\b'
            normalized = re.sub(pattern, suffix, normalized)
        
        # Handle common abbreviations
        abbreviations = {
            'intl': 'international',
            'natl': 'national', 
            'mgmt': 'management',
            'tech': 'technology',
            'sys': 'systems',
            'svcs': 'services',
            'fin': 'financial'
        }
        
        for abbrev, full_form in abbreviations.items():
            normalized = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, normalized)
            normalized = re.sub(rf'\b{re.escape(full_form)}\b', abbrev, normalized)
        
        return normalized

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein (edit) distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity score (0-1) based on Levenshtein distance."""
        if not s1 or not s2:
            return 0.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = self._levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)

    def generate_llm_response(self, prompt: str) -> str:
        """
        Uses the agent's client to make a simple, one-shot call to the LLM.
        """
        try:
            # --- ADDED LOGGING ---
            print("    -> EntityResolutionAgent: Sending prompt to LLM for parsing:")
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt],
                config=self.generation_config # Apply deterministic config
            )
            
            # Extract token usage from the response
            self._extract_token_usage(response)
            
            # --- ADDED LOGGING ---
            print(f"    -> EntityResolutionAgent: Received raw LLM response.")
            return response.text
        except Exception as e:
            print(f"    -> Error during EntityResolutionAgent LLM call: {e}")
            return f"Error: {e}"

    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculates Jaccard similarity between two sets of words."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _calculate_combined_similarity(self, query: str, candidate: str) -> float:
        """
        Calculate a combined similarity score using multiple algorithms.
        """
        query_norm = self._normalize_entity_name(query)
        candidate_norm = self._normalize_entity_name(candidate)
        
        # 1. Exact match (highest weight)
        if query_norm == candidate_norm:
            return 1.0
        
        # 2. Check for exact word matches (NEW - prioritize exact word overlap)
        query_words = set(query_norm.split())
        candidate_words = set(candidate_norm.split())
        
        # Remove common corporate words to avoid false matches
        common_corporate_words = {'inc', 'corporation', 'corp', 'company', 'co', 'ltd', 'llc', 'group', 'holdings'}
        
        # Get meaningful words (excluding generic corporate terms)
        query_meaningful = query_words - common_corporate_words
        candidate_meaningful = candidate_words - common_corporate_words
        
        # If we have meaningful word overlap, prioritize it heavily
        if query_meaningful and candidate_meaningful:
            meaningful_jaccard = self._jaccard_similarity(query_meaningful, candidate_meaningful)
            if meaningful_jaccard >= 0.7:  # High overlap of meaningful words
                return 0.95 + (meaningful_jaccard * 0.05)  # Score between 0.95-1.0
        
        # 3. Substring containment (high weight) - but be more careful
        substring_score = 0.0
        if query_norm in candidate_norm or candidate_norm in query_norm:
            # Make sure it's not just matching generic corporate terms
            shorter = query_norm if len(query_norm) < len(candidate_norm) else candidate_norm
            longer = candidate_norm if len(query_norm) < len(candidate_norm) else query_norm
            
            # If the shorter string is mostly corporate words, penalize
            shorter_words = set(shorter.split())
            if len(shorter_words - common_corporate_words) >= 1:  # At least one meaningful word
                length_ratio = min(len(query_norm), len(candidate_norm)) / max(len(query_norm), len(candidate_norm))
                substring_score = 0.8 * length_ratio
        
        # 4. Jaccard similarity on all words
        jaccard_score = self._jaccard_similarity(query_words, candidate_words)
        
        # 5. Levenshtein similarity on full strings
        levenshtein_score = self._levenshtein_similarity(query_norm, candidate_norm)
        
        # 6. Weighted combination - prioritize meaningful word matches
        if query_meaningful and candidate_meaningful:
            meaningful_overlap = len(query_meaningful.intersection(candidate_meaningful))
            if meaningful_overlap > 0:
                # Boost score significantly for meaningful word matches
                combined_score = max(
                    substring_score,
                    0.7 * meaningful_jaccard + 0.3 * levenshtein_score + 0.1 * (meaningful_overlap / max(len(query_meaningful), len(candidate_meaningful)))
                )
            else:
                # No meaningful overlap - heavily penalize
                combined_score = max(0.1, 0.3 * jaccard_score + 0.2 * levenshtein_score)
        else:
            # Fallback to original logic for edge cases
            combined_score = max(
                substring_score,
                0.6 * jaccard_score + 0.4 * levenshtein_score
            )
        
        return combined_score

    def _find_best_match_from_llm_output(self, llm_entity: Optional[str], canonical_names: List[str], entity_type: str) -> Optional[str]:
        """
        Takes the entity name suggested by the LLM and finds the best fuzzy match
        in our database list using multiple similarity algorithms.
        """
        print(f"    -> EntityResolutionAgent: Attempting to match LLM's suggested {entity_type}: '{llm_entity}'")
        if not llm_entity:
            print(f"       - LLM provided no suggestion for {entity_type}.")
            return None

        # Calculate similarity scores for all candidates
        candidates_with_scores = []
        for name in canonical_names:
            if not name:
                continue
            
            score = self._calculate_combined_similarity(llm_entity, name)
            if score > 0:  # Only consider candidates with some similarity
                candidates_with_scores.append((name, score))
        
        # Sort by score (highest first)
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log top candidates for debugging
        print(f"       - Top matching candidates:")
        for name, score in candidates_with_scores[:5]:  # Show top 5
            print(f"         '{name}': {score:.3f}")
        
        # Use a lower threshold for the improved algorithm
        confidence_threshold = 0.25  # Lowered from 0.4
        
        if candidates_with_scores and candidates_with_scores[0][1] >= confidence_threshold:
            best_match, best_score = candidates_with_scores[0]
            print(f"       - Selected best match: '{llm_entity}' -> '{best_match}' (score: {best_score:.3f})")
            return best_match
        
        best_score = candidates_with_scores[0][1] if candidates_with_scores else 0.0
        print(f"       - WARNING: No confident match found in database for LLM's suggested {entity_type} '{llm_entity}' (best score: {best_score:.3f}).")
        return None

    def resolve(self, user_question: str, all_company_names: List[str], all_investor_names: List[str], conversation_history: str = None) -> Dict[str, Any]:
        """
        Parses the user question, using the LLM to resolve ambiguous entities first,
        then grounds the suggestions against the database.
        
        Args:
            user_question: The current user question
            all_company_names: List of all company names from database
            all_investor_names: List of all investor names from database
            conversation_history: Previous conversation context
        """
        print("    -> EntityResolutionAgent: Starting resolution process...")
        try:
            prompt_template = PROMPTS['entity_resolution_prompt']
        except KeyError:
            return {"error": "Could not find 'entity_resolution_prompt' in prompts.yaml"}

        prompt = format_prompt(
            prompt_template,
            user_question=user_question,
            conversation_history=conversation_history or "This is the start of a new conversation."
        )
        
        llm_response_str = self.generate_llm_response(prompt)

        try:
            clean_response = llm_response_str.strip().replace("```json", "").replace("```", "")
            # print(f"    -> EntityResolutionAgent: Cleaned LLM JSON response: {clean_response[:300]}...")
            parsed_json_from_llm = json.loads(clean_response)
            
            suggested_company = parsed_json_from_llm.get("suggested_company_name")
            suggested_investor = parsed_json_from_llm.get("suggested_investor_name")
            
            print(f"    -> EntityResolutionAgent: LLM suggested company: '{suggested_company}'")
            print(f"    -> EntityResolutionAgent: LLM suggested investor: '{suggested_investor}'")

            final_company = self._find_best_match_from_llm_output(suggested_company, all_company_names, "company")
            final_investor = self._find_best_match_from_llm_output(suggested_investor, all_investor_names, "investor")

            # Construct the final output, preserving other metrics from the LLM.
            final_output = parsed_json_from_llm.copy() # Start with LLM's parsed structure
            final_output['original_question'] = user_question
            final_output['resolved_company_name'] = final_company
            final_output['resolved_investor_name'] = final_investor
            
            final_output.pop("suggested_company_name", None)
            final_output.pop("suggested_investor_name", None)
            
            print(f"    -> EntityResolutionAgent: Final resolved entities: Company='{final_company}', Investor='{final_investor}'")
            return final_output, final_company
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"    -> [!] EntityResolutionAgent failed to parse LLM response. Error: {e}")
            print(f"        Raw LLM response was: {llm_response_str}")
            return {
                "resolved_company_name": None,
                "resolved_investor_name": None,
                "original_question": user_question,
                "error": "Failed to parse a structured response from the language model."
            }