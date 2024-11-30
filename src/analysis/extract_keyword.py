import logging
import re
import json
from typing import List, Dict, Any

import dspy

from src.assertions import (
    assert_keyword_realness,
    assert_keyword_richness,
    assert_keyword_repetition,
    assert_keyword_rationale,
    assert_keyword_repartee,
    assert_keyword_regal
)

logger = logging.getLogger(__name__)

class KeywordExtractionSignature(dspy.Signature):
    """
    A signature for extracting and analyzing keywords from qualitative data
    following the 6Rs framework (Realness, Richness, Repetition, Rationale, Repartee, Regal).
    """
    
    # Input Fields
    research_objectives: str = dspy.InputField(
        desc="Research goals and questions guiding the keyword analysis"
    )
    
    transcript_chunk: str = dspy.InputField(
        desc="Primary transcript segment for keyword extraction"
    )
    
    contextualized_contents: List[str] = dspy.InputField(
        desc="Additional contextual content to support interpretation"
    )
    
    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc="""Theoretical foundation including:
        - theory: Primary theoretical approach
        - philosophical_approach: Underlying philosophical foundation
        - rationale: Justification for chosen approach"""
    )
    
    # Output Fields
    keywords: List[Dict[str, Any]] = dspy.OutputField(
        desc="""Extracted keywords with detailed analysis:
        - keyword: The extracted keyword or phrase
        - category: Which of the 6Rs this keyword represents
        - context:
            * surrounding_text: Text surrounding the keyword
            * usage_pattern: How the keyword is used
        - analysis_value:
            * realness: How it reflects genuine experiences
            * richness: Depth of meaning provided
            * repetition: Frequency and pattern of use
            * rationale: Connection to theoretical framework
            * repartee: Insightfulness and discussion potential
            * regal: Centrality to phenomenon understanding"""
    )
    
    analysis: Dict[str, Any] = dspy.OutputField(
        desc="""Comprehensive keyword analysis including:
        - methodology_reflection: Analysis approach used
        - patterns_identified: Key patterns in keywords
        - theoretical_alignment: Framework connection
        - research_implications: Research impact
        - practical_implications: Applied insights"""
    )

    def create_prompt(self, research_objectives: str, transcript_chunk: str,
                     contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> str:
        """Creates the prompt for the language model."""
        
        # Format contextualized contents
        chunks_formatted = "\n\n".join([
            f"Content {i+1}:\n{content}" 
            for i, content in enumerate([transcript_chunk] + contextualized_contents)
        ])
        
        prompt = (
            f"You are conducting keyword extraction and analysis using the 6Rs framework "
            f"(Realness, Richness, Repetition, Rationale, Repartee, Regal). Your task is to "
            f"identify and analyze keywords from the provided transcript chunks.\n\n"
            
            f"Research Objectives:\n{research_objectives}\n\n"
            f"Transcript Content:\n{chunks_formatted}\n\n"
            f"Theoretical Framework:\n{json.dumps(theoretical_framework, indent=2)}\n\n"
            
            f"Please analyze the text following these criteria:\n\n"
            f"1. Realness: Words reflecting genuine experiences\n"
            f"2. Richness: Words providing detailed understanding\n"
            f"3. Repetition: Frequently occurring significant words\n"
            f"4. Rationale: Words connected to theoretical framework\n"
            f"5. Repartee: Insightful, evocative words\n"
            f"6. Regal: Central, phenomenon-defining words\n\n"
            
            f"Format your response as JSON with this structure:\n"
            "{\n"
            "  \"keywords\": [\n"
            "    {\n"
            "      \"keyword\": \"example_word\",\n"
            "      \"category\": \"realness\",\n"
            "      \"context\": {\n"
            "        \"surrounding_text\": \"\",\n"
            "        \"usage_pattern\": \"\"\n"
            "      },\n"
            "      \"analysis_value\": {\n"
            "        \"realness\": \"\",\n"
            "        \"richness\": \"\",\n"
            "        \"repetition\": \"\",\n"
            "        \"rationale\": \"\",\n"
            "        \"repartee\": \"\",\n"
            "        \"regal\": \"\"\n"
            "      }\n"
            "    }\n"
            "  ],\n"
            "  \"analysis\": {\n"
            "    \"methodology_reflection\": \"\",\n"
            "    \"patterns_identified\": [],\n"
            "    \"theoretical_alignment\": \"\",\n"
            "    \"research_implications\": \"\",\n"
            "    \"practical_implications\": \"\"\n"
            "  }\n"
            "}\n"
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parses the complete response from the language model."""
        try:
            # Extract JSON content
            json_match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
            if not json_match:
                logger.error("No valid JSON found in response")
                return {}
            
            json_string = json_match.group(1)
            return json.loads(json_string)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}

    def forward(self, research_objectives: str, transcript_chunk: str,
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Executes the keyword extraction and analysis process."""
        try:
            # Generate prompt
            prompt = self.create_prompt(
                research_objectives,
                transcript_chunk,
                contextualized_contents,
                theoretical_framework
            )
            
            # Generate response
            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7
            ).strip()
            
            # Parse response
            parsed_response = self.parse_response(response)
            
            if not parsed_response:
                raise ValueError("Failed to parse response")
            
            # Extract components
            keywords = parsed_response.get("keywords", [])
            analysis = parsed_response.get("analysis", {})
            
            # Apply assertions
            assert_keyword_realness(keywords)
            assert_keyword_richness(keywords)
            assert_keyword_repetition(keywords)
            assert_keyword_rationale(keywords, theoretical_framework)
            assert_keyword_repartee(keywords)
            assert_keyword_regal(keywords)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error in KeywordExtractionSignature.forward: {e}")
            return {}