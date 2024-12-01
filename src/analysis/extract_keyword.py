import logging
from typing import List, Dict, Any
import dspy
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class KeywordAnalysisValue:
    """Analysis values for each of the 6Rs framework dimensions."""
    realness: str
    richness: str
    repetition: str
    rationale: str
    repartee: str
    regal: str

class KeywordExtractionSignature(dspy.Signature):
    """
    A comprehensive signature for conducting thematic keyword extraction from interview transcripts,
    following advanced qualitative methodologies. This signature supports systematic keyword analysis
    with robust pattern recognition and theoretical integration, aligned with the 6Rs framework.
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
    transcript_info: Dict[str, Any] = dspy.OutputField(
        desc="""Comprehensive context information including:
        - transcript_chunk: Selected content
        - research_objectives: Analysis goals
        - theoretical_framework: Complete framework details"""
    )

    retrieved_chunks: List[str] = dspy.OutputField(
        desc="Collection of relevant transcript segments retrieved for analysis"
    )

    retrieved_chunks_count: int = dspy.OutputField(
        desc="Number of transcript chunks retrieved and analyzed"
    )

    used_chunk_ids: List[str] = dspy.OutputField(
        desc="Identifiers of transcript chunks utilized in the analysis"
    )

    keywords: List[Dict[str, Any]] = dspy.OutputField(
        desc="""Extracted keywords with detailed analysis:
        - keyword: The keyword text
        - category: Classification of the keyword
        - context:
            * surrounding_text: Text surrounding the keyword
            * usage_pattern: How the keyword is used across the data
            * source_location: Where the keyword was found
        - analysis_value:
            * realness: Reflection of genuine experiences
            * richness: Depth of meaning
            * repetition: Recurrence in data
            * rationale: Connection to theoretical foundations
            * repartee: Contribution to discussions
            * regal: Centrality to the topic"""
    )

    analysis: Dict[str, Any] = dspy.OutputField(
        desc="""Comprehensive keyword analysis including:
        - patterns_identified: Key patterns discovered
        - theoretical_interpretation: Framework application
        - methodological_reflection:
            * pattern_robustness
            * theoretical_alignment
            * researcher_reflexivity
        - practical_implications: Applied insights"""
    )

    answer: Dict[str, Any] = dspy.OutputField(
        desc="""Analysis synthesis and contributions:
        - summary: Key findings
        - theoretical_contribution: Theory advancement
        - methodological_contribution:
            * approach
            * pattern_validity
            * theoretical_integration"""
    )

    def create_prompt(self, research_objectives: str, transcript_chunk: str,
                      contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> str:
        """Creates the prompt for enhanced keyword extraction."""

        chunks_formatted = "\n\n".join([
            f"Content {i+1}:\n{content}"
            for i, content in enumerate([transcript_chunk] + contextualized_contents)
        ])

        theory = theoretical_framework.get("theory", "")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "")
        rationale = theoretical_framework.get("rationale", "")

        prompt = (
            f"You are an experienced qualitative researcher conducting keyword analysis using "
            f"an advanced framework that integrates the 6Rs methodology. Your task is to identify "
            f"and analyze meaningful keywords while maintaining methodological rigor.\n\n"

            f"Review the following content:\n\n"
            f"{chunks_formatted}\n\n"

            f"Research Objectives:\n"
            f"{research_objectives}\n\n"

            f"Theoretical Framework:\n"
            f"Theory: {theory}\n"
            f"Philosophical Approach: {philosophical_approach}\n"
            f"Rationale: {rationale}\n\n"

            f"Your analysis should follow the 6Rs framework:\n\n"
            f"1. Realness: Select words reflecting genuine experiences.\n"
            f"2. Richness: Identify words with deep meaning.\n"
            f"3. Repetition: Note recurring patterns.\n"
            f"4. Rationale: Connect to theoretical foundations.\n"
            f"5. Repartee: Consider discussion value.\n"
            f"6. Regal: Focus on centrality to topic.\n\n"

            f"Provide a comprehensive analysis including pattern identification, theoretical interpretation, "
            f"methodological reflection, and practical implications. Reflect on your researcher's reflexivity "
            f"and the robustness of the patterns identified.\n\n"

            f"Expected JSON structure:\n"
            "{\n"
            '  "transcript_info": {\n'
            '    "transcript_chunk": "...",\n'
            '    "research_objectives": "...",\n'
            '    "theoretical_framework": {\n'
            '      "theory": "...",\n'
            '      "philosophical_approach": "...",\n'
            '      "rationale": "..." \n'
            '    }\n'
            '  },\n'
            '  "retrieved_chunks": ["..."],\n'
            '  "retrieved_chunks_count": 0,\n'
            '  "used_chunk_ids": ["..."],\n'
            '  "keywords": [{\n'
            '    "keyword": "...",\n'
            '    "category": "...",\n'
            '    "context": {\n'
            '      "surrounding_text": "...",\n'
            '      "usage_pattern": "...",\n'
            '      "source_location": "..." \n'
            '    },\n'
            '    "analysis_value": {\n'
            '      "realness": "...",\n'
            '      "richness": "...",\n'
            '      "repetition": "...",\n'
            '      "rationale": "...",\n'
            '      "repartee": "...",\n'
            '      "regal": "..." \n'
            '    }\n'
            '  }],\n'
            '  "analysis": {\n'
            '    "patterns_identified": {\n'
            '      "...": "..." \n'
            '    },\n'
            '    "theoretical_interpretation": {\n'
            '      "...": "..." \n'
            '    },\n'
            '    "methodological_reflection": {\n'
            '      "pattern_robustness": "...",\n'
            '      "theoretical_alignment": "...",\n'
            '      "researcher_reflexivity": "..." \n'
            '    },\n'
            '    "practical_implications": "..." \n'
            '  },\n'
            '  "answer": {\n'
            '    "summary": "...",\n'
            '    "theoretical_contribution": "...",\n'
            '    "methodological_contribution": {\n'
            '      "approach": "...",\n'
            '      "pattern_validity": "...",\n'
            '      "theoretical_integration": "..." \n'
            '    }\n'
            '  }\n'
            "}\n"
        )
        return prompt


    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parses LLM response with error handling."""
        try:
            import re
            import json
            
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                logger.error("No valid JSON found in response")
                return {}
                
            parsed = json.loads(json_match.group(0))
            return parsed
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}

    def forward(self, research_objectives: str, transcript_chunk: str,
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Executes keyword extraction with retry mechanism."""
        for attempt in range(3):  # Same retry mechanism as QuotationSelection
            try:
                logger.debug(f"Attempt {attempt + 1} - Starting keyword extraction")
                
                prompt = self.create_prompt(
                    research_objectives,
                    transcript_chunk,
                    contextualized_contents,
                    theoretical_framework
                )
                
                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=3000,
                    temperature=0.7
                ).strip()
                
                parsed_response = self.parse_response(response)
                
                if not parsed_response:
                    raise ValueError("Failed to parse response")
                
                # Extract components for validation
                keywords = parsed_response.get("keywords", [])
                analysis = parsed_response.get("analysis", {})
                
                # Validate using 6Rs framework
                self._validate_keywords(keywords)
                self._validate_analysis(analysis)
                
                logger.info(f"Successfully extracted {len(keywords)} keywords")
                return parsed_response

            except AssertionError as ae:
                logger.warning(f"Attempt {attempt + 1} - Assertion failed: {ae}")
                parsed_response = self.handle_failed_assertion(
                    ae, research_objectives, transcript_chunk,
                    contextualized_contents, theoretical_framework
                )
                if parsed_response:
                    return parsed_response
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error: {e}", exc_info=True)
                
        logger.error("Failed to generate valid output after multiple attempts")
        return {}

    def handle_failed_assertion(self, assertion_failure: AssertionError,
                              research_objectives: str, transcript_chunk: str,
                              contextualized_contents: List[str],
                              theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Handles failed assertions by refining the prompt."""
        try:
            logger.debug("Handling failed assertion")
            
            focused_prompt = self.create_prompt(
                research_objectives,
                transcript_chunk,
                contextualized_contents,
                theoretical_framework
            )
            
            focused_prompt += (
                f"\n\nPrevious attempt failed: {assertion_failure}\n"
                f"Please ensure your analysis addresses this specific issue while "
                f"maintaining the 6Rs framework requirements."
            )

            response = self.language_model.generate(
                prompt=focused_prompt,
                max_tokens=3000,
                temperature=0.5
            ).strip()

            parsed_response = self.parse_response(response)
            
            if not parsed_response:
                raise ValueError("Failed to parse refined response")
            
            keywords = parsed_response.get("keywords", [])
            analysis = parsed_response.get("analysis", {})
            
            self._validate_keywords(keywords)
            self._validate_analysis(analysis)
            
            return parsed_response

        except Exception as e:
            logger.error(f"Error in handle_failed_assertion: {e}", exc_info=True)
            return {}

    def _validate_keywords(self, keywords: List[Dict[str, Any]]) -> None:
        """Validates keywords against 6Rs framework."""
        if not keywords:
            raise AssertionError("No keywords extracted")
            
        for keyword in keywords:
            analysis_value = keyword.get("analysis_value", {})
            
            # Check all 6Rs dimensions
            for dimension in ["realness", "richness", "repetition", "rationale", "repartee", "regal"]:
                if not analysis_value.get(dimension):
                    raise AssertionError(f"Missing {dimension} analysis for keyword: {keyword.get('keyword')}")

    def _validate_analysis(self, analysis: Dict[str, Any]) -> None:
        """Validates overall analysis structure."""
        required_sections = [
            "patterns_identified",
            "theoretical_interpretation",
            "methodological_reflection"
        ]
        
        for section in required_sections:
            if not analysis.get(section):
                raise AssertionError(f"Missing required analysis section: {section}")

class EnhancedKeywordExtractionModule(dspy.Module):
    """DSPy module implementing enhanced keyword extraction."""
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)

    def forward(self, research_objectives: str, transcript_chunk: str,
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        try:
            logger.debug("Running EnhancedKeywordExtractionModule")
            response = self.chain(
                research_objectives=research_objectives,
                transcript_chunk=transcript_chunk,
                contextualized_contents=contextualized_contents,
                theoretical_framework=theoretical_framework
            )
            return response
        except Exception as e:
            logger.error(f"Error in EnhancedKeywordExtractionModule.forward: {e}", exc_info=True)
            return {}