import logging
from typing import Dict, Any, List, Optional
import dspy
from dataclasses import dataclass
import json
import asyncio

from src.assertions_keyword import validate_keywords_dspy, AssertionConfig
from src.config.keyword_config import KeywordExtractionConfig

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
    """Signature for conducting thematic keyword extraction from quotations."""
    research_objectives: str = dspy.InputField(
        desc="Research goals and questions guiding the keyword analysis"
    )
    quotation: str = dspy.InputField(
        desc="Selected quotation for keyword extraction"
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

    quotation_info: Dict[str, Any] = dspy.OutputField(
        desc="Comprehensive context information"
    )
    keywords: List[Dict[str, Any]] = dspy.OutputField(
        desc="Extracted keywords with detailed analysis"
    )
    analysis: Dict[str, Any] = dspy.OutputField(
        desc="Comprehensive keyword analysis output"
    )

class KeywordExtractionModule(dspy.Module):
    """Enhanced DSPy module for keyword extraction with configurable validation."""
    
    def __init__(self, config: Optional[KeywordExtractionConfig] = None):
        """Initialize the module with optional configuration."""
        super().__init__()
        self.config = config or KeywordExtractionConfig()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)
        logger.info("Initialized KeywordExtractionModule with config: %s", self.config)

    def create_prompt(self, research_objectives: str, quotation: str,
                     contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> str:
        """Creates the enhanced prompt for keyword extraction."""
        logger.debug("Creating prompt for keyword extraction")
        
        context_formatted = "\n\n".join([
            f"Context {i+1}:\n{content}"
            for i, content in enumerate(contextualized_contents)
        ])

        theory = theoretical_framework.get("theory", "")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "")
        rationale = theoretical_framework.get("rationale", "")

        prompt = (
            f"As an experienced qualitative researcher, analyze the following quotation "
            f"using the 6Rs framework to extract meaningful keywords.\n\n"

            f"Quotation:\n{quotation}\n\n"
            f"Additional Context:\n{context_formatted}\n\n"
            f"Research Objectives:\n{research_objectives}\n\n"

            f"Theoretical Framework:\n"
            f"Theory: {theory}\n"
            f"Philosophical Approach: {philosophical_approach}\n"
            f"Rationale: {rationale}\n\n"

            f"6Rs Framework Guidelines:\n"
            f"1. Realness: Select words reflecting genuine experiences\n"
            f"2. Richness: Identify words with deep meaning\n"
            f"3. Repetition: Note recurring patterns\n"
            f"4. Rationale: Connect to theoretical foundations\n"
            f"5. Repartee: Consider discussion value\n"
            f"6. Regal: Focus on centrality to topic\n\n"

            f"Requirements:\n"
            "1. Each keyword must be analyzed across all 6Rs dimensions\n"
            "2. Include theoretical alignment justification\n"
            "3. Consider contextual significance\n"
            "4. Note pattern frequency\n"
            f"5. Maximum keywords: {self.config.max_keywords}\n"
            f"6. Minimum confidence: {self.config.min_confidence}\n\n"

            "Expected Output Format:\n"
            "{\n"
            '  "quotation_info": {\n'
            '    "quotation": "...",\n'
            '    "research_objectives": "...",\n'
            '    "theoretical_framework": {...}\n'
            "  },\n"
            '  "keywords": [\n'
            "    {\n"
            '      "keyword": "...",\n'
            '      "category": "...",\n'
            '      "6Rs_framework": ["..."],\n'
            '      "analysis_value": {\n'
            '        "realness": "...",\n'
            '        "richness": "...",\n'
            '        "repetition": "...",\n'
            '        "rationale": "...",\n'
            '        "repartee": "...",\n'
            '        "regal": "..."\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "analysis": {\n'
            '    "patterns_identified": [...],\n'
            '    "theoretical_interpretation": "...",\n'
            '    "methodological_reflection": {...},\n'
            '    "practical_implications": "..."\n'
            "  }\n"
            "}\n"
        )
        
        logger.debug("Created prompt with length: %d characters", len(prompt))
        return prompt

    async def forward(self, 
                     research_objectives: str, 
                     quotation: str,
                     contextualized_contents: List[str], 
                     theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Execute keyword extraction with validation and retry mechanism."""
        attempt = 0
        max_retries = self.config.max_retries
        
        while attempt < max_retries:
            attempt += 1
            logger.info("Attempt %d/%d - Starting keyword extraction", attempt, max_retries)
            
            try:
                prompt = self.create_prompt(
                    research_objectives,
                    quotation,
                    contextualized_contents,
                    theoretical_framework
                )

                start_time = asyncio.get_event_loop().time()
                response = await self.chain(
                    research_objectives=research_objectives,
                    quotation=quotation,
                    contextualized_contents=contextualized_contents,
                    theoretical_framework=theoretical_framework
                )
                generation_time = asyncio.get_event_loop().time() - start_time
                logger.debug("LLM response generated in %.2f seconds", generation_time)

                # Extract keywords and validate
                keywords = response.keywords if hasattr(response, 'keywords') else []
                if not keywords:
                    raise ValueError("No keywords found in response")

                logger.info("Extracted %d keywords", len(keywords))

                # Validate using new assertion system
                validation_result = validate_keywords_dspy(
                    keywords=keywords,
                    quotation=quotation,
                    contextualized_contents=contextualized_contents,
                    research_objectives=research_objectives,
                    theoretical_framework=theoretical_framework,
                    config=self.config.assertion_config
                )

                if not validation_result["passed"]:
                    if self.config.strict_mode:
                        raise ValueError(
                            f"Keyword validation failed: {validation_result['failed_assertions']}"
                        )
                    logger.warning(
                        "Keyword validation produced warnings: %s", 
                        validation_result["warnings"]
                    )

                # Apply configuration limits
                if len(keywords) > self.config.max_keywords:
                    logger.warning(
                        "Truncating keywords to maximum limit of %d", 
                        self.config.max_keywords
                    )
                    keywords = keywords[:self.config.max_keywords]

                # Prepare final response
                final_response = {
                    "quotation_info": {
                        "quotation": quotation,
                        "research_objectives": research_objectives,
                        "theoretical_framework": theoretical_framework
                    },
                    "keywords": keywords,
                    "analysis": response.analysis if hasattr(response, 'analysis') else {},
                    "validation_report": validation_result
                }

                logger.info("Successfully extracted and validated keywords on attempt %d", attempt)
                return final_response

            except Exception as e:
                logger.error("Error in attempt %d: %s", attempt, str(e))
                if attempt == max_retries:
                    logger.error("Max retries reached. Extraction failed.")
                    return {
                        "error": str(e),
                        "keywords": [],
                        "quotation_info": {
                            "quotation": quotation,
                            "research_objectives": research_objectives,
                            "theoretical_framework": theoretical_framework
                        },
                        "analysis": {
                            "error": "Failed to complete keyword extraction after maximum retries"
                        }
                    }
                logger.info("Retrying keyword extraction...")
                await asyncio.sleep(1)  # Brief delay before retry

        return {}