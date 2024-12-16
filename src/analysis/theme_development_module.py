# src/analysis/theme_development_module.py

import logging
from typing import Dict, Any, List
import dspy
from .theme_development import ThemedevelopmentAnalysisSignature

logger = logging.getLogger(__name__)

class ThemedevelopmentAnalysisModule(dspy.Module):
    """
    DSPy module for developing and refining themes from previously derived codes.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.ChainOfThought(ThemedevelopmentAnalysisSignature)  # Updated to use ChainOfThought

    def forward(self,
                research_objectives: str,
                quotation: str,
                keywords: List[str],
                codes: List[Dict[str, Any]],
                theoretical_framework: Dict[str, str],
                transcript_chunk: str) -> Dict[str, Any]:
        """
        Execute theme development analysis to transform codes into higher-order themes.

        Args:
            research_objectives (str): The research aims and guiding questions.
            quotation (str): The original excerpt or passage analyzed.
            keywords (List[str]): The extracted keywords from the quotation.
            codes (List[Dict[str, Any]]): The previously developed and validated codes.
            theoretical_framework (Dict[str, str]): The theoretical foundation and rationale.
            transcript_chunk (str): The contextual transcript segment associated with the quotation.

        Returns:
            Dict[str, Any]: Thematic analysis results including identified themes and their analysis.
        """
        logger.info("Starting theme development analysis.")
        logger.debug(f"Inputs: Research Objectives='{research_objectives[:100]}', Quotation='{quotation[:100]}', "
                     f"Keywords={keywords}, Codes={len(codes)}, Theoretical Framework={theoretical_framework}, "
                     f"Transcript Chunk Length={len(transcript_chunk)}")

        try:
            response = self.chain(
                research_objectives=research_objectives,
                quotation=quotation,
                keywords=keywords,
                codes=codes,
                theoretical_framework=theoretical_framework,
                transcript_chunk=transcript_chunk
            )

            logger.info("Successfully completed theme development analysis.")
            logger.debug(f"Response: {response}")

            if not response.get("themes"):
                logger.warning("No themes were generated. Possible issue with inputs or prompt formulation.")
            
            return response
        except Exception as e:
            logger.error(f"Error during theme development analysis: {e}", exc_info=True)
            return {}
