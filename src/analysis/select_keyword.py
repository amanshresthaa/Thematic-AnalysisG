# File: select_keyword.py
import logging
from typing import List, Dict, Any
import dspy
import json

logger = logging.getLogger(__name__)

class KeywordExtractionSignature(dspy.Signature):
    """
    Signature for extracting keywords from transcript chunks using the 6Rs framework.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives that provide focus for conducting the analysis"
    )
    transcript_chunks: List[str] = dspy.InputField(
        desc="Chunks of transcript from which keywords are to be extracted"
    )
    theoretical_framework: str = dspy.InputField(
        desc="The theoretical and philosophical framework guiding the analysis (optional)"
    )
    keywords: List[str] = dspy.OutputField(
        desc="List of extracted keywords from the transcript chunks"
    )

    def parse_response(self, response: str) -> List[str]:
        """Parses the LM response into a list of keywords."""
        try:
            response_json = json.loads(response)
            keywords = response_json.get("keywords", [])
            return keywords
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []

    def create_prompt(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: str = None) -> str:
        """Creates the prompt for the language model."""
        chunks_formatted = "\n".join([f"Chunk {i+1}:\n{chunk}\n" for i, chunk in enumerate(transcript_chunks)])
        theoretical_part = f"Theoretical Framework:\n{theoretical_framework}\n\n" if theoretical_framework else ""
        prompt = (
            f"You are conducting a thematic analysis using keyword extraction based on the 6Rs framework.\n\n"
            f"Research Objectives:\n{research_objectives}\n\n"
            f"{theoretical_part}"
            f"Transcript Chunks:\n{chunks_formatted}\n"
            f"Task:\n"
            f"Extract meaningful keywords from the transcript chunks using the 6Rs framework, which includes:\n"
            f"1. Realness: Keywords that reflect the genuine experiences and perceptions of the participants.\n"
            f"2. Richness: Keywords that are rich in meaning and provide a detailed understanding of the phenomenon.\n"
            f"3. Repetition: Keywords that frequently occur in the data, indicating their significance.\n"
            f"4. Rationale: Keywords connected to the theoretical or philosophical foundation of the research.\n"
            f"5. Repartee: Keywords that are insightful, evocative, and stimulate further discussion.\n"
            f"6. Regal: Keywords that are central to understanding the phenomenon and contribute significantly to the literature.\n\n"
            f"Return the list of extracted keywords in valid JSON format:\n"
            f"{{\n"
            f"  'keywords': List[str]\n"
            f"}}\n"
        )
        return prompt

    def forward(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: str = None) -> Dict[str, Any]:
        try:
            logger.debug("Starting keyword extraction process.")

            # Generate the prompt
            prompt = self.create_prompt(research_objectives, transcript_chunks, theoretical_framework)

            # Generate response
            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            ).strip()

            # Parse response
            keywords = self.parse_response(response)

            # Apply assertions (basic validation)
            if not keywords:
                error_msg = "No keywords were extracted from the transcript chunks."
                logger.error(error_msg)
                raise AssertionError(error_msg)

            logger.info(f"Successfully extracted {len(keywords)} keywords.")
            return {
                "keywords": keywords
            }

        except AssertionError as af:
            logger.warning(f"Assertion failed during keyword extraction: {af}")
            return self.handle_failed_assertion(af, research_objectives, transcript_chunks, theoretical_framework)
        except Exception as e:
            logger.error(f"Error in KeywordExtractionSignature.forward: {e}", exc_info=True)
            return {
                "keywords": [],
                "error": f"Error occurred during keyword extraction: {str(e)}"
            }

    def handle_failed_assertion(self, assertion_failure: AssertionError,
                                research_objectives: str, transcript_chunks: List[str],
                                theoretical_framework: str = None) -> Dict[str, Any]:
        """Handles failed assertions by attempting to generate improved keywords."""
        try:
            theoretical_part = f"Theoretical Framework:\n{theoretical_framework}\n\n" if theoretical_framework else ""
            focused_prompt = (
                f"The previous attempt failed because: {assertion_failure}\n\n"
                f"Research Objectives:\n{research_objectives}\n\n"
                f"{theoretical_part}"
                "Please ensure the following in your next attempt:\n"
                "Extract meaningful keywords from the transcript chunks using the 6Rs framework.\n\n"
                "Transcript Chunks:\n" +
                "\n".join([f"Chunk {i+1}:\n{chunk}\n" for i, chunk in enumerate(transcript_chunks)])
            )

            response = self.language_model.generate(
                prompt=focused_prompt,
                max_tokens=500,
                temperature=0.5
            ).strip()

            keywords = self.parse_response(response)

            # Re-apply assertions
            if not keywords:
                error_msg = "No keywords were extracted after refinement."
                logger.error(error_msg)
                raise AssertionError(error_msg)

            logger.info("Keywords successfully extracted after refinement.")
            return {
                "keywords": keywords,
                "note": f"Keywords were refined to address: {assertion_failure}"
            }

        except AssertionError as af_inner:
            logger.error(f"Refined keywords still failed assertions: {af_inner}")
            return {
                "keywords": [],
                "error": f"Failed to extract appropriate keywords after refinement: {str(af_inner)}"
            }
        except Exception as e:
            logger.error(f"Error in handle_failed_assertion: {e}", exc_info=True)
            return {
                "keywords": [],
                "error": f"Failed to extract appropriate keywords: {str(assertion_failure)}"
            }
