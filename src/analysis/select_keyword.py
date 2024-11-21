# src/analysis/select_keyword.py
import logging
from typing import List, Dict, Any
import dspy
import json

from src.assertions_keyword import (
    assert_keywords_extracted,
    assert_realness,
    assert_keywords_not_exclusive_to_context,
    assert_richness,
    assert_repetition,
    assert_rationale,
    assert_repartee,
    assert_regal
)

logger = logging.getLogger(__name__)

class KeywordExtractionSignature(dspy.Signature):
    """
    Signature for extracting keywords from individual quotations using the 6Rs framework.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives that provide focus for conducting the analysis"
    )
    quotation: str = dspy.InputField(
        desc="The specific quotation from which to extract keywords"
    )
    contextual_info: List[str] = dspy.InputField(
        desc="List of contextualized content providing background for the quotation"
    )
    theoretical_framework: str = dspy.InputField(
        desc="The theoretical and philosophical framework guiding the analysis (optional)"
    )
    keywords: List[Dict[str, str]] = dspy.OutputField(
        desc="List of extracted keywords with their type and context"
    )

    def parse_response(self, response: str) -> List[Dict[str, str]]:
        """Parses the LM response into a list of keyword objects."""
        try:
            response_json = json.loads(response)
            keywords = response_json.get("keywords", [])
            # Ensure each keyword has 'keywords_to_use_coding', 'type', and 'context'
            validated_keywords = []
            for kw in keywords:
                if all(key in kw for key in ['keywords_to_use_coding', 'type', 'context']):
                    validated_keywords.append({
                        "keywords_to_use_coding": kw['keywords_to_use_coding'],
                        "type": kw['type'],
                        "context": kw['context']
                    })
                else:
                    logger.warning(f"Keyword entry missing fields: {kw}")
            return validated_keywords
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []

    def create_prompt(self, research_objectives: str, quotation: str, contextual_info: List[str], theoretical_framework: str = None) -> str:
        """Creates the prompt for the language model."""
        contextual_text = "\n".join(contextual_info) if contextual_info else ""
        theoretical_part = f"Theoretical Framework:\n{theoretical_framework}\n\n" if theoretical_framework else ""
        prompt = (
            f"You are conducting a thematic analysis using keyword extraction based on the 6Rs framework.\n\n"
            f"Research Objectives:\n{research_objectives}\n\n"
            f"{theoretical_part}"
            f"Quotation:\n{quotation}\n\n"
            f"Contextual Information:\n{contextual_text}\n\n"
            f"Task:\n"
            f"**Extract meaningful keywords exclusively from the Quotation provided above using the 6Rs framework.**\n"
            f"Use the Contextual Information only to better understand the Quotation.\n\n"
            f"Ensure that the keywords satisfy the following criteria:\n"
            f"1. Realness: Keywords that reflect the genuine experiences and perceptions of the participants.\n"
            f"2. Richness: Keywords that are rich in meaning and provide a detailed understanding of the phenomenon.\n"
            f"3. Repetition: Keywords that frequently occur in the data, indicating their significance.\n"
            f"4. Rationale: Keywords connected to the theoretical or philosophical foundation of the research.\n"
            f"5. Repartee: Keywords that are insightful, evocative, and stimulate further discussion.\n"
            f"6. Regal: Keywords that are central to understanding the phenomenon and contribute significantly to the literature.\n\n"
            f"Return the list of extracted keywords with their types and contexts in valid JSON format:\n"
            f"{{\n"
            f"  'keywords': [\n"
            f"    {{\n"
            f"      'keywords_to_use_coding': 'string',\n"
            f"      'type': 'string',\n"
            f"      'context': 'string'\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n"
        )
        return prompt

    def forward(self, research_objectives: str, quotation: str, contextual_info: List[str], theoretical_framework: str = None) -> Dict[str, Any]:
        try:
            logger.debug("Starting keyword extraction process for individual quotation.")

            # Generate the prompt
            prompt = self.create_prompt(research_objectives, quotation, contextual_info, theoretical_framework)

            # Generate response
            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            ).strip()

            # Parse response
            keywords = self.parse_response(response)

            # Extract keyword texts for assertions
            extracted_keyword_texts = [kw['keywords_to_use_coding'] for kw in keywords]

            # Apply assertions
            assert_keywords_extracted(extracted_keyword_texts)
            assert_realness(extracted_keyword_texts, quotation)
            assert_keywords_not_exclusive_to_context(extracted_keyword_texts, quotation, contextual_info)
            assert_richness(extracted_keyword_texts)
            assert_repetition(extracted_keyword_texts, quotation)
            assert_rationale(extracted_keyword_texts, theoretical_framework)
            assert_repartee(extracted_keyword_texts)
            assert_regal(extracted_keyword_texts, research_objectives)

            logger.info(f"Successfully extracted {len(keywords)} keywords.")

            return {
                "keywords": keywords
            }

        except AssertionError as af:
            logger.warning(f"Assertion failed during keyword extraction: {af}")
            return self.handle_failed_assertion(af, research_objectives, quotation, contextual_info, theoretical_framework)
        except Exception as e:
            logger.error(f"Error in KeywordExtractionSignature.forward: {e}", exc_info=True)
            return {
                "keywords": [],
                "error": f"Error occurred during keyword extraction: {str(e)}"
            }

    def handle_failed_assertion(self, assertion_failure: AssertionError,
                                research_objectives: str, quotation: str,
                                contextual_info: List[str],
                                theoretical_framework: str = None) -> Dict[str, Any]:
        """Handles failed assertions by attempting to generate improved keywords."""
        try:
            contextual_text = "\n".join(contextual_info) if contextual_info else ""
            theoretical_part = f"Theoretical Framework:\n{theoretical_framework}\n\n" if theoretical_framework else ""
            focused_prompt = (
                f"The previous attempt failed because: {assertion_failure}\n\n"
                f"Research Objectives:\n{research_objectives}\n\n"
                f"{theoretical_part}"
                f"Quotation:\n{quotation}\n\n"
                f"Contextual Information:\n{contextual_text}\n\n"
                "Please ensure the following in your next attempt:\n"
                "Extract meaningful keywords **only** from the quotation using the 6Rs framework, utilizing the contextual information to better understand the quotation.\n"
                "Make sure the keywords satisfy the following criteria:\n"
                f"- Realness: Keywords must be present in the quotation and reflect participants' experiences.\n"
                f"- Richness: Keywords should be meaningful and provide detailed understanding.\n"
                f"- Repetition: Keywords should occur frequently in the data.\n"
                f"- Rationale: Keywords should be connected to the theoretical framework.\n"
                f"- Repartee: Keywords should be insightful and evocative.\n"
                f"- Regal: Keywords should be central to understanding the phenomenon.\n\n"
                f"Quotation:\n{quotation}"
            )

            response = self.language_model.generate(
                prompt=focused_prompt,
                max_tokens=500,
                temperature=0.5
            ).strip()

            keywords = self.parse_response(response)

            # Re-apply assertions
            extracted_keywords = []
            for kw in keywords:
                keyword = kw.get('keywords_to_use_coding', '').strip()
                kw_type = kw.get('type', '').strip()
                kw_context = kw.get('context', '').strip()
                if keyword:
                    extracted_keywords.append({
                        "keywords_to_use_coding": keyword,
                        "type": kw_type,
                        "context": kw_context
                    })

            # Extract keyword texts for assertions
            extracted_keyword_texts = [kw['keywords_to_use_coding'] for kw in extracted_keywords]

            assert_keywords_extracted(extracted_keyword_texts)
            assert_realness(extracted_keyword_texts, quotation)
            assert_keywords_not_exclusive_to_context(extracted_keyword_texts, quotation, contextual_info)
            assert_richness(extracted_keyword_texts)
            assert_repetition(extracted_keyword_texts, quotation)
            assert_rationale(extracted_keyword_texts, theoretical_framework)
            assert_repartee(extracted_keyword_texts)
            assert_regal(extracted_keyword_texts, research_objectives)

            logger.info("Keywords successfully extracted after refinement.")
            return {
                "keywords": extracted_keywords,
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
