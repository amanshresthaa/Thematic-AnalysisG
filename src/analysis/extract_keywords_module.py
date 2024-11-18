import logging
from typing import Dict, Any, List
import dspy
import os

from .extract_keywords import KeywordExtractionSignature, load_quotations, save_keywords
from ..core.contextual_vector_db import ContextualVectorDB
from ..core.elasticsearch_bm25 import ElasticsearchBM25

logger = logging.getLogger(__name__)

class KeywordExtractionModule(dspy.Module):
    """
    Module to extract keywords from quotations using contextual retrieval and 6Rs enhancement.
    """
    def __init__(self, input_file: str, output_file: str,
                 contextual_db: ContextualVectorDB, es_bm25: ElasticsearchBM25):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)
        self.input_file = input_file
        self.output_file = output_file
        self.contextual_db = contextual_db
        self.es_bm25 = es_bm25

    def process_file(self, input_file: str, research_objectives: str) -> Dict[str, Any]:
        try:
            logger.debug(f"Processing file: {input_file}")
            # Load quotations
            quotations = load_quotations(input_file)
            if not quotations:
                logger.warning(f"No quotations found in {input_file}")
                return {"keywords": []}

            keywords_mapping = []
            for idx, quote_dict in enumerate(quotations):
                quote = quote_dict.get("quote", "")
                if not quote:
                    logger.warning(f"Quote at index {idx} is empty.")
                    keywords_mapping.append({"quote": quote, "keywords_data": []})
                    continue

                logger.debug(f"Extracting keywords for quote {idx+1}: {quote}")
                # Retrieve additional context using Elasticsearch BM25
                query = quote
                relevant_docs = self.es_bm25.search(query, k=5)
                additional_context = " ".join([doc['content'] for doc in relevant_docs])

                # Combine the quote with additional context
                combined_quote = f"{quote} {additional_context}"

                # Extract keywords using the combined quote
                response = self.chain(
                    research_objectives=research_objectives,
                    quotations=[{"quote": combined_quote}]
                )
                keywords_data = response.get("keywords_data", [])

                keywords_mapping.append({"quote": quote, "keywords_data": keywords_data})

            # Save the mapping with enhanced keywords data
            save_keywords(keywords_mapping, self.output_file)

            return {"keywords": keywords_mapping, "output_file": self.output_file}

        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}", exc_info=True)
            return {"keywords": []}