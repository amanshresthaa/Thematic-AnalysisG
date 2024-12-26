# src/processing/query_processor.py

import os
import json
from typing import List, Dict, Any
import time

from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType

# Import local modules
from .logger import get_logger
from .config import COHERE_API_KEY, ST_WEIGHT
from .validators import get_validator_for_module
from .factories import get_handler_for_module
from .handlers import BaseHandler
from .base import BaseValidator

import dspy  # Presumably your library
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions  # or you can define your own

setup_logging()
logger = get_logger(__name__)

def save_results(results: List[Dict[str, Any]], output_file: str):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4)
        logger.info(f"Saved results to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving results to '{output_file}': {e}", exc_info=True)

@handle_exceptions
async def process_queries(
    transcripts: List[Dict[str, Any]],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    output_file: str,
    optimized_program: dspy.Program,  # Not fully used in this code, but left for consistency
    module: dspy.Module
):
    """
    Main function that processes transcripts according to the specified module.
    - Retrieves a matching validator and handler for the module
    - Validates transcripts
    - Optionally retrieves relevant docs from DB/ES
    - Processes each transcript with the appropriate handler
    - Saves results
    """

    logger.info(f"Processing transcripts for '{output_file}'.")

    # ----------------------------------------------------------------
    # 1. Get a validator for the module and validate transcripts
    # ----------------------------------------------------------------
    validator: BaseValidator = get_validator_for_module(module)
    logger.debug(f"Validator obtained: {type(validator).__name__} for module: {type(module).__name__}")
    valid_transcripts = validator.validate(transcripts)
    if not valid_transcripts:
        logger.warning("No valid transcripts found after validation. Exiting.")
        return

    # ----------------------------------------------------------------
    # 2. Get a handler for the module
    # ----------------------------------------------------------------
    handler: BaseHandler = get_handler_for_module(module)
    logger.debug(f"Handler obtained: {type(handler).__name__} for module: {type(module).__name__}")

    # ----------------------------------------------------------------
    # 3. Prepare reranker configuration
    # ----------------------------------------------------------------
    reranker_config = RerankerConfig(
        reranker_type=RerankerType.COHERE,
        cohere_api_key=COHERE_API_KEY,
        st_weight=ST_WEIGHT
    )

    # ----------------------------------------------------------------
    # 4. Process each transcript
    # ----------------------------------------------------------------
    all_results = []
    for idx, transcript_item in enumerate(tqdm(valid_transcripts, desc="Processing transcripts")):
        try:
            # For modules that require retrieval, we look up the query
            # (SelectQuotationModule, EnhancedQuotationModule, KeywordExtractionModule, CodingAnalysisModule)
            # For grouping/theme modules, we skip retrieval.
            retrieve_docs_flag = hasattr(module, 'requires_retrieval') and module.requires_retrieval
            # If your actual modules don't have a 'requires_retrieval' property,
            # you can explicitly check instance types:
            if type(handler).__name__ not in ["GroupingHandler", "ThemeHandler"]:
                query = transcript_item.get('query') \
                        or transcript_item.get('transcript_chunk') \
                        or transcript_item.get('quotation', '')
                if not query.strip():
                    logger.warning(f"Transcript at index {idx} has no valid query. Skipping.")
                    continue

                retrieved_docs = retrieve_with_reranking(
                    query=query,
                    db=db,
                    es_bm25=es_bm25,
                    k=k,
                    reranker_config=reranker_config
                )
            else:
                retrieved_docs = []

            # ----------------------------------------------------------------
            # 5. Process the single transcript
            # ----------------------------------------------------------------
            result = await handler.process_single_transcript(
                transcript_item=transcript_item,
                retrieved_docs=retrieved_docs,
                module=module
            )

            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing transcript at index {idx}: {e}", exc_info=True)

    # ----------------------------------------------------------------
    # 6. Save results
    # ----------------------------------------------------------------
    save_results(all_results, output_file)
