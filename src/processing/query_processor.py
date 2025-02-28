# src/processing/query_processor.py

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.core.retrieval.reranking import (
    retrieve_with_reranking, RerankerConfig, RerankerType
)

# Local module imports
from .logger import get_logger
from .config import COHERE_API_KEY, ST_WEIGHT
from .validators import get_validator_for_module
from .factories import get_handler_for_module
from .handlers import BaseHandler
from .base import BaseValidator

import dspy  # DSPy library with built-in parallel support
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions  # or your custom decorator

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


def _get_query_from_transcript(transcript: Dict[str, Any]) -> Optional[str]:
    """
    Extract a non-empty query from the transcript.
    """
    query = transcript.get('query') or transcript.get('transcript_chunk') or transcript.get('quotation', '')
    return query.strip() if query and query.strip() else None


def _requires_retrieval(handler: BaseHandler) -> bool:
    """
    Determine whether the current handler requires a retrieval step.
    """
    return type(handler).__name__ not in ["GroupingHandler", "ThemeHandler"]


def _retrieve_documents(
    query: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    reranker_config: RerankerConfig
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using reranking.
    """
    return retrieve_with_reranking(
        query=query,
        db=db,
        es_bm25=es_bm25,
        k=k,
        reranker_config=reranker_config
    )


def _process_transcript(
    transcript_item: Dict[str, Any],
    handler: BaseHandler,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    reranker_config: RerankerConfig,
    module: dspy.Module
) -> Any:
    """
    Process a single transcript by optionally retrieving documents and then
    running the module-specific processing.
    """
    try:
        if _requires_retrieval(handler):
            query = _get_query_from_transcript(transcript_item)
            if query is None:
                logger.warning("Transcript has no valid query. Skipping.")
                return None
            retrieved_docs = _retrieve_documents(query, db, es_bm25, k, reranker_config)
        else:
            retrieved_docs = []
        # Run the async processing synchronously in this thread.
        result = asyncio.run(
            handler.process_single_transcript(
                transcript_item=transcript_item,
                retrieved_docs=retrieved_docs,
                module=module
            )
        )
        return result
    except Exception as e:
        logger.error(f"Error processing a transcript: {e}", exc_info=True)
        return None


@handle_exceptions
async def process_queries(
    transcripts: List[Dict[str, Any]],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    output_file: str,
    optimized_program: dspy.Program,  # Provided for consistency
    module: dspy.Module
):
    """
    Processes transcripts by validating, optionally retrieving documents, and
    then handling each transcript via DSPy's parallel executor.
    """
    logger.info(f"Processing transcripts for '{output_file}'.")

    # Validate transcripts.
    validator: BaseValidator = get_validator_for_module(module)
    logger.debug(f"Validator obtained: {type(validator).__name__} for module: {type(module).__name__}")
    valid_transcripts = validator.validate(transcripts)
    if not valid_transcripts:
        logger.warning("No valid transcripts found after validation. Exiting.")
        return

    # Get module handler.
    handler: BaseHandler = get_handler_for_module(module)
    logger.debug(f"Handler obtained: {type(handler).__name__} for module: {type(module).__name__}")

    # Prepare reranker configuration.
    reranker_config = RerankerConfig(
        reranker_type=RerankerType.COHERE,
        cohere_api_key=COHERE_API_KEY,
        st_weight=ST_WEIGHT
    )

    # Build execution pairs for DSPy's Parallel executor.
    exec_pairs: List[Tuple] = [
        (_process_transcript, (transcript, handler, db, es_bm25, k, reranker_config, module))
        for transcript in valid_transcripts
    ]

    # Create a Parallel executor instance.
    parallel_executor = dspy.Parallel(num_threads=8, max_errors=10, disable_progress_bar=False)

    # Execute the processing in parallel without blocking the event loop.
    results = await asyncio.to_thread(parallel_executor.forward, exec_pairs)

    # Filter out transcripts that failed or were skipped.
    all_results = [res for res in results if res is not None]

    # Save the results.
    save_results(all_results, output_file)
