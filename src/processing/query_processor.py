# src/processing/query_processor.py

import os
import json
from typing import List, Dict, Any
import asyncio

from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.core.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType

# Import local modules
from .logger import get_logger
from .config import COHERE_API_KEY, ST_WEIGHT
from .validators import get_validator_for_module
from .factories import get_handler_for_module
from .handlers import BaseHandler
from .base import BaseValidator

import dspy  # DSPy library with built-in parallel support
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
    This version leverages DSPy's Parallel class to perform multi-threaded, parallel API calls.
    """

    logger.info(f"Processing transcripts for '{output_file}'.")

    # 1. Get a validator for the module and validate transcripts
    validator: BaseValidator = get_validator_for_module(module)
    logger.debug(f"Validator obtained: {type(validator).__name__} for module: {type(module).__name__}")
    valid_transcripts = validator.validate(transcripts)
    if not valid_transcripts:
        logger.warning("No valid transcripts found after validation. Exiting.")
        return

    # 2. Get a handler for the module
    handler: BaseHandler = get_handler_for_module(module)
    logger.debug(f"Handler obtained: {type(handler).__name__} for module: {type(module).__name__}")

    # 3. Prepare reranker configuration
    reranker_config = RerankerConfig(
        reranker_type=RerankerType.COHERE,
        cohere_api_key=COHERE_API_KEY,
        st_weight=ST_WEIGHT
    )

    # 4. Define a synchronous helper function to process one transcript.
    #    This function will be run in a separate thread.
    def process_transcript_sync(transcript_item: Dict[str, Any]):
        try:
            # For modules that require retrieval, we look up the query.
            # For grouping/theme modules, we skip retrieval.
            if type(handler).__name__ not in ["GroupingHandler", "ThemeHandler"]:
                query = transcript_item.get('query') or transcript_item.get('transcript_chunk') or transcript_item.get('quotation', '')
                if not query or not query.strip():
                    logger.warning("Transcript has no valid query. Skipping.")
                    return None

                retrieved_docs = retrieve_with_reranking(
                    query=query,
                    db=db,
                    es_bm25=es_bm25,
                    k=k,
                    reranker_config=reranker_config
                )
            else:
                retrieved_docs = []

            # Process the single transcript by calling the async function synchronously.
            # We use asyncio.run to create a new event loop in this thread.
            result = asyncio.run(handler.process_single_transcript(
                transcript_item=transcript_item,
                retrieved_docs=retrieved_docs,
                module=module
            ))
            return result
        except Exception as e:
            logger.error(f"Error processing a transcript: {e}", exc_info=True)
            return None

    # 5. Build execution pairs for DSPy's Parallel executor.
    #    Each pair is a tuple: (function, (transcript_item,))
    exec_pairs = [(process_transcript_sync, (transcript_item,)) for transcript_item in valid_transcripts]

    # 6. Create a Parallel executor instance with the desired number of threads.
    parallel_executor = dspy.Parallel(num_threads=8, max_errors=10, disable_progress_bar=False)

    # 7. Execute the processing in parallel.
    #    We wrap the blocking call using asyncio.to_thread so as not to block the event loop.
    results = await asyncio.to_thread(parallel_executor.forward, exec_pairs)

    # 8. Filter out None results (i.e. transcripts that failed or were skipped).
    all_results = [res for res in results if res is not None]

    # 9. Save results to the specified output file.
    save_results(all_results, output_file)
