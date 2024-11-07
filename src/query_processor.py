# File: /Users/amankumarshrestha/Downloads/Example/src/query_processor.py
import logging
from typing import List, Dict, Any, Callable
from contextual_vector_db import ContextualVectorDB
from elasticsearch_bm25 import ElasticsearchBM25  # Import ElasticsearchBM25 class

# Ensure that multi_stage_retrieval is correctly defined in retrieval.py
from retrieval import multi_stage_retrieval  # Use multi_stage_retrieval as default
from reranking import retrieve_with_reranking  # Ensure this function is correctly defined and imported

from answer_generator import generate_answer_dspy
from utils.logger import setup_logging
import json
from tqdm import tqdm
from src.decorators import handle_exceptions

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


def validate_queries(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates the structure of input queries.

    Args:
        queries (List[Dict[str, Any]]): List of query items.

    Returns:
        List[Dict[str, Any]]: List of validated query items.
    """
    valid_queries = []
    for idx, query in enumerate(queries):
        if 'query' not in query or not query['query'].strip():
            logger.warning(f"Query at index {idx} is missing the 'query' field or is empty. Skipping.")
            continue

        # Check for optional golden fields
        has_golden_doc_uuids = 'golden_doc_uuids' in query
        has_golden_chunk_uuids = 'golden_chunk_uuids' in query
        has_golden_documents = 'golden_documents' in query

        # If some but not all golden fields are present, log a warning
        if any([has_golden_doc_uuids, has_golden_chunk_uuids, has_golden_documents]) and not all([has_golden_doc_uuids, has_golden_chunk_uuids, has_golden_documents]):
            logger.warning(f"Query at index {idx} has incomplete golden data. All golden fields should be present together.")

        valid_queries.append(query)

    logger.info(f"Validated {len(valid_queries)} queries out of {len(queries)} provided.")
    return valid_queries


def retrieve_documents(query: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int) -> List[Dict[str, Any]]:
    """
    Retrieves documents using multi-stage retrieval with contextual BM25.

    Args:
        query (str): The search query.
        db (ContextualVectorDB): Contextual vector database instance.
        es_bm25 (ElasticsearchBM25): Elasticsearch BM25 instance.
        k (int): Number of top documents to retrieve.

    Returns:
        List[Dict[str, Any]]: List of retrieved documents.
    """
    logger.debug(f"Retrieving documents for query: '{query}' with top {k} results using multi-stage retrieval with contextual BM25.")
    final_results = multi_stage_retrieval(query, db, es_bm25, k)
    logger.debug(f"Multi-stage retrieval with contextual BM25 returned {len(final_results)} results.")
    return final_results

@handle_exceptions
async def process_single_query(query_item: Dict[str, Any], db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int) -> Dict[str, Any]:
    query_text = query_item.get('query', '').strip()
    if not query_text:
        logger.warning(f"Query is empty. Skipping.")
        return {}

    logger.info(f"Processing query: {query_text}")

    # Retrieve relevant chunks/documents using multi-stage retrieval with contextual BM25
    retrieved_chunks = retrieve_documents(query_text, db, es_bm25, k)
    logger.info(f"Total chunks retrieved for query '{query_text}': {len(retrieved_chunks)}")
    logger.info(f"Retrieved chunk IDs: {[chunk['chunk']['chunk_id'] for chunk in retrieved_chunks]}")

    # Generate answer using DSPy with assertions
    qa_response = await generate_answer_dspy(query_text, retrieved_chunks)
    answer = qa_response.get("answer", "")
    used_chunks_info = qa_response.get("used_chunks", [])
    retrieved_chunks_count = qa_response.get("num_chunks_used", 0)

    result = {
        "query": query_text,
        "retrieved_chunks": used_chunks_info,
        "retrieved_chunks_count": retrieved_chunks_count,
        "used_chunk_ids": [chunk['chunk_id'] for chunk in used_chunks_info],
        "answer": {
            "answer": answer
        }
    }

    # Log the chunks used for this query
    logger.info(f"Query '{query_text}' used {retrieved_chunks_count} chunks in answer generation.")

    return result


def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Saves the results of the processed queries to a specified output file.

    Args:
        results (List[Dict[str, Any]]): List of results to save.
        output_file (str): Path to the output file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4)
        logger.info(f"All query results have been saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving results to '{output_file}': {e}", exc_info=True)

@handle_exceptions
async def process_queries(
    queries: List[Dict[str, Any]],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    output_file: str
):
    logger.info("Starting to process queries.")

    all_results = []

    try:
        for idx, query_item in enumerate(tqdm(queries, desc="Processing queries")):
            try:
                result = await process_single_query(query_item, db, es_bm25, k)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing query at index {idx}: {e}", exc_info=True)

        save_results(all_results, output_file)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Saving partial results.")
        save_results(all_results, output_file)
        raise

    except Exception as e:
        logger.error(f"Error processing queries: {e}", exc_info=True)
        raise
