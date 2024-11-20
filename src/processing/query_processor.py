# File: src/processing/query_processor.py
import logging
from typing import List, Dict, Any
import json
from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.retrieval import multi_stage_retrieval
from src.retrieval.reranking import retrieve_with_reranking
from src.processing.answer_generator import generate_answer_dspy
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions
import dspy

# Importing both quotation modules
from src.analysis.select_quotation_module import SelectQuotationModule
from src.analysis.select_quotation_module_alt import SelectQuotationModuleAlt

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
async def process_single_query(
    query_item: Dict[str, Any],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    quotation_module: dspy.Module
) -> Dict[str, Any]:
    """
    Processes a single query to retrieve documents, select quotations, and generate an answer.

    Args:
        query_item (Dict[str, Any]): The query item containing the query text and other relevant information.
        db (ContextualVectorDB): Contextual vector database instance.
        es_bm25 (ElasticsearchBM25): Elasticsearch BM25 instance.
        k (int): Number of top documents to retrieve.
        quotation_module (dspy.Module): Module to select quotations.

    Returns:
        Dict[str, Any]: The result of processing the query, including retrieved chunks, quotations, and generated answer.
    """
    query_text = query_item.get('query', '').strip()
    if not query_text:
        logger.warning("Query is empty. Skipping.")
        return {}

    logger.info(f"Processing query: {query_text}")

    # Retrieve relevant chunks/documents using multi-stage retrieval with contextual BM25
    retrieved_chunks = retrieve_documents(query_text, db, es_bm25, k)
    logger.info(f"Total chunks retrieved for query '{query_text}': {len(retrieved_chunks)}")
    logger.info(f"Retrieved chunk IDs: {[chunk['chunk']['chunk_id'] for chunk in retrieved_chunks]}")

    # Extract transcript chunks from retrieved documents
    transcript_chunks = [chunk['chunk']['original_content'] for chunk in retrieved_chunks]

    # Define research objectives (this could be part of the query_item or defined elsewhere)
    research_objectives = query_item.get('research_objectives', 'Extract relevant quotations based on the provided objectives.')

    # Define theoretical framework (this should be part of the query_item or configuration)
    theoretical_framework = query_item.get('theoretical_framework', 'Constructivist')

    # Select quotations using the provided DSPy module
    quotations_response = quotation_module.forward(
        research_objectives=research_objectives,
        transcript_chunks=transcript_chunks,
        theoretical_framework=theoretical_framework
    )
    quotations = quotations_response.get("quotations", [])
    analysis = quotations_response.get("analysis", "")

    # Prepare the result
    result = {
        "query": query_text,
        "research_objectives": research_objectives,
        "theoretical_framework": theoretical_framework,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_chunks_count": len(retrieved_chunks),
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in retrieved_chunks],
        "quotations": quotations,
        "analysis": analysis,
        "answer": ""  # Placeholder for the answer
    }

    # Determine if any quotations are selected
    if not quotations:
        logger.warning(f"No quotations selected for query '{query_text}'. Skipping answer generation.")
        result["answer"] = "No relevant quotations were found to generate an answer."
    else:
        # Generate answer using DSPy with assertions
        qa_response = await generate_answer_dspy(query_text, retrieved_chunks)
        answer = qa_response.get("answer", "")
        used_chunks_info = qa_response.get("used_chunks", [])
        retrieved_chunks_count = qa_response.get("num_chunks_used", 0)

        result["answer"] = {
            "answer": answer
        }

    # Log the chunks used for this query
    logger.info(f"Query '{query_text}' used {retrieved_chunks_count} chunks in answer generation.")

    # Log the number of quotations selected
    logger.info(f"Selected {len(quotations)} quotations for query '{query_text}'.")

    return result

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Saves the processed query results to a specified output file.

    Args:
        results (List[Dict[str, Any]]): List of query results to save.
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
    output_file: str,
    optimized_program: dspy.Program,
    quotation_module: dspy.Module
):
    """
    Processes a list of queries to retrieve documents, select quotations, and generate answers.
    Saves results into a specified output file.

    Args:
        queries (List[Dict[str, Any]]): List of query items.
        db (ContextualVectorDB): Contextual vector database instance.
        es_bm25 (ElasticsearchBM25): Elasticsearch BM25 instance.
        k (int): Number of top documents to retrieve.
        output_file (str): Path to the output file to save results.
        optimized_program (dspy.Program): The optimized DSPy program.
        quotation_module (dspy.Module): Module to select quotations.
    """
    logger.info(f"Starting to process queries for output file '{output_file}'.")

    all_results = []
    try:
        for idx, query_item in enumerate(tqdm(queries, desc="Processing queries")):
            try:
                result = await process_single_query(
                    query_item,
                    db,
                    es_bm25,
                    k,
                    quotation_module
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing query at index {idx}: {e}", exc_info=True)

        # Save results to the specified output file
        save_results(all_results, output_file)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Saving partial results.")
        save_results(all_results, output_file)
        raise

    except Exception as e:
        logger.error(f"Error processing queries: {e}", exc_info=True)
        raise
