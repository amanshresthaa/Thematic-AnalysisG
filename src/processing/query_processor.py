# File: src/processing/query_processor.py
import logging
from typing import List, Dict, Any, Callable
import json
from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.retrieval import multi_stage_retrieval
from src.retrieval.reranking import retrieve_with_reranking
from src.analysis.select_quotation_module import SelectQuotationModule
from src.analysis.select_quotation_module_alt import SelectQuotationModuleAlt  # Import the new module
from src.processing.answer_generator import generate_answer_dspy
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions
import dspy

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
async def process_single_query(query_item: Dict[str, Any], db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int, quotation_module: SelectQuotationModule, quotation_module_alt: SelectQuotationModuleAlt) -> Dict[str, Any]:
    """
    Processes a single query to retrieve documents, select quotations, and generate an answer.

    Args:
        query_item (Dict[str, Any]): The query item containing the query text and other relevant information.
        db (ContextualVectorDB): Contextual vector database instance.
        es_bm25 (ElasticsearchBM25): Elasticsearch BM25 instance.
        k (int): Number of top documents to retrieve.
        quotation_module (SelectQuotationModule): Primary module to select quotations.
        quotation_module_alt (SelectQuotationModuleAlt): Alternative module to select quotations.

    Returns:
        Dict[str, Any]: The result of processing the query, including retrieved chunks, quotations, and generated answer.
    """
    query_text = query_item.get('query', '').strip()
    if not query_text:
        logger.warning(f"Query is empty. Skipping.")
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

    # Select quotations using the primary DSPy module
    quotations_response = quotation_module.forward(
        research_objectives=research_objectives,
        transcript_chunks=transcript_chunks,
        theoretical_framework=theoretical_framework
    )
    quotations = quotations_response.get("quotations", [])
    analysis = quotations_response.get("analysis", "")

    # Select quotations using the alternative DSPy module
    quotations_alt_response = quotation_module_alt.forward(
        research_objectives=research_objectives,
        transcript_chunks=transcript_chunks,
        theoretical_framework=theoretical_framework
    )
    quotations_alt = quotations_alt_response.get("quotations", [])
    analysis_alt = quotations_alt_response.get("analysis", "")

    # Prepare separate results for primary and alternative modules
    result_primary = {
        "query": query_text,
        "research_objectives": research_objectives,
        "theoretical_framework": theoretical_framework,
        "retrieved_chunks": retrieved_chunks,  # Use the original retrieved_chunks
        "retrieved_chunks_count": len(retrieved_chunks),  # Use the length of retrieved_chunks
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in retrieved_chunks],  # Fixed nested dictionary access
        "quotations": quotations,  # Quotations from primary module
        "analysis": analysis,
        "answer": ""  # Placeholder for the answer
    }

    result_alt = {
        "query": query_text,
        "research_objectives": research_objectives,
        "theoretical_framework": theoretical_framework,
        "retrieved_chunks": retrieved_chunks,  # Use the original retrieved_chunks
        "retrieved_chunks_count": len(retrieved_chunks),  # Use the length of retrieved_chunks
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in retrieved_chunks],  # Fixed nested dictionary access
        "quotations": quotations_alt,  # Quotations from alternative module
        "analysis": analysis_alt,
        "answer": ""  # Placeholder for the answer
    }

    # Determine if any quotations are selected
    if not quotations and not quotations_alt:
        logger.warning(f"No quotations selected for query '{query_text}'. Skipping answer generation.")
        result_primary["answer"] = "No relevant quotations were found to generate an answer."
        result_alt["answer"] = "No relevant quotations were found to generate an answer."
    else:
        # Generate answer using DSPy with assertions
        qa_response = await generate_answer_dspy(query_text, retrieved_chunks)
        answer = qa_response.get("answer", "")
        used_chunks_info = qa_response.get("used_chunks", [])
        retrieved_chunks_count = qa_response.get("num_chunks_used", 0)

        result_primary["answer"] = {
            "answer": answer
        }
        result_alt["answer"] = {
            "answer": answer
        }

    # Log the chunks used for this query
    logger.info(f"Query '{query_text}' used {retrieved_chunks_count} chunks in answer generation.")
    logger.info(f"Selected {len(quotations)} primary quotations and {len(quotations_alt)} alternative quotations for query '{query_text}'.")

    return {
        "primary": result_primary,
        "alternative": result_alt
    }

def save_results_primary(results: List[Dict[str, Any]], output_file: str):
    """
    Saves the primary results of the processed queries to a specified output file.

    Args:
        results (List[Dict[str, Any]]): List of primary results to save.
        output_file (str): Path to the primary output file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4)
        logger.info(f"All primary query results have been saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving primary results to '{output_file}': {e}", exc_info=True)

def save_results_alternative(results: List[Dict[str, Any]], output_file: str):
    """
    Saves the alternative results of the processed queries to a specified output file.

    Args:
        results (List[Dict[str, Any]]): List of alternative results to save.
        output_file (str): Path to the alternative output file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4)
        logger.info(f"All alternative query results have been saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving alternative results to '{output_file}': {e}", exc_info=True)

@handle_exceptions
async def process_queries(
    queries: List[Dict[str, Any]],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    output_file_primary: str,
    output_file_alt: str,
    optimized_program: dspy.Program,
    quotation_module: SelectQuotationModule,
    quotation_module_alt: SelectQuotationModuleAlt  # Add the new module as a parameter
):
    """
    Processes a list of queries to retrieve documents, select quotations, and generate answers.
    Saves primary and alternative results into separate output files.

    Args:
        queries (List[Dict[str, Any]]): List of query items.
        db (ContextualVectorDB): Contextual vector database instance.
        es_bm25 (ElasticsearchBM25): Elasticsearch BM25 instance.
        k (int): Number of top documents to retrieve.
        output_file_primary (str): Path to the primary output file to save results.
        output_file_alt (str): Path to the alternative output file to save results.
        optimized_program (dspy.Program): The optimized DSPy program.
        quotation_module (SelectQuotationModule): Primary module to select quotations.
        quotation_module_alt (SelectQuotationModuleAlt): Alternative module to select quotations.
    """
    logger.info("Starting to process queries.")

    all_results_primary = []
    all_results_alt = []
    try:
        for idx, query_item in enumerate(tqdm(queries, desc="Processing queries")):
            try:
                result = await process_single_query(
                    query_item,
                    db,
                    es_bm25,
                    k,
                    quotation_module,
                    quotation_module_alt  # Pass the new module
                )
                if result:
                    # Separate the results
                    all_results_primary.append(result.get("primary", {}))
                    all_results_alt.append(result.get("alternative", {}))
            except Exception as e:
                logger.error(f"Error processing query at index {idx}: {e}", exc_info=True)

        # Save primary and alternative results to separate files
        save_results_primary(all_results_primary, output_file_primary)
        save_results_alternative(all_results_alt, output_file_alt)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Saving partial results.")
        save_results_primary(all_results_primary, output_file_primary)
        save_results_alternative(all_results_alt, output_file_alt)
        raise

    except Exception as e:
        logger.error(f"Error processing queries: {e}", exc_info=True)
        raise
