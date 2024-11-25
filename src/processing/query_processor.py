# src/processing/query_processor.py
import logging
from typing import List, Dict, Any
import json
from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.retrieval import multi_stage_retrieval
from src.processing.answer_generator import generate_answer_dspy
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions
import dspy

# Import quotation module
from src.analysis.select_quotation_module import SelectQuotationModule

setup_logging()
logger = logging.getLogger(__name__)

def validate_queries(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates the structure of input queries.
    """
    valid_queries = []
    for idx, query in enumerate(queries):
        if 'query' not in query or not query['query'].strip():
            logger.warning(f"Query at index {idx} is missing the 'query' field or is empty. Skipping.")
            continue
        valid_queries.append(query)

    logger.info(f"Validated {len(valid_queries)} queries out of {len(queries)} provided.")
    return valid_queries

def retrieve_documents(query: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int) -> List[Dict[str, Any]]:
    """
    Retrieves documents using multi-stage retrieval with contextual BM25.
    """
    logger.debug(f"Retrieving documents for query: '{query}' with top {k} results.")
    final_results = multi_stage_retrieval(query, db, es_bm25, k)
    logger.debug(f"Multi-stage retrieval returned {len(final_results)} results.")
    return final_results

@handle_exceptions
async def process_single_query(
    query_item: Dict[str, Any],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    module: dspy.Module,
    theoretical_analysis_module: dspy.Module = None
) -> Dict[str, Any]:
    """
    Processes a single query to retrieve documents and select quotations.
    Returns a result dictionary.
    """
    query_text = query_item.get('query', '').strip()
    if not query_text:
        logger.warning("Query is empty. Skipping.")
        return {}

    logger.info(f"Processing query: {query_text}")

    # Retrieve relevant chunks/documents
    retrieved_chunks = retrieve_documents(query_text, db, es_bm25, k)
    logger.info(f"Total chunks retrieved for query '{query_text}': {len(retrieved_chunks)}")

    # Extract transcript chunks from retrieved documents
    transcript_chunks = [chunk['chunk']['original_content'] for chunk in retrieved_chunks]

    # Define research objectives
    research_objectives = query_item.get('research_objectives', 'Extract relevant quotations based on the provided objectives.')

    # Define theoretical framework
    theoretical_framework = query_item.get('theoretical_framework', {})

    # Select quotations using the provided DSPy module
    quotations_response = module.forward(
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
        "analysis": {},  # Will be filled after theoretical analysis
        "answer": {}     # Will be filled after synthesis
    }

    # Determine if any quotations are selected
    if not quotations:
        logger.warning(f"No quotations selected for query '{query_text}'. Skipping theoretical analysis and synthesis.")
        result["answer"] = "No relevant quotations were found to generate an answer."
    else:
        # Perform theoretical analysis
        theoretical_analysis_response = theoretical_analysis_module.forward(
            quotations=quotations,
            theoretical_framework=theoretical_framework,
            research_objectives=research_objectives
        )

        # Synthesize the final output
        result["analysis"] = {
            "patterns_identified": theoretical_analysis_response.get("patterns_identified", []),
            "theoretical_interpretation": theoretical_analysis_response.get("theoretical_interpretation", ""),
            "practical_implications": theoretical_analysis_response.get("practical_implications", "")
        }

        # Generate answer
        answer_text = theoretical_analysis_response.get("theoretical_interpretation", "")
        result["answer"] = {
            "answer": answer_text,
            "theoretical_contribution": theoretical_analysis_response.get("research_alignment", "")
        }

    logger.info(f"Selected {len(quotations)} quotations for query '{query_text}'.")
    return result

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Saves the processed query results to a specified output file.
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
    module: dspy.Module,
    theoretical_analysis_module: dspy.Module = None
):
    """
    Processes a list of queries to retrieve documents and select quotations.
    Saves results into a specified output file.
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
                    module,
                    theoretical_analysis_module
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