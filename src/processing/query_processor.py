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

# Import quotation modules
from src.analysis.select_quotation_module import SelectQuotationModule
from src.analysis.select_quotation_module_alt import SelectQuotationModuleAlt
from src.analysis.select_keyword_module import SelectKeywordModule  # Import the keyword module

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
    # Limiting is handled outside this function
    logger.debug(f"Multi-stage retrieval returned {len(final_results)} results.")
    return final_results

@handle_exceptions
async def process_single_query(
    query_item: Dict[str, Any],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    module: dspy.Module,
    is_keyword_extraction: bool = False
) -> List[Dict[str, Any]]:
    """
    Processes a single query to retrieve documents, select quotations, or extract keywords.
    Returns a list of result dictionaries, one for each quotation.
    """
    results = []
    if is_keyword_extraction:
        # Set k to 2 for keyword extraction
        retrieval_k = 2
        # Extract quotations intended for keyword extraction
        quotations = query_item.get('quotations', [])
        quotation_texts = [
            q.get('quotation', '').strip() 
            for q in quotations 
            if 'quotation' in q and q.get('quotation', '').strip()
        ]

        if not quotation_texts:
            logger.warning("No 'quotation' found for keyword extraction. Skipping.")
            return results  # Empty list

        logger.info(f"Processing keyword extraction for query: {query_item.get('query', '')[:50]}...")

        # Define research objectives and theoretical framework
        research_objectives = query_item.get('research_objectives', 'Extract relevant keywords based on the provided objectives.')
        theoretical_framework = query_item.get('theoretical_framework', '')

        # Retrieve top 2 relevant chunks
        retrieved_chunks = retrieve_documents(query_item.get('query', ''), db, es_bm25, retrieval_k)
        retrieved_chunks_count = len(retrieved_chunks)
        used_chunk_ids = [chunk['chunk']['chunk_id'] for chunk in retrieved_chunks]
        analysis = query_item.get('analysis', '')
        answer = query_item.get('answer', {})

        # Extract contextualized_content from the top 2 chunks
        contextual_info = [chunk['chunk']['contextualized_content'] for chunk in retrieved_chunks if 'contextualized_content' in chunk['chunk'] and chunk['chunk']['contextualized_content'].strip()]

        for idx, quotation in enumerate(quotation_texts, start=1):
            logger.info(f"Extracting keywords for quotation {idx}: {quotation[:50]}...")

            # Extract keywords using the provided DSPy module
            response = module.forward(
                research_objectives=research_objectives,
                quotation=quotation,
                contextual_info=contextual_info,
                theoretical_framework=theoretical_framework
            )
            keywords = response.get("keywords", [])
            error = response.get("error", "")

            if error:
                logger.error(f"Error extracting keywords for quotation {idx}: {error}")
                # Skip to the next quotation
                continue

            # Prepare the result for this quotation
            result = {
                "quotation": quotation,
                "research_objectives": research_objectives,
                "theoretical_framework": theoretical_framework,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_chunks_count": retrieved_chunks_count,
                "used_chunk_ids": used_chunk_ids,
                "keywords": keywords,  # List of keyword dictionaries
                "analysis": analysis,
                "answer": answer
            }

            # Remove 'contextualized_content' from retrieved_chunks if not needed in output
            for chunk in result["retrieved_chunks"]:
                if 'contextualized_content' in chunk['chunk']:
                    del chunk['chunk']['contextualized_content']

            # Log the number of keywords extracted
            logger.info(f"Extracted {len(keywords)} keywords for quotation {idx}.")

            results.append(result)

        return results  # List of result dicts
    else:
        # Existing logic for processing quotations and generating answers
        query_text = query_item.get('query', '').strip()
        if not query_text:
            logger.warning("Query is empty. Skipping.")
            return results  # Empty list

        logger.info(f"Processing query: {query_text}")

        # Retrieve relevant chunks/documents
        retrieved_chunks = retrieve_documents(query_text, db, es_bm25, k)
        logger.info(f"Total chunks retrieved for query '{query_text}': {len(retrieved_chunks)}")

        # Extract transcript chunks from retrieved documents
        transcript_chunks = [chunk['chunk']['original_content'] for chunk in retrieved_chunks]

        # Define research objectives
        research_objectives = query_item.get('research_objectives', 'Extract relevant quotations based on the provided objectives.')

        # Define theoretical framework
        theoretical_framework = query_item.get('theoretical_framework', '')

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
            retrieved_chunks_count = qa_response.get("num_chunks_used", 0)

            result["answer"] = {
                "answer": answer
            }

        # Log the chunks used for this query
        logger.info(f"Query '{query_text}' used {retrieved_chunks_count} chunks in answer generation.")

        # Log the number of quotations selected
        logger.info(f"Selected {len(quotations)} quotations for query '{query_text}'.")

        results.append(result)
        return results

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
    is_keyword_extraction: bool = False
):
    """
    Processes a list of queries to retrieve documents, select quotations, or extract keywords.
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
                    is_keyword_extraction
                )
                if result:
                    # 'result' is a list of dicts
                    all_results.extend(result)
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
