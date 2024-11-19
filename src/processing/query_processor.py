import logging
from typing import List, Dict, Any
import json
from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.retrieval import multi_stage_retrieval
from src.analysis.select_quotation_module import SelectQuotationModule
from src.processing.answer_generator import generate_answer_dspy
from src.utils.logger import setup_logging

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

        if 'research_objectives' not in query:
            logger.warning(f"Query at index {idx} missing research_objectives. Using default.")
            query['research_objectives'] = "Extract relevant insights and themes from the provided text."

        valid_queries.append(query)

    logger.info(f"Validated {len(valid_queries)} queries out of {len(queries)} provided.")
    return valid_queries

def retrieve_documents(query: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int) -> List[Dict[str, Any]]:
    """
    Retrieves documents using multi-stage retrieval with contextual BM25.
    """
    logger.debug(f"Retrieving documents for query: '{query}' with top {k} results.")
    final_results = multi_stage_retrieval(query, db, es_bm25, k)
    logger.debug(f"Retrieved {len(final_results)} results.")
    return final_results

async def process_single_query(
    query_item: Dict[str, Any],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    quotation_module: SelectQuotationModule
) -> Dict[str, Any]:
    """
    Processes a single query to retrieve documents, select quotations, and generate an answer.
    """
    try:
        query_text = query_item.get('query', '').strip()
        if not query_text:
            logger.warning("Empty query received. Skipping.")
            return {}

        logger.info(f"Processing query: {query_text}")

        # Retrieve relevant chunks
        retrieved_chunks = retrieve_documents(query_text, db, es_bm25, k)
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")

        # Extract transcript chunks
        transcript_chunks = [chunk['chunk']['original_content'] for chunk in retrieved_chunks]

        # Get research objectives
        research_objectives = query_item.get('research_objectives', 
            'Extract relevant insights and themes from the provided text.')

        # Select quotations
        quotations_response = quotation_module.forward(
            research_objectives=research_objectives,
            transcript_chunks=transcript_chunks
        )
        
        # Generate answer
        qa_response = await generate_answer_dspy(query_text, retrieved_chunks)

        # Construct result
        result = {
            "query": query_text,
            "research_objectives": research_objectives,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_chunks_count": len(retrieved_chunks),
            "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in retrieved_chunks],
            "quotations": quotations_response.get("quotations", []),
            "purpose": quotations_response.get("purpose", ""),
            "answer": {
                "answer": qa_response.get("answer", "")
            }
        }

        logger.info(f"Processed query with {len(result['quotations'])} quotations selected.")
        return result

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {}

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Saves the results of the processed queries to a specified output file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)
        raise

async def process_queries(
    queries: List[Dict[str, Any]],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    output_file: str
) -> None:
    """
    Processes a list of queries to retrieve documents, select quotations, and generate answers.
    """
    logger.info("Starting query processing.")
    
    all_results = []
    quotation_module = SelectQuotationModule()

    try:
        for idx, query_item in enumerate(tqdm(queries, desc="Processing queries")):
            try:
                result = await process_single_query(
                    query_item=query_item,
                    db=db,
                    es_bm25=es_bm25,
                    k=k,
                    quotation_module=quotation_module
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing query {idx}: {e}", exc_info=True)
                continue

        save_results(all_results, output_file)
        logger.info("Query processing completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Process interrupted. Saving partial results.")
        save_results(all_results, output_file)
        raise
    except Exception as e:
        logger.error(f"Fatal error in query processing: {e}", exc_info=True)
        raise