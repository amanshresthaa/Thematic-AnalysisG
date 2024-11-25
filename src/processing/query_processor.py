# processing/query_processor.py
import logging
from typing import List, Dict, Any
import json
from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.retrieval import multi_stage_retrieval
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions
import dspy

setup_logging()
logger = logging.getLogger(__name__)

def validate_queries(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates the structure of input transcripts.
    """
    valid_transcripts = []
    for idx, transcript in enumerate(transcripts):
        if 'transcript_chunk' not in transcript or not transcript['transcript_chunk'].strip():
            logger.warning(f"Transcript at index {idx} is missing the 'transcript_chunk' field or is empty. Skipping.")
            continue
        valid_transcripts.append(transcript)

    logger.info(f"Validated {len(valid_transcripts)} transcripts out of {len(transcripts)} provided.")
    return valid_transcripts

def retrieve_documents(transcript_chunk: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int) -> List[Dict[str, Any]]:
    """
    Retrieves documents using multi-stage retrieval with contextual BM25.
    """
    logger.debug(f"Retrieving documents for transcript chunk: '{transcript_chunk[:100]}...' with top {k} results.")
    final_results = multi_stage_retrieval(transcript_chunk, db, es_bm25, k)
    logger.debug(f"Multi-stage retrieval returned {len(final_results)} results.")
    return final_results

@handle_exceptions
async def process_single_transcript(
    transcript_item: Dict[str, Any],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    module: dspy.Module
) -> Dict[str, Any]:
    """
    Processes a single transcript chunk to retrieve documents and perform enhanced quotation analysis.
    Returns a result dictionary.
    """
    transcript_chunk = transcript_item.get('transcript_chunk', '').strip()
    if not transcript_chunk:
        logger.warning("Transcript chunk is empty. Skipping.")
        return {}

    logger.info(f"Processing transcript chunk: {transcript_chunk[:100]}...")

    # Retrieve relevant chunks/documents
    retrieved_chunks = retrieve_documents(transcript_chunk, db, es_bm25, k)
    logger.info(f"Total chunks retrieved for transcript chunk: {len(retrieved_chunks)}")

    # Extract contextualized contents from retrieved documents
    contextualized_contents = [chunk['chunk']['original_content'] for chunk in retrieved_chunks]

    # Get research objectives and theoretical framework
    research_objectives = transcript_item.get('research_objectives', 
        'Extract relevant quotations based on the provided objectives.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    # Process transcript using the enhanced quotation module
    response = module.forward(
        research_objectives=research_objectives,
        transcript_chunk=transcript_chunk,
        contextualized_contents=contextualized_contents,
        theoretical_framework=theoretical_framework
    )
    
    # Prepare the result dictionary
    result = {
        "transcript_chunk": transcript_chunk,
        "research_objectives": research_objectives,
        "theoretical_framework": theoretical_framework,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_chunks_count": len(retrieved_chunks),
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in retrieved_chunks],
        "transcript_info": response.get("transcript_info", {}),
        "quotations": response.get("quotations", []),
        "analysis": response.get("analysis", {}),
        "answer": response.get("answer", {})
    }

    if not result["quotations"]:
        logger.warning(f"No quotations selected for transcript chunk: '{transcript_chunk[:100]}...'")
        result["answer"] = {"answer": "No relevant quotations were found to generate an answer."}

    logger.info(f"Selected {len(result['quotations'])} quotations for transcript chunk.")
    return result

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Saves the processed transcript results to a specified output file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4)
        logger.info(f"All transcript results have been saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving results to '{output_file}': {e}", exc_info=True)

@handle_exceptions
async def process_queries(
    transcripts: List[Dict[str, Any]],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    output_file: str,
    optimized_program: dspy.Program,
    module: dspy.Module
):
    """
    Processes a list of transcript chunks to retrieve documents and perform enhanced quotation analysis.
    Saves results into a specified output file.
    """
    logger.info(f"Starting to process transcripts for output file '{output_file}'.")

    all_results = []
    try:
        for idx, transcript_item in enumerate(tqdm(transcripts, desc="Processing transcripts")):
            try:
                result = await process_single_transcript(
                    transcript_item,
                    db,
                    es_bm25,
                    k,
                    module
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing transcript at index {idx}: {e}", exc_info=True)

        # Save results to the specified output file
        save_results(all_results, output_file)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Saving partial results.")
        save_results(all_results, output_file)
        raise

    except Exception as e:
        logger.error(f"Error processing transcripts: {e}", exc_info=True)
        raise