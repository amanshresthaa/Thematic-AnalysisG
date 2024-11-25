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
    module: dspy.Module,
    theoretical_analysis_module: dspy.Module = None
) -> Dict[str, Any]:
    """
    Processes a single transcript chunk to retrieve documents and select quotations.
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

    # Extract transcript chunks from retrieved documents
    transcript_chunks = [chunk['chunk']['original_content'] for chunk in retrieved_chunks]

    # Add the original transcript chunk to the list
    transcript_chunks.insert(0, transcript_chunk)

    # Define research objectives
    research_objectives = transcript_item.get('research_objectives', 'Extract relevant quotations based on the provided objectives.')

    # Define theoretical framework
    theoretical_framework = transcript_item.get('theoretical_framework', {})

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
        "transcript_chunk": transcript_chunk,
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
        logger.warning(f"No quotations selected for transcript chunk: '{transcript_chunk[:100]}...' Skipping theoretical analysis and synthesis.")
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

    logger.info(f"Selected {len(quotations)} quotations for transcript chunk.")
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
    module: dspy.Module,
    theoretical_analysis_module: dspy.Module = None
):
    """
    Processes a list of transcript chunks to retrieve documents and select quotations.
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
                    module,
                    theoretical_analysis_module
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