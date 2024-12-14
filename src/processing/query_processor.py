
#query_processor.py
import logging
from typing import List, Dict, Any
import json
import os
import dspy
from tqdm import tqdm
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions

from src.analysis.select_quotation_module import SelectQuotationModule, EnhancedQuotationModule
from src.analysis.extract_keyword_module import KeywordExtractionModule
from src.analysis.coding_module import CodingAnalysisModule
from src.analysis.theme_development_module import ThemedevelopmentAnalysisModule

setup_logging()
logger = logging.getLogger(__name__)

def validate_queries(transcripts: List[Dict[str, Any]], module: dspy.Module) -> List[Dict[str, Any]]:
    valid_transcripts = []
    for idx, transcript in enumerate(transcripts):
        if isinstance(module, (SelectQuotationModule, EnhancedQuotationModule)):
            if 'transcript_chunk' not in transcript or not isinstance(transcript['transcript_chunk'], str) or not transcript['transcript_chunk'].strip():
                logger.warning(f"Transcript at index {idx} missing 'transcript_chunk'. Skipping.")
                continue
        elif isinstance(module, KeywordExtractionModule):
            if 'quotation' not in transcript or not isinstance(transcript['quotation'], str) or not transcript['quotation'].strip():
                logger.warning(f"Transcript at index {idx} missing 'quotation' for keyword extraction. Skipping.")
                continue
        elif isinstance(module, CodingAnalysisModule):
            required_string_fields = ['quotation']
            required_list_fields = ['keywords']
            missing_fields = []
            for field in required_string_fields:
                if field not in transcript or not isinstance(transcript[field], str) or not transcript[field].strip():
                    missing_fields.append(field)
            for field in required_list_fields:
                if field not in transcript or not isinstance(transcript[field], list) or not transcript[field]:
                    missing_fields.append(field)
            if missing_fields:
                logger.warning(f"Coding analysis transcript at {idx} missing required fields {missing_fields}. Skipping.")
                continue
        elif isinstance(module, ThemedevelopmentAnalysisModule):
            # For theme development now only:
            # research_objectives (str), codes (list), theoretical_framework (dict)
            if 'research_objectives' not in transcript or not isinstance(transcript['research_objectives'], str) or not transcript['research_objectives'].strip():
                logger.warning(f"Theme transcript at index {idx} missing or empty 'research_objectives'. Skipping.")
                continue
            if 'codes' not in transcript or not isinstance(transcript['codes'], list) or not transcript['codes']:
                logger.warning(f"Theme transcript at index {idx} missing or empty 'codes'. Skipping.")
                continue
            if 'theoretical_framework' not in transcript or not isinstance(transcript['theoretical_framework'], dict):
                logger.warning(f"Theme transcript at index {idx} missing or invalid 'theoretical_framework'. Skipping.")
                continue
        else:
            logger.warning(f"Unknown module type for transcript at index {idx}. Skipping.")
            continue
        valid_transcripts.append(transcript)

    logger.info(f"Validated {len(valid_transcripts)} transcripts out of {len(transcripts)} provided.")
    return valid_transcripts

@handle_exceptions
async def process_single_transcript_quotation(
    transcript_item: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    module: dspy.Module
) -> Dict[str, Any]:
    transcript_chunk = transcript_item.get('transcript_chunk', '').strip()
    if not transcript_chunk:
        logger.warning("Transcript chunk is empty. Skipping.")
        return {}

    logger.info(f"Processing transcript chunk for quotation: {transcript_chunk[:100]}...")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    research_objectives = transcript_item.get('research_objectives', 'Extract relevant quotations based on the provided objectives.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    response = module.forward(
        research_objectives=research_objectives,
        transcript_chunk=transcript_chunk,
        contextualized_contents=contextualized_contents,
        theoretical_framework=theoretical_framework
    )
    
    result = {
        "transcript_info": response.get("transcript_info", {
            "transcript_chunk": transcript_chunk,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        }),
        "retrieved_chunks": retrieved_docs,
        "retrieved_chunks_count": len(retrieved_docs),
        "filtered_chunks_count": len(filtered_chunks),
        "contextualized_contents": contextualized_contents,
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "quotations": response.get("quotations", []),
        "analysis": response.get("analysis", {}),
        "answer": response.get("answer", {})
    }

    if not result["quotations"]:
        logger.warning(f"No quotations selected for transcript chunk: '{transcript_chunk[:100]}...'")
        result["answer"] = {"answer": "No relevant quotations were found to generate an answer."}

    logger.info(f"Selected {len(result['quotations'])} quotations for transcript chunk.")
    return result

@handle_exceptions
async def process_single_transcript_keyword(
    transcript_item: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    module: dspy.Module
) -> Dict[str, Any]:
    quotation = transcript_item.get('quotation', '').strip()
    if not quotation:
        logger.warning("Quotation is empty. Skipping.")
        return {}

    logger.info(f"Processing quotation for keywords: {quotation[:100]}...")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    research_objectives = transcript_item.get('research_objectives', 'Extract relevant keywords based on the provided objectives.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    response = module.forward(
        research_objectives=research_objectives,
        quotation=quotation,
        contextualized_contents=contextualized_contents,
        theoretical_framework=theoretical_framework
    )
    
    result = {
        "quotation_info": response.get("quotation_info", {
            "quotation": quotation,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        }),
        "retrieved_chunks": retrieved_docs,
        "retrieved_chunks_count": len(retrieved_docs),
        "filtered_chunks_count": len(filtered_chunks),
        "contextualized_contents": contextualized_contents,
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "keywords": response.get("keywords", []),
        "analysis": response.get("analysis", {})
    }

    if not result["keywords"]:
        logger.warning(f"No keywords extracted for quotation: '{quotation[:100]}...'")
        result["analysis"]["error"] = "No relevant keywords were found to analyze."

    logger.info(f"Extracted {len(result['keywords'])} keywords for quotation.")
    return result

@handle_exceptions
async def process_single_transcript_coding(
    transcript_item: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    module: dspy.Module
) -> Dict[str, Any]:
    quotation = transcript_item.get('quotation', '').strip()
    keywords = transcript_item.get('keywords', [])
    if not quotation:
        logger.warning("Quotation is missing or empty. Skipping.")
        return {}
    if not keywords:
        logger.warning("Keywords are missing or empty. Skipping.")
        return {}

    logger.info(f"Processing transcript for coding analysis: Quotation='{quotation[:100]}...', Keywords={keywords}")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    research_objectives = transcript_item.get('research_objectives', 'Perform comprehensive coding analysis based on the provided quotation and keywords.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    response = module.forward(
        research_objectives=research_objectives,
        quotation=quotation,
        keywords=keywords,
        contextualized_contents=contextualized_contents,
        theoretical_framework=theoretical_framework
    )

    result = {
        "coding_info": response.get("coding_info", {
            "quotation": quotation,
            "keywords": keywords,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        }),
        "retrieved_chunks": retrieved_docs,
        "retrieved_chunks_count": len(retrieved_docs),
        "filtered_chunks_count": len(filtered_chunks),
        "contextualized_contents": contextualized_contents,
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "codes": response.get("codes", []),
        "analysis": response.get("analysis", {})
    }

    if not result["codes"]:
        logger.warning(f"No codes developed for quotation: '{quotation[:100]}...'")
        result["analysis"]["error"] = "No relevant codes were developed for analysis."

    logger.info(f"Developed {len(result['codes'])} codes for coding analysis.")
    return result

@handle_exceptions
async def process_single_transcript_theme(
    transcript_item: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    module: dspy.Module
) -> Dict[str, Any]:
    """
    Processes a single transcript item for theme development without quotation and keywords.
    """
    research_objectives = transcript_item.get('research_objectives', '').strip()
    theoretical_framework = transcript_item.get('theoretical_framework', {})
    codes = transcript_item.get('codes', [])

    if not research_objectives:
        logger.warning("Research objectives are missing or empty for theme development. Skipping.")
        return {}
    if not theoretical_framework:
        logger.warning("Theoretical framework is missing or empty for theme development. Skipping.")
        return {}
    if not codes:
        logger.warning("Codes are missing or empty for theme development. Skipping.")
        return {}

    logger.info(f"Processing transcript for theme development: Research Objectives='{research_objectives[:100]}...', Codes Count={len(codes)}")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    response = module.forward(
        research_objectives=research_objectives,
        codes=codes,
        theoretical_framework=theoretical_framework
    )

    result = {
        "theme_info": {
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        },
        "retrieved_chunks": retrieved_docs,
        "retrieved_chunks_count": len(retrieved_docs),
        "filtered_chunks_count": len(filtered_chunks),
        "contextualized_contents": contextualized_contents,
        "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "themes": response.get("themes", []),
        "analysis": response.get("analysis", {})
    }

    if not result["themes"]:
        logger.warning("No themes developed.")
        result["analysis"]["error"] = "No themes were developed."

    logger.info(f"Developed {len(result['themes'])} themes for theme development.")
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
    logger.info(f"Starting to process transcripts for output file '{output_file}'.")
    reranker_config = RerankerConfig(
        reranker_type=RerankerType.COHERE,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        st_weight=0.5
    )
    all_results = []

    if isinstance(module, (SelectQuotationModule, EnhancedQuotationModule)):
        process_func = process_single_transcript_quotation
    elif isinstance(module, KeywordExtractionModule):
        process_func = process_single_transcript_keyword
    elif isinstance(module, CodingAnalysisModule):
        process_func = process_single_transcript_coding
    elif isinstance(module, ThemedevelopmentAnalysisModule):
        process_func = process_single_transcript_theme
    else:
        logger.error("Unsupported module type provided.")
        return

    for idx, transcript_item in enumerate(tqdm(transcripts, desc="Processing transcripts")):
        try:
            # Define query based on module type
            if isinstance(module, (SelectQuotationModule, EnhancedQuotationModule)):
                query = transcript_item.get('transcript_chunk', '')
            elif isinstance(module, KeywordExtractionModule):
                query = transcript_item.get('quotation', '')
            elif isinstance(module, CodingAnalysisModule):
                query = transcript_item.get('quotation', '')  # or another relevant field
            elif isinstance(module, ThemedevelopmentAnalysisModule):
                query = transcript_item.get('research_objectives', '')
            else:
                query = ''

            if not query:
                logger.warning(f"No query found for transcript at index {idx}. Skipping retrieval.")
                retrieved_docs = []
            else:
                retrieved_docs = retrieve_with_reranking(
                    query=query,
                    db=db,
                    es_bm25=es_bm25,
                    k=k,
                    reranker_config=reranker_config
                )

            result = await process_func(
                transcript_item=transcript_item,
                retrieved_docs=retrieved_docs,
                module=module
            )

            if result:
                all_results.append(result)
                
        except Exception as e:
            logger.error(f"Error processing transcript at index {idx}: {e}", exc_info=True)

    # Save all results to the output file
    save_results(all_results, output_file)
