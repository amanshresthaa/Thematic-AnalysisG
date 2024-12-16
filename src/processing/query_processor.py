# src/processing/query_processor.py

import logging
import time
from typing import List, Dict, Any
import json
from tqdm import tqdm
import os

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions
import dspy

from src.analysis.select_quotation_module import SelectQuotationModule, EnhancedQuotationModule
from src.analysis.extract_keyword_module import KeywordExtractionModule
from src.analysis.coding_module import CodingAnalysisModule
from src.analysis.theme_development_module import ThemedevelopmentAnalysisModule
from src.analysis.grouping_module import GroupingAnalysisModule

setup_logging()
logger = logging.getLogger(__name__)

def validate_queries(transcripts: List[Dict[str, Any]], module: dspy.Module) -> List[Dict[str, Any]]:
    valid_transcripts = []
    for idx, transcript in enumerate(transcripts):
        if isinstance(module, (SelectQuotationModule, EnhancedQuotationModule)):
            if 'transcript_chunk' not in transcript or not isinstance(transcript['transcript_chunk'], str) or not transcript['transcript_chunk'].strip():
                logger.warning(f"Transcript at index {idx} missing or invalid 'transcript_chunk'. Skipping.")
                continue
        elif isinstance(module, KeywordExtractionModule):
            if 'quotation' not in transcript or not isinstance(transcript['quotation'], str) or not transcript['quotation'].strip():
                logger.warning(f"Transcript at index {idx} missing or invalid 'quotation'. Skipping.")
                continue
        elif isinstance(module, CodingAnalysisModule):
            required_string_fields = ['quotation']
            required_list_fields = ['keywords']
            missing_fields = []

            for field in required_string_fields:
                if field not in transcript or not isinstance(transcript[field], str) or not transcript[field].strip():
                    missing_fields.append(field)
            for field in required_list_fields:
                if field not in transcript or not isinstance(transcript[field], list) or not all(isinstance(kw, str) and kw.strip() for kw in transcript[field]):
                    missing_fields.append(field)

            if missing_fields:
                logger.warning(f"Transcript at index {idx} missing required fields {missing_fields}. Skipping.")
                continue
        elif isinstance(module, ThemedevelopmentAnalysisModule):
            if 'quotation' not in transcript or not isinstance(transcript['quotation'], str) or not transcript['quotation'].strip():
                logger.warning(f"Transcript at index {idx} missing or invalid 'quotation'. Skipping.")
                continue
            if 'keywords' not in transcript or not isinstance(transcript['keywords'], list) or not transcript['keywords']:
                logger.warning(f"Transcript at index {idx} missing or invalid 'keywords'. Skipping.")
                continue
            if 'codes' not in transcript or not isinstance(transcript['codes'], list) or not transcript['codes']:
                logger.warning(f"Transcript at index {idx} missing or invalid 'codes'. Skipping.")
                continue
        elif isinstance(module, GroupingAnalysisModule):
            if 'codes' not in transcript or not isinstance(transcript['codes'], list) or not transcript['codes']:
                logger.warning(f"Transcript at index {idx} missing or invalid 'codes' for grouping. Skipping.")
                continue
        else:
            logger.warning(f"Unknown module type for transcript at index {idx}. Skipping.")
            continue

        valid_transcripts.append(transcript)

    logger.info(f"Validated {len(valid_transcripts)}/{len(transcripts)} transcripts.")
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

    logger.debug(f"Processing transcript chunk for quotation: {transcript_chunk[:100]}...")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    research_objectives = transcript_item.get('research_objectives', 'Extract relevant quotations.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    response = module.forward(
        research_objectives=research_objectives,
        transcript_chunk=transcript_chunk,
        contextualized_contents=contextualized_contents,
        theoretical_framework=theoretical_framework
    )
    
    result = {
        "transcriptInfo": response.get("transcript_info", {
            "transcript_chunk": transcript_chunk,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        }),
        "retrievedChunks": retrieved_docs,
        "contextualizedContents": contextualized_contents,
        "usedChunkIds": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "quotations": response.get("quotations", []),
        "analysis": response.get("analysis", {}),
        "answer": response.get("answer", {})
    }

    if not result["quotations"]:
        logger.warning("No quotations selected.")
        result["answer"] = {"answer": "No relevant quotations were found."}

    logger.debug(f"Selected {len(result['quotations'])} quotations.")
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

    logger.debug(f"Processing quotation for keywords: {quotation[:100]}...")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    research_objectives = transcript_item.get('research_objectives', 'Extract relevant keywords.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    response = module.forward(
        research_objectives=research_objectives,
        quotation=quotation,
        contextualized_contents=contextualized_contents,
        theoretical_framework=theoretical_framework
    )
    
    result = {
        "quotationInfo": response.get("quotation_info", {
            "quotation": quotation,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        }),
        "retrievedChunks": retrieved_docs,
        "contextualizedContents": contextualized_contents,
        "usedChunkIds": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "keywords": response.get("keywords", []),
        "analysis": response.get("analysis", {})
    }

    if not result["keywords"]:
        logger.warning("No keywords extracted.")
        result["analysis"]["error"] = "No relevant keywords found."

    logger.debug(f"Extracted {len(result['keywords'])} keywords.")
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
        logger.warning("Quotation missing. Skipping.")
        return {}
    if not keywords:
        logger.warning("Keywords missing. Skipping.")
        return {}

    logger.debug(f"Processing coding analysis for quotation: '{quotation[:100]}', Keywords: {keywords}")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    research_objectives = transcript_item.get('research_objectives', 'Perform coding analysis.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    response = module.forward(
        research_objectives=research_objectives,
        quotation=quotation,
        keywords=keywords,
        contextualized_contents=contextualized_contents,
        theoretical_framework=theoretical_framework
    )

    result = {
        "codingInfo": response.get("coding_info", {
            "quotation": quotation,
            "keywords": keywords,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        }),
        "retrievedChunks": retrieved_docs,
        "contextualizedContents": contextualized_contents,
        "usedChunkIds": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "codes": response.get("codes", []),
        "analysis": response.get("analysis", {})
    }

    if not result["codes"]:
        logger.warning("No codes developed.")
        result["analysis"]["error"] = "No codes were developed."

    logger.debug(f"Developed {len(result['codes'])} codes.")
    return result

@handle_exceptions
async def process_single_transcript_grouping(
    transcript_item: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    module: dspy.Module
) -> Dict[str, Any]:
    codes = transcript_item.get('codes', [])
    if not codes:
        logger.warning("No codes provided for grouping. Skipping.")
        return {}

    logger.debug(f"Processing grouping analysis for {len(codes)} codes.")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    info_path = 'data/input/info.json'
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    else:
        info = {
            "research_objectives": "Group codes into themes.",
            "theoretical_framework": {}
        }

    research_objectives = info.get('research_objectives', 'Group codes into themes.')
    theoretical_framework = info.get('theoretical_framework', {})

    batch_size = 20
    batched_groupings = []
    for i in range(0, len(codes), batch_size):
        code_batch = codes[i:i+batch_size]
        response = module.forward(
            research_objectives=research_objectives,
            theoretical_framework=theoretical_framework,
            codes=code_batch
        )
        batch_groupings = response.get("groupings", [])
        batched_groupings.extend(batch_groupings)

    result = {
        "groupingInfo": {
            "codes": codes,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        },
        "retrievedChunks": retrieved_docs,
        "contextualizedContents": contextualized_contents,
        "usedChunkIds": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "groupings": batched_groupings
    }

    if not batched_groupings:
        logger.warning("No groupings formed.")
        result["error"] = "No groupings could be formed."

    logger.debug(f"Developed {len(batched_groupings)} groupings.")
    return result

@handle_exceptions
async def process_single_transcript_theme(
    transcript_item: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    module: dspy.Module
) -> Dict[str, Any]:
    groupings = transcript_item.get('groupings', [])
    if not groupings:
        logger.warning("No groupings provided for theme development. Skipping.")
        return {}

    logger.debug(f"Processing theme development for {len(groupings)} groupings.")
    filtered_chunks = [chunk for chunk in retrieved_docs if chunk['score'] >= 0.7]
    contextualized_contents = [chunk['chunk']['contextualized_content'] for chunk in filtered_chunks]

    research_objectives = transcript_item.get('research_objectives', 'Develop themes from groupings.')
    theoretical_framework = transcript_item.get('theoretical_framework', {})

    quotation = transcript_item.get("quotation", "")
    keywords = transcript_item.get("keywords", [])
    codes = transcript_item.get("codes", [])
    transcript_chunk = transcript_item.get("transcript_chunk", "")

    response = module.forward(
        research_objectives=research_objectives,
        quotation=quotation,
        keywords=keywords,
        codes=codes,
        theoretical_framework=theoretical_framework,
        transcript_chunk=transcript_chunk
    )

    result = {
        "themeInfo": response.get("theme_info", {
            "groupings": groupings,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework
        }),
        "retrievedChunks": retrieved_docs,
        "contextualizedContents": contextualized_contents,
        "usedChunkIds": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
        "themes": response.get("themes", []),
        "analysis": response.get("analysis", {})
    }

    if not result["themes"]:
        logger.warning("No themes developed.")
        if "analysis" in result and isinstance(result["analysis"], dict):
            result["analysis"]["error"] = "No themes were developed."
        else:
            result["analysis"] = {"error": "No themes were developed."}

    logger.debug(f"Developed {len(result['themes'])} themes.")
    return result

def save_results(results: List[Dict[str, Any]], output_file: str, single_object: bool = False):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as outfile:
            if single_object and len(results) == 1:
                json.dump(results[0], outfile, indent=4)
                logger.info(f"Saved single optimized result to '{output_file}'")
            else:
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
    optimized_program: dspy.Program,
    module: dspy.Module,
    single_output: bool = False  # New parameter to handle single object output
):
    logger.info(f"Processing transcripts for '{output_file}'.")
    
    reranker_config = RerankerConfig(
        reranker_type=RerankerType.COHERE,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        st_weight=0.5
    )

    if isinstance(module, (SelectQuotationModule, EnhancedQuotationModule)):
        process_func = process_single_transcript_quotation
    elif isinstance(module, KeywordExtractionModule):
        process_func = process_single_transcript_keyword
    elif isinstance(module, CodingAnalysisModule):
        process_func = process_single_transcript_coding
    elif isinstance(module, GroupingAnalysisModule):
        process_func = process_single_transcript_grouping
    elif isinstance(module, ThemedevelopmentAnalysisModule):
        process_func = process_single_transcript_theme
    else:
        logger.error("Unsupported module type.")
        return

    all_results = []
    for idx, transcript_item in enumerate(tqdm(transcripts, desc="Processing transcripts")):
        try:
            if not isinstance(module, (GroupingAnalysisModule, ThemedevelopmentAnalysisModule)):
                query = transcript_item.get('query', transcript_item.get('transcript_chunk', transcript_item.get('quotation', '')))
                if not query.strip():
                    logger.warning(f"Transcript at index {idx} has no valid query. Skipping.")
                    continue
                retrieved_docs = retrieve_with_reranking(
                    query=query,
                    db=db,
                    es_bm25=es_bm25,
                    k=k,
                    reranker_config=reranker_config
                )
            else:
                retrieved_docs = []

            result = await process_func(
                transcript_item=transcript_item,
                retrieved_docs=retrieved_docs,
                module=module
            )

            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing transcript at index {idx}: {e}", exc_info=True)

    if single_output and len(all_results) == 1:
        save_results(all_results, output_file, single_object=True)
    else:
        save_results(all_results, output_file)
