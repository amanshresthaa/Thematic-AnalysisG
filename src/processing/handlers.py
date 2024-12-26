# src/processing/handlers.py

from typing import List, Dict, Any
import os
import json

import dspy

from .base import BaseHandler
from .logger import get_logger
from .config import INFO_PATH, DEFAULT_RESEARCH_OBJECTIVES, DEFAULT_THEORETICAL_FRAMEWORK
from src.decorators import handle_exceptions  # Reuse your existing decorator if needed

logger = get_logger(__name__)


class QuotationHandler(BaseHandler):
    @handle_exceptions
    async def process_single_transcript(
        self,
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
            logger.warning("No quotations selected.")
            result["answer"] = {"answer": "No relevant quotations were found."}

        logger.debug(f"Selected {len(result['quotations'])} quotations.")
        return result


class KeywordHandler(BaseHandler):
    @handle_exceptions
    async def process_single_transcript(
        self,
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
            logger.warning("No keywords extracted.")
            result["analysis"]["error"] = "No relevant keywords found."

        logger.debug(f"Extracted {len(result['keywords'])} keywords.")
        return result


class CodingHandler(BaseHandler):
    @handle_exceptions
    async def process_single_transcript(
        self,
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
            logger.warning("No codes developed.")
            result["analysis"]["error"] = "No codes were developed."

        logger.debug(f"Developed {len(result['codes'])} codes.")
        return result


class GroupingHandler(BaseHandler):
    @handle_exceptions
    async def process_single_transcript(
        self,
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

        if os.path.exists(INFO_PATH):
            with open(INFO_PATH, 'r', encoding='utf-8') as f:
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
            "grouping_info": {
                "codes": codes,
                "research_objectives": research_objectives,
                "theoretical_framework": theoretical_framework
            },
            "retrieved_chunks": retrieved_docs,
            "retrieved_chunks_count": len(retrieved_docs),
            "filtered_chunks_count": len(filtered_chunks),
            "used_chunk_ids": [chunk['chunk']['chunk_id'] for chunk in filtered_chunks],
            "groupings": batched_groupings
        }

        if not batched_groupings:
            logger.warning("No groupings formed.")
            result["error"] = "No groupings could be formed."

        logger.debug(f"Developed {len(batched_groupings)} groupings.")
        return result


class ThemeHandler(BaseHandler):
    @handle_exceptions
    async def process_single_transcript(
        self,
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
            "theme_info": response.get("theme_info", {
                "groupings": groupings,
                "research_objectives": research_objectives,
                "theoretical_framework": theoretical_framework
            }),
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
            if "analysis" in result and isinstance(result["analysis"], dict):
                result["analysis"]["error"] = "No themes were developed."
            else:
                result["analysis"] = {"error": "No themes were developed."}

        logger.debug(f"Developed {len(result['themes'])} themes.")
        return result
