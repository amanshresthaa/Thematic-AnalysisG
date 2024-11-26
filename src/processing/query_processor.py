# processing/query_processor.py
import logging
from typing import List, Dict, Any
import json
from tqdm import tqdm
import asyncio

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.retrieval.retrieval import multi_stage_retrieval
from src.utils.logger import setup_logging
from src.decorators import handle_exceptions
import dspy

from src.analysis.select_quotation_module import EnhancedQuotationModule
from src.evaluation.evaluation import PipelineEvaluator
from src.analysis.metrics import comprehensive_metric

logger = logging.getLogger(__name__)

class QuotationStructure:
    """Defines the exact structure for quotation analysis output."""
    def __init__(self):
        self.structure = {
            "transcript_info": {
                "transcript_chunk": "",                    # Selected transcript content
                "research_objectives": "",                 # Research goals guiding analysis
                "theoretical_framework": {
                    "theory": "",                         # Primary theoretical approach
                    "philosophical_approach": "",          # Philosophical foundation
                    "rationale": ""                       # Justification for approach
                }
            },
            "retrieved_chunks": [],                       # Retrieved document chunks
            "retrieved_chunks_count": 0,                  # Count of retrieved chunks
            "filtered_chunks_count": 0,                   # Count of filtered chunks
            "contextualized_contents": [],                # List of contextualized contents
            "used_chunk_ids": [],                        # List of used chunk IDs
            "quotations": [],                            # List of analyzed quotations
            "analysis": {
                "philosophical_underpinning": "",         # Analysis approach
                "patterns_identified": [""],              # Key patterns found
                "theoretical_interpretation": "",         # Framework application
                "methodological_reflection": {
                    "pattern_robustness": "",            # Pattern evidence
                    "theoretical_alignment": "",          # Framework fit
                    "researcher_reflexivity": ""         # Interpretation awareness
                },
                "practical_implications": ""             # Applied insights
            },
            "answer": {
                "summary": "",                           # Key findings
                "theoretical_contribution": "",          # Theory advancement
                "methodological_contribution": {
                    "approach": "",                     # Method used
                    "pattern_validity": "",             # Evidence quality
                    "theoretical_integration": ""       # Theory-data synthesis
                }
            }
        }

    def create_quotation(self) -> Dict[str, Any]:
        """Create a properly structured quotation entry."""
        return {
            "quotation": "",                             # Exact quote text
            "creswell_category": "",                     # longer/discrete/embedded
            "classification": "",                        # Content type
            "context": {
                "preceding_question": "",                # Prior question
                "situation": "",                         # Context description
                "pattern_representation": ""             # Pattern linkage
            },
            "analysis_value": {
                "relevance": "",                        # Research objective alignment
                "pattern_support": "",                  # Pattern evidence
                "theoretical_alignment": ""             # Framework connection
            }
        }

    def get_empty_structure(self) -> Dict[str, Any]:
        """Return a copy of the empty structure."""
        return json.loads(json.dumps(self.structure))  # Deep copy

def validate_queries(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validates the structure of input transcripts."""
    valid_transcripts = []
    required_fields = ['transcript_chunk', 'research_objectives', 'theoretical_framework']
    framework_fields = ['theory', 'philosophical_approach', 'rationale']
    
    for idx, transcript in enumerate(transcripts):
        # Check for required fields
        if not all(field in transcript for field in required_fields):
            logger.warning(f"Transcript at index {idx} missing required fields. Skipping.")
            continue
            
        # Validate transcript chunk
        if not transcript['transcript_chunk'].strip():
            logger.warning(f"Transcript at index {idx} has empty transcript_chunk. Skipping.")
            continue
            
        # Validate theoretical framework structure
        framework = transcript.get('theoretical_framework', {})
        if not isinstance(framework, dict) or not all(field in framework for field in framework_fields):
            logger.warning(f"Transcript at index {idx} has invalid theoretical_framework structure. Skipping.")
            continue
            
        valid_transcripts.append(transcript)

    logger.info(f"Validated {len(valid_transcripts)} transcripts out of {len(transcripts)} provided.")
    return valid_transcripts

async def process_single_transcript(
    transcript_item: Dict[str, Any],
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    module: dspy.Module
) -> Dict[str, Any]:
    """
    Processes a single transcript chunk with exact output structure.
    """
    structure = QuotationStructure()
    result = structure.get_empty_structure()
    
    try:
        # Extract transcript information
        transcript_chunk = transcript_item.get('transcript_chunk', '').strip()
        if not transcript_chunk:
            logger.warning("Empty transcript chunk provided.")
            return result

        # Update transcript info
        result['transcript_info'].update({
            'transcript_chunk': transcript_chunk,
            'research_objectives': transcript_item.get('research_objectives', 'N/A'),
            'theoretical_framework': transcript_item.get('theoretical_framework', {
                "theory": "N/A",
                "philosophical_approach": "N/A",
                "rationale": "N/A"
            })
        })

        # Retrieve and process chunks
        retrieved_chunks = retrieve_documents(transcript_chunk, db, es_bm25, k)
        result['retrieved_chunks'] = retrieved_chunks
        result['retrieved_chunks_count'] = len(retrieved_chunks)

        # Filter chunks
        filtered_chunks = [chunk for chunk in retrieved_chunks if chunk['score'] >= 0.7]
        result['filtered_chunks_count'] = len(filtered_chunks)

        # Process contextualized contents
        contextualized_contents = [
            chunk['chunk'].get('contextualized_content', 'N/A') 
            for chunk in filtered_chunks
        ]
        result['contextualized_contents'] = contextualized_contents
        result['used_chunk_ids'] = [chunk['chunk']['chunk_id'] for chunk in filtered_chunks]

        # Process with module
        module_response = module.forward(
            research_objectives=result['transcript_info']['research_objectives'],
            transcript_chunk=transcript_chunk,
            contextualized_contents=contextualized_contents,
            theoretical_framework=result['transcript_info']['theoretical_framework']
        )

        # Update with module response
        if module_response:
            # Process quotations
            result['quotations'] = []
            for q in module_response.get('quotations', []):
                quotation = structure.create_quotation()
                quotation.update({
                    'quotation': q.get('quotation', 'N/A'),
                    'creswell_category': q.get('creswell_category', 'N/A'),
                    'classification': q.get('classification', 'N/A'),
                    'context': q.get('context', {
                        "preceding_question": "N/A",
                        "situation": "N/A",
                        "pattern_representation": "N/A"
                    }),
                    'analysis_value': q.get('analysis_value', {
                        "relevance": "N/A",
                        "pattern_support": "N/A",
                        "theoretical_alignment": "N/A"
                    })
                })
                result['quotations'].append(quotation)

            # Update analysis and answer sections
            if 'analysis' in module_response:
                analysis = module_response['analysis']
                result['analysis'].update({
                    'philosophical_underpinning': analysis.get('philosophical_underpinning', 'N/A'),
                    'patterns_identified': analysis.get('patterns_identified', ["N/A"]),
                    'theoretical_interpretation': analysis.get('theoretical_interpretation', 'N/A'),
                    'methodological_reflection': analysis.get('methodological_reflection', {
                        "pattern_robustness": "N/A",
                        "theoretical_alignment": "N/A",
                        "researcher_reflexivity": "N/A"
                    }),
                    'practical_implications': analysis.get('practical_implications', 'N/A')
                })
            if 'answer' in module_response:
                answer = module_response['answer']
                result['answer'].update({
                    'summary': answer.get('summary', 'N/A'),
                    'theoretical_contribution': answer.get('theoretical_contribution', 'N/A'),
                    'methodological_contribution': answer.get('methodological_contribution', {
                        "approach": "N/A",
                        "pattern_validity": "N/A",
                        "theoretical_integration": "N/A"
                    })
                })

        if not result['quotations']:
            logger.warning(f"No quotations selected for transcript chunk: '{transcript_chunk[:100]}...'")
            result['answer']['summary'] = "No relevant quotations were found to generate an answer."

        logger.info(f"Selected {len(result['quotations'])} quotations for transcript chunk.")
        return result

    except Exception as e:
        logger.error(f"Error processing transcript: {e}", exc_info=True)
        return result

def retrieve_documents(
    transcript_chunk: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int
) -> List[Dict[str, Any]]:
    """Retrieves documents using multi-stage retrieval."""
    try:
        logger.debug(f"Retrieving documents for transcript chunk: '{transcript_chunk[:100]}...'")
        final_results = multi_stage_retrieval(transcript_chunk, db, es_bm25, k)
        logger.debug(f"Retrieved {len(final_results)} results")
        return final_results
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        return []

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Saves results with exact structure."""
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)

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
    Process transcripts with exact output structure.
    """
    logger.info(f"Processing {len(transcripts)} transcripts")
    
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

        save_results(all_results, output_file)

    except KeyboardInterrupt:
        logger.warning("Process interrupted. Saving partial results.")
        save_results(all_results, output_file)
        raise
    except Exception as e:
        logger.error(f"Error in process_queries: {e}", exc_info=True)
        raise

    return all_results