"""
Pipeline evaluation module.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from typing import List, Dict, Any, Callable
from tqdm import tqdm

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.utils.logger import setup_logging
from src.analysis.metrics import comprehensive_metric, is_answer_fully_correct

setup_logging()
logger = logging.getLogger(__name__)

class PipelineEvaluator:
    def __init__(self, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, retrieval_function: Callable):
        self.db = db
        self.es_bm25 = es_bm25
        self.retrieval_function = retrieval_function

    def evaluate_pipeline(self, queries: List[Dict[str, Any]], k: int = 20) -> Dict[str, float]:
        total_score = 0
        total_queries = len(queries)
        queries_with_golden = 0
        queries_without_golden = 0

        semantic_success = 0
        bm25_contextual_success = 0
        total_semantic_hits = 0
        total_bm25_contextual_hits = 0

        logger.info(f"Starting evaluation of {total_queries} queries.")

        for query_item in tqdm(queries, desc="Evaluating retrieval"):
            query = query_item.get('query', '').strip()

            has_golden_data = all([
                'golden_doc_uuids' in query_item,
                'golden_chunk_uuids' in query_item,
                'golden_documents' in query_item
            ])

            if has_golden_data:
                queries_with_golden += 1
                golden_chunk_uuids = query_item.get('golden_chunk_uuids', [])
                golden_contents = []

                for doc_uuid, chunk_index in golden_chunk_uuids:
                    golden_doc = next((doc for doc in query_item.get('golden_documents', []) if doc.get('uuid') == doc_uuid), None)
                    if not golden_doc:
                        logger.debug(f"No document found with UUID '{doc_uuid}' for query '{query}'.")
                        continue
                    golden_chunk = next((chunk for chunk in golden_doc.get('chunks', []) if chunk.get('index') == chunk_index), None)
                    if not golden_chunk:
                        logger.debug(f"No chunk found with index '{chunk_index}' in document '{doc_uuid}' for query '{query}'.")
                        continue
                    golden_contents.append(golden_chunk.get('content', '').strip())

                if not golden_contents:
                    logger.warning(f"No golden contents found for query '{query}'. Skipping evaluation for this query.")
                    continue

                retrieved_docs = self.retrieval_function(query, self.db, self.es_bm25, k)

                chunks_found = 0
                semantic_hits = 0
                bm25_contextual_hits = 0

                for golden_content in golden_contents:
                    for doc in retrieved_docs[:k]:
                        retrieved_content = doc.get('chunk', {}).get('original_content', '').strip()
                        contextualized_content = doc.get('chunk', {}).get('contextualized_content', '').strip()
                        if retrieved_content == golden_content:
                            chunks_found += 1
                            semantic_hits += 1
                            break
                        elif contextualized_content == golden_content:
                            chunks_found += 1
                            bm25_contextual_hits += 1
                            break

                query_score = chunks_found / len(golden_contents)
                total_score += query_score
                logger.debug(f"Query '{query}' score: {query_score}")

                semantic_success += semantic_hits
                bm25_contextual_success += bm25_contextual_hits
                total_semantic_hits += semantic_hits
                total_bm25_contextual_hits += bm25_contextual_hits
            else:
                queries_without_golden += 1
                logger.debug(f"Query '{query}' does not contain golden data. Skipping evaluation metrics for this query.")
                continue

        average_score = (total_score / queries_with_golden) if queries_with_golden > 0 else 0
        pass_at_n = average_score * 100

        semantic_percentage = (semantic_success / total_semantic_hits) * 100 if total_semantic_hits > 0 else 0
        bm25_contextual_percentage = (bm25_contextual_success / total_bm25_contextual_hits) * 100 if total_bm25_contextual_hits > 0 else 0

        logger.info(f"Evaluation completed.")
        logger.info(f"Total Queries: {total_queries}")
        logger.info(f"Queries with Golden Data: {queries_with_golden}")
        logger.info(f"Queries without Golden Data: {queries_without_golden}")
        logger.info(f"Pass@{k}: {pass_at_n:.2f}%, Average Score: {average_score:.4f}")
        logger.info(f"Semantic Hits: {semantic_percentage:.2f}%")
        logger.info(f"BM25 Contextual Hits: {bm25_contextual_percentage:.2f}%")

        return {
            "pass_at_n": pass_at_n,
            "average_score": average_score,
            "semantic_hit_percentage": semantic_percentage,
            "bm25_contextual_hit_percentage": bm25_contextual_percentage,
            "total_queries": total_queries,
            "queries_with_golden": queries_with_golden,
            "queries_without_golden": queries_without_golden
        }

    def evaluate_complete_pipeline(self, k_values: List[int], evaluation_set: List[Dict[str, Any]]):
        for k in k_values:
            logger.info(f"Starting evaluation for Pass@{k}")
            results = self.evaluate_pipeline(evaluation_set, k)
            logger.info(f"Pass@{k}: {results['pass_at_n']:.2f}%")
            logger.info(f"Average Score: {results['average_score']:.4f}")
            logger.info(f"Semantic Hit Percentage: {results['semantic_hit_percentage']:.2f}%")
            logger.info(f"BM25 Contextual Hit Percentage: {results['bm25_contextual_hit_percentage']:.2f}%")
            logger.info(f"Total Queries: {results['total_queries']}")
            logger.info(f"Queries with Golden Data: {results.get('queries_with_golden', 0)}")
            logger.info(f"Queries without Golden Data: {results.get('queries_without_golden', 0)}\n")