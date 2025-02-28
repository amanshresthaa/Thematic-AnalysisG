# File: retrieval/retrieval.py
# ------------------------------------------------------------------------------
import logging
import time
from typing import List, Dict, Any, Tuple, Callable
import dspy

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.utils.logger import setup_logging, log_execution_time
from src.core.retrieval.query_generator import QueryGeneratorSignature
from src.utils.utils import compute_similarity

setup_logging()
logger = logging.getLogger(__name__)


def _search_with_weights(
    query: str,
    k: int,
    search_func: Callable[[str, int], List[Dict[str, Any]]]
) -> Tuple[List[str], Dict[str, float]]:
    """
    Helper to perform a search and return both ranked chunk IDs and their scores.
    """
    results = search_func(query, k=k)
    ranked_ids = [result['chunk_id'] for result in results]
    scores = {result['chunk_id']: result['score'] for result in results}
    return ranked_ids, scores


def _compute_rrf_score(chunk_id: str, ranked_list: List[str], weight: float) -> float:
    """
    Compute the reciprocal rank fusion (RRF) score for a given chunk ID.
    """
    if chunk_id in ranked_list:
        index = ranked_list.index(chunk_id)
        return weight * (1 / (index + 1))
    return 0.0


def _combine_rrf_scores(
    ranked_semantic: List[str],
    ranked_bm25_content: List[str],
    ranked_bm25_contextual: List[str],
    semantic_weight: float,
    bm25_content_weight: float,
    bm25_contextual_weight: float
) -> Dict[str, float]:
    """
    Combine scores from multiple retrieval strategies using RRF.
    """
    combined_scores = {}
    all_chunk_ids = set(ranked_semantic + ranked_bm25_content + ranked_bm25_contextual)
    for chunk_id in all_chunk_ids:
        score = (
            _compute_rrf_score(chunk_id, ranked_semantic, semantic_weight) +
            _compute_rrf_score(chunk_id, ranked_bm25_content, bm25_content_weight) +
            _compute_rrf_score(chunk_id, ranked_bm25_contextual, bm25_contextual_weight)
        )
        combined_scores[chunk_id] = score
    return combined_scores


def _generate_new_query(
    current_query: str,
    accumulated_context: str,
    query_generator: dspy.ChainOfThought
) -> str:
    """
    Uses the query generator to create a refined query based on the accumulated context.
    """
    response = query_generator(question=current_query, context=accumulated_context)
    new_query = response.get('new_query', '').strip()
    return new_query


@log_execution_time(logger)
def hybrid_retrieval(
    query: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    semantic_weight: float = 0.2,
    bm25_content_weight: float = 0.2,
    bm25_contextual_weight: float = 0.6,
    min_chunks: int = 1
) -> List[Dict[str, Any]]:
    """
    Performs hybrid retrieval by combining FAISS semantic search and dual BM25
    contextual search using Reciprocal Rank Fusion.
    """
    logger.debug(
        f"Entering hybrid_retrieval with query='{query}', k={k}, "
        f"semantic_weight={semantic_weight}, bm25_content_weight={bm25_content_weight}, "
        f"bm25_contextual_weight={bm25_contextual_weight}, min_chunks={min_chunks}."
    )
    start_time = time.time()
    num_chunks_to_recall = k * 10  # Retrieve more to improve chances

    while True:
        # Perform searches using the helper function.
        logger.debug(f"Performing semantic search using FAISS for query: '{query}'")
        ranked_semantic, _ = _search_with_weights(query, num_chunks_to_recall, db.search)
        logger.debug(f"Semantic search retrieved {len(ranked_semantic)} chunk IDs.")

        logger.debug(f"Performing BM25 search on 'content' for query: '{query}'")
        ranked_bm25_content, _ = _search_with_weights(query, num_chunks_to_recall, es_bm25.search_content)
        logger.debug(f"BM25 'content' search retrieved {len(ranked_bm25_content)} chunk IDs.")

        logger.debug(f"Performing BM25 search on 'contextualized_content' for query: '{query}'")
        ranked_bm25_contextual, _ = _search_with_weights(query, num_chunks_to_recall, es_bm25.search_contextualized)
        logger.debug(f"BM25 'contextualized_content' search retrieved {len(ranked_bm25_contextual)} chunk IDs.")

        # Combine the scores from different search methods.
        combined_scores = _combine_rrf_scores(
            ranked_semantic, ranked_bm25_content, ranked_bm25_contextual,
            semantic_weight, bm25_content_weight, bm25_contextual_weight
        )
        sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True)
        logger.debug("Sorted chunk IDs based on RRF scores.")

        # Filter final results by matching metadata.
        final_results = []
        filtered_count = 0
        for chunk_id in sorted_chunk_ids[:k]:
            chunk_metadata = next((chunk for chunk in db.metadata if chunk['chunk_id'] == chunk_id), None)
            if not chunk_metadata:
                filtered_count += 1
                logger.warning(f"Chunk metadata not found for chunk_id {chunk_id}")
                continue
            final_results.append({
                'chunk': chunk_metadata,
                'score': combined_scores[chunk_id]
            })

        logger.info(f"Filtered {filtered_count} chunks due to missing metadata.")
        logger.info(f"Total chunks retrieved after filtering: {len(final_results)} (required min_chunks={min_chunks})")

        if len(final_results) >= min_chunks or k >= num_chunks_to_recall:
            break
        else:
            k += 5  # Increment k to retrieve more chunks
            logger.info(
                f"Retrieved chunks ({len(final_results)}) less than min_chunks ({min_chunks}). "
                f"Increasing k to {k} and retrying."
            )

    elapsed_time = time.time() - start_time
    logger.debug(f"Exiting hybrid_retrieval method. Time taken: {elapsed_time:.2f} seconds.")
    logger.debug(f"Hybrid retrieval returning {len(final_results)} chunks.")
    logger.info(
        f"Chunks used for hybrid retrieval for query '{query}': "
        f"[{', '.join([res['chunk']['chunk_id'] for res in final_results])}]"
    )
    return final_results


@log_execution_time(logger)
def multi_stage_retrieval(
    query: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    max_hops: int = 3,
    max_results: int = 5,
    similarity_threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Performs multi-stage retrieval by iteratively refining the query based on retrieved context.
    """
    accumulated_context = ""
    all_retrieved_chunks = {}
    current_query = query
    query_generator = dspy.ChainOfThought(QueryGeneratorSignature)

    for hop in range(max_hops):
        logger.info(f"Starting hop {hop+1} with query: '{current_query}'")
        retrieved_chunks = hybrid_retrieval(current_query, db, es_bm25, k)

        # Merge retrieved chunks.
        for chunk in retrieved_chunks:
            chunk_id = chunk['chunk']['chunk_id']
            if chunk_id not in all_retrieved_chunks:
                all_retrieved_chunks[chunk_id] = chunk

        if len(all_retrieved_chunks) >= max_results:
            logger.info(f"Retrieved sufficient chunks ({len(all_retrieved_chunks)}). Terminating.")
            break

        # Accumulate new context from retrieved chunks.
        new_context = "\n\n".join([
            chunk['chunk'].get('contextualized_content', '') or chunk['chunk'].get('original_content', '')
            for chunk in retrieved_chunks
        ])
        accumulated_context += "\n\n" + new_context

        # Generate a new query based on the accumulated context.
        new_query = _generate_new_query(current_query, accumulated_context, query_generator)
        if not new_query:
            logger.info("No new query generated. Terminating multi-stage retrieval.")
            break

        similarity = compute_similarity(current_query, new_query)
        logger.debug(f"Similarity between queries: {similarity:.4f}")
        if similarity >= similarity_threshold:
            logger.info("New query is too similar to the current query. Terminating multi-stage retrieval.")
            break

        current_query = new_query

    final_results = sorted(all_retrieved_chunks.values(), key=lambda x: x['score'], reverse=True)[:max_results]
    logger.info(f"Multi-stage retrieval completed with {len(final_results)} chunks.")
    return final_results
