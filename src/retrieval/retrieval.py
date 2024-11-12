# File: retrieval.py
import logging
import time
from typing import List, Dict, Any
import dspy

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.utils.logger import setup_logging
from src.retrieval.query_generator import QueryGeneratorSignature
from src.utils.utils import compute_similarity
# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


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
    Performs hybrid retrieval by combining FAISS semantic search and dual BM25 contextual search using Reciprocal Rank Fusion.
    """
    logger.debug(
        f"Entering hybrid_retrieval with query='{query}', k={k}, "
        f"semantic_weight={semantic_weight}, bm25_content_weight={bm25_content_weight}, "
        f"bm25_contextual_weight={bm25_contextual_weight}, min_chunks={min_chunks}."
    )
    start_time = time.time()
    num_chunks_to_recall = k * 10  # Retrieve more to improve chances

    # Initialize reciprocal rank fusion score dictionary
    chunk_id_to_score = {}

    while True:
        # Semantic search
        logger.debug(f"Performing semantic search using FAISS for query: '{query}'")
        semantic_results = db.search(query, k=num_chunks_to_recall)
        ranked_semantic = [result['chunk_id'] for result in semantic_results]
        semantic_scores = {result['chunk_id']: result['score'] for result in semantic_results}
        logger.debug(f"Semantic search retrieved {len(ranked_semantic)} chunk IDs.")

        # BM25 search on 'content'
        logger.debug(f"Performing BM25 search on 'content' for query: '{query}'")
        bm25_content_results = es_bm25.search_content(query, k=num_chunks_to_recall)
        ranked_bm25_content = [result['chunk_id'] for result in bm25_content_results]
        bm25_content_scores = {result['chunk_id']: result['score'] for result in bm25_content_results}
        logger.debug(f"BM25 'content' search retrieved {len(ranked_bm25_content)} chunk IDs.")

        # BM25 search on 'contextualized_content'
        logger.debug(f"Performing BM25 search on 'contextualized_content' for query: '{query}'")
        bm25_contextual_results = es_bm25.search_contextualized(query, k=num_chunks_to_recall)
        ranked_bm25_contextual = [result['chunk_id'] for result in bm25_contextual_results]
        bm25_contextual_scores = {result['chunk_id']: result['score'] for result in bm25_contextual_results}
        logger.debug(f"BM25 'contextualized_content' search retrieved {len(ranked_bm25_contextual)} chunk IDs.")

        # Combine all unique chunk IDs
        chunk_ids = list(set(ranked_semantic + ranked_bm25_content + ranked_bm25_contextual))
        logger.debug(f"Total unique chunk IDs after combining: {len(chunk_ids)}")

        # Calculate Reciprocal Rank Fusion scores
        for chunk_id in chunk_ids:
            score = 0
            if chunk_id in ranked_semantic:
                index = ranked_semantic.index(chunk_id)
                score += semantic_weight * (1 / (index + 1))
                logger.debug(
                    f"Added semantic RRF score for chunk_id {chunk_id}: "
                    f"{semantic_weight * (1 / (index + 1))}"
                )
            if chunk_id in ranked_bm25_content:
                index = ranked_bm25_content.index(chunk_id)
                score += bm25_content_weight * (1 / (index + 1))
                logger.debug(
                    f"Added BM25 'content' RRF score for chunk_id {chunk_id}: "
                    f"{bm25_content_weight * (1 / (index + 1))}"
                )
            if chunk_id in ranked_bm25_contextual:
                index = ranked_bm25_contextual.index(chunk_id)
                score += bm25_contextual_weight * (1 / (index + 1))
                logger.debug(
                    f"Added BM25 'contextualized_content' RRF score for chunk_id {chunk_id}: "
                    f"{bm25_contextual_weight * (1 / (index + 1))}"
                )
            chunk_id_to_score[chunk_id] = score

        # Sort chunk IDs by their RRF scores in descending order
        sorted_chunk_ids = sorted(
            chunk_id_to_score.keys(),
            key=lambda x: chunk_id_to_score[x],
            reverse=True
        )
        logger.debug(f"Sorted chunk IDs based on RRF scores.")

        # Select top k chunks, ensuring at least min_chunks are returned
        final_results = []
        filtered_count = 0
        for chunk_id in sorted_chunk_ids[:k]:
            chunk_metadata = next(
                (chunk for chunk in db.metadata if chunk['chunk_id'] == chunk_id),
                None
            )
            if not chunk_metadata:
                filtered_count += 1
                logger.warning(f"Chunk metadata not found for chunk_id {chunk_id}")
                continue
            final_results.append({
                'chunk': chunk_metadata,
                'score': chunk_id_to_score[chunk_id]
            })

        logger.info(f"Filtered {filtered_count} chunks due to missing metadata.")
        logger.info(
            f"Total chunks retrieved after filtering: {len(final_results)} "
            f"(required min_chunks={min_chunks})"
        )

        if len(final_results) >= min_chunks or k >= num_chunks_to_recall:
            break
        else:
            k += 5  # Increment k to retrieve more chunks
            logger.info(
                f"Number of retrieved chunks ({len(final_results)}) is less than min_chunks ({min_chunks}). "
                f"Increasing k to {k} and retrying retrieval."
            )

    logger.debug(f"Hybrid retrieval returning {len(final_results)} chunks.")
    logger.info(
        f"Chunks used for hybrid retrieval for query '{query}': "
        f"[{', '.join([res['chunk']['chunk_id'] for res in final_results])}]"
    )
    end_time = time.time()
    logger.debug(f"Exiting hybrid_retrieval method. Time taken: {end_time - start_time:.2f} seconds.")
    return final_results


def multi_stage_retrieval(
    query: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    max_hops: int = 3,
    max_results: int = 5,
    similarity_threshold: float = 0.9
) -> List[Dict[str, Any]]:
    accumulated_context = ""
    all_retrieved_chunks = {}
    current_query = query
    previous_query = ""
    query_generator = dspy.TypedChainOfThought(QueryGeneratorSignature)

    for hop in range(max_hops):
        logger.info(f"Starting hop {hop+1} with query: '{current_query}'")

        retrieved_chunks = hybrid_retrieval(current_query, db, es_bm25, k)

        # Update accumulated retrieved chunks
        for chunk in retrieved_chunks:
            chunk_id = chunk['chunk']['chunk_id']
            if chunk_id not in all_retrieved_chunks:
                all_retrieved_chunks[chunk_id] = chunk

        # Check if we have reached the desired number of results
        if len(all_retrieved_chunks) >= max_results:
            logger.info(f"Retrieved sufficient chunks ({len(all_retrieved_chunks)}). Terminating.")
            break

        # Accumulate new context from retrieved chunks
        new_context = "\n\n".join([
            chunk['chunk'].get('contextualized_content', '') or chunk['chunk'].get('original_content', '')
            for chunk in retrieved_chunks
        ])
        accumulated_context += "\n\n" + new_context

        # Generate a new query based on the accumulated context
        response = query_generator(question=query, context=accumulated_context)
        new_query = response.get('new_query', '').strip()
        if not new_query:
            logger.info("No new query generated. Terminating multi-stage retrieval.")
            break

        # Compute similarity between the new query and the previous query
        similarity = compute_similarity(current_query, new_query)
        logger.debug(f"Similarity between queries: {similarity:.4f}")
        if similarity >= similarity_threshold:
            logger.info("New query is too similar to the current query. Terminating multi-stage retrieval.")
            break

        # Update queries for the next iteration
        previous_query = current_query
        current_query = new_query

    # Sort and return the top results
    final_results = sorted(
        all_retrieved_chunks.values(),
        key=lambda x: x['score'],
        reverse=True
    )[:max_results]

    logger.info(f"Multi-stage retrieval completed with {len(final_results)} chunks.")
    return final_results
