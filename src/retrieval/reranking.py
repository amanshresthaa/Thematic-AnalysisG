# File: src/retrieval/reranking.py

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
import torch
import time
import cohere
import os
from enum import Enum

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.utils.logger import setup_logging
from src.retrieval.retrieval import hybrid_retrieval

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

class RerankerType(str, Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    COHERE = "cohere"
    COMBINED = "combined"

class RerankerConfig:
    def __init__(self,
                 reranker_type: RerankerType = RerankerType.SENTENCE_TRANSFORMER,
                 st_model_name: str = 'all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 cohere_api_key: Optional[str] = None,
                 st_weight: float = 0.5):
        """
        Configuration for rerankers.

        Args:
            reranker_type (RerankerType): Which reranker to use.
            st_model_name (str): Model name for SentenceTransformer reranker.
            device (str, optional): Device to run SentenceTransformer on ('cuda', 'mps', or 'cpu').
            cohere_api_key (str, optional): Cohere API key.
            st_weight (float): Weight for ST scores when using combined reranker. Cohere weight = 1 - st_weight.
        """
        self.reranker_type = reranker_type
        self.st_model_name = st_model_name
        self.device = device
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.st_weight = st_weight

class SentenceTransformerReRanker:
    """
    Re-ranker using Sentence Transformers for semantic similarity.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initializes the Sentence Transformer re-ranker.
        
        Args:
            model_name (str): Pre-trained Sentence Transformer model name.
            device (str, optional): Device to run the model on ('cuda', 'mps', or 'cpu').
        """
        self.model = SentenceTransformer(model_name)
        if device:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.model.to(self.device)
        logger.info(f"SentenceTransformerReRanker initialized with model '{model_name}' on device '{self.device}'.")

    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Re-ranks documents based on semantic similarity to the query.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of document contents to re-rank.
            top_k (int, optional): Number of top documents to return after re-ranking. Defaults to 20.
        
        Returns:
            List[Dict[str, Any]]: List of re-ranked documents with similarity scores.
        """
        if not query or not documents:
            logger.warning("Query and documents must be provided for re-ranking.")
            return []

        try:
            logger.debug("Encoding query and documents.")
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
            
            logger.debug("Computing cosine similarities.")
            cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            
            num_docs = len(documents)
            actual_k = min(top_k, num_docs)
            if actual_k == 0:
                logger.warning("No documents available for re-ranking.")
                return []
            
            logger.debug(f"Selecting top {actual_k} documents out of {num_docs}.")
            top_results = torch.topk(cosine_scores, k=actual_k)
            
            re_ranked_docs = []
            for score, idx in zip(top_results.values, top_results.indices):
                re_ranked_docs.append({
                    "document": documents[idx],
                    "score": score.item()
                })
            
            logger.info(f"Re-ranked {actual_k} documents based on semantic similarity.")
            return re_ranked_docs
        except Exception as e:
            logger.error(f"Error during re-ranking with Sentence Transformers: {e}", exc_info=True)
            return []

class CohereReRanker:
    """
    Re-ranker using Cohere's reranking model.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the Cohere re-ranker.
        
        Args:
            api_key (str, optional): Cohere API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key must be provided or set in COHERE_API_KEY environment variable")
        self.client = cohere.Client(self.api_key)
        logger.info("CohereReRanker initialized successfully.")

    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Re-ranks documents using Cohere's reranking model.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of document contents to re-rank.
            top_k (int, optional): Number of top documents to return. Defaults to 20.
        
        Returns:
            List[Dict[str, Any]]: List of re-ranked documents with relevance scores.
        """
        if not query or not documents:
            logger.warning("Query and documents must be provided for Cohere re-ranking.")
            return []

        try:
            logger.debug("Performing Cohere reranking.")
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents))
            )
            time.sleep(0.1)  # Rate limiting protection
            
            re_ranked_docs = []
            for result in response.results:
                re_ranked_docs.append({
                    "document": documents[result.index],
                    "score": result.relevance_score
                })
            
            logger.info(f"Re-ranked {len(re_ranked_docs)} documents using Cohere.")
            return re_ranked_docs
        except Exception as e:
            logger.error(f"Error during re-ranking with Cohere: {e}", exc_info=True)
            return []

class CombinedReRanker:
    """
    Combined re-ranker using both SentenceTransformer and Cohere.
    """
    def __init__(self, 
                 st_model_name: str = 'all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 cohere_api_key: Optional[str] = None,
                 st_weight: float = 0.5):
        """
        Initializes both rerankers and sets weight for score combination.
        
        Args:
            st_model_name (str): Model name for SentenceTransformer
            device (str, optional): Device for SentenceTransformer
            cohere_api_key (str, optional): Cohere API key
            st_weight (float): Weight for SentenceTransformer scores (1 - st_weight for Cohere)
        """
        self.st_reranker = SentenceTransformerReRanker(st_model_name, device)
        self.cohere_reranker = CohereReRanker(cohere_api_key)
        self.st_weight = st_weight
        logger.info(f"CombinedReRanker initialized with ST weight: {st_weight}")

    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Re-ranks documents using both rerankers and combines scores.
        
        Args:
            query (str): The search query
            documents (List[str]): Documents to rerank
            top_k (int): Number of top documents to return
        
        Returns:
            List[Dict[str, Any]]: Combined and reranked documents
        """
        try:
            # Get rankings from both rerankers
            st_results = self.st_reranker.rerank(query, documents, top_k=len(documents))
            cohere_results = self.cohere_reranker.rerank(query, documents, top_k=len(documents))
            
            # Create score maps for both results
            st_scores = {doc['document']: doc['score'] for doc in st_results}
            cohere_scores = {doc['document']: doc['score'] for doc in cohere_results}
            
            # Combine scores
            combined_scores = {}
            for doc in documents:
                st_score = st_scores.get(doc, 0.0)
                cohere_score = cohere_scores.get(doc, 0.0)
                combined_scores[doc] = (st_score * self.st_weight) + (cohere_score * (1 - self.st_weight))
            
            # Sort by combined score and get top_k
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            return [{"document": doc, "score": score} for doc, score in sorted_docs]
        except Exception as e:
            logger.error(f"Error in combined reranking: {e}", exc_info=True)
            # Fallback to SentenceTransformer results
            return self.st_reranker.rerank(query, documents, top_k=top_k)

class RerankerFactory:
    @staticmethod
    def create_reranker(config: RerankerConfig):
        if config.reranker_type == RerankerType.SENTENCE_TRANSFORMER:
            return SentenceTransformerReRanker(model_name=config.st_model_name, device=config.device)
        elif config.reranker_type == RerankerType.COHERE:
            return CohereReRanker(api_key=config.cohere_api_key)
        elif config.reranker_type == RerankerType.COMBINED:
            return CombinedReRanker(
                st_model_name=config.st_model_name,
                device=config.device,
                cohere_api_key=config.cohere_api_key,
                st_weight=config.st_weight
            )
        else:
            raise ValueError(f"Unknown reranker type: {config.reranker_type}")

def retrieve_with_reranking(query: str,
                            db: ContextualVectorDB,
                            es_bm25: ElasticsearchBM25,
                            k: int,
                            reranker_config: Optional[RerankerConfig] = None) -> List[Dict[str, Any]]:
    """
    Retrieves documents using hybrid retrieval and re-ranks them using the configured reranker.
    
    Args:
        query (str): The search query.
        db (ContextualVectorDB): Contextual vector database instance.
        es_bm25 (ElasticsearchBM25): Elasticsearch BM25 instance.
        k (int): Number of documents to retrieve.
        reranker_config (RerankerConfig, optional): Reranker configuration. Defaults to SentenceTransformer if not provided.
    
    Returns:
        List[Dict[str, Any]]: List of re-ranked documents.
    """
    logger.debug(f"Entering retrieve_with_reranking method with query='{query}' and k={k}.")
    start_time = time.time()

    if reranker_config is None:
        # Default to SentenceTransformer reranker if no config provided
        reranker_config = RerankerConfig(reranker_type=RerankerType.SENTENCE_TRANSFORMER)

    try:
        logger.debug(f"Performing hybrid retrieval for query: '{query}'")
        initial_results = hybrid_retrieval(query, db, es_bm25, k=k*10)
        logger.debug(f"Initial hybrid retrieval returned {len(initial_results)} results.")

        if not initial_results:
            logger.warning(f"No initial results retrieved for query '{query}'. Skipping reranking.")
            return []

        # Extract document contents
        documents = [doc['chunk']['original_content'] for doc in initial_results]

        # Create the appropriate reranker
        reranker = RerankerFactory.create_reranker(reranker_config)

        # Perform re-ranking
        re_ranked = reranker.rerank(query, documents, top_k=k)

        # Map results back to original document structure
        final_results = []
        for r in re_ranked:
            doc_index = documents.index(r['document'])
            final_results.append({
                "chunk": initial_results[doc_index]['chunk'],
                "score": r['score']
            })

        end_time = time.time()
        logger.debug(f"Exiting retrieve_with_reranking method. Time taken: {end_time - start_time:.2f} seconds.")
        return final_results

    except Exception as e:
        logger.error(f"Error during retrieval or re-ranking for query '{query}': {e}", exc_info=True)
        return []
