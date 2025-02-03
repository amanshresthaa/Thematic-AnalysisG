# File: retrieval/reranking.py
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
from src.core.retrieval.retrieval import hybrid_retrieval

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
        self.reranker_type = reranker_type
        self.st_model_name = st_model_name
        self.device = device
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.st_weight = st_weight

class SentenceTransformerReRanker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
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
        logger.info(f"SentenceTransformerReRanker initialized with model '{model_name}' on device '{self.device}'")

    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        if not query or not documents:
            logger.warning("Query and documents must be provided for re-ranking.")
            return []

        try:
            start_time = time.time()
            logger.info(f"Starting ST reranking for query: '{query[:100]}...' with {len(documents)} documents")
            
            logger.debug("Encoding query and documents with SentenceTransformer")
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
            
            logger.debug("Computing cosine similarities")
            cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            
            num_docs = len(documents)
            actual_k = min(top_k, num_docs)
            
            logger.debug(f"Selecting top {actual_k} documents out of {num_docs}")
            top_results = torch.topk(cosine_scores, k=actual_k)
            
            re_ranked_docs = []
            for score, idx in zip(top_results.values, top_results.indices):
                re_ranked_docs.append({
                    "document": documents[idx],
                    "score": score.item()
                })
            
            elapsed_time = time.time() - start_time
            logger.info(f"ST reranking completed in {elapsed_time:.2f}s. Top score: {re_ranked_docs[0]['score']:.4f}, Bottom score: {re_ranked_docs[-1]['score']:.4f}")
            return re_ranked_docs
            
        except Exception as e:
            logger.error(f"Error during ST reranking: {e}", exc_info=True)
            return []

class CohereReRanker:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key must be provided or set in COHERE_API_KEY environment variable")
        self.client = cohere.Client(self.api_key)
        logger.info("CohereReRanker initialized successfully")

    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        if not query or not documents:
            logger.warning("Query and documents must be provided for Cohere re-ranking.")
            return []

        try:
            start_time = time.time()
            logger.info(f"Starting Cohere reranking for query: '{query[:100]}...' with {len(documents)} documents")
            
            logger.debug("Calling Cohere rerank API")
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents))
            )
            
            re_ranked_docs = []
            for result in response.results:
                re_ranked_docs.append({
                    "document": documents[result.index],
                    "score": result.relevance_score
                })
            
            elapsed_time = time.time() - start_time
            logger.info(f"Cohere reranking completed in {elapsed_time:.2f}s. Top score: {re_ranked_docs[0]['score']:.4f}, Bottom score: {re_ranked_docs[-1]['score']:.4f}")
            return re_ranked_docs
            
        except Exception as e:
            logger.error(f"Error during Cohere reranking: {e}", exc_info=True)
            return []

class CombinedReRanker:
    def __init__(self, 
                 st_model_name: str = 'all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 cohere_api_key: Optional[str] = None,
                 st_weight: float = 0.5):
        self.st_reranker = SentenceTransformerReRanker(st_model_name, device)
        self.cohere_reranker = CohereReRanker(cohere_api_key)
        self.st_weight = st_weight
        logger.info(f"CombinedReRanker initialized with ST weight: {st_weight}")

    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        try:
            start_time = time.time()
            logger.info(f"Starting combined reranking for query: '{query[:100]}...' with {len(documents)} documents")
            
            logger.debug("Getting ST rankings")
            st_results = self.st_reranker.rerank(query, documents, top_k=len(documents))
            logger.debug("Getting Cohere rankings")
            cohere_results = self.cohere_reranker.rerank(query, documents, top_k=len(documents))
            
            logger.debug("Creating score maps and combining results")
            st_scores = {doc['document']: doc['score'] for doc in st_results}
            cohere_scores = {doc['document']: doc['score'] for doc in cohere_results}
            
            combined_scores = {}
            for doc in documents:
                st_score = st_scores.get(doc, 0.0)
                cohere_score = cohere_scores.get(doc, 0.0)
                combined_score = (st_score * self.st_weight) + (cohere_score * (1 - self.st_weight))
                combined_scores[doc] = combined_score
                logger.debug(f"Document scores - ST: {st_score:.4f}, Cohere: {cohere_score:.4f}, Combined: {combined_score:.4f}")
            
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            final_results = [{"document": doc, "score": score} for doc, score in sorted_docs]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Combined reranking completed in {elapsed_time:.2f}s. Top score: {final_results[0]['score']:.4f}, Bottom score: {final_results[-1]['score']:.4f}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in combined reranking: {e}", exc_info=True)
            logger.debug("Falling back to ST reranker")
            return self.st_reranker.rerank(query, documents, top_k=top_k)

class RerankerFactory:
    @staticmethod
    def create_reranker(config: RerankerConfig):
        logger.debug(f"Creating reranker of type: {config.reranker_type}")
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
            error_msg = f"Unknown reranker type: {config.reranker_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

def retrieve_with_reranking(query: str,
                          db: ContextualVectorDB,
                          es_bm25: ElasticsearchBM25,
                          k: int,
                          reranker_config: Optional[RerankerConfig] = None) -> List[Dict[str, Any]]:
    start_time = time.time()
    logger.info(f"Starting retrieval and reranking for query: '{query[:100]}...'")

    if reranker_config is None:
        logger.debug("No reranker config provided, defaulting to SentenceTransformer")
        reranker_config = RerankerConfig(reranker_type=RerankerType.SENTENCE_TRANSFORMER)
    
    logger.debug(f"Using reranker type: {reranker_config.reranker_type}")

    try:
        logger.debug(f"Performing initial hybrid retrieval with k={k*10}")
        initial_results = hybrid_retrieval(query, db, es_bm25, k=k*10)
        logger.info(f"Initial retrieval returned {len(initial_results)} results in {time.time() - start_time:.2f}s")

        if not initial_results:
            logger.warning(f"No initial results retrieved for query '{query[:100]}...'. Skipping reranking.")
            return []

        documents = [doc['chunk']['original_content'] for doc in initial_results]
        
        logger.debug(f"Creating {reranker_config.reranker_type} reranker")
        reranker = RerankerFactory.create_reranker(reranker_config)
        
        logger.debug(f"Performing reranking with k={k}")
        re_ranked = reranker.rerank(query, documents, top_k=k)

        final_results = []
        for r in re_ranked:
            doc_index = documents.index(r['document'])
            final_results.append({
                "chunk": initial_results[doc_index]['chunk'],
                "score": r['score']
            })

        elapsed_time = time.time() - start_time
        logger.info(f"Retrieval and reranking completed in {elapsed_time:.2f}s. Returned {len(final_results)} results")
        if final_results:
            logger.debug(f"Top score: {final_results[0]['score']:.4f}, Bottom score: {final_results[-1]['score']:.4f}")
        return final_results

    except Exception as e:
        logger.error(f"Error during retrieval/reranking for query '{query[:100]}...': {e}", exc_info=True)
        return []
