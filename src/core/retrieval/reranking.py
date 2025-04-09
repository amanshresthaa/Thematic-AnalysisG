

# File: retrieval/reranking.py
# ------------------------------------------------------------------------------
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import torch
import time
import cohere
import os
from enum import Enum

from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.utils.logger import setup_logging, log_execution_time
from src.core.retrieval.retrieval import hybrid_retrieval

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

    @log_execution_time(logger)
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
            if re_ranked_docs:
                logger.info(
                    f"ST reranking completed in {elapsed_time:.2f}s. "
                    f"Top score: {re_ranked_docs[0]['score']:.4f}, "
                    f"Bottom score: {re_ranked_docs[-1]['score']:.4f}"
                )
            else:
                logger.info(f"ST reranking completed in {elapsed_time:.2f}s with no documents returned.")
            return re_ranked_docs
            
        except Exception as e:
            logger.error(f"Error during ST reranking: {e}", exc_info=True)
            return []

class CohereReRanker:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Cohere reranker with the provided API key."""
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Cohere API key must be provided or set in COHERE_API_KEY environment variable")
        
        try:
            # Test the API key with a minimal initialization
            self.client = cohere.Client(self.api_key)
            # Optional: could make a lightweight test call here if desired
            logger.debug("Successfully initialized Cohere client")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {str(e)}")
            raise ValueError(f"Invalid or expired Cohere API key: {str(e)}")

    @log_execution_time(logger)
    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        if not query or not documents:
            logger.warning("Query and documents must be provided for Cohere re-ranking.")
            return []

        try:
            start_time = time.time()
            logger.info(f"Starting Cohere reranking for query: '{query[:100]}...' with {len(documents)} documents")
            
            logger.debug("Calling Cohere rerank API")
            # Try to use the Cohere API
            try:
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
                if re_ranked_docs:
                    logger.info(
                        f"Cohere reranking completed in {elapsed_time:.2f}s. "
                        f"Top score: {re_ranked_docs[0]['score']:.4f}, "
                        f"Bottom score: {re_ranked_docs[-1]['score']:.4f}"
                    )
                else:
                    logger.info(f"Cohere reranking completed in {elapsed_time:.2f}s with no documents returned.")
                return re_ranked_docs
            
            except Exception as auth_error:
                # Check if it's an authentication error
                error_msg = str(auth_error)
                if "invalid api token" in error_msg or "unauthorized" in error_msg.lower():
                    # Log at debug level only - avoids showing in error logs
                    logger.debug(f"Cohere API authentication issue: {error_msg}")
                else:
                    # For other types of errors, log more details but without the full traceback
                    logger.debug(f"Cohere API error (non-authentication): {error_msg}")
                # Return empty list (will fall back to other rerankers in combined mode)
                return []
                
        except Exception as e:
            # General exception handler for any other issues
            logger.debug(f"Unexpected error in Cohere reranking: {str(e)}")
            return []

class CombinedReRanker:
    def __init__(self, 
                 st_model_name: str = 'all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 cohere_api_key: Optional[str] = None,
                 st_weight: float = 0.5):
        self.st_reranker = SentenceTransformerReRanker(st_model_name, device)
        self.st_weight = st_weight
        
        # Try to initialize Cohere reranker, but don't show errors if it fails
        self.cohere_available = False
        self.cohere_reranker = None
        try:
            if cohere_api_key:
                self.cohere_reranker = CohereReRanker(cohere_api_key)
                self.cohere_available = True
                logger.info(f"CombinedReRanker initialized with both ST and Cohere, ST weight: {st_weight}")
            else:
                logger.info(f"CombinedReRanker initialized with ST only (no Cohere API key provided), ST weight: {st_weight}")
        except Exception as e:
            # Log at debug level only to avoid error messages
            logger.debug(f"Could not initialize Cohere reranker: {str(e)}. Using SentenceTransformer only.")
            logger.info(f"CombinedReRanker initialized with ST only, ST weight: {st_weight}")

    @log_execution_time(logger)
    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        try:
            start_time = time.time()
            logger.info(f"Starting reranking for query: '{query[:100]}...' with {len(documents)} documents")
            
            # Get SentenceTransformer rankings
            logger.debug("Getting ST rankings")
            st_results = self.st_reranker.rerank(query, documents, top_k=len(documents))
            st_scores = {doc['document']: doc['score'] for doc in st_results}
            
            # If Cohere is not available, just use SentenceTransformer
            if not self.cohere_available or self.cohere_reranker is None:
                logger.debug("Using ST rankings only (Cohere not available)")
                sorted_docs = sorted(
                    st_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                final_results = [{"document": doc, "score": score} for doc, score in sorted_docs]
            else:
                # Try to get Cohere rankings and combine
                try:
                    logger.debug("Getting Cohere rankings")
                    cohere_results = self.cohere_reranker.rerank(query, documents, top_k=len(documents))
                    cohere_scores = {doc['document']: doc['score'] for doc in cohere_results}
                    
                    logger.debug("Creating score maps and combining results")
                    combined_scores = {}
                    for doc in documents:
                        st_score = st_scores.get(doc, 0.0)
                        cohere_score = cohere_scores.get(doc, 0.0)
                        combined_score = (st_score * self.st_weight) + (cohere_score * (1 - self.st_weight))
                        combined_scores[doc] = combined_score
                        logger.debug(
                            f"Document scores - ST: {st_score:.4f}, "
                            f"Cohere: {cohere_score:.4f}, "
                            f"Combined: {combined_score:.4f}"
                        )
                    
                    sorted_docs = sorted(
                        combined_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:top_k]
                    final_results = [{"document": doc, "score": score} for doc, score in sorted_docs]
                    
                except Exception as e:
                    # If anything goes wrong with Cohere, silently fall back to ST
                    logger.debug(f"Could not use Cohere rankings: {str(e)}. Using ST rankings only.")
                    sorted_docs = sorted(
                        st_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:top_k]
                    final_results = [{"document": doc, "score": score} for doc, score in sorted_docs]
            
            # Log results
            elapsed_time = time.time() - start_time
            if final_results:
                logger.info(
                    f"Reranking completed in {elapsed_time:.2f}s. "
                    f"Top score: {final_results[0]['score']:.4f}, "
                    f"Bottom score: {final_results[-1]['score']:.4f}"
                )
            else:
                logger.info(f"Reranking completed in {elapsed_time:.2f}s with no documents returned.")
            return final_results
            
        except Exception as e:
            # Last resort fallback
            logger.debug(f"Error in reranking: {str(e)}. Falling back to ST reranker.")
            return self.st_reranker.rerank(query, documents, top_k=top_k)

class RerankerFactory:
    @staticmethod
    def create_reranker(config: RerankerConfig):
        """Create a reranker based on the configuration with graceful fallback."""
        try:
            logger.debug(f"Creating reranker of type: {config.reranker_type}")
            if config.reranker_type == RerankerType.SENTENCE_TRANSFORMER:
                return SentenceTransformerReRanker(model_name=config.st_model_name, device=config.device)
            elif config.reranker_type == RerankerType.COHERE:
                # Try to initialize Cohere reranker with fallback to SentenceTransformer
                try:
                    return CohereReRanker(api_key=config.cohere_api_key)
                except (ValueError, Exception) as e:
                    logger.warning(f"Failed to initialize Cohere reranker: {str(e)}. Falling back to SentenceTransformer.")
                    return SentenceTransformerReRanker(model_name=config.st_model_name, device=config.device)
            elif config.reranker_type == RerankerType.COMBINED:
                # For combined reranker, try using Cohere if available
                try:
                    # Test if Cohere API key is valid by initializing it
                    cohere_reranker = CohereReRanker(api_key=config.cohere_api_key)
                    # If we get here, Cohere key is valid, so use combined reranker
                    return CombinedReRanker(
                        st_model_name=config.st_model_name,
                        device=config.device,
                        cohere_api_key=config.cohere_api_key,
                        st_weight=config.st_weight
                    )
                except (ValueError, Exception) as e:
                    logger.warning(f"Cohere API key issue: {str(e)}. Using SentenceTransformer only.")
                    return SentenceTransformerReRanker(model_name=config.st_model_name, device=config.device)
            else:
                error_msg = f"Unknown reranker type: {config.reranker_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error creating reranker: {str(e)}. Using default SentenceTransformer.")
            return SentenceTransformerReRanker()  # Use default sentence transformer as last resort

@log_execution_time(logger)
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

