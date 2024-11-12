import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch

from utils.logger import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

class SentenceTransformerReRanker:
    """
    Re-ranker using Sentence Transformers for semantic similarity.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initializes the Sentence Transformer re-ranker.
        
        Args:
            model_name (str): Pre-trained Sentence Transformer model name.
            device (str, optional): Device to run the model on ('cuda', 'mps', or 'cpu'). Defaults to automatic selection.
        """
        self.model = SentenceTransformer(model_name)
        
        # Set device
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
            actual_k = min(top_k, num_docs)  # Adjust k to the number of available documents
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

def rerank_documents_sentence_transformer(query: str, retrieved_docs: List[Dict[str, Any]], k: int = 20) -> List[Dict[str, Any]]:
    """
    Re-ranks the retrieved documents using Sentence Transformers.
    
    Args:
        query (str): The search query.
        retrieved_docs (List[Dict[str, Any]]): List of retrieved documents.
        k (int, optional): Number of top documents to return after re-ranking. Defaults to 20.
    
    Returns:
        List[Dict[str, Any]]: List of re-ranked documents.
    """
    logger.info(f"Starting re-ranking of {len(retrieved_docs)} documents for query: '{query}' using Sentence Transformers.")
    if not query or not retrieved_docs:
        logger.warning(f"Query and retrieved documents must be provided for re-ranking.")
        return []
    try:
        # Initialize the re-ranker
        reranker = SentenceTransformerReRanker()
        
        # Extract document contents
        documents = [doc['chunk']['original_content'] for doc in retrieved_docs]
        
        # Perform re-ranking
        re_ranked = reranker.rerank(query, documents, top_k=k)
        
        # Attach scores back to the documents
        re_ranked_docs = []
        for r, original_doc in zip(re_ranked, retrieved_docs[:len(re_ranked)]):  # Ensure matching length
            re_ranked_docs.append({
                "chunk": original_doc['chunk'],
                "score": r['score']
            })
        
        logger.info(f"Re-ranking completed using Sentence Transformers. Top {k} documents selected.")
        return re_ranked_docs
    except Exception as e:
        logger.error(f"Error during document re-ranking with Sentence Transformers: {e}", exc_info=True)
        return retrieved_docs[:k]  # Fallback to original ranking if re-ranking fails
