# File: contextual_vector_db.py
# ------------------------------------------------------------------------------
"""
Module: contextual_vector_db

This module contains the ContextualVectorDB class which manages the creation,
storage, and search functionalities for a vector database enhanced with contextual
information. It leverages external APIs (OpenAI) and libraries (FAISS, DSPy) while
implementing robust error handling and detailed logging.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import faiss
import dspy
import time

from src.core.openai_client import OpenAIClient
from src.utils.logger import setup_logging, log_execution_time, get_logger

setup_logging()
logger = logging.getLogger(__name__)


def log_exception(logger_obj: logging.Logger, message: str, e: Exception, context: Dict[str, Any] = None) -> None:
    """
    Helper function for logging exceptions with additional context.
    """
    context_str = ", ".join(f"{k}={v}" for k, v in (context or {}).items())
    logger_obj.error(f"{message}. Context: {context_str}. Exception: {e}", exc_info=True)


class SituateContextSignature(dspy.Signature):
    doc = dspy.InputField(desc="Full document content")
    chunk = dspy.InputField(desc="Specific chunk content")
    reasoning = dspy.OutputField(desc="Chain of thought reasoning")
    contextualized_content = dspy.OutputField(desc="Contextualized content for the chunk")


class SituateContext(dspy.Module):
    def __init__(self):
        super().__init__()
        self.chain = dspy.ChainOfThought(SituateContextSignature)
        logger.debug("SituateContext module initialized.")

    def forward(self, doc: str, chunk: str):
        prompt = f"""
                <document>
                {doc}
                </document>
            
                CHUNK_CONTEXT_PROMPT = 
                Here is the chunk we want to situate within the whole document
                <chunk>
                {chunk}
                </chunk>

                Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
                Answer only with the succinct context and nothing else.
        """
        logger.debug("Generating contextualized content for a chunk.")
        return self.chain(doc=doc, chunk=chunk, prompt=prompt)


class ContextualVectorDB:
    def __init__(self, name: str, openai_api_key: str = None):
        if openai_api_key is None:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.critical("OPENAI_API_KEY is not set in environment variables.")
                raise ValueError("OPENAI_API_KEY is required but not set.")

        self.name = name
        self.embeddings = []
        self.metadata = []
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"
        self.faiss_index_path = f"./data/{name}/faiss_index.bin"

        try:
            self.client = OpenAIClient(api_key=openai_api_key)
            logger.debug(f"Initialized OpenAIClient for ContextualVectorDB '{self.name}'.")
        except Exception as e:
            log_exception(logger, "Failed to initialize OpenAIClient", e, {"name": self.name})
            raise

        self.index = None  # FAISS index
        self.situate_context_module = None

    def _is_db_loaded_in_memory(self) -> bool:
        return bool(self.embeddings and self.metadata and os.path.exists(self.faiss_index_path))

    def _is_db_saved_on_disk(self) -> bool:
        return os.path.exists(self.db_path) and os.path.exists(self.faiss_index_path)

    def _load_from_disk(self) -> None:
        self.load_db()
        self.load_faiss_index()

    def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        try:
            start_time = time.time()
            logger.debug("Generating embedding for the query.")
            response = self.client.create_embeddings(
                model="text-embedding-3-small",
                input=[query]
            )
            query_embedding = response['data'][0]['embedding']
            elapsed_time = time.time() - start_time
            logger.debug(f"Generated embedding for query in {elapsed_time:.2f} seconds.")
            query_embedding_np = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding_np)
            return query_embedding_np
        except Exception as e:
            log_exception(logger, "Error generating query embedding", e, {"query": query})
            return None

    def _extract_chunk_details(self, doc: Dict[str, Any], chunk: Any) -> Optional[Tuple[str, str, int]]:
        """
        Extracts and returns (chunk_id, content, original_index) for the given chunk.
        """
        if isinstance(chunk, dict):
            chunk_id = chunk.get('chunk_id')
            if not chunk_id:
                chunk_index = chunk.get('index', 0)
                chunk_id = f"{doc.get('doc_id', 'unknown_doc_id')}_{chunk_index}"
                chunk['chunk_id'] = chunk_id
            content = chunk.get('content', '')
            original_index = chunk.get('original_index', chunk.get('index', 0))
            return chunk_id, content, original_index
        elif isinstance(chunk, str):
            logger.warning("Chunk is a string. Expected a dict. Assigning default values.")
            chunk_id = f"{doc.get('doc_id', 'unknown_doc_id')}_0"
            return chunk_id, chunk, 0
        else:
            logger.error(f"Unsupported chunk type: {type(chunk)}. Skipping chunk.")
            return None

    @log_execution_time(logger)
    def situate_context(self, doc: str, chunk: str) -> Tuple[str, Any]:
        logger.debug(f"Entering situate_context with doc length={len(doc)} and chunk length={len(chunk)}.")
        try:
            if self.situate_context_module is None:
                self.situate_context_module = SituateContext()
                logger.debug("Initialized SituateContext module.")

            start_time = time.time()
            response = self.situate_context_module(doc=doc, chunk=chunk)
            elapsed_time = time.time() - start_time
            logger.debug(f"Generated contextualized content using DSPy in {elapsed_time:.2f} seconds.")

            contextualized_content = response.contextualized_content
            usage_metrics = {}  # Populate usage metrics if needed.
            return contextualized_content, usage_metrics
        except Exception as e:
            log_exception(
                logger,
                "Error during DSPy situate_context",
                e,
                {"doc_length": len(doc), "chunk_length": len(chunk)}
            )
            return "", None

    @log_execution_time(logger)
    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 8):
        logger.debug("Entering load_data method.")
        if self._is_db_loaded_in_memory():
            logger.info("Vector database is already loaded in memory. Skipping data loading.")
            return

        if self._is_db_saved_on_disk():
            logger.info("Loading vector database and FAISS index from disk.")
            self._load_from_disk()
            return

        texts_to_embed, metadata = self._process_dataset(dataset, parallel_threads)

        if not texts_to_embed:
            logger.warning("No texts to embed after processing the dataset.")
            return

        self._embed_and_store(texts_to_embed, metadata, max_workers=parallel_threads)
        self.save_db()
        self._build_faiss_index()
        logger.info(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}.")

    @log_execution_time(logger)
    def _process_dataset(self, dataset: List[Dict[str, Any]], parallel_threads: int) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc.get('chunks', [])) for doc in dataset)
        logger.info(f"Total chunks to process: {total_chunks}.")
        logger.info(f"Processing {total_chunks} chunks with {parallel_threads} threads.")

        try:
            with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                futures = [
                    executor.submit(self._generate_contextualized_content, doc, chunk)
                    for doc in dataset
                    for chunk in doc.get('chunks', [])
                ]
                for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                    result = future.result()
                    if result:
                        texts_to_embed.append(result['text_to_embed'])
                        metadata.append(result['metadata'])
            logger.debug("Completed processing all chunks.")
        except Exception as e:
            log_exception(logger, "Error during processing chunks", e)
            return [], []

        return texts_to_embed, metadata

    @log_execution_time(logger)
    def _generate_contextualized_content(self, doc: Dict[str, Any], chunk: Any) -> Optional[Dict[str, Any]]:
        """
        Generates contextualized content for a single doc-chunk pair.
        """
        if not isinstance(doc, dict):
            logger.error(f"Document is not a dictionary: {doc}")
            return None

        details = self._extract_chunk_details(doc, chunk)
        if details is None:
            return None
        chunk_id, content, original_index = details

        logger.debug(f"Processing chunk_id='{chunk_id}' in doc_id='{doc.get('doc_id', 'unknown_doc_id')}'.")
        contextualized_text, usage = self.situate_context(doc.get('content', ''), content)
        if not contextualized_text:
            logger.warning(f"Contextualized content is empty for chunk_id='{chunk_id}'.")
            return None

        return {
            'text_to_embed': f"{content}\n\n{contextualized_text}",
            'metadata': {
                'doc_id': doc.get('doc_id', ''),
                'original_uuid': doc.get('original_uuid', ''),
                'chunk_id': chunk_id,
                'original_index': original_index,
                'original_content': content,
                'contextualized_content': contextualized_text
            }
        }

    @log_execution_time(logger)
    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]], max_workers: int = 4):
        logger.debug("Entering _embed_and_store method.")
        batch_size = 128
        embeddings = []
        logger.info("Starting embedding generation.")

        def embed_batch(batch: List[str]) -> List[List[float]]:
            try:
                logger.debug(f"Generating embeddings for batch of size {len(batch)}.")
                response = self.client.create_embeddings(
                    model="text-embedding-3-small",
                    input=batch
                )
                return [item['embedding'] for item in response['data']]
            except Exception as e:
                log_exception(logger, "Error during OpenAI embeddings for batch", e, {"batch_size": len(batch)})
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(embed_batch, texts[i:i + batch_size])
                for i in range(0, len(texts), batch_size)
            ]
            for future in as_completed(futures):
                try:
                    embeddings_batch = future.result()
                    if embeddings_batch:
                        embeddings.extend(embeddings_batch)
                        logger.debug(f"Processed a batch with {len(embeddings_batch)} embeddings.")
                except Exception as e:
                    log_exception(logger, "Error retrieving embeddings from future", e)

        if not embeddings:
            logger.warning("No embeddings were generated.")
            return

        self.embeddings = embeddings
        self.metadata = data
        self.save_db()
        self._build_faiss_index()
        logger.info(f"Embedding generation completed. Total embeddings: {len(self.embeddings)}.")

    @log_execution_time(logger)
    def _build_faiss_index(self):
        logger.debug("Entering _build_faiss_index method.")
        start_time = time.time()
        try:
            self.create_faiss_index()
            self.save_faiss_index()
        except Exception as e:
            log_exception(logger, "Error building FAISS index", e)
            raise
        elapsed_time = time.time() - start_time
        logger.info(f"FAISS index built and saved in {elapsed_time:.2f} seconds.")

    @log_execution_time(logger)
    def create_faiss_index(self):
        logger.debug("Entering create_faiss_index method.")
        if not self.embeddings:
            logger.error("No embeddings available to create FAISS index.")
            raise ValueError("Embeddings list is empty.")

        embedding_dim = len(self.embeddings[0])
        logger.info(f"Embedding dimension: {embedding_dim}.")
        embeddings_np = np.array(self.embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings_np)
        logger.info(f"FAISS index created with {self.index.ntotal} vectors.")

    @log_execution_time(logger)
    def save_faiss_index(self):
        logger.debug("Entering save_faiss_index method.")
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        try:
            faiss.write_index(self.index, self.faiss_index_path)
            logger.info(f"FAISS index saved to '{self.faiss_index_path}'.")
        except Exception as e:
            log_exception(logger, "Error saving FAISS index", e, {"faiss_index_path": self.faiss_index_path})

    @log_execution_time(logger)
    def load_faiss_index(self):
        logger.debug("Entering load_faiss_index method.")
        if not os.path.exists(self.faiss_index_path):
            logger.error(f"FAISS index file not found at '{self.faiss_index_path}'.")
            raise ValueError("FAISS index file not found.")
        try:
            self.index = faiss.read_index(self.faiss_index_path)
            logger.info(f"FAISS index loaded from '{self.faiss_index_path}' with {self.index.ntotal} vectors.")
        except Exception as e:
            log_exception(logger, "Error loading FAISS index", e, {"faiss_index_path": self.faiss_index_path})
            raise

    @log_execution_time(logger)
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"Entering search method with query='{query}' and k={k}.")
        if not self.embeddings or not self.metadata:
            logger.error("Embeddings or metadata are not loaded. Cannot perform search.")
            return []
        if self.index is None:
            logger.error("FAISS index is not loaded.")
            return []

        query_embedding_np = self._generate_query_embedding(query)
        if query_embedding_np is None:
            return []

        distances, indices = self._perform_faiss_search(query_embedding_np, k)
        if distances is None or indices is None:
            return []

        top_results = self._retrieve_top_results(indices, distances)
        logger.info(f"FAISS search returned {len(top_results)} results for query: '{query}'.")
        logger.debug(f"Chunks retrieved: {[res['chunk_id'] for res in top_results]}.")
        return top_results

    def _perform_faiss_search(self, query_embedding_np: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Performing FAISS search.")
        try:
            start_time = time.time()
            distances, indices = self.index.search(query_embedding_np, k)
            elapsed_time = time.time() - start_time
            logger.debug(f"FAISS search completed in {elapsed_time:.2f} seconds.")
            return distances.flatten(), indices.flatten()
        except Exception as e:
            log_exception(logger, "Error during FAISS search", e, {"query_embedding_np": query_embedding_np, "k": k})
            return None, None

    def _retrieve_top_results(self, indices: np.ndarray, distances: np.ndarray) -> List[Dict[str, Any]]:
        top_results = []
        for idx, score in zip(indices, distances):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                result = {
                    "doc_id": meta['doc_id'],
                    "chunk_id": meta['chunk_id'],
                    "original_index": meta.get('original_index', 0),
                    "content": meta['original_content'],
                    "contextualized_content": meta.get('contextualized_content'),
                    "score": float(score),
                    "metadata": meta
                }
                top_results.append(result)
                logger.debug(f"Retrieved chunk_id='{meta['chunk_id']}' with score={score:.4f}.")
            else:
                logger.warning(f"Index {idx} out of bounds for metadata.")
        return top_results

    @log_execution_time(logger)
    def save_db(self):
        logger.debug("Entering save_db method.")
        data = {
            "metadata": self.metadata,
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            with open(self.db_path, "wb") as file:
                pickle.dump(data, file)
            logger.info(f"Vector database metadata saved to '{self.db_path}'.")
        except Exception as e:
            log_exception(logger, "Error saving vector database metadata", e, {"db_path": self.db_path})

    @log_execution_time(logger)
    def load_db(self):
        logger.debug("Entering load_db method.")
        if not os.path.exists(self.db_path):
            logger.error(f"Vector database file not found at '{self.db_path}'. Use load_data to create a new database.")
            raise ValueError("Vector database file not found.")
        try:
            with open(self.db_path, "rb") as file:
                data = pickle.load(file)
            self.metadata = data.get("metadata", [])
            logger.info(f"Vector database metadata loaded from '{self.db_path}' with {len(self.metadata)} entries.")
            logger.debug(f"Chunks loaded: {[meta['chunk_id'] for meta in self.metadata]}.")
        except Exception as e:
            log_exception(logger, "Error loading vector database metadata", e, {"db_path": self.db_path})
            raise
