import os
import pickle
import numpy as np
import threading
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

from openai_client import OpenAIClient
import dspy
import logging
import faiss
from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class SituateContext(dspy.Module):
    def __init__(self):
        super().__init__()
        self.chain = dspy.ChainOfThought(SituateContextSignature)

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
        
        return self.chain(doc=doc, chunk=chunk, prompt=prompt)


class SituateContextSignature(dspy.Signature):
    doc = dspy.InputField(desc="Full document content")
    chunk = dspy.InputField(desc="Specific chunk content")
    reasoning = dspy.OutputField(desc="Chain of thought reasoning")
    contextualized_content = dspy.OutputField(desc="Contextualized content for the chunk")
        

class ContextualVectorDB:
    def __init__(self, name: str, openai_api_key: str = None):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"
        self.faiss_index_path = f"./data/{name}/faiss_index.bin"

        self.client = OpenAIClient(api_key=openai_api_key)
        logger.debug(f"Initialized OpenAIClient for ContextualVectorDB '{self.name}'")

    def situate_context(self, doc: str, chunk: str) -> Tuple[str, Any]:
        logger.debug(f"Entering situate_context with doc length={len(doc)} and chunk length={len(chunk)}.")
        try:
            if not hasattr(self, 'situate_context_module'):
                self.situate_context_module = SituateContext()
                logger.debug("Initialized SituateContext module")

            response = self.situate_context_module(doc=doc, chunk=chunk)
            contextualized_content = response.contextualized_content
            logger.debug("Generated contextualized_content using DSPy.")
            usage_metrics = {}
            return contextualized_content, usage_metrics
        except Exception as e:
            logger.error(f"Error during DSPy situate_context: {e}", exc_info=True)
            return "", None

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 1):
        logger.debug("Entering load_data method.")
        if self.embeddings and self.metadata and os.path.exists(self.faiss_index_path):
            logger.info("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path) and os.path.exists(self.faiss_index_path):
            logger.info("Loading vector database and FAISS index from disk.")
            self.load_db()
            self.load_faiss_index()
            return

        texts_to_embed, metadata = self._process_dataset(dataset, parallel_threads)
        
        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()
        self._build_faiss_index()

        logger.info(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")


    def _process_dataset(self, dataset: List[Dict[str, Any]], parallel_threads: int) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc.get('chunks', [])) for doc in dataset)
        logger.info(f"Total chunks to process: {total_chunks}")

        logger.info(f"Processing {total_chunks} chunks with {parallel_threads} threads.")
        try:
            with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                futures = []
                for doc in dataset:
                    for chunk in doc.get('chunks', []):
                        futures.append(executor.submit(self._generate_contextualized_content, doc, chunk))
                
                for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                    result = future.result()
                    if result:
                        texts_to_embed.append(result['text_to_embed'])
                        metadata.append(result['metadata'])
        except Exception as e:
            logger.error(f"Error during processing chunks: {e}", exc_info=True)
            return [], []

        return texts_to_embed, metadata

    def _generate_contextualized_content(self, doc: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(doc, dict):
            logger.error(f"Document is not a dictionary: {doc}")
            return None

        if isinstance(chunk, dict):
            chunk_id = chunk.get('chunk_id')
            if not chunk_id:
                # Assign a unique chunk_id combining doc_id and chunk's index
                chunk_index = chunk.get('index', 0)
                chunk_id = f"{doc.get('doc_id', 'unknown_doc_id')}_{chunk_index}"
                chunk['chunk_id'] = chunk_id
            content = chunk.get('content', '')
            original_index = chunk.get('original_index', chunk.get('index', 0))
        elif isinstance(chunk, str):
            # Handle case where chunk is a string
            content = chunk
            chunk_id = f"{doc.get('doc_id', 'unknown_doc_id')}_0"
            original_index = 0
            logger.warning(f"Chunk is a string. Expected a dict. Assigning default values.")
        else:
            logger.error(f"Unsupported chunk type: {type(chunk)}. Skipping chunk.")
            return None

        logger.debug(f"Processing chunk_id='{chunk_id}' in doc_id='{doc.get('doc_id', 'unknown_doc_id')}'")
        contextualized_text, usage = self.situate_context(doc.get('content', ''), content)
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

    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        logger.debug("Entering _embed_and_store method.")
        batch_size = 128
        embeddings = []
        logger.info("Starting embedding generation.")
        try:
            with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    try:
                        response = self.client.create_embeddings(
                            model="text-embedding-3-small",
                            input=batch
                        )
                        embeddings_batch = [item['embedding'] for item in response['data']]
                        embeddings.extend(embeddings_batch)
                        pbar.update(len(batch))
                        logger.debug(f"Processed batch {i // batch_size + 1}: {len(batch)} embeddings.")
                    except Exception as e:
                        logger.error(f"Error during OpenAI embeddings for batch starting at index {i}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
        
        self.embeddings = embeddings
        self.metadata = data
        logger.info("Embedding generation completed.")

    def _build_faiss_index(self):
        self.create_faiss_index()
        self.save_faiss_index()

    def create_faiss_index(self):
        logger.debug("Entering create_faiss_index method.")
        try:
            embedding_dim = len(self.embeddings[0])
            logger.info(f"Embedding dimension: {embedding_dim}")
            embeddings_np = np.array(self.embeddings).astype('float32')
            faiss.normalize_L2(embeddings_np)
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(embeddings_np)
            logger.info(f"FAISS index created with {self.index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}", exc_info=True)
            raise

    def save_faiss_index(self):
        logger.debug("Entering save_faiss_index method.")
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        try:
            faiss.write_index(self.index, self.faiss_index_path)
            logger.info(f"FAISS index saved to '{self.faiss_index_path}'")
        except Exception as e:
            logger.error(f"Error saving FAISS index to '{self.faiss_index_path}': {e}", exc_info=True)

    def load_faiss_index(self):
        logger.debug("Entering load_faiss_index method.")
        if not os.path.exists(self.faiss_index_path):
            logger.error(f"FAISS index file not found at '{self.faiss_index_path}'.")
            raise ValueError("FAISS index file not found.")
        try:
            self.index = faiss.read_index(self.faiss_index_path)
            logger.info(f"FAISS index loaded from '{self.faiss_index_path}' with {self.index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Error loading FAISS index from '{self.faiss_index_path}': {e}", exc_info=True)
            raise

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"Entering search method with query='{query}' and k={k}.")
        if not self.embeddings or not self.metadata:
            logger.error("Embeddings or metadata are not loaded. Cannot perform search.")
            return []
        if not hasattr(self, 'index'):
            logger.error("FAISS index is not loaded.")
            return []
        
        try:
            response = self.client.create_embeddings(
                model="text-embedding-3-small",
                input=[query]
            )
            query_embedding = response['data'][0]['embedding']
            logger.debug(f"Generated embedding for query: '{query}'")
        except Exception as e:
            logger.error(f"Error generating embedding for query '{query}': {e}", exc_info=True)
            return []

        query_embedding_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_np)

        logger.debug("Performing FAISS search.")
        try:
            distances, indices = self.index.search(query_embedding_np, k)
            indices = indices.flatten()
            distances = distances.flatten()
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

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
            else:
                logger.warning(f"Index {idx} out of bounds for metadata.")
        logger.debug(f"FAISS search returned {len(top_results)} results for query: '{query}'")
        logger.info(f"Chunks retrieved for query '{query}': {[res['chunk_id'] for res in top_results]}")

        return top_results

    def save_db(self):
        logger.debug("Entering save_db method.")
        data = {
            "metadata": self.metadata,
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            with open(self.db_path, "wb") as file:
                pickle.dump(data, file)
            logger.info(f"Vector database metadata saved to '{self.db_path}'")
        except Exception as e:
            logger.error(f"Error saving vector database metadata to '{self.db_path}': {e}", exc_info=True)

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
            logger.info(f"Chunks loaded: {[meta['chunk_id'] for meta in self.metadata]}")
        except Exception as e:
            logger.error(f"Error loading vector database metadata from '{self.db_path}': {e}", exc_info=True)
            raise
