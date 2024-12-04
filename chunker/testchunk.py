# chunker.py
import os
import json
import uuid
from typing import List, Dict
import datetime
import concurrent.futures
import functools
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
import logging
import re

from utils.logger import setup_logging  # Assuming this works with your local utils module

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return None  # Return None instead of an error dict
    return wrapper

class SmartChunker:
    def __init__(self, chunk_size: int, max_topic_distance: float, overlap_size: int = 1, nlp=None, embedding_model=None, tokenizer=None):
        self.chunk_size = chunk_size
        self.max_topic_distance = max_topic_distance
        self.overlap_size = overlap_size
        self.nlp = nlp
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.max_block_tokens = int(chunk_size * 0.8)
        self.max_chunk_tokens = chunk_size
        self.token_cache = {}

    def tokenize(self, text: str) -> int:
        if text in self.token_cache:
            return self.token_cache[text]
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        self.token_cache[text] = token_count
        return token_count

    def preprocess_text(self, text: str) -> str:
        text = ' '.join(text.split())
        return text

    def split_document_into_blocks(self, paragraphs: List[str]) -> List[str]:
        blocks = []
        current_block = ''
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_tokens = self.tokenize(paragraph)
            if paragraph_tokens > self.max_block_tokens:
                logger.warning("Paragraph exceeds maximum block tokens and will be split into sentences.")
                sentences = [sent.text.strip() for sent in self.nlp(paragraph).sents if sent.text.strip()]
                for sentence in sentences:
                    sentence_tokens = self.tokenize(sentence)
                    if sentence_tokens > self.max_block_tokens:
                        logger.warning("Sentence exceeds maximum block tokens and will be split further.")
                        clauses = self.split_long_sentence(sentence)
                        for clause in clauses:
                            clause_tokens = self.tokenize(clause)
                            if clause_tokens > self.max_block_tokens:
                                logger.warning("Clause exceeds maximum block tokens and will be skipped.")
                                continue
                            if current_tokens + clause_tokens > self.max_block_tokens:
                                if current_block:
                                    blocks.append(current_block.strip())
                                current_block = clause + ' '
                                current_tokens = clause_tokens
                            else:
                                current_block += clause + ' '
                                current_tokens += clause_tokens
                        continue
                    if current_tokens + sentence_tokens > self.max_block_tokens:
                        if current_block:
                            blocks.append(current_block.strip())
                        current_block = sentence + ' '
                        current_tokens = sentence_tokens
                    else:
                        current_block += sentence + ' '
                        current_tokens += sentence_tokens
            else:
                if current_tokens + paragraph_tokens > self.max_block_tokens:
                    if current_block:
                        blocks.append(current_block.strip())
                    current_block = paragraph + ' '
                    current_tokens = paragraph_tokens
                else:
                    current_block += paragraph + ' '
                    current_tokens += paragraph_tokens

        if current_block:
            blocks.append(current_block.strip())

        return blocks

    def split_long_sentence(self, sentence: str) -> List[str]:
        # Split sentence by commas, semicolons, or conjunctions
        clauses = re.split(r'[;,:]|\band\b|\bor\b', sentence)
        clauses = [clause.strip() for clause in clauses if clause.strip()]
        return clauses

    def split_block_into_chunks(self, block: str, previous_sentences: List[str]) -> List[str]:
        sentences = previous_sentences[-self.overlap_size:] if previous_sentences else []
        block_sentences = [sent.text.strip() for sent in self.nlp(block).sents if sent.text.strip()]
        sentences.extend(block_sentences)
        if not sentences:
            return []

        # Compute embeddings in batches
        embeddings = np.array(self.embedding_model.encode(sentences, batch_size=64))
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # Compute cosine similarities between adjacent sentences
        similarities = np.sum(normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1)

        # Dynamic thresholding using the 25th percentile
        if similarities.size > 0:
            threshold = np.percentile(similarities, 25)
        else:
            threshold = 0.0  # Default threshold if no similarities

        # Identify boundaries
        boundaries = [0]
        boundaries.extend([i + 1 for i, sim in enumerate(similarities) if sim < threshold])
        boundaries.append(len(sentences))

        # Form chunks
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)
            tokens = self.tokenize(chunk_text)
            if tokens > self.max_chunk_tokens:
                sub_chunks = self._split_chunk_by_sentence_tokens(chunk_sentences)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return chunks

    def _split_chunk_by_sentence_tokens(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_chunk = ''
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = self.tokenize(sentence)
            if current_tokens + sentence_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + ' '
                current_tokens += sentence_tokens
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

@handle_exceptions
def process_document(doc_id: str, content: str, chunk_size: int, max_topic_distance: float, nlp, embedding_model, tokenizer) -> Dict:
    paragraphs = content.strip().split('\n\n')
    cleaned_content = ' '.join(paragraphs)

    processed_doc = {
        "doc_id": doc_id,
        "original_uuid": uuid.uuid4().hex,
        "content": content,
        "chunks": []
    }

    chunker = SmartChunker(chunk_size=chunk_size, max_topic_distance=max_topic_distance, overlap_size=1, nlp=nlp, embedding_model=embedding_model, tokenizer=tokenizer)
    blocks = chunker.split_document_into_blocks(paragraphs)
    all_chunks = []
    previous_sentences = []

    for block in blocks:
        chunks = chunker.split_block_into_chunks(block, previous_sentences)
        if chunks:
            all_chunks.extend(chunks)
            # Update previous_sentences for overlap
            previous_sentences = [sent.text.strip() for sent in chunker.nlp(block).sents if sent.text.strip()]

    for idx, chunk_text in enumerate(all_chunks):
        chunk_id = f"{doc_id}_chunk_{idx}"
        processed_doc["chunks"].append({
            "chunk_id": chunk_id,
            "original_index": idx,
            "content": chunk_text
        })

    return processed_doc

@handle_exceptions
def process_single_document(doc: Dict, chunk_size: int, max_topic_distance: float) -> Dict:
    """
    This function initializes the necessary models and processes a single document.
    It's defined at the top level to ensure it can be pickled by ProcessPoolExecutor.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model...")
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        return None

    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Error loading Sentence-BERT model: {e}")
        return None

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return None

    doc_id = doc.get("doc_id", f"doc_{uuid.uuid4().hex}")
    content = doc.get("content", "").strip()

    if not content:
        logger.warning(f"Document '{doc_id}' is empty. Skipping.")
        return None

    return process_document(doc_id, content, chunk_size, max_topic_distance, nlp, embedding_model, tokenizer)

@handle_exceptions
def process_documents(documents: List[Dict], chunk_size: int, max_topic_distance: float = 0.5) -> List[Dict]:
    processed_docs = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Prepare arguments as tuples
        futures = [
            executor.submit(process_single_document, doc, chunk_size, max_topic_distance)
            for doc in documents
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                processed_docs.append(result)

    return processed_docs

@handle_exceptions
def load_documents_from_folder(folder_path: str) -> List[Dict]:
    documents = []
    if not os.path.isdir(folder_path):
        logger.error(f"Invalid directory: {folder_path}")
        return documents

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_id = os.path.splitext(filename)[0]
                documents.append({"doc_id": doc_id, "content": content})
                logger.info(f"Loaded document: {doc_id}")
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

    return documents

@handle_exceptions
def save_json(data: List[Dict], filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved output to {filename}")
    except Exception as e:
        logger.error(f"Error saving output: {e}")

@handle_exceptions
def main():
    folder = "documents"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"output_{timestamp}"
    output_filename = os.path.join(output_folder, f"chunked_documents_{timestamp}.json")
    CHUNK_SIZE = 512  # Set your desired chunk size here
    max_topic_distance = 0.5

    documents = load_documents_from_folder(folder)
    if not documents:
        logger.error("No documents found. Exiting.")
        return

    processed_docs = process_documents(documents, CHUNK_SIZE, max_topic_distance)

    if not processed_docs:
        logger.error("No documents were processed successfully. Exiting.")
        return

    save_json(processed_docs, output_filename)

    # Print the processed documents directly from memory
    print(json.dumps(processed_docs, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
