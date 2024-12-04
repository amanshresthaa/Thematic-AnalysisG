import os
print("Current Working Directory:", os.getcwd())
# chunker.py
import os
import json
import uuid
from typing import List, Dict, Optional
import datetime
import concurrent.futures
import functools
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

def handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return {"error": "An error occurred. Please try again later."}
    return wrapper

def validate_metadata(metadata: Dict) -> Dict:
    """
    Ensure that all required metadata fields are present. If not, set them to empty values.

    :param metadata: The loaded metadata dictionary.
    :return: A validated metadata dictionary with all required fields.
    """
    research_objectives = metadata.get("research_objectives", "")
    theoretical_framework = metadata.get("theoretical_framework", {})
    # Ensure nested fields are present
    theoretical_framework_complete = {
        "theory": theoretical_framework.get("theory", ""),
        "philosophical_approach": theoretical_framework.get("philosophical_approach", ""),
        "rationale": theoretical_framework.get("rationale", "")
    }
    return {
        "research_objectives": research_objectives,
        "theoretical_framework": theoretical_framework_complete
    }

@handle_exceptions
def load_metadata_for_document(doc_id: str, folder_path: str) -> Optional[Dict]:
    """
    Attempt to load a JSON metadata file corresponding to the document.

    :param doc_id: The document ID (filename without extension).
    :param folder_path: The path to the folder containing the documents.
    :return: Validated metadata dictionary if available, else None.
    """
    metadata_filename = os.path.join(folder_path, f"{doc_id}.json")
    logger.debug(f"Looking for metadata file: {metadata_filename}")
    if os.path.isfile(metadata_filename):
        try:
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata for document: {doc_id}")
            validated_metadata = validate_metadata(metadata)
            logger.debug(f"Validated metadata for {doc_id}: {validated_metadata}")
            return validated_metadata
        except Exception as e:
            logger.error(f"Error loading metadata for {doc_id}: {e}")
            return None
    else:
        logger.info(f"No metadata found for document: {doc_id}")
        return None

@handle_exceptions
def load_documents_from_folder(folder_path: str) -> List[Dict]:
    """
    Load all .txt documents from the specified folder.

    :param folder_path: Path to the folder containing documents.
    :return: List of documents with 'doc_id' and 'content'.
    """
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

# Initialize spaCy and SentenceTransformer models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Error loading Sentence-BERT model: {e}")
    raise

tokenizer = tiktoken.get_encoding("cl100k_base")

class SmartChunker:
    def __init__(self, 
                 chunk_size: int, 
                 min_chunk_size: int = 512, 
                 max_topic_distance: float = 0.5, 
                 overlap_size: int = 1, 
                 delimiter: str = "\n---\n"):
        """
        Initialize the SmartChunker.

        :param chunk_size: Maximum number of tokens per chunk.
        :param min_chunk_size: Minimum number of tokens per chunk.
        :param max_topic_distance: Threshold for topic shift detection.
        :param overlap_size: Number of sentences to overlap between chunks.
        :param delimiter: Delimiter to insert at topic shifts.
        """
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_topic_distance = max_topic_distance
        self.overlap_size = overlap_size
        self.delimiter = delimiter
        self.delimiter_tokens = self.tokenize(delimiter)
        self.max_block_tokens = int(chunk_size * 0.8)
        self.max_chunk_tokens = chunk_size - self.delimiter_tokens  # Reserve space for delimiters

    def tokenize(self, text: str) -> int:
        tokens = tokenizer.encode(text)
        return len(tokens)

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
                sentences = [sent.text.strip() for sent in nlp(paragraph).sents if sent.text.strip()]
                for sentence in sentences:
                    sentence_tokens = self.tokenize(sentence)
                    if sentence_tokens > self.max_block_tokens:
                        logger.warning("Sentence exceeds maximum block tokens and will be skipped.")
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

    def split_block_into_chunks(self, block: str, previous_sentences: List[str]) -> List[str]:
        sentences = previous_sentences[-self.overlap_size:] if previous_sentences else []
        block_sentences = [sent.text.strip() for sent in nlp(block).sents if sent.text.strip()]
        sentences.extend(block_sentences)
        if not sentences:
            return []

        embeddings = embedding_model.encode(sentences)

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        mean_similarity = np.mean(similarities) if similarities else 1.0
        threshold = mean_similarity - self.max_topic_distance

        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = self.delimiter.join(chunk_sentences)  # Insert delimiter at topic shift
            tokens = self.tokenize(chunk_text)
            if tokens > self.max_chunk_tokens:
                sub_chunks = self._split_chunk_by_sentence_tokens(chunk_sentences)
                # Insert delimiter between sub_chunks
                for j in range(len(sub_chunks) - 1):
                    sub_chunks[j] += self.delimiter
                chunks.extend(sub_chunks)
            elif tokens < self.min_chunk_size and i < len(boundaries) - 2:
                # Attempt to merge with the next chunk if current chunk is too small
                next_start = end
                next_end = boundaries[i + 2]
                merged_sentences = chunk_sentences + sentences[next_start:next_end]
                merged_text = self.delimiter.join(merged_sentences)
                merged_tokens = self.tokenize(merged_text)
                if merged_tokens <= self.max_chunk_tokens:
                    chunks.append(merged_text)
                    boundaries[i + 1] = next_end  # Skip the next chunk as it's merged
                else:
                    chunks.append(chunk_text)
            else:
                chunks.append(chunk_text)

        # Remove any chunks smaller than min_chunk_size by merging with the previous chunk
        final_chunks = []
        buffer = ""
        for chunk in chunks:
            chunk_tokens = self.tokenize(chunk)
            if chunk_tokens < self.min_chunk_size:
                buffer += " " + chunk if buffer else chunk
            else:
                if buffer:
                    combined = buffer + " " + chunk
                    combined_tokens = self.tokenize(combined)
                    if combined_tokens <= self.max_chunk_tokens:
                        final_chunks.append(combined)
                        buffer = ""
                    else:
                        final_chunks.append(buffer)
                        final_chunks.append(chunk)
                        buffer = ""
                else:
                    final_chunks.append(chunk)
        if buffer:
            final_chunks.append(buffer)

        if previous_sentences and final_chunks:
            overlap_text = self.delimiter.join(previous_sentences[-self.overlap_size:])
            if final_chunks[0].startswith(overlap_text):
                final_chunks[0] = final_chunks[0][len(overlap_text):].strip()

        return final_chunks

    def _split_chunk_by_sentence_tokens(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_chunk = ''
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = self.tokenize(sentence)
            if current_tokens + sentence_tokens + self.delimiter_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + self.delimiter
                current_tokens = sentence_tokens + self.delimiter_tokens
            else:
                current_chunk += sentence + self.delimiter
                current_tokens += sentence_tokens + self.delimiter_tokens
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

@handle_exceptions
def process_document(
    doc_id: str, 
    content: str, 
    chunk_size: int, 
    min_chunk_size: int, 
    max_topic_distance: float = 0.5, 
    delimiter: str = "\n---\n",
    metadata: Optional[Dict] = None
) -> Dict:
    paragraphs = content.strip().split('\n\n')
    cleaned_content = ' '.join(paragraphs)

    # Original Output Structure
    original_output = {
        "doc_id": doc_id,
        "original_uuid": uuid.uuid4().hex,
        "content": content,
        "chunks": []
    }

    # New Output Structure
    new_output = []
    # Initialize metadata fields to empty if metadata is None
    if metadata:
        research_objectives = metadata.get("research_objectives", "")
        theoretical_framework = metadata.get("theoretical_framework", {})
        # Ensure nested fields are present
        theoretical_framework_complete = {
            "theory": theoretical_framework.get("theory", ""),
            "philosophical_approach": theoretical_framework.get("philosophical_approach", ""),
            "rationale": theoretical_framework.get("rationale", "")
        }
    else:
        research_objectives = ""
        theoretical_framework_complete = {
            "theory": "",
            "philosophical_approach": "",
            "rationale": ""
        }

    chunker = SmartChunker(
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        max_topic_distance=max_topic_distance,
        overlap_size=1,
        delimiter=delimiter
    )
    blocks = chunker.split_document_into_blocks(paragraphs)
    all_chunks = []
    previous_sentences = []

    for block in blocks:
        chunks = chunker.split_block_into_chunks(block, previous_sentences)
        if chunks:
            all_chunks.extend(chunks)
            previous_sentences = [sent.text.strip() for sent in nlp(block).sents if sent.text.strip()]

    for idx, chunk_text in enumerate(all_chunks):
        chunk_id = f"{doc_id}_chunk_{idx}"
        # Populate Original Output
        original_output["chunks"].append({
            "chunk_id": chunk_id,
            "original_index": idx,
            "content": chunk_text
        })
        # Populate New Output
        new_output.append({
            "transcript_chunk": chunk_text,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework_complete
        })

    return {
        "original_output": original_output,
        "new_output": new_output
    }

@handle_exceptions
def process_documents(
    documents: List[Dict], 
    folder_path: str,
    chunk_size_original: int,
    min_chunk_size_original: int,
    chunk_size_new: int,
    min_chunk_size_new: int,
    max_topic_distance: float = 0.5, 
    delimiter_original: str = "\n---\n",
    delimiter_new: str = "\n***\n"
) -> Dict[str, List[Dict]]:
    original_processed_docs = []
    new_processed_docs = []

    def process_single_document(doc):
        doc_id = doc.get("doc_id", f"doc_{len(original_processed_docs)+1}")
        content = doc.get("content", "").strip()

        if not content:
            logger.warning(f"Document '{doc_id}' is empty. Skipping.")
            return None

        # Attempt to load associated metadata JSON file
        metadata = load_metadata_for_document(doc_id, folder_path)

        # Process for Original Output
        processed_original = process_document(
            doc_id=doc_id,
            content=content,
            chunk_size=chunk_size_original,
            min_chunk_size=min_chunk_size_original,
            max_topic_distance=max_topic_distance,
            delimiter=delimiter_original,
            metadata=metadata
        )
        # Process for New Output with different chunk sizes and delimiter
        processed_new = process_document(
            doc_id=doc_id,
            content=content,
            chunk_size=chunk_size_new,
            min_chunk_size=min_chunk_size_new,
            max_topic_distance=max_topic_distance,
            delimiter=delimiter_new,
            metadata=metadata
        )

        return (processed_original, processed_new)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_document, doc) for doc in documents]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                processed_original, processed_new = result
                original_processed_docs.append(processed_original["original_output"])
                new_processed_docs.extend(processed_new["new_output"])  # Flatten the list

    return {
        "original_output": original_processed_docs,
        "new_output": new_processed_docs
    }

@handle_exceptions
def save_json(data: Dict[str, List[Dict]], original_filename: str, new_filename: str):
    """
    Save the original and new output structures into separate JSON files.

    :param data: Dictionary containing both output structures.
    :param original_filename: Filename for the original output.
    :param new_filename: Filename for the new output.
    """
    try:
        # Save Original Output
        os.makedirs(os.path.dirname(original_filename), exist_ok=True)
        with open(original_filename, 'w', encoding='utf-8') as f:
            json.dump(data["original_output"], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved original output to {original_filename}")

        # Save New Output
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
        with open(new_filename, 'w', encoding='utf-8') as f:
            json.dump(data["new_output"], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved new output to {new_filename}")

    except Exception as e:
        logger.error(f"Error saving output: {e}")

@handle_exceptions
def main():
    # Load configuration
    config_filename = 'chunker/config.json'
    if not os.path.isfile(config_filename):
        logger.error(f"Configuration file '{config_filename}' not found. Exiting.")
        return

    try:
        with open(config_filename, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        logger.info(f"Loaded configuration from {config_filename}")
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        return

    folder = config.get("folder_path", "documents")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output filenames
    output_folder_original = f"{config.get('output_prefix_original', 'chunks_for_CE_original')}_{timestamp}"
    output_folder_new = f"{config.get('output_prefix_new', 'chunks_for_CE_new')}_{timestamp}"
    output_filename_original = os.path.join(output_folder_original, f"{config.get('output_prefix_original', 'chunks_for_CE_original')}_{timestamp}.json")
    output_filename_new = os.path.join(output_folder_new, f"{config.get('output_prefix_new', 'chunks_for_CE_new')}_{timestamp}.json")
    
    CHUNK_SIZE_ORIGINAL = config.get("chunk_size_original", 2048)  # Maximum chunk size for original output
    MIN_CHUNK_SIZE_ORIGINAL = config.get("min_chunk_size_original", 1024)  # Minimum chunk size for original output
    CHUNK_SIZE_NEW = config.get("chunk_size_new", 1024)  # Maximum chunk size for new output
    MIN_CHUNK_SIZE_NEW = config.get("min_chunk_size_new", 512)  # Minimum chunk size for new output
    max_topic_distance = config.get("max_topic_distance", 0.5)
    delimiter_original = config.get("delimiter_original", "\n---\n")
    delimiter_new = config.get("delimiter_new", "\n***\n")

    documents = load_documents_from_folder(folder)
    if not documents:
        logger.error("No documents found. Exiting.")
        return

    processed_docs = process_documents(
        documents, 
        folder_path=folder,
        chunk_size_original=CHUNK_SIZE_ORIGINAL,
        min_chunk_size_original=MIN_CHUNK_SIZE_ORIGINAL,
        chunk_size_new=CHUNK_SIZE_NEW,
        min_chunk_size_new=MIN_CHUNK_SIZE_NEW,
        max_topic_distance=max_topic_distance, 
        delimiter_original=delimiter_original,
        delimiter_new=delimiter_new
    )

    save_json(processed_docs, output_filename_original, output_filename_new)

    # Optionally, print the outputs
    try:
        with open(output_filename_original, 'r', encoding='utf-8') as f:
            print("Original Output:")
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error printing original output: {e}")

    try:
        with open(output_filename_new, 'r', encoding='utf-8') as f:
            print("\nNew Output:")
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error printing new output: {e}")

if __name__ == "__main__":
    main()
