import os
import json
import uuid
from typing import List, Dict, Optional, Tuple, Any
import datetime
import concurrent.futures
import functools
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict

# Setup logging first
def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging with customizable level and optional file output."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers
    )

setup_logging()
logger = logging.getLogger(__name__)

# Load required NLP libraries
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import tiktoken
    import numpy as np
    logger.info("Successfully imported all required libraries")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    raise

def handle_exceptions(func):
    """Decorator to handle exceptions gracefully and provide detailed error logs."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return {"error": f"Error in {func.__name__}: {str(e)}"}
    return wrapper

@dataclass
class TheoreticalFramework:
    """Data class for theoretical framework structure."""
    theory: str = ""
    philosophical_approach: str = ""
    rationale: str = ""

@dataclass
class DocumentMetadata:
    """Data class for document metadata."""
    research_objectives: str = ""
    theoretical_framework: TheoreticalFramework = field(default_factory=TheoreticalFramework)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create a DocumentMetadata instance from a dictionary."""
        if not data:
            return cls()
            
        tf_data = data.get('theoretical_framework', {})
        tf = TheoreticalFramework(
            theory=tf_data.get('theory', ''),
            philosophical_approach=tf_data.get('philosophical_approach', ''),
            rationale=tf_data.get('rationale', '')
        )
        
        return cls(
            research_objectives=data.get('research_objectives', ''),
            theoretical_framework=tf
        )

@dataclass
class ChunkConfig:
    """Configuration parameters for chunking."""
    chunk_size: int
    min_chunk_size: int
    max_topic_distance: float = 0.5
    overlap_size: int = 1
    delimiter: str = "\n---\n"

@dataclass
class Chunk:
    """Data class for document chunks."""
    chunk_id: str
    content: str
    original_index: int
    document_id: str
    chunk_tokens: int = 0
    metadata: Optional[DocumentMetadata] = None

@dataclass
class ProcessedDocument:
    """Data class for processed documents."""
    doc_id: str
    original_uuid: str
    content: str
    chunks: List[Chunk] = field(default_factory=list)

# Initialize NLP components
def initialize_nlp_components():
    """Initialize and return NLP components with proper error handling."""
    # Initialize spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        logger.info("Downloaded and loaded spaCy model: en_core_web_sm")

    # Initialize SentenceTransformer
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded SentenceTransformer model: all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Error loading Sentence-BERT model: {e}")
        raise

    # Initialize tiktoken
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("Loaded tokenizer: cl100k_base")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

    return nlp, embedding_model, tokenizer

# Load NLP models (doing this at module level)
try:
    nlp, embedding_model, tokenizer = initialize_nlp_components()
except Exception as e:
    logger.error(f"Failed to initialize NLP components: {e}")
    # We'll continue and fail later if these components are needed

@handle_exceptions
def load_metadata_for_document(doc_id: str, folder_path: str) -> Optional[DocumentMetadata]:
    """
    Attempt to load a JSON metadata file corresponding to the document.

    Args:
        doc_id: The document ID (filename without extension)
        folder_path: The path to the folder containing the documents

    Returns:
        DocumentMetadata if available, else None
    """
    metadata_path = Path(folder_path) / f"{doc_id}.json"
    logger.debug(f"Looking for metadata file: {metadata_path}")
    
    if metadata_path.is_file():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            logger.info(f"Loaded metadata for document: {doc_id}")
            return DocumentMetadata.from_dict(metadata_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata file for {doc_id}: {e}")
            return None
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

    Args:
        folder_path: Path to the folder containing documents

    Returns:
        List of documents with 'doc_id' and 'content'
    """
    documents = []
    folder = Path(folder_path)
    
    if not folder.is_dir():
        logger.error(f"Invalid directory: {folder_path}")
        return documents

    for file_path in folder.glob('*.txt'):
        try:
            content = file_path.read_text(encoding='utf-8')
            doc_id = file_path.stem
            documents.append({"doc_id": doc_id, "content": content})
            logger.info(f"Loaded document: {doc_id} ({len(content)} characters)")
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")

    logger.info(f"Loaded {len(documents)} documents from {folder_path}")
    return documents

class SmartChunker:
    """Advanced document chunking system with topic-based segmentation."""
    
    def __init__(self, config: ChunkConfig):
        """
        Initialize the SmartChunker with configuration parameters.

        Args:
            config: ChunkConfig object with chunking parameters
        """
        self.config = config
        self.delimiter_tokens = self._tokenize(config.delimiter)
        self.max_block_tokens = int(config.chunk_size * 0.8)
        self.max_chunk_tokens = config.chunk_size - self.delimiter_tokens

    def _tokenize(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        tokens = tokenizer.encode(text)
        return len(tokens)

    def _get_token_count(self, text: str) -> int:
        """Get token count for a text string."""
        return self._tokenize(text)
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text

    def split_document_into_blocks(self, paragraphs: List[str]) -> List[str]:
        """
        Split document into semantic blocks that respect token limits.
        
        Args:
            paragraphs: List of paragraphs from the document
            
        Returns:
            List of text blocks
        """
        blocks = []
        current_block = ''
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_tokens = self._tokenize(paragraph)
            
            # Handle paragraphs that exceed max block size
            if paragraph_tokens > self.max_block_tokens:
                logger.warning(f"Paragraph exceeds maximum block tokens ({paragraph_tokens} > {self.max_block_tokens}) and will be split")
                
                # First add current block if it exists
                if current_block:
                    blocks.append(current_block.strip())
                    current_block = ''
                    current_tokens = 0
                
                # Split large paragraph into sentences
                doc = nlp(paragraph)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                
                # Process each sentence
                for sentence in sentences:
                    sentence_tokens = self._tokenize(sentence)
                    
                    if sentence_tokens > self.max_block_tokens:
                        logger.warning(f"Sentence exceeds maximum block tokens ({sentence_tokens} > {self.max_block_tokens}) and will be truncated")
                        # Try to split by clauses or just add as is if we can't do better
                        blocks.append(sentence)
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
                # Handle normal-sized paragraphs
                if current_tokens + paragraph_tokens > self.max_block_tokens:
                    if current_block:
                        blocks.append(current_block.strip())
                    current_block = paragraph + ' '
                    current_tokens = paragraph_tokens
                else:
                    current_block += paragraph + ' '
                    current_tokens += paragraph_tokens

        # Add any remaining content
        if current_block:
            blocks.append(current_block.strip())

        logger.info(f"Split document into {len(blocks)} blocks")
        return blocks

    def detect_topic_shifts(self, sentences: List[str]) -> List[int]:
        """
        Detect topic shifts within a list of sentences based on semantic similarity.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of indices where topic shifts occur
        """
        if len(sentences) <= 1:
            return [0, len(sentences)]
            
        # Generate embeddings for sentences
        embeddings = embedding_model.encode(sentences)
        
        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
            
        # Adaptive thresholding based on document characteristics
        mean_similarity = np.mean(similarities) if similarities else 1.0
        std_similarity = np.std(similarities) if len(similarities) > 1 else 0.2
        
        # Use a more adaptive threshold based on statistics
        threshold = max(0.3, mean_similarity - (self.config.max_topic_distance * std_similarity))
        
        # Identify topic shift boundaries
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1)
                logger.debug(f"Topic shift detected at sentence {i+1} (similarity: {sim:.4f}, threshold: {threshold:.4f})")
        
        # Always include the end
        if boundaries[-1] != len(sentences):
            boundaries.append(len(sentences))
            
        return boundaries

    def split_block_into_chunks(self, block: str, previous_sentences: List[str]) -> List[Tuple[str, int]]:
        """
        Split a block into chunks based on topic shifts and token limits.
        
        Args:
            block: Text block to split
            previous_sentences: Sentences from previous block for context
            
        Returns:
            List of (chunk_text, token_count) tuples
        """
        # Get overlap sentences from previous block
        overlap_sentences = previous_sentences[-self.config.overlap_size:] if previous_sentences else []
        
        # Extract sentences from current block
        doc = nlp(block)
        block_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not block_sentences:
            return []
            
        # Combine overlap and current sentences
        all_sentences = overlap_sentences + block_sentences
        
        # Detect topic boundaries
        boundaries = self.detect_topic_shifts(all_sentences)
        
        # Adjust boundaries to account for overlap sentences
        adjusted_boundaries = []
        overlap_count = len(overlap_sentences)
        for boundary in boundaries:
            if boundary > overlap_count:
                adjusted_boundaries.append(boundary - overlap_count)
            elif boundary == overlap_count:
                adjusted_boundaries.append(0)
                
        if not adjusted_boundaries:
            adjusted_boundaries = [0, len(block_sentences)]
            
        # Ensure the last boundary includes all sentences
        if adjusted_boundaries[-1] != len(block_sentences):
            adjusted_boundaries.append(len(block_sentences))
            
        # Create chunks based on boundaries
        chunks_with_tokens = []
        for i in range(len(adjusted_boundaries) - 1):
            start = adjusted_boundaries[i]
            end = adjusted_boundaries[i + 1]
            
            # Skip empty ranges
            if start == end:
                continue
                
            chunk_sentences = block_sentences[start:end]
            chunk_text = self.config.delimiter.join(chunk_sentences)
            chunk_tokens = self._tokenize(chunk_text)
            
            # Handle chunks that exceed token limit
            if chunk_tokens > self.max_chunk_tokens:
                sub_chunks = self._split_chunk_by_token_budget(chunk_sentences)
                chunks_with_tokens.extend(sub_chunks)
            else:
                chunks_with_tokens.append((chunk_text, chunk_tokens))
                
        # Apply post-processing to optimize chunk sizes
        optimized_chunks = self._optimize_chunk_sizes(chunks_with_tokens)
        
        return optimized_chunks

    def _split_chunk_by_token_budget(self, sentences: List[str]) -> List[Tuple[str, int]]:
        """
        Split sentences into chunks that fit within token budget.
        
        Args:
            sentences: List of sentences to split
            
        Returns:
            List of (chunk_text, token_count) tuples
        """
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._tokenize(sentence)
            delimiter_tokens = 0 if not current_chunk_sentences else self.delimiter_tokens
            
            # If adding this sentence would exceed the limit
            if current_tokens + sentence_tokens + delimiter_tokens > self.max_chunk_tokens:
                # Save current chunk if it exists
                if current_chunk_sentences:
                    chunk_text = self.config.delimiter.join(current_chunk_sentences)
                    chunks.append((chunk_text, current_tokens))
                    current_chunk_sentences = []
                    current_tokens = 0
                
                # Handle sentences that are too long on their own
                if sentence_tokens > self.max_chunk_tokens:
                    logger.warning(f"Sentence exceeds max chunk tokens ({sentence_tokens} > {self.max_chunk_tokens})")
                    # Try to truncate or split the sentence
                    truncated = sentence[:int(len(sentence) * 0.9)]  # Simple truncation strategy
                    truncated_tokens = self._tokenize(truncated)
                    chunks.append((truncated, truncated_tokens))
                else:
                    current_chunk_sentences.append(sentence)
                    current_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens + delimiter_tokens
                
        # Add any remaining content
        if current_chunk_sentences:
            chunk_text = self.config.delimiter.join(current_chunk_sentences)
            chunks.append((chunk_text, current_tokens))
            
        return chunks

    def _optimize_chunk_sizes(self, chunks_with_tokens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Optimize chunk sizes by merging small chunks or splitting large ones.
        
        Args:
            chunks_with_tokens: List of (chunk_text, token_count) tuples
            
        Returns:
            Optimized list of (chunk_text, token_count) tuples
        """
        if not chunks_with_tokens:
            return []
            
        optimized_chunks = []
        i = 0
        
        while i < len(chunks_with_tokens):
            current_chunk, current_tokens = chunks_with_tokens[i]
            
            # Try to merge small chunks with next chunk
            if (current_tokens < self.config.min_chunk_size and 
                i < len(chunks_with_tokens) - 1):
                
                next_chunk, next_tokens = chunks_with_tokens[i + 1]
                combined_text = current_chunk + self.config.delimiter + next_chunk
                combined_tokens = self._tokenize(combined_text)
                
                if combined_tokens <= self.max_chunk_tokens:
                    # Merge chunks
                    optimized_chunks.append((combined_text, combined_tokens))
                    i += 2  # Skip the next chunk since we merged it
                else:
                    # Can't merge, keep as is
                    optimized_chunks.append((current_chunk, current_tokens))
                    i += 1
            else:
                # Current chunk is good size or is the last chunk
                optimized_chunks.append((current_chunk, current_tokens))
                i += 1
                
        return optimized_chunks

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int
    min_chunk_size: int
    max_topic_distance: float = 0.5
    delimiter: str = "\n---\n"
    
@handle_exceptions
def process_document(
    doc_id: str, 
    content: str,
    config: ProcessingConfig,
    metadata: Optional[DocumentMetadata] = None
) -> Dict:
    """
    Process a document into chunks using smart chunking.
    
    Args:
        doc_id: Document identifier
        content: Document content
        config: Processing configuration
        metadata: Optional document metadata
        
    Returns:
        Dictionary with processed document information
    """
    logger.info(f"Processing document {doc_id} ({len(content)} chars)")
    
    # Split content into paragraphs
    paragraphs = [p for p in content.strip().split('\n\n') if p.strip()]
    
    # Create chunking configuration
    chunk_config = ChunkConfig(
        chunk_size=config.chunk_size,
        min_chunk_size=config.min_chunk_size,
        max_topic_distance=config.max_topic_distance,
        overlap_size=1,
        delimiter=config.delimiter
    )
    
    # Initialize chunker
    chunker = SmartChunker(chunk_config)
    
    # Split document into blocks
    blocks = chunker.split_document_into_blocks(paragraphs)
    
    # Process blocks into chunks
    all_chunks = []
    previous_sentences = []
    chunk_contents = []
    
    for block in blocks:
        chunks_with_tokens = chunker.split_block_into_chunks(block, previous_sentences)
        
        # Update context for next block
        if block:
            doc = nlp(block)
            previous_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Add chunks with their token counts    
        for chunk_text, token_count in chunks_with_tokens:
            chunk_contents.append((chunk_text, token_count))
    
    # Create chunk objects
    for idx, (chunk_text, token_count) in enumerate(chunk_contents):
        chunk_id = f"{doc_id}_chunk_{idx}"
        
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            original_index=idx,
            content=chunk_text,
            chunk_tokens=token_count,
            metadata=metadata
        )
        all_chunks.append(chunk)
    
    # Create original output format
    original_output = ProcessedDocument(
        doc_id=doc_id,
        original_uuid=uuid.uuid4().hex,
        content=content,
        chunks=all_chunks
    )
    
    # Create new output format
    new_output = []
    for chunk in all_chunks:
        if metadata:
            new_output.append({
                "transcript_chunk": chunk.content,
                "research_objectives": metadata.research_objectives,
                "theoretical_framework": asdict(metadata.theoretical_framework)
            })
        else:
            new_output.append({
                "transcript_chunk": chunk.content,
                "research_objectives": "",
                "theoretical_framework": {
                    "theory": "",
                    "philosophical_approach": "",
                    "rationale": ""
                }
            })
    
    logger.info(f"Document {doc_id} split into {len(all_chunks)} chunks")
    
    return {
        "original_output": asdict(original_output),
        "new_output": new_output
    }

@handle_exceptions
def process_documents(
    documents: List[Dict],
    folder_path: str,
    original_config: ProcessingConfig,
    new_config: ProcessingConfig,
    max_workers: int = None
) -> Dict[str, List[Dict]]:
    """
    Process multiple documents with concurrent execution.
    
    Args:
        documents: List of documents to process
        folder_path: Path to the documents folder
        original_config: Configuration for original output format
        new_config: Configuration for new output format
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Dictionary with both output formats
    """
    if not documents:
        logger.warning("No documents to process")
        return {"original_output": [], "new_output": []}
    
    original_processed_docs = []
    new_processed_docs = []
    
    def process_single_document(doc):
        doc_id = doc.get("doc_id", f"doc_{uuid.uuid4().hex[:8]}")
        content = doc.get("content", "").strip()
        
        if not content:
            logger.warning(f"Document '{doc_id}' is empty. Skipping.")
            return None
        
        # Load metadata
        metadata = load_metadata_for_document(doc_id, folder_path)
        
        # Process for original output
        processed_original = process_document(
            doc_id=doc_id,
            content=content,
            config=original_config,
            metadata=metadata
        )
        
        # Process for new output with different configuration
        processed_new = process_document(
            doc_id=doc_id,
            content=content,
            config=new_config,
            metadata=metadata
        )
        
        return (processed_original, processed_new)
    
    # Determine optimal number of workers based on CPU count
    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4)
    
    logger.info(f"Processing {len(documents)} documents with {max_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_document, doc) for doc in documents]
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if result:
                    processed_original, processed_new = result
                    original_processed_docs.append(processed_original["original_output"])
                    new_processed_docs.extend(processed_new["new_output"])
                    
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == len(documents):
                    logger.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error processing document: {e}", exc_info=True)
    
    logger.info(f"Completed processing {len(documents)} documents")
    logger.info(f"Generated {len(original_processed_docs)} documents in original format")
    logger.info(f"Generated {len(new_processed_docs)} chunks in new format")
    
    return {
        "original_output": original_processed_docs,
        "new_output": new_processed_docs
    }

@handle_exceptions
def save_json(data: Dict[str, List[Dict]], original_filename: str, new_filename: str):
    """
    Save the original and new output structures into separate JSON files.
    
    Args:
        data: Dictionary containing both output structures
        original_filename: Filename for the original output
        new_filename: Filename for the new output
    """
    try:
        # Ensure output directories exist
        os.makedirs(os.path.dirname(original_filename), exist_ok=True)
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
        
        # Save original output
        original_stats = {
            "documents": len(data["original_output"]),
            "total_chunks": sum(len(doc.get("chunks", [])) for doc in data["original_output"])
        }
        
        with open(original_filename, 'w', encoding='utf-8') as f:
            json.dump(data["original_output"], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved original output to {original_filename}")
        logger.info(f"Original output stats: {original_stats}")
        
        # Save new output
        with open(new_filename, 'w', encoding='utf-8') as f:
            json.dump(data["new_output"], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved new output to {new_filename} ({len(data['new_output'])} chunks)")
        
    except Exception as e:
        logger.error(f"Error saving output: {e}", exc_info=True)
        raise

@dataclass
class AppConfig:
    """Application configuration."""
    folder_path: str = "documents"
    output_prefix_original: str = "chunks_for_CE_original"
    output_prefix_new: str = "chunks_for_CE_new"
    chunk_size_original: int = 2048
    min_chunk_size_original: int = 1024
    chunk_size_new: int = 1024
    min_chunk_size_new: int = 512
    max_topic_distance: float = 0.5
    delimiter_original: str = "\n---\n"
    delimiter_new: str = "\n***\n"
    max_workers: int = None
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_file(cls, filename: str) -> 'AppConfig':
        """Load configuration from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as config_file:
                config_dict = json.load(config_file)
            logger.info(f"Loaded configuration from {filename}")
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}", exc_info=True)
            return cls()  # Return default config on error

@handle_exceptions
def main():
    """Main application entry point."""
    # Load configuration
    config_filename = 'chunker/config.json'
    if not os.path.isfile(config_filename):
        logger.error(f"Configuration file '{config_filename}' not found. Using defaults.")
        config = AppConfig()
    else:
        config = AppConfig.from_file(config_filename)
    
    # Configure logging based on config
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    setup_logging(log_level=log_level, log_file=config.log_file)
    
    # Generate timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output paths
    output_folder_original = f"{config.output_prefix_original}_{timestamp}"
    output_folder_new = f"{config.output_prefix_new}_{timestamp}"
    output_filename_original = os.path.join(output_folder_original, f"{config.output_prefix_original}_{timestamp}.json")
    output_filename_new = os.path.join(output_folder_new, f"{config.output_prefix_new}_{timestamp}.json")
    
    # Create processing configurations
    original_config = ProcessingConfig(
        chunk_size=config.chunk_size_original,
        min_chunk_size=config.min_chunk_size_original,
        max_topic_distance=config.max_topic_distance,
        delimiter=config.delimiter_original
    )
    
    new_config = ProcessingConfig(
        chunk_size=config.chunk_size_new,
        min_chunk_size=config.min_chunk_size_new,
        max_topic_distance=config.max_topic_distance,
        delimiter=config.delimiter_new
    )
    
    # Print configuration summary
    logger.info("=== Configuration Summary ===")
    logger.info(f"Documents folder: {config.folder_path}")
    logger.info(f"Original output: chunk_size={config.chunk_size_original}, min_size={config.min_chunk_size_original}")
    logger.info(f"New output: chunk_size={config.chunk_size_new}, min_size={config.min_chunk_size_new}")
    logger.info(f"Max topic distance: {config.max_topic_distance}")
    logger.info(f"Max workers: {config.max_workers or 'auto'}")
    logger.info("===========================")
    
    # Load documents
    documents = load_documents_from_folder(config.folder_path)
    if not documents:
        logger.error("No documents found. Exiting.")
        return
    
    # Process documents
    processed_docs = process_documents(
        documents,
        folder_path=config.folder_path,
        original_config=original_config,
        new_config=new_config,
        max_workers=config.max_workers
    )
    
    # Save results
    save_json(processed_docs, output_filename_original, output_filename_new)
    
    logger.info("Processing complete!")
    logger.info(f"Original output saved to: {output_filename_original}")
    logger.info(f"New output saved to: {output_filename_new}")
    
    # Print basic stats about the outputs
    try:
        with open(output_filename_original, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            total_original_chunks = sum(len(doc.get("chunks", [])) for doc in original_data)
            logger.info(f"Original output: {len(original_data)} documents with {total_original_chunks} total chunks")
    except Exception as e:
        logger.error(f"Error analyzing original output: {e}")

    try:
        with open(output_filename_new, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
            logger.info(f"New output: {len(new_data)} total chunks")
    except Exception as e:
        logger.error(f"Error analyzing new output: {e}")

if __name__ == "__main__":
    main()