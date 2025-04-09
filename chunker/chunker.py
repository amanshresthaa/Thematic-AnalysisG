#!/usr/bin/env python
# chunker/chunker.py

import json
import os
import re
import logging
import tiktoken
import datetime
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, 
                 documents_dir="documents/", 
                 chunk_size=1024, 
                 chunk_overlap=200,
                 encoding_name="cl100k_base"):
        """
        Initialize the document chunker.
        
        Args:
            documents_dir (str): Directory containing documents to process
            chunk_size (int): Maximum number of tokens per chunk
            chunk_overlap (int): Number of overlapping tokens between chunks
            encoding_name (str): Tokenizer encoding to use
        """
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.output_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = os.path.join("data", "chunker_output", self.output_timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ensure other required directories exist
        os.makedirs(os.path.join("data", "codebase_chunks"), exist_ok=True)
        os.makedirs(os.path.join("data", "input"), exist_ok=True)
    
    def load_documents(self):
        """Load documents from the documents directory."""
        documents = []
        
        if not os.path.exists(self.documents_dir):
            logger.error(f"Documents directory {self.documents_dir} does not exist")
            return documents
        
        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            
            # Skip directories and non-text/json files
            if os.path.isdir(file_path):
                continue
                
            if filename.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document object
                    document = {
                        "id": os.path.splitext(filename)[0],
                        "content": content,
                        "metadata": self.extract_metadata_from_text(content)
                    }
                    documents.append(document)
                    logger.info(f"Loaded text document: {filename}")
                except Exception as e:
                    logger.error(f"Error loading document {filename}: {e}")
            
            elif filename.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                    
                    # Ensure the document has required fields
                    if "id" not in document:
                        document["id"] = os.path.splitext(filename)[0]
                    
                    if "content" not in document and "text" in document:
                        document["content"] = document["text"]
                    
                    if "metadata" not in document:
                        document["metadata"] = self.extract_metadata_from_text(document.get("content", ""))
                    
                    documents.append(document)
                    logger.info(f"Loaded JSON document: {filename}")
                except Exception as e:
                    logger.error(f"Error loading JSON document {filename}: {e}")
        
        return documents
    
    def extract_metadata_from_text(self, text):
        """
        Extract metadata from document text including research objectives and theoretical framework.
        
        Args:
            text (str): Document text content
            
        Returns:
            dict: Extracted metadata
        """
        metadata = {
            "research_objectives": [],
            "theoretical_framework": [],
            "document_type": "interview"
        }
        
        # Extract research objectives (simplified - in a real implementation, this would be more sophisticated)
        objective_pattern = re.compile(r'(?:objective|goal|aim|purpose)s?:?\s*(.*?)(?:\n\n|\n(?=[A-Z]))', re.IGNORECASE)
        objective_matches = objective_pattern.findall(text)
        if objective_matches:
            metadata["research_objectives"] = [match.strip() for match in objective_matches if match.strip()]
        
        # Extract theoretical framework mentions
        framework_keywords = [
            "grounded theory", "phenomenology", "ethnography", "case study", 
            "narrative analysis", "discourse analysis", "content analysis",
            "thematic analysis", "framework analysis"
        ]
        
        for framework in framework_keywords:
            if re.search(r'\b' + re.escape(framework) + r'\b', text, re.IGNORECASE):
                metadata["theoretical_framework"].append(framework)
        
        return metadata
    
    def tokenize_text(self, text):
        """Convert text to tokens using the specified encoding."""
        return self.encoding.encode(text)
    
    def chunk_document(self, document):
        """
        Split a document into overlapping chunks.
        
        Args:
            document (dict): Document with content and metadata
            
        Returns:
            list: List of chunk dictionaries
        """
        content = document.get("content", "")
        if not content:
            logger.warning(f"Document {document.get('id', 'unknown')} has no content to chunk")
            return []
        
        tokens = self.tokenize_text(content)
        chunks = []
        
        i = 0
        chunk_id = 0
        
        while i < len(tokens):
            # Get chunk tokens
            chunk_end = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:chunk_end]
            
            # Convert tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk = {
                "id": f"{document.get('id', 'doc')}_chunk_{chunk_id}",
                "document_id": document.get('id', 'unknown'),
                "text": chunk_text,
                "metadata": document.get("metadata", {}),
                "chunk_id": chunk_id,
                "token_count": len(chunk_tokens)
            }
            
            chunks.append(chunk)
            
            # Move to next chunk position, accounting for overlap
            i += self.chunk_size - self.chunk_overlap
            
            # Ensure we make progress even with large overlaps
            if i <= 0:
                i = min(self.chunk_size, len(tokens))
                
            chunk_id += 1
        
        return chunks
    
    def process_all_documents(self):
        """Process all documents and create chunks."""
        documents = self.load_documents()
        all_chunks = []
        
        for document in documents:
            document_chunks = self.chunk_document(document)
            all_chunks.extend(document_chunks)
            logger.info(f"Created {len(document_chunks)} chunks for document {document.get('id', 'unknown')}")
        
        if not all_chunks:
            logger.warning("No chunks were created from documents")
            return None
        
        return all_chunks
    
    def save_chunks(self, chunks):
        """
        Save chunks to output files.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            dict: Paths to saved files
        """
        if not chunks:
            logger.error("No chunks to save")
            return None
        
        # Create metadata about this chunking session
        info = {
            "timestamp": self.output_timestamp,
            "total_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "documents_processed": list(set(chunk["document_id"] for chunk in chunks))
        }
        
        # Flatten chunks for queries_quotation.json format
        flattened_chunks = []
        for chunk in chunks:
            flattened_chunks.append({
                "id": chunk["id"],
                "query": chunk["text"],
                "document_id": chunk["document_id"],
                "metadata": chunk["metadata"]
            })
        
        # Save original chunks to timestamped directory
        original_chunks_path = os.path.join(self.output_dir, "chunks_original.json")
        with open(original_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        # Create info.json in the output directory
        info_path = os.path.join(self.output_dir, "info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        # Copy to pipeline input locations
        codebase_chunks_path = os.path.join("data", "codebase_chunks", "codebase_chunks.json")
        with open(codebase_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        queries_path = os.path.join("data", "input", "queries_quotation.json")
        with open(queries_path, 'w', encoding='utf-8') as f:
            json.dump(flattened_chunks, f, indent=2)
        
        # Update latest symlink
        latest_path = os.path.join("data", "chunker_output", "latest")
        if os.path.exists(latest_path) or os.path.islink(latest_path):
            os.remove(latest_path)
        
        with open(latest_path, 'w') as f:
            f.write(self.output_timestamp)
        
        logger.info(f"Chunks saved successfully to {self.output_dir}")
        logger.info(f"Copied to pipeline locations: {codebase_chunks_path} and {queries_path}")
        
        return {
            "original_chunks": original_chunks_path,
            "info": info_path,
            "codebase_chunks": codebase_chunks_path,
            "queries": queries_path
        }

def run_chunker(documents_dir="documents/", chunk_size=1024, chunk_overlap=200):
    """
    Run the document chunking process.
    
    Args:
        documents_dir (str): Directory containing documents to process
        chunk_size (int): Maximum tokens per chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        dict: Information about the chunking process results
    """
    logger.info(f"Starting document chunking process with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    chunker = DocumentChunker(
        documents_dir=documents_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = chunker.process_all_documents()
    if not chunks:
        logger.error("Document chunking failed - no chunks produced")
        return {"success": False, "error": "No chunks produced"}
    
    output_paths = chunker.save_chunks(chunks)
    if not output_paths:
        logger.error("Failed to save chunks")
        return {"success": False, "error": "Failed to save chunks"}
    
    return {
        "success": True, 
        "timestamp": chunker.output_timestamp,
        "paths": output_paths,
        "chunk_count": len(chunks)
    }

if __name__ == "__main__":
    # Run the chunker with default settings
    result = run_chunker()
    if result["success"]:
        print(f"Chunking completed successfully. Created {result['chunk_count']} chunks.")
        print(f"Output saved to: {result['paths']['original_chunks']}")
    else:
        print(f"Chunking failed: {result.get('error', 'Unknown error')}")
