"""
Chunker module for preprocessing documents before the Thematic Analysis Pipeline.
This module handles loading raw documents, splitting them into chunks,
and saving both the original chunks and pipeline-ready flattened data.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

def chunk_document(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    doc_id: str = None
) -> List[Dict[str, Any]]:
    """
    Splits a raw document into overlapping chunks of a given size.
    
    Args:
        text: The document text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        doc_id: Optional document identifier
        
    Returns:
        List of dictionaries with structure:
        [
          {"index": 0, "content": "...", "doc_id": "..."}, 
          {"index": 1, "content": "...", "doc_id": "..."},
          ...
        ]
    """
    chunks = []
    start = 0
    index = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        chunk_info = {
            "index": index,
            "content": chunk_text.strip()
        }
        
        if doc_id:
            chunk_info["doc_id"] = doc_id
            
        chunks.append(chunk_info)
        
        index += 1
        # Move forward by chunk_size minus overlap
        start += (chunk_size - chunk_overlap)
    
    return chunks


def generate_timestamped_dir(base_dir: str = "data/chunker_output") -> str:
    """
    Creates a timestamped folder in the specified base directory.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Path to the newly created timestamped directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created timestamped directory: {output_dir}")
    return output_dir


def run_chunker(
    documents_folder: str = "documents",
    output_base_dir: str = "data/chunker_output",
    pipeline_input_path: str = "data/input/queries_quotation.json",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    research_objective: str = "Extract meaningful quotations from these interview transcripts."
) -> Tuple[str, str]:
    """
    Main chunker function that processes documents into chunks for the pipeline.
    
    Args:
        documents_folder: Folder containing raw document files
        output_base_dir: Base directory to save chunker output
        pipeline_input_path: Path to save flattened pipeline input
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        research_objective: Default research objective for pipeline entries
        
    Returns:
        Tuple of (original_chunks_path, pipeline_input_path)
    """
    # Ensure documents folder exists
    if not os.path.exists(documents_folder):
        logger.warning(f"Documents folder {documents_folder} does not exist. Creating it.")
        os.makedirs(documents_folder, exist_ok=True)
    
    # Create timestamped directory for chunk outputs
    chunker_out_dir = generate_timestamped_dir(output_base_dir)
    original_chunks_path = os.path.join(chunker_out_dir, "chunks_original.json")
    
    all_original_docs = []
    pipeline_entries = []
    
    # Get list of documents
    document_files = [f for f in os.listdir(documents_folder) 
                     if f.lower().endswith((".txt", ".md"))]
    
    if not document_files:
        logger.warning(f"No text documents found in {documents_folder}")
    
    for filename in document_files:
        doc_id = os.path.splitext(filename)[0]  # Remove extension
        full_path = os.path.join(documents_folder, filename)
        
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # Chunk the document
            doc_chunks = chunk_document(
                raw_text, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                doc_id=doc_id
            )
            
            # Store original chunks
            all_original_docs.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunks": doc_chunks
            })
            
            # Flatten for pipeline usage
            for chunk in doc_chunks:
                pipeline_entry = {
                    "transcript_chunk": chunk["content"],
                    "research_objectives": research_objective,
                    "theoretical_framework": {
                        "theory": "Thematic Analysis",
                        "philosophical_approach": "Interpretive",
                        "rationale": "To identify patterns and themes across the dataset"
                    }
                }
                
                # Add document metadata if needed
                pipeline_entry["metadata"] = {
                    "doc_id": doc_id,
                    "chunk_index": chunk["index"]
                }
                
                pipeline_entries.append(pipeline_entry)
                
            logger.info(f"Processed document: {filename} - Created {len(doc_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
    
    # Save original chunk data
    os.makedirs(os.path.dirname(original_chunks_path), exist_ok=True)
    with open(original_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_original_docs, f, indent=2)
    logger.info(f"Saved original chunks to: {original_chunks_path}")
    
    # Save the pipeline flattened data
    os.makedirs(os.path.dirname(pipeline_input_path), exist_ok=True)
    with open(pipeline_input_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_entries, f, indent=2)
    logger.info(f"Saved flattened chunks to pipeline input: {pipeline_input_path}")
    
    return original_chunks_path, pipeline_input_path
