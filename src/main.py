# File: /Users/amankumarshrestha/Downloads/Example/src/main.py

import gc

# Clear any existing caches
gc.collect()

import logging
import os
from typing import List, Dict, Any
from utils.logger import setup_logging
from contextual_vector_db import ContextualVectorDB
from elasticsearch_bm25 import ElasticsearchBM25  # Import ElasticsearchBM25 class

from data_loader import load_codebase_chunks, load_queries
from query_processor import validate_queries, process_queries
from evaluation import evaluate_complete_pipeline
from metrics import comprehensive_metric
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
import dspy
from dspy.datasets import DataLoader
from answer_generator import generate_answer_dspy, QuestionAnswerSignature
from reranking import retrieve_with_reranking
import answer_generator
import asyncio
from src.decorators import handle_exceptions

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def create_elasticsearch_bm25_index(db: ContextualVectorDB) -> ElasticsearchBM25:
    """
    Create and index documents in Elasticsearch BM25.

    Args:
        db (ContextualVectorDB): Contextual vector database instance.

    Returns:
        ElasticsearchBM25: Initialized ElasticsearchBM25 instance.
    """
    logger.debug("Entering create_elasticsearch_bm25_index method.")
    try:
        es_bm25 = ElasticsearchBM25()
        logger.info("ElasticsearchBM25 instance created.")
        indexed_count = es_bm25.index_documents(db.metadata)
        logger.info(f"Elasticsearch BM25 index created successfully with {indexed_count} documents.")
    except Exception as e:
        logger.error(f"Error creating Elasticsearch BM25 index: {e}", exc_info=True)
        raise
    return es_bm25

@handle_exceptions
async def main():
    """
    Main function to load data, process queries, and generate outputs.
    """
    logger.debug("Entering main function.")
    try:
        # Configure DSPy Language Model using LiteLLM
        logger.info("Configuring DSPy Language Model with LiteLLM")

        lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
        dspy.configure(lm=lm)

        # Define file paths
        codebase_chunks_file = 'data/codebase_chunks.json'  # Ensure this file exists
        queries_file = 'data/queries.json'                  # Ensure this file exists
        evaluation_set_file = 'data/evaluation_set.jsonl'   # Ensure this file exists
        output_filename = 'query_results.json'              # Modified to use a single output file

        dl = DataLoader()
        # Initialize the DataLoader

        # Load the training data from CSV
        logger.info(f"Loading training data from 'training_data.csv'")
        train_dataset = dl.from_csv(
            "data/training_data.csv",
            fields=("question", "context", "answer"),
            input_keys=("question", "context")
        )

        # Load the codebase chunks
        logger.info(f"Loading codebase chunks from '{codebase_chunks_file}'")
        codebase_chunks = load_codebase_chunks(codebase_chunks_file)

        # Initialize the ContextualVectorDB
        logger.info("Initializing ContextualVectorDB")
        contextual_db = ContextualVectorDB("contextual_db")

        # Load and process the data with parallel threads
        try:
            logger.info("Loading data into ContextualVectorDB with parallel threads")
            contextual_db.load_data(codebase_chunks, parallel_threads=5)
        except Exception as e:
            logger.error(f"Error loading data into ContextualVectorDB: {e}", exc_info=True)
            return

        # Create the Elasticsearch BM25 index
        try:
            logger.info("Creating Elasticsearch BM25 index")
            es_bm25 = create_elasticsearch_bm25_index(contextual_db)
        except Exception as e:
            logger.error(f"Error creating Elasticsearch BM25 index: {e}", exc_info=True)
            return

        # Load the queries
        logger.info(f"Loading queries from '{queries_file}'")
        queries = load_queries(queries_file)

        if not queries:
            logger.error("No queries found to process.")
            return

        # Validate queries
        logger.info("Validating input queries")
        validated_queries = validate_queries(queries)

        if not validated_queries:
            logger.error("No valid queries to process after validation. Exiting.")
            return

        # Load the evaluation set
        logger.info(f"Loading evaluation set from '{evaluation_set_file}'")
        evaluation_set = load_queries(evaluation_set_file)

        if not evaluation_set:
            logger.error("No evaluation queries found. Ensure the evaluation set file is correctly formatted and exists.")
            return

        # Define k value (number of top documents/chunks to retrieve)
        k = 20  # Ensure k is set correctly and not overridden elsewhere

        # Initialize 'qa_module' to None before the try-except block
        qa_module = None

        # Attempt to load the optimized program
        logger.info("Attempting to load optimized DSPy program")
        try:
            qa_module = dspy.Program.load("optimized_program.json")
            logger.info("Optimized DSPy program loaded successfully.")
        except Exception as e:
            # If loading fails, use the unoptimized module
            logger.warning("Failed to load optimized program. Using unoptimized module.")
            try:
                qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)
                logger.info("Unoptimized DSPy module initialized successfully.")
            except Exception as inner_e:
                logger.error(f"Error initializing unoptimized DSPy module: {inner_e}", exc_info=True)
                raise

        # Initialize DSPy Optimizer with Comprehensive Metric
        logger.info("Initializing DSPy Optimizer with Comprehensive Metric")
        try:
            optimizer_config = {
                'max_bootstrapped_demos': 4,       # Number of generated demonstrations
                'max_labeled_demos': 4,            # Number of labeled demonstrations from trainset
                'num_candidate_programs': 10,      # Number of candidate programs to evaluate
                'num_threads': 4                    # Number of parallel threads
            }
            teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,  # Use the comprehensive metric
                **optimizer_config
            )
            logger.info("DSPy Optimizer initialized successfully with Comprehensive Metric.")
        except Exception as e:
            logger.error(f"Error initializing DSPy Optimizer: {e}", exc_info=True)
            return

        # Compile the program using the optimizer
        logger.info("Compiling the program using the optimizer")
        try:
            optimized_program = teleprompter.compile(
                student=qa_module,
                teacher=qa_module,  # Assuming teacher is the same as student; adjust as needed
                trainset=train_dataset
            )
            logger.info("Program compiled and optimized successfully.")
        except Exception as e:
            logger.error(f"Error during program compilation: {e}", exc_info=True)
            return

        # Save the optimized program
        try:
            optimized_program.save("optimized_program.json")
            logger.info("Optimized program saved to 'optimized_program.json'")
        except Exception as e:
            logger.error(f"Error saving optimized program: {e}", exc_info=True)

        # Assign the optimized program to answer_generator.qa_module
        try:
            answer_generator.qa_module = optimized_program
            logger.info("Assigned optimized_program to answer_generator.qa_module successfully.")
        except Exception as e:
            logger.error(f"Error assigning optimized program to answer_generator.qa_module: {e}", exc_info=True)
            return

        # Proceed with processing queries using the optimized program
        logger.info("Starting to process queries with the optimized program")
        try:
            await process_queries(
                validated_queries,
                contextual_db,
                es_bm25,
                k,
                output_filename  # Modified to pass only one output file
            )
        except Exception as e:
            logger.error(f"Error processing queries: {e}", exc_info=True)
            return

        # Define k values for evaluation
        k_values = [5, 10, 20]

        # Perform evaluation
        logger.info("Starting evaluation of the retrieval pipeline")
        try:
            evaluate_complete_pipeline(
                contextual_db,
                es_bm25,
                k_values,
                evaluation_set,
                retrieve_with_reranking  # Pass the correct retrieval function
            )
            logger.info("Evaluation completed successfully.")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)

        logger.info("All operations completed successfully.")
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
