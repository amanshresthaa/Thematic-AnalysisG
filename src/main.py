import gc
import logging
import os
from typing import List, Dict, Any
import asyncio
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.datasets import DataLoader

from src.utils.logger import setup_logging
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.data.data_loader import load_codebase_chunks, load_queries
from src.processing.query_processor import validate_queries, process_queries
from src.evaluation.evaluation import PipelineEvaluator
from src.analysis.metrics import comprehensive_metric
from src.processing.answer_generator import generate_answer_dspy, QuestionAnswerSignature
from src.retrieval.reranking import retrieve_with_reranking
from src.analysis.select_quotation_module import SelectQuotationModule
from src.analysis.extract_keywords_module import KeywordExtractionModule
from src.decorators import handle_exceptions
from src.analysis.select_quotation import SelectQuotationSignature
from src.analysis.select_quotation_module import SelectQuotationModule
from src.processing.answer_generator import generate_answer_dspy, QuestionAnswerSignature
from src.processing.query_processor import validate_queries, process_queries

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class ThematicAnalysisPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.contextual_db = None
        self.es_bm25 = None
        self.qa_module = None
        self.teleprompter = None
        self.optimized_program = None
        self.quotation_module = None
        self.keyword_module = None  # Added KeywordExtractionModule

    def create_elasticsearch_bm25_index(self) -> ElasticsearchBM25:
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
            success_count, failed_docs = es_bm25.index_documents(self.contextual_db.metadata)
            logger.info(f"Elasticsearch BM25 index created successfully with {success_count} documents indexed.")
            if failed_docs:
                logger.warning(f"{len(failed_docs)} documents failed to index.")
        except Exception as e:
            logger.error(f"Error creating Elasticsearch BM25 index: {e}", exc_info=True)
            raise
        return es_bm25

    @handle_exceptions
    async def run_pipeline(self):
        """
        Main function to load data, process queries, extract keywords, and generate outputs.
        """
        logger.debug("Entering main function.")
        try:
            # Configure DSPy Language Model using LiteLLM
            logger.info("Configuring DSPy Language Model with LiteLLM")

            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)

            # Define file paths
            codebase_chunks_file = self.config['codebase_chunks_file']
            queries_file = self.config['queries_file']
            evaluation_set_file = self.config['evaluation_set_file']
            output_filename = self.config['output_filename']
            quotation_file = self.config['quotation_file']
            keywords_output_file = self.config['keywords_output_file']

            dl = DataLoader()
            # Initialize the DataLoader

            # Load the training data from the new CSV format
            logger.info(f"Loading training data from 'new_training_data.csv'")
            train_dataset = dl.from_csv(
                "data/new_training_data.csv",
                fields=("input", "output"),
                input_keys=("input",)
            )
            # Load the codebase chunks
            logger.info(f"Loading codebase chunks from '{codebase_chunks_file}'")
            codebase_chunks = load_codebase_chunks(codebase_chunks_file)

            # Initialize the ContextualVectorDB
            logger.info("Initializing ContextualVectorDB")
            self.contextual_db = ContextualVectorDB("contextual_db")

            # Load and process the data with parallel threads
            try:
                logger.info("Loading data into ContextualVectorDB with parallel threads")
                self.contextual_db.load_data(codebase_chunks, parallel_threads=5)
            except Exception as e:
                logger.error(f"Error loading data into ContextualVectorDB: {e}", exc_info=True)
                return

            # Create the Elasticsearch BM25 index
            try:
                logger.info("Creating Elasticsearch BM25 index")
                self.es_bm25 = self.create_elasticsearch_bm25_index()
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
            self.qa_module = None

            # Attempt to load the optimized program
            logger.info("Attempting to load optimized DSPy program")
            try:
                self.qa_module = dspy.Program.load("optimized_program.json")
                logger.info("Optimized DSPy program loaded successfully.")
            except Exception as e:
                # If loading fails, use the unoptimized module
                logger.warning("Failed to load optimized program. Using unoptimized module.")
                try:
                    self.qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)
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
                self.teleprompter = BootstrapFewShotWithRandomSearch(
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
                self.optimized_program = self.teleprompter.compile(
                    student=self.qa_module,
                    teacher=self.qa_module,  # Assuming teacher is the same as student; adjust as needed
                    trainset=train_dataset
                )
                logger.info("Program compiled and optimized successfully.")
            except Exception as e:
                logger.error(f"Error during program compilation: {e}", exc_info=True)
                return

            # Save the optimized program
            try:
                self.optimized_program.save("optimized_program.json")
                logger.info("Optimized program saved to 'optimized_program.json'")
            except Exception as e:
                logger.error(f"Error saving optimized program: {e}", exc_info=True)

            # Assign the optimized program to answer_generator.qa_module
            try:
                # The optimized program will be used directly in process_queries
                logger.info("Optimized program ready for query processing.")
            except Exception as e:
                logger.error(f"Error preparing optimized program: {e}", exc_info=True)
                return

            # Initialize SelectQuotationModule
            logger.info("Initializing SelectQuotationModule")
            try:
                self.quotation_module = SelectQuotationModule()
                logger.info("SelectQuotationModule initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing SelectQuotationModule: {e}", exc_info=True)
                return

            # Initialize KeywordExtractionModule
            logger.info("Initializing KeywordExtractionModule")
            try:
                self.keyword_module = KeywordExtractionModule(
                    input_file=quotation_file,
                    output_file=keywords_output_file
                )
                logger.info("KeywordExtractionModule initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing KeywordExtractionModule: {e}", exc_info=True)
                return

            # Proceed with processing queries using the optimized program and quotation selection
            logger.info("Starting to process queries with the optimized program and quotation selection")
            try:
                await process_queries(
                    validated_queries,
                    self.contextual_db,
                    self.es_bm25,
                    k,
                    output_filename  # Modified to pass only one output file
                )
            except Exception as e:
                logger.error(f"Error processing queries: {e}", exc_info=True)
                return

            # Extract Keywords from Quotations
            logger.info("Starting keyword extraction from quotations")
            try:
                keywords_result = self.keyword_module.process_file(
                    input_file=quotation_file,
                    research_objectives="Analyze potato farming and supply chain"
                )
                if keywords_result.get("keywords"):
                    logger.info(f"Keywords extracted successfully and saved to {keywords_result.get('output_file')}")
                else:
                    logger.warning("No keywords were extracted.")
            except Exception as e:
                logger.error(f"Error during keyword extraction: {e}", exc_info=True)
                return

            # Define k values for evaluation
            k_values = [5, 10, 20]

            # Initialize the evaluator
            logger.info("Starting evaluation of the retrieval pipeline")
            try:
                evaluator = PipelineEvaluator(
                    db=self.contextual_db,
                    es_bm25=self.es_bm25,
                    retrieval_function=retrieve_with_reranking
                )
                
                # Perform evaluation
                evaluator.evaluate_complete_pipeline(
                    k_values=k_values,
                    evaluation_set=evaluation_set
                )
                logger.info("Evaluation completed successfully.")
            except Exception as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)

            logger.info("All operations completed successfully.")
        except Exception as e:
            logger.error(f"Unexpected error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    config = {
        'codebase_chunks_file': 'data/codebase_chunks.json',
        'queries_file': 'data/queries.json',
        'evaluation_set_file': 'data/evaluation_set.jsonl',
        'output_filename': 'query_results.json',
        'quotation_file': 'data/quotation.json',
        'keywords_output_file': 'data/keywords.json'
    }
    pipeline = ThematicAnalysisPipeline(config)
    asyncio.run(pipeline.run_pipeline())
