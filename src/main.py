# File: main.py
import gc
import logging
import os
from typing import List, Dict, Any
import asyncio
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.datasets import DataLoader
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import threading

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
from src.analysis.select_quotation_module_alt import SelectQuotationModuleAlt
from src.analysis.select_keyword_module import SelectKeywordModule  # Import the keyword module
from src.decorators import handle_exceptions

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Configure DSPy settings
dspy.settings.configure(main_thread_only=True)

# Introduce a thread lock mechanism
thread_lock = threading.Lock()

class ThematicAnalysisPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.contextual_db = None
        self.es_bm25 = None
        self.qa_module = None
        self.teleprompter = None
        self.optimized_program = None
        self.quotation_module = None
        self.quotation_module_alt = None
        self.keyword_module = None  # Initialize the keyword module

    def create_elasticsearch_bm25_index(self) -> ElasticsearchBM25:
        """
        Create and index documents in Elasticsearch BM25.
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
        Main function to load data, process queries, and generate outputs.
        """
        logger.debug("Entering main function.")
        try:
            # Configure DSPy Language Model
            logger.info("Configuring DSPy Language Model")

            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)

            # Define file paths from config
            codebase_chunks_file = self.config['codebase_chunks_file']
            queries_file_standard = self.config['queries_file_standard']
            queries_file_alt = self.config['queries_file_alt']
            queries_file_keyword = self.config['queries_file_keyword']  # Add keyword queries
            evaluation_set_file = self.config['evaluation_set_file']
            output_filename_primary = self.config['output_filename_primary']
            output_filename_alt = self.config['output_filename_alt']
            output_filename_keyword = self.config['output_filename_keyword']  # Add keyword output

            dl = DataLoader()

            # Load the training data
            logger.info(f"Loading training data from 'data/new_training_data.csv'")
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

            # Load and process the data
            try:
                logger.info("Loading data into ContextualVectorDB")
                self.contextual_db.load_data(codebase_chunks, parallel_threads=1)  # Reduced to single thread
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

            # Load queries
            logger.info(f"Loading standard queries from '{queries_file_standard}'")
            standard_queries = load_queries(queries_file_standard)

            logger.info(f"Loading alternative queries from '{queries_file_alt}'")
            alternative_queries = load_queries(queries_file_alt)

            logger.info(f"Loading keyword queries from '{queries_file_keyword}'")
            keyword_queries = load_queries(queries_file_keyword)

            if not standard_queries:
                logger.error("No standard queries found to process.")
            if not alternative_queries:
                logger.error("No alternative queries found to process.")
            if not keyword_queries:
                logger.error("No keyword queries found to process.")

            # Validate queries
            logger.info("Validating standard queries")
            validated_standard_queries = validate_queries(standard_queries)

            logger.info("Validating alternative queries")
            validated_alternative_queries = validate_queries(alternative_queries)

            logger.info("Validating keyword queries")
            validated_keyword_queries = validate_queries(keyword_queries)

            # Define k value
            k = 20

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
                    'max_bootstrapped_demos': 4,
                    'max_labeled_demos': 4,
                    'num_candidate_programs': 10,
                    'num_threads': 1  # Reduced to single thread
                }
                self.teleprompter = BootstrapFewShotWithRandomSearch(
                    metric=comprehensive_metric,
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
                    teacher=self.qa_module,
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

            # Activate assertions in the optimized program
            logger.info("Activating assertions in the optimized program.")
            try:
                self.optimized_program = assert_transform_module(self.optimized_program, backtrack_handler)
                logger.info("Assertions activated in the optimized program.")
            except Exception as e:
                logger.error(f"Error activating assertions in optimized program: {e}", exc_info=True)
                return

            # Initialize SelectQuotationModule with activated assertions
            logger.info("Initializing SelectQuotationModule with activated assertions")
            try:
                self.quotation_module = SelectQuotationModule()
                self.quotation_module = assert_transform_module(self.quotation_module, backtrack_handler)
                logger.info("SelectQuotationModule initialized successfully with assertions activated.")
            except Exception as e:
                logger.error(f"Error initializing SelectQuotationModule: {e}", exc_info=True)
                return

            # Initialize SelectQuotationModuleAlt with activated assertions
            logger.info("Initializing SelectQuotationModuleAlt with activated assertions")
            try:
                self.quotation_module_alt = SelectQuotationModuleAlt()
                self.quotation_module_alt = assert_transform_module(self.quotation_module_alt, backtrack_handler)
                logger.info("SelectQuotationModuleAlt initialized successfully with assertions activated.")
            except Exception as e:
                logger.error(f"Error initializing SelectQuotationModuleAlt: {e}", exc_info=True)
                return

            # Initialize SelectKeywordModule without assertions (since it's only keyword extraction)
            logger.info("Initializing SelectKeywordModule")
            try:
                self.keyword_module = SelectKeywordModule()
                logger.info("SelectKeywordModule initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing SelectKeywordModule: {e}", exc_info=True)
                return

            # Process standard queries with SelectQuotationModule
            logger.info("Processing standard queries with SelectQuotationModule")
            await process_queries(
                validated_standard_queries,
                self.contextual_db,
                self.es_bm25,
                k,
                output_filename_primary,
                self.optimized_program,
                self.quotation_module
            )

            # Process alternative queries with SelectQuotationModuleAlt
            logger.info("Processing alternative queries with SelectQuotationModuleAlt")
            await process_queries(
                validated_alternative_queries,
                self.contextual_db,
                self.es_bm25,
                k,
                output_filename_alt,
                self.optimized_program,
                self.quotation_module_alt
            )

            # Process keyword queries with SelectKeywordModule
            logger.info("Processing keyword queries with SelectKeywordModule")
            await process_queries(
                validated_keyword_queries,
                self.contextual_db,
                self.es_bm25,
                k,
                output_filename_keyword,
                self.optimized_program,
                self.keyword_module,
                is_keyword_extraction=True  # Indicate that we're only extracting keywords
            )

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
                    evaluation_set=load_queries(evaluation_set_file)
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
        'queries_file_standard': 'data/queries.json',
        'queries_file_alt': 'data/queries_alt.json',
        'queries_file_keyword': 'data/queries_keyword.json',
        'evaluation_set_file': 'data/evaluation_set.jsonl',
        'output_filename_primary': 'query_results_primary.json',
        'output_filename_alt': 'query_results_alternative.json',
        'output_filename_keyword': 'query_results_keyword.json'  # Add output file for keyword extraction
    }
    pipeline = ThematicAnalysisPipeline(config)
    asyncio.run(pipeline.run_pipeline())