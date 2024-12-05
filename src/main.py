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
import time

from src.utils.logger import setup_logging
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.data.data_loader import load_codebase_chunks, load_queries
from src.processing.query_processor import process_queries, validate_queries
from src.evaluation.evaluation import PipelineEvaluator
from src.analysis.metrics import comprehensive_metric
from src.processing.answer_generator import generate_answer_dspy, QuestionAnswerSignature
from src.retrieval.reranking import retrieve_with_reranking
from src.analysis.select_quotation_module import EnhancedQuotationModule as EnhancedQuotationModuleStandard
# Removed Alternate Quotation Module import
from src.analysis.extract_keyword_module import KeywordExtractionModule  # Import KeywordExtractionModule
from src.analysis.coding_module import CodingAnalysisModule  # Import CodingAnalysisModule
from src.decorators import handle_exceptions

# Import the conversion functions from src.convert/
from src.convert.convertquotationforkeyword import convert_query_results as convert_quotation_to_keyword
from src.convert.convertkeywordforcoding import convert_query_results as convert_keyword_to_coding
from src.convert.convertcodingfortheme import process_input_file as convert_coding_to_theme

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Configure DSPy settings
dspy.settings.configure(main_thread_only=True)

# Introduce a thread lock mechanism
thread_lock = threading.Lock()


class ThematicAnalysisPipeline:
    def __init__(self):
        self.contextual_db = ContextualVectorDB("contextual_db")
        self.es_bm25 = None
        self.optimized_coding_program = None  # Added for Coding Analysis
        logger.debug("ThematicAnalysisPipeline instance created with ContextualVectorDB initialized.")

    def create_elasticsearch_bm25_index(self, index_name: str) -> ElasticsearchBM25:
        """
        Create and index documents in Elasticsearch BM25.
        """
        logger.debug("Entering create_elasticsearch_bm25_index method.")
        try:
            es_bm25 = ElasticsearchBM25(index_name=index_name)
            logger.info(f"ElasticsearchBM25 instance created with index '{index_name}'.")
            start_time = time.time()
            success_count, failed_docs = es_bm25.index_documents(self.contextual_db.metadata)
            elapsed_time = time.time() - start_time
            logger.info(
                f"Elasticsearch BM25 index '{index_name}' created successfully with {success_count} documents indexed "
                f"in {elapsed_time:.2f} seconds."
            )
            if failed_docs:
                logger.warning(f"{len(failed_docs)} documents failed to index.")
                for doc in failed_docs:
                    logger.warning(f"Failed to index document ID: {doc.get('doc_id', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error creating Elasticsearch BM25 index '{index_name}': {e}", exc_info=True)
            raise  # Re-raise the exception after logging
        finally:
            logger.debug("Exiting create_elasticsearch_bm25_index method.")
        return es_bm25

    async def initialize_quotation_optimizer(self, config):
        """
        Initialize the quotation selection optimizer.
        """
        logger.info("Initializing quotation selection optimizer.")
        start_time = time.time()
        try:
            dl = DataLoader()
            quotation_train_dataset = dl.from_csv(
                config['quotation_training_data'],
                fields=("input", "output"),
                input_keys=("input",)
            )
            logger.debug(f"Quotation training dataset loaded with {len(quotation_train_dataset)} samples.")

            self.quotation_qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)

            optimizer_config = {
                'max_bootstrapped_demos': 4,
                'max_labeled_demos': 4,
                'num_candidate_programs': 10,
                'num_threads': 1
            }

            self.quotation_teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,
                **optimizer_config
            )

            self.optimized_quotation_program = self.quotation_teleprompter.compile(
                student=self.quotation_qa_module,
                teacher=self.quotation_qa_module,
                trainset=quotation_train_dataset
            )

            self.optimized_quotation_program.save(config['optimized_quotation_program'])
            elapsed_time = time.time() - start_time
            logger.info(f"Quotation optimizer initialized successfully in {elapsed_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error initializing quotation optimizer: {e}", exc_info=True)
            raise  # Re-raise the exception after logging

    async def initialize_keyword_optimizer(self, config):
        """
        Initialize the keyword extraction optimizer.
        """
        logger.info("Initializing keyword extraction optimizer.")
        start_time = time.time()
        try:
            dl = DataLoader()
            keyword_train_dataset = dl.from_csv(
                config['keyword_training_data'],
                fields=("input", "output"),
                input_keys=("input",)
            )
            logger.debug(f"Keyword training dataset loaded with {len(keyword_train_dataset)} samples.")

            self.keyword_qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)

            optimizer_config = {
                'max_bootstrapped_demos': 4,
                'max_labeled_demos': 4,
                'num_candidate_programs': 10,
                'num_threads': 1
            }

            self.keyword_teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,
                **optimizer_config
            )

            self.optimized_keyword_program = self.keyword_teleprompter.compile(
                student=self.keyword_qa_module,
                teacher=self.keyword_qa_module,
                trainset=keyword_train_dataset
            )

            self.optimized_keyword_program.save(config['optimized_keyword_program'])
            elapsed_time = time.time() - start_time
            logger.info(f"Keyword extraction optimizer initialized successfully in {elapsed_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error initializing keyword extraction optimizer: {e}", exc_info=True)
            raise  # Re-raise the exception after logging

    async def initialize_coding_optimizer(self, config):
        """
        Initialize the coding analysis optimizer.
        """
        logger.info("Initializing coding analysis optimizer.")
        start_time = time.time()
        try:
            dl = DataLoader()
            coding_train_dataset = dl.from_csv(
                config['coding_training_data'],
                fields=("input", "output"),
                input_keys=("input",)
            )
            logger.debug(f"Coding training dataset loaded with {len(coding_train_dataset)} samples.")

            self.coding_qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)

            optimizer_config = {
                'max_bootstrapped_demos': 4,
                'max_labeled_demos': 4,
                'num_candidate_programs': 10,
                'num_threads': 1
            }

            self.coding_teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,
                **optimizer_config
            )

            self.optimized_coding_program = self.coding_teleprompter.compile(
                student=self.coding_qa_module,
                teacher=self.coding_qa_module,
                trainset=coding_train_dataset
            )

            self.optimized_coding_program.save(config['optimized_coding_program'])
            elapsed_time = time.time() - start_time
            logger.info(f"Coding analysis optimizer initialized successfully in {elapsed_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error initializing coding analysis optimizer: {e}", exc_info=True)
            raise  # Re-raise the exception after logging

    @handle_exceptions
    async def run_pipeline_with_config(self, config, module_class, optimizer_init_func):
        """
        Main function to load data, process queries, and generate outputs.
        :param config: Configuration dictionary for the pipeline.
        :param module_class: The DSPy module class to use (Quotation, Keyword Extraction, or Coding Analysis).
        :param optimizer_init_func: Function to initialize the optimizer (Quotation, Keyword Extraction, or Coding Analysis).
        """
        logger.debug("Entering run_pipeline_with_config method.")
        try:
            # Configure DSPy Language Model
            logger.info("Configuring DSPy Language Model.")
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)
            dspy.Cache = False
            logger.debug("DSPy Language Model configured.")

            # Define file paths from config
            codebase_chunks_file = config['codebase_chunks_file']
            queries_file_standard = config['queries_file_standard']
            evaluation_set_file = config['evaluation_set_file']
            output_filename_primary = config['output_filename_primary']
            training_data_file = config.get(f"{module_class.__name__.lower().replace('module', '')}_training_data", None)
            optimized_program_file = config.get(f"optimized_{module_class.__name__.lower().replace('module', '')}_program", None)

            dl = DataLoader()

            # Load the codebase chunks
            logger.info(f"Loading codebase chunks from '{codebase_chunks_file}'.")
            start_time = time.time()
            codebase_chunks = load_codebase_chunks(codebase_chunks_file)
            elapsed_time = time.time() - start_time
            logger.info(f"Loaded {len(codebase_chunks)} documents in {elapsed_time:.2f} seconds.")

            # Load and process the data
            try:
                logger.info("Loading data into ContextualVectorDB.")
                start_time = time.time()
                self.contextual_db.load_data(codebase_chunks, parallel_threads=4)
                elapsed_time = time.time() - start_time
                logger.info(f"Data loaded into ContextualVectorDB in {elapsed_time:.2f} seconds.")
                logger.info(f"Total embeddings: {len(self.contextual_db.embeddings)}.")
                logger.info(f"Total metadata entries: {len(self.contextual_db.metadata)}.")
            except Exception as e:
                logger.error(f"Error loading data into ContextualVectorDB: {e}", exc_info=True)
                return

            # Create the Elasticsearch BM25 index
            try:
                logger.info(f"Creating Elasticsearch BM25 index '{config['index_name']}'.")
                start_time = time.time()
                self.es_bm25 = self.create_elasticsearch_bm25_index(config['index_name'])
                elapsed_time = time.time() - start_time
                logger.info(f"Elasticsearch BM25 index '{config['index_name']}' created in {elapsed_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Error creating Elasticsearch BM25 index: {e}", exc_info=True)
                return

            # Load queries
            logger.info(f"Loading standard queries from '{queries_file_standard}'.")
            start_time = time.time()
            standard_queries = load_queries(queries_file_standard)
            elapsed_time = time.time() - start_time
            logger.info(f"Loaded {len(standard_queries)} standard queries in {elapsed_time:.2f} seconds.")

            if not standard_queries:
                logger.error("No standard queries found to process.")
                return

            # Validate queries
            logger.info("Validating standard queries.")
            start_time = time.time()
            validated_standard_queries = validate_queries(standard_queries, module_class())
            elapsed_time = time.time() - start_time
            logger.info(f"Validated {len(validated_standard_queries)} queries in {elapsed_time:.2f} seconds.")

            # Initialize optimizer (Quotation, Keyword Extraction, or Coding Analysis)
            logger.info(f"Initializing optimizer for {module_class.__name__}.")
            start_time = time.time()
            await optimizer_init_func(config)
            elapsed_time = time.time() - start_time
            logger.info(f"Optimizer for {module_class.__name__} initialized in {elapsed_time:.2f} seconds.")

            # Initialize the module with assertions
            try:
                logger.info(f"Initializing {module_class.__name__}.")
                module_instance = module_class()
                module_instance = assert_transform_module(
                    module_instance,
                    backtrack_handler
                )
                logger.info(f"{module_class.__name__} initialized successfully with assertions activated.")
            except Exception as e:
                logger.error(f"Error initializing {module_class.__name__}: {e}", exc_info=True)
                return

            # Define k value for queries
            k_standard = 20
            logger.debug(f"Set k_standard to {k_standard} for query processing.")

            # Determine the optimized program based on module type
            if 'quotation' in module_class.__name__.lower():
                optimized_program = self.optimized_quotation_program
                logger.debug("Using optimized quotation program.")
            elif 'keyword' in module_class.__name__.lower():
                optimized_program = self.optimized_keyword_program
                logger.debug("Using optimized keyword extraction program.")
            elif 'coding' in module_class.__name__.lower():
                optimized_program = self.optimized_coding_program
                logger.debug("Using optimized coding analysis program.")
            else:
                logger.error("No optimized program found for the given module type.")
                return

            # Process queries with the module
            logger.info(f"Processing queries with {module_class.__name__}.")
            start_time = time.time()
            await process_queries(
                validated_standard_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_standard,
                output_file=config['output_filename_primary'],
                optimized_program=optimized_program,
                module=module_instance
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Processed queries with {module_class.__name__} in {elapsed_time:.2f} seconds.")

            # Define k values for evaluation
            k_values = [5, 10, 20]
            logger.debug(f"Set k_values for evaluation: {k_values}.")

            # Initialize the evaluator
            logger.info("Starting evaluation of the retrieval pipeline.")
            try:
                evaluator = PipelineEvaluator(
                    db=self.contextual_db,
                    es_bm25=self.es_bm25,
                    retrieval_function=retrieve_with_reranking
                )
                logger.debug("PipelineEvaluator instance created.")

                # Perform evaluation
                logger.info(f"Loading evaluation set from '{config['evaluation_set_file']}'.")
                start_time = time.time()
                evaluation_set = load_queries(config['evaluation_set_file'])
                logger.info(f"Loaded {len(evaluation_set)} evaluation queries in {time.time() - start_time:.2f} seconds.")

                logger.info("Evaluating the complete retrieval pipeline.")
                start_time = time.time()
                evaluator.evaluate_complete_pipeline(
                    k_values=k_values,
                    evaluation_set=evaluation_set
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Evaluation completed successfully in {elapsed_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)

            logger.info(f"Pipeline for {module_class.__name__} completed successfully.")
            logger.debug("Exiting run_pipeline_with_config method.")
        except Exception as e:
            logger.error(f"Unexpected error in run_pipeline_with_config: {e}", exc_info=True)
            raise  # Re-raise the exception after logging

    async def run_pipeline(self):
        logger.info("Starting Thematic Analysis Pipeline.")
        start_time = time.time()
        # Define configurations for each pipeline
        config_standard_quotation = {
            'index_name': 'contextual_bm25_index_standard_quotation',
            'codebase_chunks_file': 'data/codebase_chunks/codebase_chunks.json',
            'queries_file_standard': 'data/input/queries_quotation.json',
            'evaluation_set_file': 'data/evaluation/evaluation_set_quotation.jsonl',
            'output_filename_primary': 'data/output/query_results_quotation.json',
            'quotation_training_data': 'data/training/quotation_training_data.csv',
            'optimized_quotation_program': 'data/optimized/optimized_quotation_program.json'
        }
        config_keyword_extraction = {
            'index_name': 'contextual_bm25_index_keyword_extraction',
            'codebase_chunks_file': 'data/codebase_chunks/codebase_chunks.json',
            'queries_file_standard': 'data/input/queries_keyword.json',  # Will be updated after conversion
            'evaluation_set_file': 'data/evaluation/evaluation_set_keyword.jsonl',
            'output_filename_primary': 'data/output/query_results_keyword_extraction.json',
            'keyword_training_data': 'data/training/keyword_training_data.csv',
            'optimized_keyword_program': 'data/optimized/optimized_keyword_program.json'
        }
        config_coding_analysis = {  # Added configuration for Coding Analysis pipeline
            'index_name': 'contextual_bm25_index_coding_analysis',
            'codebase_chunks_file': 'data/codebase_chunks/codebase_chunks.json',
            'queries_file_standard': 'data/input/queries_coding.json',  # Will be updated after conversion
            'evaluation_set_file': 'data/evaluation/evaluation_set_coding.jsonl',
            'output_filename_primary': 'data/output/query_results_coding_analysis.json',
            'coding_training_data': 'data/training/coding_training_data.csv',
            'optimized_coding_program': 'data/optimized/optimized_coding_program.json'
        }

        # Run Standard Quotation Extraction Pipeline
        logger.info("Starting Standard Quotation Extraction Pipeline.")
        await self.run_pipeline_with_config(
            config_standard_quotation,
            EnhancedQuotationModuleStandard,
            self.initialize_quotation_optimizer
        )

        # Removed Alternate Quotation Extraction Pipeline

        # After running the quotation extraction pipeline, perform the conversion

        # Define a helper function to handle the conversion asynchronously
        async def perform_conversion(func, description="", **kwargs):
            try:
                logger.info(f"Converting data: {description}")
                start_time = time.time()
                # Run the conversion in a separate thread to avoid blocking the event loop
                await asyncio.to_thread(func, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"Conversion successful: {description} in {elapsed_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Conversion failed ({description}): {e}", exc_info=True)
                raise  # Re-raise the exception to halt the pipeline

        # Perform conversion for Standard Quotation Extraction
        await perform_conversion(
            convert_quotation_to_keyword,
            description="Quotation to Keyword Conversion (Standard)",
            input_file=config_standard_quotation['output_filename_primary'],
            output_dir='data',
            output_file='queries_keyword_standard.json'
        )

        # Run Keyword Extraction Pipeline for Standard Quotation
        logger.info("Starting Keyword Extraction Pipeline for Standard Quotation.")
        config_keyword_extraction_standard = config_keyword_extraction.copy()
        config_keyword_extraction_standard['queries_file_standard'] = 'data/input/queries_keyword_standard.json'
        await self.run_pipeline_with_config(
            config_keyword_extraction_standard,
            KeywordExtractionModule,
            self.initialize_keyword_optimizer
        )

        # After Keyword Extraction, perform conversion to Coding
        await perform_conversion(
            convert_keyword_to_coding,
            description="Keyword to Coding Conversion (Standard)",
            input_file=config_keyword_extraction_standard['output_filename_primary'],
            output_dir='data',
            output_file='queries_coding_standard.json'
        )

        # Run Coding Analysis Pipeline for Standard Keyword Extraction
        logger.info("Starting Coding Analysis Pipeline for Standard Keyword Extraction.")
        config_coding_analysis_standard = config_coding_analysis.copy()
        config_coding_analysis_standard['queries_file_standard'] = 'data/input/queries_coding_standard.json'
        await self.run_pipeline_with_config(
            config_coding_analysis_standard,
            CodingAnalysisModule,
            self.initialize_coding_optimizer
        )

        # After Coding Analysis, perform conversion to Theme (without ThemeAnalysisModule)
        await perform_conversion(
            convert_coding_to_theme,
            description="Coding to Theme Conversion (Standard)",
            input_file=config_coding_analysis_standard['output_filename_primary'],
            output_dir='data',
            output_file='queries_theme_standard.json'
        )

        # Since ThemeAnalysisModule is not available, we stop here

        elapsed_time = time.time() - start_time
        logger.info(f"All pipelines and conversion steps executed successfully up to Coding Analysis in {elapsed_time:.2f} seconds.")

    # Optionally, you can add more methods or helper functions here if needed.


if __name__ == "__main__":
    logger.info("Launching Thematic Analysis Pipeline.")
    pipeline = ThematicAnalysisPipeline()
    try:
        asyncio.run(pipeline.run_pipeline())
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        logger.info("Thematic Analysis Pipeline execution completed.")
