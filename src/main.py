# main.py

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
        self.contextual_db = None
        self.es_bm25 = None
        self.optimized_coding_program = None  # Added for Coding Analysis

    def create_elasticsearch_bm25_index(self, index_name: str) -> ElasticsearchBM25:
        """
        Create and index documents in Elasticsearch BM25.
        """
        logger.debug("Entering create_elasticsearch_bm25_index method.")
        try:
            es_bm25 = ElasticsearchBM25(index_name=index_name)
            logger.info(f"ElasticsearchBM25 instance created with index '{index_name}'.")
            success_count, failed_docs = es_bm25.index_documents(self.contextual_db.metadata)
            logger.info(f"Elasticsearch BM25 index '{index_name}' created successfully with {success_count} documents indexed.")
            if failed_docs:
                logger.warning(f"{len(failed_docs)} documents failed to index.")
        except Exception as e:
            logger.error(f"Error creating Elasticsearch BM25 index '{index_name}': {e}", exc_info=True)
            raise
        return es_bm25

    async def initialize_quotation_optimizer(self, config):
        """
        Initialize the quotation selection optimizer.
        """
        logger.info("Initializing quotation selection optimizer")
        dl = DataLoader()
        quotation_train_dataset = dl.from_csv(
            config['quotation_training_data'],
            fields=("input", "output"),
            input_keys=("input",)
        )

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
        logger.info("Quotation optimizer initialized successfully")

    async def initialize_keyword_optimizer(self, config):
        """
        Initialize the keyword extraction optimizer.
        """
        logger.info("Initializing keyword extraction optimizer")
        dl = DataLoader()
        keyword_train_dataset = dl.from_csv(
            config['keyword_training_data'],
            fields=("input", "output"),
            input_keys=("input",)
        )

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
        logger.info("Keyword extraction optimizer initialized successfully")

    async def initialize_coding_optimizer(self, config):
        """
        Initialize the coding analysis optimizer.
        """
        logger.info("Initializing coding analysis optimizer")
        dl = DataLoader()
        coding_train_dataset = dl.from_csv(
            config['coding_training_data'],
            fields=("input", "output"),
            input_keys=("input",)
        )

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
        logger.info("Coding analysis optimizer initialized successfully")

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
            logger.info("Configuring DSPy Language Model")
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)
            dspy.Cache = False

            # Define file paths from config
            codebase_chunks_file = config['codebase_chunks_file']
            queries_file_standard = config['queries_file_standard']
            evaluation_set_file = config['evaluation_set_file']
            output_filename_primary = config['output_filename_primary']
            training_data_file = config.get(f"{module_class.__name__.lower().replace('module', '')}_training_data", None)
            optimized_program_file = config.get(f"optimized_{module_class.__name__.lower().replace('module', '')}_program", None)

            dl = DataLoader()

            # Load the codebase chunks
            logger.info(f"Loading codebase chunks from '{codebase_chunks_file}'")
            codebase_chunks = load_codebase_chunks(codebase_chunks_file)

            # Initialize the ContextualVectorDB
            logger.info("Initializing ContextualVectorDB")
            self.contextual_db = ContextualVectorDB("contextual_db")

            # Load and process the data
            try:
                logger.info("Loading data into ContextualVectorDB")
                self.contextual_db.load_data(codebase_chunks, parallel_threads=1)
            except Exception as e:
                logger.error(f"Error loading data into ContextualVectorDB: {e}", exc_info=True)
                return

            # Create the Elasticsearch BM25 index
            try:
                logger.info(f"Creating Elasticsearch BM25 index '{config['index_name']}'")
                self.es_bm25 = self.create_elasticsearch_bm25_index(config['index_name'])
            except Exception as e:
                logger.error(f"Error creating Elasticsearch BM25 index: {e}", exc_info=True)
                return

            # Load queries
            logger.info(f"Loading standard queries from '{queries_file_standard}'")
            standard_queries = load_queries(queries_file_standard)

            if not standard_queries:
                logger.error("No standard queries found to process.")

            # Validate queries
            logger.info("Validating standard queries")
            validated_standard_queries = validate_queries(standard_queries, module_class())

            # Initialize optimizer (Quotation, Keyword Extraction, or Coding Analysis)
            await optimizer_init_func(config)

            # Initialize the module with assertions
            try:
                logger.info(f"Initializing {module_class.__name__}")
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

            # Determine the optimized program based on module type
            if 'quotation' in module_class.__name__.lower():
                optimized_program = self.optimized_quotation_program
            elif 'keyword' in module_class.__name__.lower():
                optimized_program = self.optimized_keyword_program
            elif 'coding' in module_class.__name__.lower():
                optimized_program = self.optimized_coding_program
            else:
                logger.error("No optimized program found for the given module type.")
                return

            # Process queries with the module
            logger.info(f"Processing queries with {module_class.__name__}")
            await process_queries(
                validated_standard_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_standard,
                output_file=config['output_filename_primary'],
                optimized_program=optimized_program,
                module=module_instance
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
                evaluation_set = load_queries(config['evaluation_set_file'])
                evaluator.evaluate_complete_pipeline(
                    k_values=k_values,
                    evaluation_set=evaluation_set
                )
                logger.info("Evaluation completed successfully.")
            except Exception as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)

            logger.info(f"Pipeline for {module_class.__name__} completed successfully.")
        except Exception as e:
            logger.error(f"Unexpected error in run_pipeline_with_config: {e}", exc_info=True)

    async def run_pipeline(self):
        # Define configurations for each pipeline
        config_standard_quotation = {
            'index_name': 'contextual_bm25_index_standard_quotation',
            'codebase_chunks_file': 'data/codebase_chunks.json',
            'queries_file_standard': 'data/queries.json',
            'evaluation_set_file': 'data/evaluation_set.jsonl',
            'output_filename_primary': 'query_results_quotation.json',
            'quotation_training_data': 'data/quotation_training_data.csv',
            'optimized_quotation_program': 'optimized_quotation_program.json'
        }
        config_keyword_extraction = {
            'index_name': 'contextual_bm25_index_keyword_extraction',
            'codebase_chunks_file': 'data/codebase_chunks_keyword.json',
            'queries_file_standard': 'data/input/queries_keyword.json',  # Will be updated after conversion
            'evaluation_set_file': 'data/evaluation_set_keyword.jsonl',
            'output_filename_primary': 'query_results_keyword_extraction.json',
            'keyword_training_data': 'data/keyword_training_data.csv',
            'optimized_keyword_program': 'optimized_keyword_program.json'
        }
        config_coding_analysis = {  # Added configuration for Coding Analysis pipeline
            'index_name': 'contextual_bm25_index_coding_analysis',
            'codebase_chunks_file': 'data/codebase_chunks_coding.json',
            'queries_file_standard': 'data/input/queries_coding.json',  # Will be updated after conversion
            'evaluation_set_file': 'data/evaluation_set_coding.jsonl',
            'output_filename_primary': 'query_results_coding_analysis.json',
            'coding_training_data': 'data/coding_training_data.csv',
            'optimized_coding_program': 'optimized_coding_program.json'
        }

        # Run Standard Quotation Extraction Pipeline
        logger.info("Starting Standard Quotation Extraction Pipeline")
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
                # Run the conversion in a separate thread to avoid blocking the event loop
                await asyncio.to_thread(func, **kwargs)
                logger.info(f"Conversion successful: {description}")
            except Exception as e:
                logger.error(f"Conversion failed ({description}): {e}")
                raise  # Optionally, re-raise the exception to halt the pipeline

        # Perform conversion for Standard Quotation Extraction
        await perform_conversion(
            convert_quotation_to_keyword,
            description="Quotation to Keyword Conversion (Standard)",
            input_file=config_standard_quotation['output_filename_primary'],
            output_dir='data',
            output_file='queries_keyword_standard.json'
        )

        # Run Keyword Extraction Pipeline for Standard Quotation
        logger.info("Starting Keyword Extraction Pipeline for Standard Quotation")
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
        logger.info("Starting Coding Analysis Pipeline for Standard Keyword Extraction")
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

        logger.info("All pipelines and conversion steps executed successfully up to Coding Analysis.")

if __name__ == "__main__":
    pipeline = ThematicAnalysisPipeline()
    asyncio.run(pipeline.run_pipeline())
