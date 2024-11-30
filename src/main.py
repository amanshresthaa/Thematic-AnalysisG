# src/main.py

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
from src.analysis.select_quotation_module_alt import EnhancedQuotationModule as EnhancedQuotationModuleAlt
from src.decorators import handle_exceptions

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

    @handle_exceptions
    async def run_pipeline_with_config(self, config, module_class):
        """
        Main function to load data, process queries, and generate outputs.
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
            validated_standard_queries = validate_queries(standard_queries)

            # Initialize quotation optimizer
            await self.initialize_quotation_optimizer(config)

            # Initialize EnhancedQuotationModule with assertions
            try:
                logger.info("Initializing EnhancedQuotationModule")
                enhanced_quotation_module = module_class()
                enhanced_quotation_module = assert_transform_module(
                    enhanced_quotation_module, 
                    backtrack_handler
                )
                logger.info("EnhancedQuotationModule initialized successfully with assertions activated.")
            except Exception as e:
                logger.error(f"Error initializing EnhancedQuotationModule: {e}", exc_info=True)
                return

            # Define k value for standard queries
            k_standard = 20

            # Process standard queries with EnhancedQuotationModule
            logger.info("Processing standard queries with EnhancedQuotationModule")
            await process_queries(
                validated_standard_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_standard,
                output_file=config['output_filename_primary'],
                optimized_program=self.optimized_quotation_program,
                module=enhanced_quotation_module
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

            logger.info("All operations completed successfully.")
        except Exception as e:
            logger.error(f"Unexpected error in run_pipeline_with_config: {e}", exc_info=True)

    async def run_pipeline(self):
        # Define standard and alternate configurations
        config_standard = {
            'index_name': 'contextual_bm25_index_standard',
            'codebase_chunks_file': 'data/codebase_chunks.json',
            'queries_file_standard': 'data/queries.json',
            'evaluation_set_file': 'data/evaluation_set.jsonl',
            'output_filename_primary': 'query_results_quotation.json',
            'quotation_training_data': 'data/quotation_training_data.csv',
            'optimized_quotation_program': 'optimized_quotation_program.json'
        }
        config_alt = {
            'index_name': 'contextual_bm25_index_alt',
            'codebase_chunks_file': 'data/codebase_chunks_alt.json',
            'queries_file_standard': 'data/queries_alt.json',
            'evaluation_set_file': 'data/evaluation_set_alt.jsonl',
            'output_filename_primary': 'query_results_quotation_alt.json',
            'quotation_training_data': 'data/quotation_training_data_alt.csv',
            'optimized_quotation_program': 'optimized_quotation_program_alt.json'
        }
        # Run standard pipeline
        logger.info("Starting standard pipeline")
        await self.run_pipeline_with_config(config_standard, EnhancedQuotationModuleStandard)
        # Run alternate pipeline
        logger.info("Starting alternate pipeline")
        await self.run_pipeline_with_config(config_alt, EnhancedQuotationModuleAlt)

if __name__ == "__main__":
    pipeline = ThematicAnalysisPipeline()
    asyncio.run(pipeline.run_pipeline())
