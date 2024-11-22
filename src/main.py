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
from src.processing.query_processor import validate_queries, process_queries
from src.evaluation.evaluation import PipelineEvaluator
from src.analysis.metrics import comprehensive_metric
from src.processing.answer_generator import generate_answer_dspy, QuestionAnswerSignature
from src.retrieval.reranking import retrieve_with_reranking
from src.analysis.select_quotation_module import SelectQuotationModule
from src.analysis.select_quotation_module_alt import SelectQuotationModuleAlt
from src.analysis.select_keyword_module import SelectKeywordModule
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
        self.keyword_qa_module = None
        self.quotation_qa_module = None
        self.keyword_teleprompter = None
        self.quotation_teleprompter = None
        self.optimized_keyword_program = None
        self.optimized_quotation_program = None
        self.quotation_module = None
        self.quotation_module_alt = None
        self.keyword_module = None

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

    async def initialize_keyword_optimizer(self):
        """
        Initialize the keyword extraction optimizer.
        """
        logger.info("Initializing keyword extraction optimizer")
        dl = DataLoader()
        keyword_train_dataset = dl.from_csv(
            self.config['keyword_training_data'],
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
        
        self.optimized_keyword_program.save(self.config['optimized_keyword_program'])
        logger.info("Keyword optimizer initialized successfully")

    async def initialize_quotation_optimizer(self):
        """
        Initialize the quotation selection optimizer.
        """
        logger.info("Initializing quotation selection optimizer")
        dl = DataLoader()
        quotation_train_dataset = dl.from_csv(
            self.config['quotation_training_data'],
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
        
        self.optimized_quotation_program.save(self.config['optimized_quotation_program'])
        logger.info("Quotation optimizer initialized successfully")

    @handle_exceptions
    async def run_pipeline(self):
        """
        Main function to load data, process queries, and generate outputs.
        """
        logger.debug("Entering run_pipeline method.")
        try:
            # Configure DSPy Language Model
            logger.info("Configuring DSPy Language Model")
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)

            # Define file paths from config
            codebase_chunks_file = self.config['codebase_chunks_file']
            queries_file_standard = self.config['queries_file_standard']
            queries_file_alt = self.config['queries_file_alt']
            queries_file_keyword = self.config['queries_file_keyword']
            evaluation_set_file = self.config['evaluation_set_file']
            output_filename_primary = self.config['output_filename_primary']
            output_filename_alt = self.config['output_filename_alt']
            output_filename_keyword = self.config['output_filename_keyword']

            dl = DataLoader()

            # Load the training data for keywords and quotations
            logger.info(f"Loading keyword training data from '{self.config['keyword_training_data']}'")
            keyword_train_dataset = dl.from_csv(
                self.config['keyword_training_data'],
                fields=("input", "output"),
                input_keys=("input",)
            )

            logger.info(f"Loading quotation training data from '{self.config['quotation_training_data']}'")
            quotation_train_dataset = dl.from_csv(
                self.config['quotation_training_data'],
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

            # Initialize optimizers
            await self.initialize_keyword_optimizer()
            await self.initialize_quotation_optimizer()

            # Initialize SelectKeywordModule and SelectQuotationModules with assertions
            try:
                logger.info("Initializing SelectKeywordModule")
                self.keyword_module = SelectKeywordModule()
                self.keyword_module = assert_transform_module(self.keyword_module, backtrack_handler)
                logger.info("SelectKeywordModule initialized successfully with assertions activated.")
            except Exception as e:
                logger.error(f"Error initializing SelectKeywordModule: {e}", exc_info=True)
                return

            try:
                logger.info("Initializing SelectQuotationModule")
                self.quotation_module = SelectQuotationModule()
                self.quotation_module = assert_transform_module(self.quotation_module, backtrack_handler)
                logger.info("SelectQuotationModule initialized successfully with assertions activated.")
            except Exception as e:
                logger.error(f"Error initializing SelectQuotationModule: {e}", exc_info=True)
                return

            try:
                logger.info("Initializing SelectQuotationModuleAlt")
                self.quotation_module_alt = SelectQuotationModuleAlt()
                self.quotation_module_alt = assert_transform_module(self.quotation_module_alt, backtrack_handler)
                logger.info("SelectQuotationModuleAlt initialized successfully with assertions activated.")
            except Exception as e:
                logger.error(f"Error initializing SelectQuotationModuleAlt: {e}", exc_info=True)
                return

            # Define k value for standard and alternative queries
            k_standard = 20
            k_keyword = 2  # Fixed k=2 for keyword extraction

            # Process keyword queries with SelectKeywordModule
            logger.info("Processing keyword queries with SelectKeywordModule")
            await process_queries(
                validated_keyword_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_keyword,
                output_file=self.config['output_filename_keyword'],
                optimized_program=self.optimized_keyword_program,
                module=self.keyword_module,
                is_keyword_extraction=True
            )

            # Process standard queries with SelectQuotationModule
            logger.info("Processing standard queries with SelectQuotationModule")
            await process_queries(
                validated_standard_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_standard,
                output_file=self.config['output_filename_primary'],
                optimized_program=self.optimized_quotation_program,
                module=self.quotation_module
            )

            # Process alternative queries with SelectQuotationModuleAlt
            logger.info("Processing alternative queries with SelectQuotationModuleAlt")
            await process_queries(
                validated_alternative_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_standard,
                output_file=self.config['output_filename_alt'],
                optimized_program=self.optimized_quotation_program,
                module=self.quotation_module_alt
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
                evaluation_set = load_queries(self.config['evaluation_set_file'])
                evaluator.evaluate_complete_pipeline(
                    k_values=k_values,
                    evaluation_set=evaluation_set
                )
                logger.info("Evaluation completed successfully.")
            except Exception as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)

            logger.info("All operations completed successfully.")
        except Exception as e:
            logger.error(f"Unexpected error in run_pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    config = {
        'codebase_chunks_file': 'data/codebase_chunks.json',
        'queries_file_standard': 'data/queries.json',
        'queries_file_alt': 'data/queries_alt.json',
        'queries_file_keyword': 'data/queries_keyword.json',
        'evaluation_set_file': 'data/evaluation_set.jsonl',
        'output_filename_primary': 'query_results_primary.json',
        'output_filename_alt': 'query_results_alternative.json',
        'output_filename_keyword': 'query_results_keyword.json',
        'keyword_training_data': 'data/keyword_training_data.csv',
        'quotation_training_data': 'data/quotation_training_data.csv',
        'optimized_keyword_program': 'optimized_keyword_program.json',
        'optimized_quotation_program': 'optimized_quotation_program.json'
    }
    pipeline = ThematicAnalysisPipeline(config)
    asyncio.run(pipeline.run_pipeline())
