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
from src.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType
from src.analysis.select_quotation_module import EnhancedQuotationModule as EnhancedQuotationModuleStandard
from src.analysis.extract_keyword_module import KeywordExtractionModule
from src.analysis.coding_module import CodingAnalysisModule
from src.decorators import handle_exceptions

from src.convert.convertquotationforkeyword import convert_query_results as convert_quotation_to_keyword
from src.convert.convertkeywordforcoding import convert_query_results as convert_keyword_to_coding
from src.convert.convertcodingfortheme import process_input_file as convert_coding_to_theme

setup_logging()
logger = logging.getLogger(__name__)

dspy.settings.configure(main_thread_only=True)

thread_lock = threading.Lock()

class ThematicAnalysisPipeline:
    def __init__(self):
        logger.info("Initializing ThematicAnalysisPipeline")
        self.contextual_db = ContextualVectorDB("contextual_db")
        self.es_bm25 = None
        self.optimized_coding_program = None
        logger.info("ThematicAnalysisPipeline instance created with ContextualVectorDB initialized")

    def create_elasticsearch_bm25_index(self, index_name: str) -> ElasticsearchBM25:
        """
        Create and index documents in Elasticsearch BM25.
        """
        logger.info(f"Creating Elasticsearch BM25 index: {index_name}")
        start_time = time.time()
        
        try:
            es_bm25 = ElasticsearchBM25(index_name=index_name)
            logger.debug(f"ElasticsearchBM25 instance created with index '{index_name}'")
            
            index_start_time = time.time()
            success_count, failed_docs = es_bm25.index_documents(self.contextual_db.metadata)
            index_time = time.time() - index_start_time
            
            logger.info(f"Successfully indexed {success_count} documents in {index_time:.2f}s")
            
            if failed_docs:
                logger.warning(f"Failed to index {len(failed_docs)} documents")
                for doc in failed_docs:
                    logger.warning(f"Failed document ID: {doc.get('doc_id', 'Unknown')}")
                    
        except Exception as e:
            logger.error(f"Error creating Elasticsearch BM25 index '{index_name}': {e}", exc_info=True)
            raise
            
        total_time = time.time() - start_time
        logger.info(f"Elasticsearch BM25 index creation completed in {total_time:.2f}s")
        return es_bm25

    async def initialize_quotation_optimizer(self, config):
        """
        Initialize the quotation selection optimizer.
        """
        logger.info("Initializing quotation selection optimizer")
        start_time = time.time()
        
        try:
            dl = DataLoader()
            quotation_train_dataset = dl.from_csv(
                config['quotation_training_data'],
                fields=("input", "output"),
                input_keys=("input",)
            )
            logger.info(f"Loaded quotation training dataset: {len(quotation_train_dataset)} samples")

            self.quotation_qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)
            logger.debug("Created quotation QA module")

            optimizer_config = {
                'max_bootstrapped_demos': 4,
                'max_labeled_demos': 4,
                'num_candidate_programs': 10,
                'num_threads': 1
            }
            logger.debug(f"Optimizer configuration: {optimizer_config}")

            self.quotation_teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,
                **optimizer_config
            )

            compile_start = time.time()
            self.optimized_quotation_program = self.quotation_teleprompter.compile(
                student=self.quotation_qa_module,
                teacher=self.quotation_qa_module,
                trainset=quotation_train_dataset
            )
            compile_time = time.time() - compile_start
            logger.info(f"Compiled optimized quotation program in {compile_time:.2f}s")

            self.optimized_quotation_program.save(config['optimized_quotation_program'])
            logger.info(f"Saved optimized quotation program to {config['optimized_quotation_program']}")

        except Exception as e:
            logger.error(f"Error initializing quotation optimizer: {e}", exc_info=True)
            raise
            
        total_time = time.time() - start_time
        logger.info(f"Quotation optimizer initialization completed in {total_time:.2f}s")

    async def initialize_keyword_optimizer(self, config):
        """
        Initialize the keyword extraction optimizer.
        """
        logger.info("Initializing keyword extraction optimizer")
        start_time = time.time()
        
        try:
            dl = DataLoader()
            keyword_train_dataset = dl.from_csv(
                config['keyword_training_data'],
                fields=("input", "output"),
                input_keys=("input",)
            )
            logger.info(f"Loaded keyword training dataset: {len(keyword_train_dataset)} samples")

            self.keyword_qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)
            logger.debug("Created keyword QA module")

            optimizer_config = {
                'max_bootstrapped_demos': 4,
                'max_labeled_demos': 4,
                'num_candidate_programs': 10,
                'num_threads': 1
            }
            logger.debug(f"Optimizer configuration: {optimizer_config}")

            self.keyword_teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,
                **optimizer_config
            )

            compile_start = time.time()
            self.optimized_keyword_program = self.keyword_teleprompter.compile(
                student=self.keyword_qa_module,
                teacher=self.keyword_qa_module,
                trainset=keyword_train_dataset
            )
            compile_time = time.time() - compile_start
            logger.info(f"Compiled optimized keyword program in {compile_time:.2f}s")

            self.optimized_keyword_program.save(config['optimized_keyword_program'])
            logger.info(f"Saved optimized keyword program to {config['optimized_keyword_program']}")

        except Exception as e:
            logger.error(f"Error initializing keyword optimizer: {e}", exc_info=True)
            raise
            
        total_time = time.time() - start_time
        logger.info(f"Keyword optimizer initialization completed in {total_time:.2f}s")

    async def initialize_coding_optimizer(self, config):
        """
        Initialize the coding analysis optimizer.
        """
        logger.info("Initializing coding analysis optimizer")
        start_time = time.time()
        
        try:
            dl = DataLoader()
            coding_train_dataset = dl.from_csv(
                config['coding_training_data'],
                fields=("input", "output"),
                input_keys=("input",)
            )
            logger.info(f"Loaded coding training dataset: {len(coding_train_dataset)} samples")

            self.coding_qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)
            logger.debug("Created coding QA module")

            optimizer_config = {
                'max_bootstrapped_demos': 4,
                'max_labeled_demos': 4,
                'num_candidate_programs': 10,
                'num_threads': 1
            }
            logger.debug(f"Optimizer configuration: {optimizer_config}")

            self.coding_teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,
                **optimizer_config
            )

            compile_start = time.time()
            self.optimized_coding_program = self.coding_teleprompter.compile(
                student=self.coding_qa_module,
                teacher=self.coding_qa_module,
                trainset=coding_train_dataset
            )
            compile_time = time.time() - compile_start
            logger.info(f"Compiled optimized coding program in {compile_time:.2f}s")

            self.optimized_coding_program.save(config['optimized_coding_program'])
            logger.info(f"Saved optimized coding program to {config['optimized_coding_program']}")

        except Exception as e:
            logger.error(f"Error initializing coding optimizer: {e}", exc_info=True)
            raise
            
        total_time = time.time() - start_time
        logger.info(f"Coding optimizer initialization completed in {total_time:.2f}s")

    @handle_exceptions
    async def run_pipeline_with_config(self, config, module_class, optimizer_init_func):
        """
        Main function to load data, process queries, and generate outputs.
        """
        logger.info(f"Starting pipeline with {module_class.__name__}")
        pipeline_start_time = time.time()

        try:
            # Configure DSPy Language Model
            logger.info("Configuring DSPy Language Model")
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)
            dspy.Cache = False
            logger.debug("DSPy Language Model configured successfully")

            # Load the codebase chunks
            logger.info(f"Loading codebase chunks from {config['codebase_chunks_file']}")
            chunks_start_time = time.time()
            codebase_chunks = load_codebase_chunks(config['codebase_chunks_file'])
            chunks_time = time.time() - chunks_start_time
            logger.info(f"Loaded {len(codebase_chunks)} chunks in {chunks_time:.2f}s")

            # Load data into ContextualVectorDB
            logger.info("Loading data into ContextualVectorDB")
            db_start_time = time.time()
            self.contextual_db.load_data(codebase_chunks, parallel_threads=4)
            db_time = time.time() - db_start_time
            logger.info(f"Loaded data into ContextualVectorDB in {db_time:.2f}s")
            logger.debug(f"Total embeddings: {len(self.contextual_db.embeddings)}")
            logger.debug(f"Total metadata entries: {len(self.contextual_db.metadata)}")

            # Create Elasticsearch BM25 index
            logger.info(f"Creating Elasticsearch BM25 index: {config['index_name']}")
            es_start_time = time.time()
            self.es_bm25 = self.create_elasticsearch_bm25_index(config['index_name'])
            es_time = time.time() - es_start_time
            logger.info(f"Created Elasticsearch BM25 index in {es_time:.2f}s")

            # Load and validate queries
            logger.info(f"Loading queries from {config['queries_file_standard']}")
            queries_start_time = time.time()
            standard_queries = load_queries(config['queries_file_standard'])
            logger.info(f"Loaded {len(standard_queries)} queries")

            logger.info("Validating queries")
            validated_queries = validate_queries(standard_queries, module_class())
            queries_time = time.time() - queries_start_time
            logger.info(f"Validated {len(validated_queries)} queries in {queries_time:.2f}s")

            # Initialize optimizer
            logger.info(f"Initializing optimizer for {module_class.__name__}")
            optimizer_start_time = time.time()
            await optimizer_init_func(config)
            optimizer_time = time.time() - optimizer_start_time
            logger.info(f"Initialized optimizer in {optimizer_time:.2f}s")

            # Initialize module with assertions
            logger.info(f"Initializing {module_class.__name__}")
            module_instance = module_class()
            module_instance = assert_transform_module(
                module_instance,
                backtrack_handler
            )
            logger.info(f"Initialized {module_class.__name__} with assertions")

            # Set up reranker configuration
            reranker_config = RerankerConfig(
                reranker_type=RerankerType.COHERE,
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                st_weight=0.5
            )
            logger.debug(f"Created reranker config with type: {reranker_config.reranker_type}")

            # Process queries
            k_standard = 20
            logger.info(f"Processing queries with k={k_standard}")
            query_start_time = time.time()
            
            if module_class.__name__.lower().find('quotation') != -1:
                optimized_program = self.optimized_quotation_program
            elif module_class.__name__.lower().find('keyword') != -1:
                optimized_program = self.optimized_keyword_program
            elif module_class.__name__.lower().find('coding') != -1:
                optimized_program = self.optimized_coding_program
            else:
                logger.error(f"Unknown module type: {module_class.__name__}")
                return

            await process_queries(
                validated_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_standard,
                output_file=config['output_filename_primary'],
                optimized_program=optimized_program,
                module=module_instance
            )
            query_time = time.time() - query_start_time
            logger.info(f"Processed queries in {query_time:.2f}s")

            # Evaluation
            # Evaluation
            logger.info("Starting evaluation")
            eval_start_time = time.time()
            
            evaluator = PipelineEvaluator(
                db=self.contextual_db,
                es_bm25=self.es_bm25,
                retrieval_function=lambda query, db, es_bm25, k: retrieve_with_reranking(
                    query, db, es_bm25, k, reranker_config
                )
            )
            
            evaluation_set = load_queries(config['evaluation_set_file'])
            logger.info(f"Loaded {len(evaluation_set)} evaluation queries")

            k_values = [5, 10, 20]
            logger.debug(f"Evaluating with k values: {k_values}")
            
            evaluator.evaluate_complete_pipeline(
                k_values=k_values,
                evaluation_set=evaluation_set
            )
            eval_time = time.time() - eval_start_time
            logger.info(f"Completed evaluation in {eval_time:.2f}s")

            total_pipeline_time = time.time() - pipeline_start_time
            logger.info(f"Pipeline for {module_class.__name__} completed in {total_pipeline_time:.2f}s")

        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}", exc_info=True)
            raise

    async def run_pipeline(self):
        """
        Run the complete thematic analysis pipeline.
        """
        logger.info("Starting Thematic Analysis Pipeline")
        total_start_time = time.time()

        try:
            # Configure pipeline stages
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
                'queries_file_standard': 'data/input/queries_keyword.json',
                'evaluation_set_file': 'data/evaluation/evaluation_set_keyword.jsonl',
                'output_filename_primary': 'data/output/query_results_keyword_extraction.json',
                'keyword_training_data': 'data/training/keyword_training_data.csv',
                'optimized_keyword_program': 'data/optimized/optimized_keyword_program.json'
            }
            
            config_coding_analysis = {
                'index_name': 'contextual_bm25_index_coding_analysis',
                'codebase_chunks_file': 'data/codebase_chunks/codebase_chunks.json',
                'queries_file_standard': 'data/input/queries_coding.json',
                'evaluation_set_file': 'data/evaluation/evaluation_set_coding.jsonl',
                'output_filename_primary': 'data/output/query_results_coding_analysis.json',
                'coding_training_data': 'data/training/coding_training_data.csv',
                'optimized_coding_program': 'data/optimized/optimized_coding_program.json'
            }

            # Run Standard Quotation Extraction Pipeline
            logger.info("Starting Standard Quotation Extraction Pipeline")
            quotation_start_time = time.time()
            await self.run_pipeline_with_config(
                config_standard_quotation,
                EnhancedQuotationModuleStandard,
                self.initialize_quotation_optimizer
            )
            quotation_time = time.time() - quotation_start_time
            logger.info(f"Completed Quotation Extraction in {quotation_time:.2f}s")

            # Convert quotation results to keyword format
            logger.info("Converting quotation results to keyword format")
            try:
                await asyncio.to_thread(
                    convert_quotation_to_keyword,
                    input_file=config_standard_quotation['output_filename_primary'],
                    output_dir='data',
                    output_file='queries_keyword_standard.json'
                )
                logger.info("Quotation to keyword conversion completed")
            except Exception as e:
                logger.error(f"Error in quotation to keyword conversion: {e}", exc_info=True)
                raise

            # Run Keyword Extraction Pipeline
            logger.info("Starting Keyword Extraction Pipeline")
            keyword_start_time = time.time()
            config_keyword_extraction_standard = config_keyword_extraction.copy()
            config_keyword_extraction_standard['queries_file_standard'] = 'data/input/queries_keyword_standard.json'
            
            await self.run_pipeline_with_config(
                config_keyword_extraction_standard,
                KeywordExtractionModule,
                self.initialize_keyword_optimizer
            )
            keyword_time = time.time() - keyword_start_time
            logger.info(f"Completed Keyword Extraction in {keyword_time:.2f}s")

            # Convert keyword results to coding format
            logger.info("Converting keyword results to coding format")
            try:
                await asyncio.to_thread(
                    convert_keyword_to_coding,
                    input_file=config_keyword_extraction_standard['output_filename_primary'],
                    output_dir='data',
                    output_file='queries_coding_standard.json'
                )
                logger.info("Keyword to coding conversion completed")
            except Exception as e:
                logger.error(f"Error in keyword to coding conversion: {e}", exc_info=True)
                raise

            # Run Coding Analysis Pipeline
            logger.info("Starting Coding Analysis Pipeline")
            coding_start_time = time.time()
            config_coding_analysis_standard = config_coding_analysis.copy()
            config_coding_analysis_standard['queries_file_standard'] = 'data/input/queries_coding_standard.json'
            
            await self.run_pipeline_with_config(
                config_coding_analysis_standard,
                CodingAnalysisModule,
                self.initialize_coding_optimizer
            )
            coding_time = time.time() - coding_start_time
            logger.info(f"Completed Coding Analysis in {coding_time:.2f}s")

            # Convert coding results to theme format
            logger.info("Converting coding results to theme format")
            try:
                await asyncio.to_thread(
                    convert_coding_to_theme,
                    input_file=config_coding_analysis_standard['output_filename_primary'],
                    output_dir='data',
                    output_file='queries_theme_standard.json'
                )
                logger.info("Coding to theme conversion completed")
            except Exception as e:
                logger.error(f"Error in coding to theme conversion: {e}", exc_info=True)
                raise

            total_time = time.time() - total_start_time
            logger.info(f"All pipeline stages completed successfully in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logger.info("Launching Thematic Analysis Pipeline")
    pipeline = ThematicAnalysisPipeline()
    try:
        asyncio.run(pipeline.run_pipeline())
        logger.info("Pipeline execution completed successfully")
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        logger.info("Thematic Analysis Pipeline execution finished")
        gc.collect()  # Final cleanup