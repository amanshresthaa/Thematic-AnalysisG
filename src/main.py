# main.py

import asyncio
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type, Optional

import dspy
from dspy.datasets import DataLoader
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from src.analysis.coding_module import CodingAnalysisModule
from src.analysis.extract_keyword_module import KeywordExtractionModule
from src.analysis.metrics import comprehensive_metric
from src.analysis.select_quotation_module import EnhancedQuotationModule as EnhancedQuotationModuleStandard
from src.analysis.theme_development_module import ThemedevelopmentAnalysisModule
from src.convert.convertcodingfortheme import convert_query_results as convert_coding_to_theme
from src.convert.convertkeywordforcoding import convert_query_results as convert_keyword_to_coding
from src.convert.convertquotationforkeyword import convert_query_results as convert_quotation_to_keyword
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.data.data_loader import load_codebase_chunks, load_queries
from src.decorators import handle_exceptions
from src.evaluation.evaluation import PipelineEvaluator
from src.processing.answer_generator import QuestionAnswerSignature, generate_answer_dspy
from src.processing.query_processor import process_queries, validate_queries
from src.retrieval.reranking import RerankerConfig, RerankerType, retrieve_with_reranking
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

dspy.settings.configure(main_thread_only=True)
thread_lock = threading.Lock()


@dataclass
class OptimizerConfig:
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    num_candidate_programs: int = 10
    num_threads: int = 1


@dataclass
class ModuleConfig:
    index_name: str
    codebase_chunks_file: str
    queries_file_standard: str
    evaluation_set_file: str
    output_filename_primary: str
    training_data: str
    optimized_program_path: str
    module_class: Type[Any]
    conversion_func: Optional[Callable[[str, str, str], None]] = None  # Allow None for the last module


class ThematicAnalysisPipeline:
    def __init__(self):
        logger.info("Initializing ThematicAnalysisPipeline")
        self.contextual_db = ContextualVectorDB("contextual_db")
        self.es_bm25: ElasticsearchBM25 = None
        self.optimized_programs: Dict[str, Any] = {}
        self.lock = threading.Lock()
        logger.info("ThematicAnalysisPipeline instance created with ContextualVectorDB initialized")

    def create_elasticsearch_bm25_index(self, index_name: str) -> ElasticsearchBM25:
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

    async def initialize_optimizer(self, config: ModuleConfig, optimizer_config: OptimizerConfig):
        module_name = config.module_class.__name__.replace("Module", "").lower()
        logger.info(f"Initializing {module_name} optimizer")
        start_time = time.time()

        try:
            dl = DataLoader()
            train_dataset = dl.from_csv(
                config.training_data,
                fields=("input", "output"),
                input_keys=("input",)
            )
            logger.info(f"Loaded {module_name} training dataset: {len(train_dataset)} samples")

            qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)
            teleprompter = BootstrapFewShotWithRandomSearch(
                metric=comprehensive_metric,
                max_bootstrapped_demos=optimizer_config.max_bootstrapped_demos,
                max_labeled_demos=optimizer_config.max_labeled_demos,
                num_candidate_programs=optimizer_config.num_candidate_programs,
                num_threads=optimizer_config.num_threads
            )

            compile_start = time.time()
            optimized_program = teleprompter.compile(
                student=qa_module,
                teacher=qa_module,
                trainset=train_dataset
            )
            compile_time = time.time() - compile_start
            logger.info(f"Compiled optimized {module_name} program in {compile_time:.2f}s")

            optimized_program.save(config.optimized_program_path)
            logger.info(f"Saved optimized {module_name} program to {config.optimized_program_path}")

            self.optimized_programs[module_name] = optimized_program
        except Exception as e:
            logger.error(f"Error initializing {module_name} optimizer: {e}", exc_info=True)
            raise

        total_time = time.time() - start_time
        logger.info(f"{module_name.capitalize()} optimizer initialization completed in {total_time:.2f}s")

    @handle_exceptions
    async def run_pipeline_with_config(
        self,
        config: ModuleConfig,
        optimizer_config: OptimizerConfig
    ):
        module_name = config.module_class.__name__.replace("Module", "").lower()
        logger.info(f"Starting pipeline for {module_name.capitalize()}")
        pipeline_start_time = time.time()

        try:
            # Configure DSPy Language Model
            logger.info("Configuring DSPy Language Model")
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)
            dspy.Cache = False

            # Load codebase chunks
            logger.info(f"Loading codebase chunks from {config.codebase_chunks_file}")
            chunks_start_time = time.time()
            codebase_chunks = load_codebase_chunks(config.codebase_chunks_file)
            chunks_time = time.time() - chunks_start_time
            logger.info(f"Loaded {len(codebase_chunks)} chunks in {chunks_time:.2f}s")

            # Load data into ContextualVectorDB
            logger.info("Loading data into ContextualVectorDB")
            db_start_time = time.time()
            self.contextual_db.load_data(codebase_chunks, parallel_threads=4)
            db_time = time.time() - db_start_time
            logger.info(f"Loaded data into ContextualVectorDB in {db_time:.2f}s")

            # Create Elasticsearch BM25 index
            logger.info(f"Creating Elasticsearch BM25 index: {config.index_name}")
            es_start_time = time.time()
            self.es_bm25 = self.create_elasticsearch_bm25_index(config.index_name)
            es_time = time.time() - es_start_time
            logger.info(f"Created Elasticsearch BM25 index in {es_time:.2f}s")

            # Load and validate queries
            logger.info(f"Loading queries from {config.queries_file_standard}")
            queries_start_time = time.time()
            standard_queries = load_queries(config.queries_file_standard)
            logger.info(f"Loaded {len(standard_queries)} queries")

            logger.info("Validating queries")
            validated_queries = validate_queries(standard_queries, config.module_class())
            logger.info(f"Validated {len(validated_queries)} queries")
            queries_time = time.time() - queries_start_time
            logger.info(f"Validated queries in {queries_time:.2f}s")

            # Initialize optimizer
            logger.info(f"Initializing optimizer for {module_name.capitalize()}")
            optimizer_start_time = time.time()
            await self.initialize_optimizer(config, optimizer_config)
            optimizer_time = time.time() - optimizer_start_time
            logger.info(f"Initialized optimizer in {optimizer_time:.2f}s")

            # Initialize module with assertions
            logger.info(f"Initializing {config.module_class.__name__}")
            module_instance = config.module_class()
            module_instance = assert_transform_module(module_instance, backtrack_handler)
            logger.info(f"Initialized {config.module_class.__name__} with assertions")

            # Setup reranker
            reranker_config = RerankerConfig(
                reranker_type=RerankerType.COHERE,
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                st_weight=0.5
            )

            # Determine optimized program
            optimized_program = self.optimized_programs.get(module_name)
            if not optimized_program:
                logger.error(f"Optimized program for {module_name} not found")
                return

            # Process queries
            k_standard = 20
            logger.info(f"Processing queries with k={k_standard}")
            query_start_time = time.time()
            await process_queries(
                validated_queries,
                self.contextual_db,
                self.es_bm25,
                k=k_standard,
                output_file=config.output_filename_primary,
                optimized_program=optimized_program,
                module=module_instance
            )
            query_time = time.time() - query_start_time
            logger.info(f"Processed queries in {query_time:.2f}s")

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
            evaluation_set = load_queries(config.evaluation_set_file)
            logger.info(f"Loaded {len(evaluation_set)} evaluation queries")

            k_values = [5, 10, 20]
            evaluator.evaluate_complete_pipeline(
                k_values=k_values,
                evaluation_set=evaluation_set
            )
            eval_time = time.time() - eval_start_time
            logger.info(f"Completed evaluation in {eval_time:.2f}s")

            total_pipeline_time = time.time() - pipeline_start_time
            logger.info(f"Pipeline for {module_name.capitalize()} completed in {total_pipeline_time:.2f}s")
        except Exception as e:
            logger.error(f"Error in pipeline execution for {module_name}: {e}", exc_info=True)
            raise

    async def convert_results(
        self,
        conversion_func: Optional[Callable[[str, str, str], None]],
        input_file: str,
        output_dir: str,
        output_file: str
    ):
        if conversion_func is None:
            logger.info("No conversion function provided; skipping conversion step.")
            return

        logger.info(f"Converting results using {conversion_func.__name__}")
        try:
            await asyncio.to_thread(
                conversion_func,
                input_file=input_file,
                output_dir=output_dir,
                output_file=output_file
                # sub_dir='input'  # If needed, modify the function to accept sub_dir
            )
            logger.info(f"Conversion {conversion_func.__name__} completed successfully")
        except Exception as e:
            logger.error(f"Error in conversion {conversion_func.__name__}: {e}", exc_info=True)
            raise

    async def run_pipeline(self):
        logger.info("Starting Thematic Analysis Pipeline")
        total_start_time = time.time()

        try:
            optimizer_config = OptimizerConfig()

            # Define configurations for each module
            configs = [
                ModuleConfig(
                    index_name='contextual_bm25_index_standard_quotation',
                    codebase_chunks_file='data/codebase_chunks/codebase_chunks.json',
                    queries_file_standard='data/input/queries_quotation.json',
                    evaluation_set_file='data/evaluation/evaluation_set_quotation.jsonl',
                    output_filename_primary='data/output/query_results_quotation.json',
                    training_data='data/training/quotation_training_data.csv',
                    optimized_program_path='data/optimized/optimized_quotation_program.json',
                    module_class=EnhancedQuotationModuleStandard,
                    conversion_func=convert_quotation_to_keyword
                ),
                ModuleConfig(
                    index_name='contextual_bm25_index_keyword_extraction',
                    codebase_chunks_file='data/codebase_chunks/codebase_chunks.json',
                    queries_file_standard='data/input/queries_keyword_standard.json',
                    evaluation_set_file='data/evaluation/evaluation_set_keyword.jsonl',
                    output_filename_primary='data/output/query_results_keyword_extraction.json',
                    training_data='data/training/keyword_training_data.csv',
                    optimized_program_path='data/optimized/optimized_keyword_program.json',
                    module_class=KeywordExtractionModule,
                    conversion_func=convert_keyword_to_coding
                ),
                ModuleConfig(
                    index_name='contextual_bm25_index_coding_analysis',
                    codebase_chunks_file='data/codebase_chunks/codebase_chunks.json',
                    queries_file_standard='data/input/queries_coding_standard.json',
                    evaluation_set_file='data/evaluation/evaluation_set_coding.jsonl',
                    output_filename_primary='data/output/query_results_coding_analysis.json',
                    training_data='data/training/coding_training_data.csv',
                    optimized_program_path='data/optimized/optimized_coding_program.json',
                    module_class=CodingAnalysisModule,
                    conversion_func=convert_coding_to_theme
                ),
                ModuleConfig(
                    index_name='contextual_bm25_index_theme_development',
                    codebase_chunks_file='data/codebase_chunks/codebase_chunks.json',
                    queries_file_standard='data/input/queries_theme.json',
                    evaluation_set_file='data/evaluation/evaluation_set_theme.jsonl',
                    output_filename_primary='data/output/query_results_theme_development.json',
                    training_data='data/training/theme_training_data.csv',
                    optimized_program_path='data/optimized/optimized_theme_program.json',
                    module_class=ThemedevelopmentAnalysisModule,
                    conversion_func=None  # No further conversion after theme development
                )
            ]

            # Run each pipeline stage sequentially with appropriate conversions
            for idx, config in enumerate(configs):
                logger.info(f"Starting {config.module_class.__name__} Pipeline")
                await self.run_pipeline_with_config(config, optimizer_config)

                # Handle conversions between stages except for the last module
                if idx < len(configs) - 1 and config.conversion_func is not None:
                    next_config = configs[idx + 1]
                    await self.convert_results(
                        conversion_func=config.conversion_func,
                        input_file=config.output_filename_primary,
                        output_dir='data',  # Set to 'data' to ensure 'input' subdirectory is correctly appended
                        output_file=next_config.queries_file_standard.split('/')[-1]  # Extract filename
                    )

            total_time = time.time() - total_start_time
            logger.info(f"All pipeline stages completed successfully in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
        finally:
            logger.info("Thematic Analysis Pipeline execution finished")
            gc.collect()


if __name__ == "__main__":
    logger.info("Launching Thematic Analysis Pipeline")
    pipeline = ThematicAnalysisPipeline()
    try:
        asyncio.run(pipeline.run_pipeline())
        logger.info("Pipeline execution completed successfully")
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
