# src/main.py

import asyncio
import gc
import logging
import os
import threading
import time
import json
from dataclasses import dataclass
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
from src.analysis.grouping_module import GroupingAnalysisModule
from src.convert.convertcodingfortheme import convert_query_results as convert_coding_to_theme
from src.convert.convertkeywordforcoding import convert_query_results as convert_keyword_to_coding
from src.convert.convertquotationforkeyword import convert_query_results as convert_quotation_to_keyword
from src.convert.convertcodingforgrouping import convert_query_results as convert_coding_to_grouping
from src.convert.convertgroupingfortheme import convert_query_results as convert_grouping_to_theme
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.data.data_loader import load_codebase_chunks, load_queries
from src.decorators import handle_exceptions
from src.evaluation.evaluation import PipelineEvaluator
from src.processing.answer_generator import QuestionAnswerSignature
from src.processing.query_processor import process_queries, validate_queries
from src.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType
from src.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

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
    conversion_func: Optional[Callable[[str, str, str], None]] = None

class ThematicAnalysisPipeline:
    def __init__(self):
        logger.info("Initializing ThematicAnalysisPipeline")
        self.contextual_db = ContextualVectorDB("contextual_db")
        self.es_bm25: Optional[ElasticsearchBM25] = None
        self.optimized_programs: Dict[str, Any] = {}
        self.lock = threading.Lock()
        logger.info("ThematicAnalysisPipeline instance created with ContextualVectorDB initialized")

    def create_elasticsearch_bm25_index(self, index_name: str) -> ElasticsearchBM25:
        logger.info(f"Creating Elasticsearch BM25 index: {index_name}")
        start_time = time.time()
        try:
            es_bm25 = ElasticsearchBM25(index_name=index_name)
            success_count, failed_docs = es_bm25.index_documents(self.contextual_db.metadata)
            if failed_docs:
                logger.warning(f"Failed to index {len(failed_docs)} documents")
            total_time = time.time() - start_time
            logger.info(f"Elasticsearch BM25 index creation completed in {total_time:.2f}s")
            return es_bm25
        except Exception as e:
            logger.error(f"Error creating Elasticsearch BM25 index '{index_name}': {e}", exc_info=True)
            raise

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
            logger.info(f"Loaded {len(train_dataset)} samples for {module_name} training data")

            # Update to use ChainOfThought instead of TypedChainOfThought
            qa_module = dspy.ChainOfThought(QuestionAnswerSignature)
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
            logger.info("Configuring DSPy Language Model")
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)
            dspy.Cache = False

            logger.info(f"Loading codebase chunks from {config.codebase_chunks_file}")
            chunks_start_time = time.time()
            codebase_chunks = load_codebase_chunks(config.codebase_chunks_file)
            chunks_time = time.time() - chunks_start_time
            logger.info(f"Loaded {len(codebase_chunks)} codebase chunks in {chunks_time:.2f}s")

            logger.info("Loading data into ContextualVectorDB")
            db_start_time = time.time()
            self.contextual_db.load_data(codebase_chunks, parallel_threads=4)
            db_time = time.time() - db_start_time
            logger.info(f"Loaded data into ContextualVectorDB in {db_time:.2f}s")

            logger.info(f"Creating Elasticsearch BM25 index: {config.index_name}")
            es_start_time = time.time()
            self.es_bm25 = self.create_elasticsearch_bm25_index(config.index_name)
            es_time = time.time() - es_start_time
            logger.info(f"Created Elasticsearch BM25 index in {es_time:.2f}s")

            logger.info(f"Loading queries from {config.queries_file_standard}")
            queries_start_time = time.time()
            standard_queries = load_queries(config.queries_file_standard)
            logger.info(f"Loaded {len(standard_queries)} queries")

            logger.info("Validating queries")
            validated_queries = validate_queries(standard_queries, config.module_class())
            logger.info(f"Validated {len(validated_queries)} queries")
            queries_time = time.time() - queries_start_time
            logger.info(f"Validated queries in {queries_time:.2f}s")

            logger.info(f"Initializing optimizer for {module_name.capitalize()}")
            optimizer_start_time = time.time()
            await self.initialize_optimizer(config, optimizer_config)
            optimizer_time = time.time() - optimizer_start_time
            logger.info(f"Initialized optimizer in {optimizer_time:.2f}s")

            logger.info(f"Initializing {config.module_class().__class__.__name__}")
            module_instance = config.module_class()
            module_instance = assert_transform_module(module_instance, backtrack_handler)
            logger.info(f"Initialized {config.module_class().__class__.__name__}")

            reranker_config = RerankerConfig(
                reranker_type=RerankerType.COHERE,
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                st_weight=0.5
            )

            optimized_program = self.optimized_programs.get(module_name)
            if not optimized_program:
                logger.error(f"Optimized program for {module_name} not found. Exiting.")
                return

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
            )
            logger.info(f"Conversion {conversion_func.__name__} completed successfully")
        except Exception as e:
            logger.error(f"Error in conversion {conversion_func.__name__}: {e}", exc_info=True)
            raise

    def generate_theme_input(self):
        try:
            logger.info("Generating queries_theme.json from info.json and query_results_grouping.json")
            info_path = 'data/input/info.json'
            grouping_results_path = 'data/output/query_results_grouping.json'

            if not os.path.exists(info_path):
                logger.error("info.json not found.")
                return

            if not os.path.exists(grouping_results_path):
                logger.error("query_results_grouping.json not found.")
                return

            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)

            with open(grouping_results_path, 'r', encoding='utf-8') as f:
                grouping_results = json.load(f)

            if not grouping_results:
                logger.error("query_results_grouping.json is empty.")
                return

            first_result = grouping_results[0]

            research_objectives = info.get("research_objectives", "")
            theoretical_framework = info.get("theoretical_framework", {})

            codes = first_result.get("grouping_info", {}).get("codes", [])
            groupings = first_result.get("groupings", [])

            # The final theme module needs quotation, keywords, and transcript_chunk.
            # These should be retrieved from previous pipeline stages or stored accordingly.
            # Here, we assume placeholders for demonstration.
            quotation = "Original quotation from previous step."
            keywords = ["improvement", "reasoning", "innovation"]
            transcript_chunk = "A relevant transcript chunk providing context."

            queries_theme = [
                {
                    "quotation": quotation,
                    "keywords": keywords,
                    "codes": codes,
                    "research_objectives": research_objectives,
                    "theoretical_framework": theoretical_framework,
                    "transcript_chunk": transcript_chunk,
                    "groupings": groupings
                }
            ]

            theme_path = 'data/input/queries_theme.json'
            os.makedirs(os.path.dirname(theme_path), exist_ok=True)
            with open(theme_path, 'w', encoding='utf-8') as f:
                json.dump(queries_theme, f, indent=4)
            logger.info(f"Generated {theme_path} successfully.")
        except Exception as e:
            logger.error(f"Error generating queries_theme.json: {e}", exc_info=True)

    @handle_exceptions
    async def run_pipeline(self):
        logger.info("Starting Thematic Analysis Pipeline")
        total_start_time = time.time()

        try:
            optimizer_config = OptimizerConfig()

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
                    conversion_func=convert_coding_to_grouping
                ),
                ModuleConfig(
                    index_name='contextual_bm25_index_grouping',
                    codebase_chunks_file='data/codebase_chunks/codebase_chunks.json',
                    queries_file_standard='data/input/queries_grouping.json',
                    evaluation_set_file='data/evaluation/evaluation_set_grouping.jsonl',
                    output_filename_primary='data/output/query_results_grouping.json',
                    training_data='data/training/grouping_training_data.csv',
                    optimized_program_path='data/optimized/optimized_grouping_program.json',
                    module_class=GroupingAnalysisModule,
                    conversion_func=convert_grouping_to_theme
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
                    conversion_func=None  # No conversion after theme development
                )
            ]

            for idx, config in enumerate(configs):
                logger.info(f"Starting {config.module_class.__name__} Pipeline")
                await self.run_pipeline_with_config(config, optimizer_config)

                # Perform conversion only if not the last module
                if idx < len(configs) - 1 and config.conversion_func is not None:
                    next_config = configs[idx + 1]
                    await self.convert_results(
                        conversion_func=config.conversion_func,
                        input_file=config.output_filename_primary,
                        output_dir='data/',  # Target directory for the next stage's input
                        output_file=os.path.basename(next_config.queries_file_standard)  # Only the filename
                    )

            # Before running the theme development stage, generate queries_theme.json using info.json and query_results_grouping.json
            self.generate_theme_input()

            # Run the final stage (theme development) again after generating queries_theme.json
            final_config = configs[-1]
            await self.run_pipeline_with_config(final_config, optimizer_config)

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
