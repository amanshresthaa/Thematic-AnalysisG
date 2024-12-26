# src/pipeline/pipeline_runner.py

import asyncio
import logging
import time
import gc
import os
from typing import Optional, Callable

import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.data.data_loader import load_codebase_chunks, load_queries
from src.decorators import handle_exceptions
from src.evaluation.evaluation import PipelineEvaluator
from src.pipeline.pipeline_configs import OptimizerConfig, ModuleConfig
from src.pipeline.pipeline_data import create_directories, generate_theme_input
from src.pipeline.pipeline_optimizer import initialize_optimizer
from src.processing.query_processor import process_queries

from src.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

class ThematicAnalysisPipeline:
    def __init__(self):
        setup_logging()
        logger.info("Initializing ThematicAnalysisPipeline")

        create_directories([
            'data/input',
            'data/output',
            'data/codebase_chunks',
            'data/optimized',
            'data/training',
            'data/evaluation'
        ])

        self.contextual_db = ContextualVectorDB("contextual_db")
        self.es_bm25: Optional[ElasticsearchBM25] = None
        self.optimized_programs = {}

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

    @handle_exceptions
    async def run_pipeline_with_config(self, config: ModuleConfig, optimizer_config: OptimizerConfig):
        module_name = config.module_class.__name__.replace("Module", "").lower()
        logger.info(f"Starting pipeline stage for {module_name.capitalize()}")
        pipeline_start_time = time.time()

        try:
            logger.info("Configuring DSPy Language Model")
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
            dspy.configure(lm=lm)
            dspy.Cache = False

            logger.info(f"Loading codebase chunks from {config.codebase_chunks_file}")
            codebase_chunks = load_codebase_chunks(config.codebase_chunks_file)

            logger.info("Loading data into ContextualVectorDB")
            self.contextual_db.load_data(codebase_chunks, parallel_threads=4)

            logger.info(f"Creating Elasticsearch BM25 index: {config.index_name}")
            self.es_bm25 = self.create_elasticsearch_bm25_index(config.index_name)

            logger.info(f"Loading queries from {config.queries_file_standard}")
            standard_queries = load_queries(config.queries_file_standard)

            # Removed separate validation step as it's now handled internally within process_queries
            # logger.info("Validating queries")
            # validated_queries = validate_queries(standard_queries, config.module_class())

            logger.info(f"Initializing optimizer for {module_name.capitalize()}")
            await initialize_optimizer(config, optimizer_config, self.optimized_programs)

            module_instance = config.module_class()
            logger.debug(f"Module instance created: {type(module_instance).__name__}")
            module_instance = assert_transform_module(module_instance, backtrack_handler)

            optimized_program = self.optimized_programs.get(module_name)
            if not optimized_program:
                logger.error(f"Optimized program for {module_name} not found.")
                return

            logger.info(f"Processing queries for {module_name.capitalize()}")
            await process_queries(
                transcripts=standard_queries,  # Pass standard_queries directly without separate validation
                db=self.contextual_db,
                es_bm25=self.es_bm25,
                k=20,
                output_file=config.output_filename_primary,
                optimized_program=optimized_program,
                module=module_instance
            )

            logger.info(f"Starting evaluation for {module_name.capitalize()}")
            evaluator = PipelineEvaluator(
                db=self.contextual_db,
                es_bm25=self.es_bm25,
                retrieval_function=lambda query, db, es_bm25, k: retrieve_with_reranking(
                    query, db, es_bm25, k, RerankerConfig(
                        reranker_type=RerankerType.COHERE,
                        cohere_api_key=os.getenv("COHERE_API_KEY"),
                        st_weight=0.5
                    )
                )
            )
            evaluation_set = load_queries(config.evaluation_set_file)
            evaluator.evaluate_complete_pipeline(k_values=[5, 10, 20], evaluation_set=evaluation_set)

            total_time = time.time() - pipeline_start_time
            logger.info(f"Pipeline stage for {module_name.capitalize()} completed in {total_time:.2f}s")

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
        if not conversion_func:
            logger.info("No conversion function provided; skipping.")
            return

        create_directories([output_dir])
        logger.info(f"Converting results with {conversion_func.__name__}")
        try:
            await asyncio.to_thread(
                conversion_func,
                input_file=input_file,
                output_dir=output_dir,
                output_file=output_file
            )
        except Exception as e:
            logger.error(f"Error converting results with {conversion_func.__name__}: {e}", exc_info=True)
            raise

    @handle_exceptions
    async def run_pipeline(self, configs: list[ModuleConfig]):
        logger.info("Starting the entire Thematic Analysis Pipeline")
        total_start_time = time.time()

        try:
            optimizer_config = OptimizerConfig()

            for idx, config in enumerate(configs):
                logger.info(f"--- Running {config.module_class.__name__} stage ---")
                await self.run_pipeline_with_config(config, optimizer_config)

                # Convert step (except for the last module)
                if idx < len(configs) - 1 and config.conversion_func:
                    next_config = configs[idx + 1]
                    await self.convert_results(
                        conversion_func=config.conversion_func,
                        input_file=config.output_filename_primary,
                        output_dir='data/',
                        output_file=os.path.basename(next_config.queries_file_standard)
                    )

            # Example of generating queries_theme.json before running final stage
            generate_theme_input(
                info_path='data/input/info.json',
                grouping_path='data/output/query_results_grouping.json',
                output_path='data/input/queries_theme.json'
            )

            # Run the final stage if needed
            final_config = configs[-1]
            await self.run_pipeline_with_config(final_config, optimizer_config)

            total_time = time.time() - total_start_time
            logger.info(f"Entire pipeline completed in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
        finally:
            logger.info("Thematic Analysis Pipeline execution finished")
            gc.collect()
