#pipeline_runner.py
import asyncio
import logging
import time
import gc
import os
from typing import Optional, Callable
from dotenv import load_dotenv

import dspy
from dspy import BestOfN
from src.config.model_config import get_model_config, ModelProvider
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.data.data_loader import load_codebase_chunks, load_queries
from src.decorators import handle_exceptions
from src.evaluation.evaluation import PipelineEvaluator
from src.pipeline.pipeline_configs import OptimizerConfig, ModuleConfig
from src.pipeline.pipeline_data import create_directories, generate_theme_input
from src.pipeline.pipeline_optimizer import initialize_optimizer
from src.processing.query_processor import process_queries

from src.core.retrieval.reranking import retrieve_with_reranking, RerankerConfig, RerankerType
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

class ThematicAnalysisPipeline:
    def __init__(self):
        load_dotenv()
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

    def _setup_model_environment(self, model_config):
        """Set up environment variables for the specified model."""
        api_key = os.getenv(model_config.api_key_env)
        if not api_key:
            raise ValueError(f"{model_config.api_key_env} environment variable is not set")

        # Set provider-specific environment variables
        if model_config.provider == ModelProvider.OPENAI:
            os.environ['OPENAI_API_KEY'] = api_key
        elif model_config.provider == ModelProvider.GOOGLE:
            os.environ['GOOGLE_API_KEY'] = api_key
        elif model_config.provider == ModelProvider.DEEPSEEK:
            os.environ['DEEPSEEK_API_KEY'] = api_key

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

    def _configure_language_model(self, model_config):
        """Configure the language model based on provider."""
        logger.info(f"Configuring DSPy Language Model for provider: {model_config.provider.value}")
        
        if model_config.provider == ModelProvider.OPENAI:
            model_name = model_config.model_name
        elif model_config.provider == ModelProvider.GOOGLE:
            model_name = f"google/{model_config.model_name}"
        elif model_config.provider == ModelProvider.DEEPSEEK:
            model_name = model_config.model_name
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

        lm = dspy.LM(
            model_name,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature
        )
        dspy.configure(lm=lm)
        dspy.Cache = False
        return lm
    
    def _create_module_reward_function(self, module_name):
        """
        Create a reward function for the specified module type.
        This replaces the previous assertion logic with a scoring mechanism.
        """
        def generic_reward_fn(input_kwargs, prediction):
            """Generic reward function that validates module outputs."""
            if not prediction:
                logger.warning(f"Empty prediction from {module_name} module")
                return 0.0
                
            # Basic validation that prediction contains expected outputs
            if module_name == "selectquotation":
                # Check if quotations exist and are valid
                quotations = prediction.get("quotations", [])
                if not quotations:
                    logger.warning("No quotations found in prediction")
                    return 0.0
                
                # Check alignment with research objectives
                research_objectives = input_kwargs.get("research_objectives", "")
                if research_objectives and not self._check_relevance(quotations, research_objectives):
                    logger.warning("Quotations not aligned with research objectives")
                    return 0.5  # Partial credit
                
                # Additional checks similar to previous assertions
                transcript_chunk = input_kwargs.get("transcript_chunk", "")
                if transcript_chunk and not self._check_selective_transcription(quotations, transcript_chunk):
                    logger.warning("Quotations not properly extracted from transcript")
                    return 0.7  # Higher partial credit
                
                return 1.0  # Full credit if all checks pass
                
            elif module_name == "extractkeyword":
                # Check if keywords exist
                keywords = prediction.get("keywords", [])
                if not keywords:
                    logger.warning("No keywords found in prediction")
                    return 0.0
                    
                # Check if analysis is present
                analysis = prediction.get("analysis", {})
                if not analysis:
                    logger.warning("No analysis found in prediction")
                    return 0.5
                    
                return 1.0
                
            elif module_name == "themedevelopment":
                # Check if themes exist
                themes = prediction.get("themes", [])
                if not themes:
                    logger.warning("No themes found in prediction")
                    return 0.0
                    
                # Check if analysis exists
                analysis = prediction.get("analysis", {})
                if not analysis:
                    logger.warning("No analysis found in prediction")
                    return 0.7
                    
                return 1.0
                
            else:
                # Default validation for other module types
                # Assuming valid output has some content
                return 1.0 if bool(prediction) else 0.0
                
        return generic_reward_fn
    
    def _check_relevance(self, quotations, research_objectives):
        """Check if quotations are relevant to research objectives."""
        # Simple implementation - in a real system, this would be more sophisticated
        if not quotations or not research_objectives:
            return False
            
        # Check if at least one quotation mentions a key term from objectives
        key_terms = [term.lower() for term in research_objectives.split() if len(term) > 4]
        for quotation in quotations:
            quote_text = quotation.get("text", "").lower()
            if any(term in quote_text for term in key_terms):
                return True
                
        return False
        
    def _check_selective_transcription(self, quotations, transcript_chunk):
        """Check if quotations are properly extracted from transcript."""
        # Simple implementation - in a real system, this would be more sophisticated
        if not quotations or not transcript_chunk:
            return False
            
        # Check if at least one quotation appears in transcript
        transcript_lower = transcript_chunk.lower()
        for quotation in quotations:
            quote_text = quotation.get("text", "").lower()
            # Allow for minor variations in text extraction
            if any(segment in transcript_lower for segment in quote_text.split('. ')):
                return True
                
        return False

    @handle_exceptions
    async def run_pipeline_with_config(self, config: ModuleConfig, optimizer_config: OptimizerConfig):
        module_name = config.module_class.__name__.replace("Module", "").lower()
        logger.info(f"Starting pipeline stage for {module_name.capitalize()}")
        pipeline_start_time = time.time()

        try:
            # Get model configuration from ModuleConfig or use default
            model_config = get_model_config(getattr(config, 'model_provider', None))
            
            # Setup environment variables for the selected model
            self._setup_model_environment(model_config)
            
            # Configure the language model
            lm = self._configure_language_model(model_config)

            logger.info(f"Loading codebase chunks from {config.codebase_chunks_file}")
            codebase_chunks = load_codebase_chunks(config.codebase_chunks_file)

            logger.info("Loading data into ContextualVectorDB")
            self.contextual_db.load_data(codebase_chunks, parallel_threads=4)

            logger.info(f"Creating Elasticsearch BM25 index: {config.index_name}")
            self.es_bm25 = self.create_elasticsearch_bm25_index(config.index_name)

            logger.info(f"Loading queries from {config.queries_file_standard}")
            standard_queries = load_queries(config.queries_file_standard)

            logger.info(f"Initializing optimizer for {module_name.capitalize()}")
            # Get the optimized module
            optimized_module = initialize_optimizer(
                config.module_class,
                optimizer_config,
                self.contextual_db,
                self.es_bm25
            )
            self.optimized_programs[module_name] = optimized_module

            # Create a basic module instance without any wrappers
            module_instance = config.module_class()
            logger.debug(f"Module instance created: {type(module_instance).__name__}")

            logger.info(f"Processing queries for {module_name.capitalize()}")
            await process_queries(
                transcripts=standard_queries,
                db=self.contextual_db,
                es_bm25=self.es_bm25,
                k=20,  # Use a hardcoded value that matches typical usage
                output_file=config.output_filename_primary,
                optimized_program=optimized_module,
                module=module_instance
            )

            logger.info(f"Starting evaluation for {module_name.capitalize()}")
            retrieval_func = lambda query, db, es_bm25, k: retrieve_with_reranking(
                query, db, es_bm25, k, RerankerConfig(
                    reranker_type=RerankerType.COHERE,
                    cohere_api_key=os.getenv("COHERE_API_KEY"),
                    st_weight=0.5
                )
            )
            evaluator = PipelineEvaluator(self.contextual_db, self.es_bm25, retrieval_func)
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

                if idx < len(configs) - 1 and config.conversion_func:
                    next_config = configs[idx + 1]
                    await self.convert_results(
                        conversion_func=config.conversion_func,
                        input_file=config.output_filename_primary,
                        output_dir='data/',
                        output_file=os.path.basename(next_config.queries_file_standard)
                    )

            generate_theme_input(
                info_path='data/input/info.json',
                grouping_path='data/output/query_results_grouping.json',
                output_path='data/input/queries_theme.json'
            )

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