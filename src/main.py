# src/main.py

import asyncio
import logging
import sys
import os

# Add the parent directory to the Python path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the pipeline configuration and runner
from pipeline.pipeline_configs import ModuleConfig
from pipeline.pipeline_runner import ThematicAnalysisPipeline

# Import the analysis modules
from analysis.select_quotation_module import EnhancedQuotationModule as EnhancedQuotationModuleStandard
from analysis.extract_keyword_module import KeywordExtractionModule
from analysis.coding_module import CodingAnalysisModule
from analysis.grouping_module import GroupingAnalysisModule
from analysis.theme_development_module import ThemedevelopmentAnalysisModule

# Import the conversion functions
from convert.convertquotationforkeyword import convert_query_results as convert_quotation_to_keyword
from convert.convertkeywordforcoding import convert_query_results as convert_keyword_to_coding
from convert.convertcodingforgrouping import convert_query_results as convert_coding_to_grouping
from convert.convertgroupingfortheme import convert_query_results as convert_grouping_to_theme

if __name__ == "__main__":
    # Basic logger configuration
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Launching Refactored Thematic Analysis Pipeline")

    # Define the modules/stages in the order you want to process them
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
            conversion_func=None  # No conversion needed after the final stage
        )
    ]

    # Instantiate the pipeline
    pipeline = ThematicAnalysisPipeline()

    try:
        # Run all pipeline stages asynchronously
        asyncio.run(pipeline.run_pipeline(configs))
        logger.info("Thematic Analysis Pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)