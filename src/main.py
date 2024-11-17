import gc
import logging
import os
from typing import Dict, Any , List
import asyncio
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.datasets import DataLoader

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
from src.analysis.extract_keywords_module import KeywordExtractionModule
from src.decorators import handle_exceptions
import json

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def save_quotations(quotations: List[Dict[str, Any]], output_file: str):
    """Save extracted quotations to output JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(quotations, f, indent=4)
        logger.info(f"Quotations saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving quotations to {output_file}: {e}")

class ThematicAnalysisPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Thematic Analysis Pipeline with given configuration.
        """
        self.config = config
        self.contextual_db = None
        self.es_bm25 = None
        self.qa_module = None
        self.teleprompter = None
        self.optimized_program = None
        self.quotation_module = None
        self.keyword_module = None

    @handle_exceptions
    def initialize_components(self):
        """Initialize all pipeline components."""
        # Configure DSPy Language Model
        lm = dspy.LM('openai/gpt-4o-mini', max_tokens=8192)
        dspy.configure(lm=lm)

        # Initialize databases
        self.contextual_db = ContextualVectorDB("contextual_db")
        self.es_bm25 = ElasticsearchBM25()

        # Initialize modules with contextual retrieval components
        self.quotation_module = SelectQuotationModule(
            contextual_db=self.contextual_db,
            es_bm25=self.es_bm25
        )
        self.keyword_module = KeywordExtractionModule(
            input_file=self.config['quotation_file'],
            output_file=self.config['keywords_output_file'],
            contextual_db=self.contextual_db,
            es_bm25=self.es_bm25
        )

        # Initialize QA module
        try:
            self.qa_module = dspy.Program.load("optimized_program.json")
        except Exception:
            self.qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)

    @handle_exceptions
    def initialize_databases(self, codebase_chunks):
        """Initialize and populate databases."""
        # Load data into ContextualVectorDB
        self.contextual_db.load_data(codebase_chunks, parallel_threads=5)

        # Index documents in ElasticsearchBM25
        success_count, _ = self.es_bm25.index_documents(self.contextual_db.metadata)
        logger.info(f"Indexed {success_count} documents in Elasticsearch")

    @handle_exceptions
    def optimize_qa_module(self, train_dataset):
        """Optimize the QA module using DSPy's BootstrapFewShotWithRandomSearch."""
        optimizer_config = {
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 4,
            'num_candidate_programs': 10,
            'num_threads': 4
        }

        self.teleprompter = BootstrapFewShotWithRandomSearch(
            metric=comprehensive_metric,
            **optimizer_config
        )

        self.optimized_program = self.teleprompter.compile(
            student=self.qa_module,
            teacher=self.qa_module,
            trainset=train_dataset
        )

        self.optimized_program.save("optimized_program.json")

    @handle_exceptions
    async def process_data(self):
        """Process data through the pipeline including quotation selection and keyword extraction."""
        # Extract quotations using the updated module
        quotations_result = self.quotation_module.forward(
            research_objectives=self.config['research_objectives']
        )
        quotations = quotations_result.get("quotations", [])

        if quotations:
            # Save quotations to a file
            save_quotations(quotations, self.config['quotation_file'])

            # Extract keywords using the updated module
            keywords_result = self.keyword_module.process_file(
                input_file=self.config['quotation_file'],
                research_objectives=self.config['research_objectives']
            )

            if keywords_result.get("keywords"):
                logger.info(f"Keywords extracted and saved to {keywords_result.get('output_file')}")
        else:
            logger.warning("No quotations were extracted.")

    @handle_exceptions
    def evaluate_pipeline(self, evaluation_set):
        """Evaluate the pipeline performance."""
        evaluator = PipelineEvaluator(
            db=self.contextual_db,
            es_bm25=self.es_bm25,
            retrieval_function=retrieve_with_reranking
        )

        evaluator.evaluate_complete_pipeline(
            k_values=[5, 10, 20],
            evaluation_set=evaluation_set
        )

    @handle_exceptions
    async def run_pipeline(self):
        """Main pipeline execution method."""
        try:
            # Initialize components
            self.initialize_components()

            # Load and prepare data
            dl = DataLoader()
            train_dataset = dl.from_csv(
                "data/new_training_data.csv",
                fields=("input", "output"),
                input_keys=("input",)
            )

            # Load and validate input data
            codebase_chunks = load_codebase_chunks(self.config['codebase_chunks_file'])
            queries = load_queries(self.config['queries_file'])
            self.validated_queries = validate_queries(queries)
            evaluation_set = load_queries(self.config['evaluation_set_file'])

            # Initialize databases
            self.initialize_databases(codebase_chunks)

            # Optimize QA module
            self.optimize_qa_module(train_dataset)

            # Process data
            await self.process_data()

            # Evaluate pipeline
            self.evaluate_pipeline(evaluation_set)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            gc.collect()

def main():
    """Entry point of the application."""
    config = {
        'codebase_chunks_file': 'data/codebase_chunks.json',
        'queries_file': 'data/queries.json',
        'evaluation_set_file': 'data/evaluation_set.jsonl',
        'output_filename': 'query_results.json',
        'quotation_file': 'data/quotations.json',
        'keywords_output_file': 'data/keywords.json',
        'research_objectives': 'Your research objectives here'
    }

    pipeline = ThematicAnalysisPipeline(config)
    asyncio.run(pipeline.run_pipeline())

if __name__ == "__main__":
    main()