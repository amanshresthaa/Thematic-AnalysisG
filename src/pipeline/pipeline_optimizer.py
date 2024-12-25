# src/pipeline/pipeline_optimizer.py

import logging
import time
import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.primitives.assertions import backtrack_handler
from src.analysis.metrics import comprehensive_metric
from src.processing.answer_generator import QuestionAnswerSignature
from src.pipeline.pipeline_configs import OptimizerConfig, ModuleConfig

logger = logging.getLogger(__name__)

async def initialize_optimizer(config: ModuleConfig, optimizer_config: OptimizerConfig, optimized_programs: dict):
    """
    Initializes and trains a teleprompt-based optimizer for the specified module.
    """
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

        optimized_programs[module_name] = optimized_program
    except Exception as e:
        logger.error(f"Error initializing {module_name} optimizer: {e}", exc_info=True)
        raise

    total_time = time.time() - start_time
    logger.info(f"{module_name.capitalize()} optimizer initialization completed in {total_time:.2f}s")