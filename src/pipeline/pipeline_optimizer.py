import logging
import dspy
from typing import Dict, Any, Type
from src.core.contextual_vector_db import ContextualVectorDB
from src.core.elasticsearch_bm25 import ElasticsearchBM25
from src.pipeline.pipeline_configs import OptimizerConfig

logger = logging.getLogger(__name__)

def initialize_optimizer(
    module_class: Type[dspy.Module],
    optimizer_config: OptimizerConfig,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
) -> dspy.Module:
    """
    Initialize the pipeline optimizer with the new BestOfN approach.
    
    Args:
        module_class: The DSPy module class to optimize
        optimizer_config: Configuration for optimization
        db: Contextual vector database instance
        es_bm25: Elasticsearch BM25 instance
    
    Returns:
        Optimized DSPy module instance
    """
    try:
        logger.info(f"Initializing optimizer for {module_class.__name__}")
        
        # Create base module instance
        module_instance = module_class()
        
        # Create reward function based on module type
        def reward_fn(input_kwargs: Dict[str, Any], prediction: Any) -> float:
            """Generic reward function for the module."""
            try:
                if not prediction:
                    return 0.0
                
                module_name = module_class.__name__.lower()
                
                # Module-specific validation logic
                if "quotation" in module_name:
                    research_objectives = input_kwargs.get("research_objectives", "")
                    transcript_chunk = input_kwargs.get("transcript_chunk", "")
                    quotations = getattr(prediction, "quotations", []) or []
                    
                    # Basic validation for quotation module
                    if not quotations:
                        return 0.0
                        
                    # Check alignment with research objectives
                    if research_objectives and not any(obj.lower() in " ".join(quotations).lower() 
                                                     for obj in research_objectives.split()):
                        return 0.5  # Partial match
                        
                    return 1.0
                    
                elif "keyword" in module_name:
                    keywords = getattr(prediction, "keywords", []) or []
                    return 1.0 if keywords else 0.0
                    
                elif "theme" in module_name:
                    themes = getattr(prediction, "themes", []) or []
                    return 1.0 if themes else 0.0
                
                # Default success if we have any prediction
                return 1.0
                
            except Exception as e:
                logger.error(f"Error in reward function: {e}")
                return 0.0
        
        # Access attributes directly instead of using .get()
        max_retries = getattr(optimizer_config, "max_retries", 3)
        threshold = getattr(optimizer_config, "threshold", 0.9)
        
        # Wrap with BestOfN
        optimized_module = dspy.BestOfN(
            module_instance,
            N=max_retries,
            reward_fn=reward_fn,
            threshold=threshold
        )
        
        logger.info(f"Successfully initialized optimizer for {module_class.__name__}")
        return optimized_module
        
    except Exception as e:
        logger.error(f"Error initializing optimizer for {module_class.__name__}: {e}", exc_info=True)
        raise