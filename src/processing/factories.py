# src/processing/factories.py

import dspy
from src.analysis.select_quotation_module import SelectQuotationModule, EnhancedQuotationModule
from src.analysis.extract_keyword_module import KeywordExtractionModule
from src.analysis.coding_module import CodingAnalysisModule
from src.analysis.theme_development_module import ThemedevelopmentAnalysisModule
from src.analysis.grouping_module import GroupingAnalysisModule

from .handlers import (
    QuotationHandler,
    KeywordHandler,
    CodingHandler,
    GroupingHandler,
    ThemeHandler
)
from .base import BaseHandler
from .logger import get_logger
from typing import Type, Dict, Any, List
import inspect

logger = get_logger(__name__)

def get_handler_for_module(module: dspy.Module) -> BaseHandler:
    """
    Returns the appropriate handler instance for the given module.
    Enhanced with detailed logging for debugging purposes.
    """
    module_type = type(module).__name__
    module_class_module = type(module).__module__
    logger.debug(f"Attempting to get handler for module type: {module_type} (defined in {module_class_module})")

    # Add type debug logging
    logger.debug(f"Module type hierarchy: {type(module).__mro__}")
    
    # Check for BestOfN first, since it's a wrapper
    if isinstance(module, dspy.BestOfN):
        return BestOfNHandler(module)
    
    # More explicit type checking
    if module_type in ['SelectQuotationModule', 'EnhancedQuotationModule'] or \
       isinstance(module, (SelectQuotationModule, EnhancedQuotationModule)):
        logger.debug(f"Handler found: QuotationHandler for module type: {module_type}")
        return QuotationHandler()
    elif module_type == 'KeywordExtractionModule' or isinstance(module, KeywordExtractionModule):
        logger.debug(f"Handler found: KeywordHandler for module type: {module_type}")
        return KeywordHandler()
    elif module_type == 'CodingAnalysisModule' or isinstance(module, CodingAnalysisModule):
        logger.debug(f"Handler found: CodingHandler for module type: {module_type}")
        return CodingHandler()
    elif module_type == 'GroupingAnalysisModule' or isinstance(module, GroupingAnalysisModule):
        logger.debug(f"Handler found: GroupingHandler for module type: {module_type}")
        return GroupingHandler()
    elif module_type == 'ThemedevelopmentAnalysisModule' or isinstance(module, ThemedevelopmentAnalysisModule):
        logger.debug(f"Handler found: ThemeHandler for module type: {module_type}")
        return ThemeHandler()
    else:
        logger.error(
            f"Unsupported module type: {module_type} (defined in {module_class_module})\n"
            f"Module MRO: {type(module).__mro__}"
        )
        raise ValueError(f"Unsupported module type: {module_type}")

# Add this new handler for BestOfN
class BestOfNHandler(BaseHandler):
    """Handler for BestOfN module that delegates to appropriate handler for the inner module."""
    
    def __init__(self, module):
        super().__init__(module)
        # Extract the inner module from BestOfN
        self.inner_module = module.program
        # Try to get the appropriate handler for the inner module
        try:
            self.inner_handler = get_handler_for_module(self.inner_module)
        except ValueError:
            logger.warning(f"Could not find handler for inner module {type(self.inner_module).__name__}. Using BaseHandler.")
            self.inner_handler = BaseHandler(self.inner_module)
    
    async def process_query(self, query_data):
        """Process query using the BestOfN module, which will handle retries internally."""
        return await self.inner_handler.process_query(query_data)
        
    async def process_single_transcript(
        self,
        transcript_item: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        module: dspy.Module
    ) -> Dict[str, Any]:
        """
        Delegate processing to the inner handler's process_single_transcript method.
        BestOfN's retry logic is handled internally by DSPy when the module is called.
        """
        # We call the inner handler's process_single_transcript method
        # but pass the BestOfN wrapper module which will handle retries
        return await self.inner_handler.process_single_transcript(
            transcript_item=transcript_item,
            retrieved_docs=retrieved_docs,
            module=module
        )