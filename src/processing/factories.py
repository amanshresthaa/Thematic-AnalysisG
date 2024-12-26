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