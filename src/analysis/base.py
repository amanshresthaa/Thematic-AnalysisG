# src/analysis/base.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import json
import asyncio

logger = logging.getLogger(__name__)

class BaseAnalysisModule(ABC):
    """
    Abstract base class for all analysis modules.
    Provides common functionality for JSON handling, error recovery,
    prompt management and retry mechanisms.
    """
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the module."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def execute_with_retry(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Dict[str, Any]: The function result or error response
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{self.max_retries}")
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                wait_time = self.backoff_factor ** attempt
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}. "
                    f"Retrying in {wait_time:.1f}s"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All retry attempts failed: {str(e)}")
                    return {"error": str(e)}
    
    def validate_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        Validate and parse JSON string.
        
        Args:
            json_str: JSON string to validate
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON or None if invalid
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return None
    
    def format_error_response(self, error: Exception) -> Dict[str, Any]:
        """
        Format an error response.
        
        Args:
            error: The exception to format
            
        Returns:
            Dict[str, Any]: Formatted error response
        """
        return {
            "error": True,
            "message": str(error),
            "type": error.__class__.__name__
        }

    @abstractmethod
    async def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        pass

    @abstractmethod
    def generate_prompt(self, **kwargs) -> str:
        """
        Generate prompt for the analysis.
        
        Args:
            **kwargs: Parameters for prompt generation
            
        Returns:
            str: Generated prompt
        """
        pass

    @abstractmethod
    async def process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate analysis results.
        
        Args:
            results: Raw analysis results to process
            
        Returns:
            Dict[str, Any]: Processed results
        """
        pass

    @abstractmethod
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Perform the analysis.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        pass

    async def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete analysis workflow with error handling.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dict[str, Any]: Analysis results or error response
        """
        start_time = time.time()
        self.logger.info("Starting analysis workflow")
        
        try:
            # Validate inputs
            if not await self.validate_input(**kwargs):
                raise ValueError("Invalid input parameters")

            # Generate prompt
            prompt = self.generate_prompt(**kwargs)
            
            # Execute analysis
            results = await self.execute_with_retry(self.analyze, prompt=prompt, **kwargs)
            
            # Process results
            final_results = await self.process_results(results)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Analysis completed in {elapsed_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return self.format_error_response(e)