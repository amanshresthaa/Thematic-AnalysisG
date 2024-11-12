import logging
import os
from typing import List, Dict, Any
from openai import OpenAI

# Initialize logger
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Wrapper for OpenAI API interactions using the updated SDK.
    """
    def __init__(self, api_key: str):
        """
        Initializes the OpenAI client with the provided API key.
        
        Args:
            api_key (str): OpenAI API key.
        """
        if not api_key:
            logger.error("OpenAI API key is not provided.")
            raise ValueError("OpenAI API key must be provided.")
        self.client = OpenAI(api_key=api_key)
        logger.debug("OpenAI client initialized successfully.")

    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Creates a chat completion using OpenAI's API.
        
        Args:
            model (str): Model name to use.
            messages (List[Dict[str, str]]): List of message dictionaries.
            max_tokens (int): Maximum number of tokens in the response.
            temperature (float): Sampling temperature.
        
        Returns:
            Dict[str, Any]: API response as a dictionary.
        """
        if not model or not messages:
            logger.error("Model and messages must be provided for chat completion.")
            raise ValueError("Model and messages must be provided for chat completion.")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.debug(f"Chat completion created successfully for model '{model}'.")
            return response.model_dump()
        except Exception as e:
            logger.error(f"Error creating chat completion: {e}")
            raise

    def create_embeddings(self, model: str, input: List[str]) -> Dict[str, Any]:
        """
        Creates embeddings for the given input texts using OpenAI's API.
        
        Args:
            model (str): Embedding model to use.
            input (List[str]): List of input texts.
        
        Returns:
            Dict[str, Any]: API response containing embeddings.
        """
        if not model or not input:
            logger.error("Model and input must be provided for creating embeddings.")
            raise ValueError("Model and input must be provided for creating embeddings.")
        
        try:
            response = self.client.embeddings.create(
                model=model,
                input=input
            )
            logger.debug(f"Embeddings created successfully using model '{model}'.")
            return response.model_dump()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
