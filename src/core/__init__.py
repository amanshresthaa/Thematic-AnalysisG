# File: __init__.py
# ------------------------------------------------------------------------------
"""
Core functionality for the thematic analysis package.
Contains database and client implementations.
"""

from .contextual_vector_db import ContextualVectorDB
from .elasticsearch_bm25 import ElasticsearchBM25
from .openai_client import OpenAIClient

__all__ = ['ContextualVectorDB', 'ElasticsearchBM25', 'OpenAIClient']