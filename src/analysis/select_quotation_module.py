import logging
from typing import Dict, Any, List
import dspy

from .select_quotation import SelectQuotationSignature
from ..core.contextual_vector_db import ContextualVectorDB
from ..core.elasticsearch_bm25 import ElasticsearchBM25

logger = logging.getLogger(__name__)

class SelectQuotationModule(dspy.Module):
    def __init__(self, contextual_db: ContextualVectorDB, es_bm25: ElasticsearchBM25):
        self.chain = dspy.TypedChainOfThought(SelectQuotationSignature)
        self.contextual_db = contextual_db
        self.es_bm25 = es_bm25

    def forward(self, research_objectives: str) -> Dict[str, Any]:
        query = research_objectives
        relevant_docs = self.es_bm25.search(query, top_k=20)
        transcript_chunks = [doc['content'] for doc in relevant_docs]

        if not transcript_chunks:
            logger.warning("No relevant transcript chunks found.")
            return {"quotations": [], "purpose": ""}

        response = self.chain(
            research_objectives=research_objectives,
            transcript_chunks=transcript_chunks
        )
        return response