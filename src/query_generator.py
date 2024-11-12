# File: /Users/amankumarshrestha/Downloads/Example/src/query_generator.py
# query_generator.py

import dspy
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class QueryGeneratorSignature(dspy.Signature):
    question: str = dspy.InputField(desc="The original user question.")
    context: str = dspy.InputField(desc="The accumulated context from previous retrievals.")
    new_query: str = dspy.OutputField(desc="The generated query for the next retrieval hop.")

    def forward(self, question: str, context: str) -> Dict[str, str]:
        try:
            if not question or not context:
                raise ValueError("Both 'question' and 'context' must be provided and non-empty.")
            
            prompt = (
                f"Given the question: '{question}'\n"
                f"and the context retrieved so far:\n{context}\n"
                "Generate a search query that will help find additional information needed to answer the question."
            )
            new_query = self.language_model.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                n=1,
                stop=["\n"]
            ).strip()
            logger.info(f"Generated new query: '{new_query}'")
            return {"new_query": new_query}
        except ValueError as ve:
            logger.error(f"ValueError in QueryGeneratorSignature.forward: {ve}", exc_info=True)
            return {"new_query": question}  # Fallback to the original question
        except Exception as e:
            logger.error(f"Error in QueryGeneratorSignature.forward: {e}", exc_info=True)
            return {"new_query": question}  # Fallback to the original question
