# processing/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import dspy


class BaseValidator(ABC):
    """
    Abstract base class for validators.
    Each validator must implement a validate method
    that returns a filtered list of transcripts.
    """

    @abstractmethod
    def validate(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass


class BaseHandler(ABC):
    """
    Abstract base class for handlers (processing logic).
    Each handler must implement a `process_single_transcript`
    method that takes a transcript_item, a list of retrieved docs, and a dspy.Module.
    """

    @abstractmethod
    async def process_single_transcript(
        self,
        transcript_item: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        module: dspy.Module
    ) -> Dict[str, Any]:
        pass
