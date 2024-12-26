# src/processing/validators.py

from typing import List, Dict, Any
from .base import BaseValidator
from .logger import get_logger

import dspy

logger = get_logger(__name__)

# Import your actual modules so we can check their types
from src.analysis.select_quotation_module import SelectQuotationModule, EnhancedQuotationModule
from src.analysis.extract_keyword_module import KeywordExtractionModule
from src.analysis.coding_module import CodingAnalysisModule
from src.analysis.theme_development_module import ThemedevelopmentAnalysisModule
from src.analysis.grouping_module import GroupingAnalysisModule


class QuotationValidator(BaseValidator):
    """
    Validator for SelectQuotationModule and EnhancedQuotationModule.
    Checks for a valid 'transcript_chunk'.
    """

    def validate(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_transcripts = []
        for idx, transcript in enumerate(transcripts):
            if 'transcript_chunk' not in transcript or not isinstance(transcript['transcript_chunk'], str) or not transcript['transcript_chunk'].strip():
                logger.warning(f"Transcript at index {idx} missing or invalid 'transcript_chunk'. Skipping.")
                continue
            valid_transcripts.append(transcript)
        logger.info(f"QuotationValidator: Validated {len(valid_transcripts)}/{len(transcripts)} transcripts.")
        return valid_transcripts


class KeywordValidator(BaseValidator):
    """
    Validator for KeywordExtractionModule.
    Checks for a valid 'quotation'.
    """

    def validate(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_transcripts = []
        for idx, transcript in enumerate(transcripts):
            if 'quotation' not in transcript or not isinstance(transcript['quotation'], str) or not transcript['quotation'].strip():
                logger.warning(f"Transcript at index {idx} missing or invalid 'quotation'. Skipping.")
                continue
            valid_transcripts.append(transcript)
        logger.info(f"KeywordValidator: Validated {len(valid_transcripts)}/{len(transcripts)} transcripts.")
        return valid_transcripts


class CodingValidator(BaseValidator):
    """
    Validator for CodingAnalysisModule.
    Checks for a valid 'quotation' and valid 'keywords'.
    """

    def validate(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_transcripts = []
        for idx, transcript in enumerate(transcripts):
            required_string_fields = ['quotation']
            required_list_fields = ['keywords']
            missing_fields = []

            for field in required_string_fields:
                if field not in transcript or not isinstance(transcript[field], str) or not transcript[field].strip():
                    missing_fields.append(field)

            for field in required_list_fields:
                if field not in transcript or not isinstance(transcript[field], list) \
                   or not all(isinstance(kw, str) and kw.strip() for kw in transcript[field]):
                    missing_fields.append(field)

            if missing_fields:
                logger.warning(f"Transcript at index {idx} missing required fields {missing_fields}. Skipping.")
                continue

            valid_transcripts.append(transcript)

        logger.info(f"CodingValidator: Validated {len(valid_transcripts)}/{len(transcripts)} transcripts.")
        return valid_transcripts


class ThemeValidator(BaseValidator):
    """
    Validator for ThemedevelopmentAnalysisModule.
    Checks for a valid 'quotation', 'keywords', and 'codes'.
    """

    def validate(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_transcripts = []
        for idx, transcript in enumerate(transcripts):
            if 'quotation' not in transcript or not isinstance(transcript['quotation'], str) or not transcript['quotation'].strip():
                logger.warning(f"Transcript at index {idx} missing or invalid 'quotation'. Skipping.")
                continue
            if 'keywords' not in transcript or not isinstance(transcript['keywords'], list) or not transcript['keywords']:
                logger.warning(f"Transcript at index {idx} missing or invalid 'keywords'. Skipping.")
                continue
            if 'codes' not in transcript or not isinstance(transcript['codes'], list) or not transcript['codes']:
                logger.warning(f"Transcript at index {idx} missing or invalid 'codes'. Skipping.")
                continue
            valid_transcripts.append(transcript)

        logger.info(f"ThemeValidator: Validated {len(valid_transcripts)}/{len(transcripts)} transcripts.")
        return valid_transcripts


class GroupingValidator(BaseValidator):
    """
    Validator for GroupingAnalysisModule.
    Checks for valid 'codes'.
    """

    def validate(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_transcripts = []
        for idx, transcript in enumerate(transcripts):
            if 'codes' not in transcript or not isinstance(transcript['codes'], list) or not transcript['codes']:
                logger.warning(f"Transcript at index {idx} missing or invalid 'codes' for grouping. Skipping.")
                continue
            valid_transcripts.append(transcript)

        logger.info(f"GroupingValidator: Validated {len(valid_transcripts)}/{len(transcripts)} transcripts.")
        return valid_transcripts


def get_validator_for_module(module: dspy.Module) -> BaseValidator:
    """
    Returns the appropriate validator instance for the given module.
    """
    if isinstance(module, (SelectQuotationModule, EnhancedQuotationModule)):
        return QuotationValidator()
    elif isinstance(module, KeywordExtractionModule):
        return KeywordValidator()
    elif isinstance(module, CodingAnalysisModule):
        return CodingValidator()
    elif isinstance(module, ThemedevelopmentAnalysisModule):
        return ThemeValidator()
    elif isinstance(module, GroupingAnalysisModule):
        return GroupingValidator()
    else:
        logger.warning("No specific validator found for given module. Returning a pass-through validator.")
        return PassThroughValidator()


class PassThroughValidator(BaseValidator):
    """
    A pass-through validator that doesn't filter out anything.
    Used as a fallback for unknown module types.
    """
    def validate(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return transcripts
