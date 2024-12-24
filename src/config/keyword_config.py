# src/config/keyword_config.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class AssertionThresholds:
    """Configuration thresholds for keyword assertions."""
    min_analysis_words: int = 10
    min_keywords_per_r: int = 2
    min_theoretical_alignment: float = 0.7
    min_repetition_count: int = 2
    min_framework_aspects: int = 2
    min_total_keywords: int = 3

@dataclass
class AssertionConfig:
    """Complete configuration for keyword assertions."""
    enabled_assertions: List[str]
    required_passes: int = 3
    thresholds: AssertionThresholds = None
    strict_mode: bool = False
    detailed_logging: bool = True

@dataclass
class KeywordExtractionConfig:
    """Configuration for keyword extraction module."""
    max_keywords: int = 10
    min_confidence: float = 0.7
    batch_size: int = 100
    assertion_config: AssertionConfig = None

@dataclass
class PipelineConfig:
    """Overall pipeline configuration."""
    keyword_config: KeywordExtractionConfig
    optimization_enabled: bool = True
    parallel_processing: bool = True
    cache_results: bool = True
    max_retries: int = 3