# src/pipeline/pipeline_configs.py

from dataclasses import dataclass
from typing import Any, Callable, Type, Optional

@dataclass
class OptimizerConfig:
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    num_candidate_programs: int = 10
    num_threads: int = 1

@dataclass
class ModuleConfig:
    index_name: str
    codebase_chunks_file: str
    queries_file_standard: str
    evaluation_set_file: str
    output_filename_primary: str
    training_data: str
    optimized_program_path: str
    module_class: Type[Any]
    conversion_func: Optional[Callable[[str, str, str], None]] = None