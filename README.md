# Example Project

## Purpose and Functionality

This project is designed to process and chunk documents, perform hybrid retrieval using semantic search and BM25, and generate answers to queries using DSPy. The main functionalities include:

- Document chunking and processing
- Hybrid retrieval combining semantic search and BM25
- Answer generation using DSPy

## Directory Structure and File Descriptions

### `chunker/`
- `chunker.py`: Contains functions for document chunking and processing.
- `config.json`: Configuration settings for the chunker module in JSON format.
- `config.yaml`: Configuration settings for the chunker module in YAML format.

### `config/`
- `config.json`: Configuration settings for the application in JSON format.
- `logging_config.yaml`: Configuration settings for logging in YAML format.

### `src/`
- `answer_generator.py`: Contains functions for generating answers to queries using DSPy.
- `copycode.py`: Script to copy code files into a consolidated file.
- `contextual_vector_db.py`: Manages the contextual vector database, including embedding generation and storage.
- `data_loader.py`: Functions to load JSON data files.
- `elasticsearch_bm25.py`: Handles Elasticsearch BM25 indexing and searching functionalities.
- `evaluation.py`: Functions to evaluate the retrieval pipeline.
- `evaluator.py`: Functions to evaluate the retrieval pipeline.
- `main.py`: Main script to load data, process queries, and generate outputs.
- `openai_client.py`: Wrapper for OpenAI API interactions.
- `query_processor.py`: Functions to validate and process queries.
- `reranking.py`: Functions to perform hybrid retrieval and reranking using Cohere.
- `retrieval.py`: Functions to perform hybrid retrieval combining semantic search and BM25.
- `utils/logger.py`: Functions to set up logging configuration.

### Root Directory
- `.env`: Environment variables for API keys.
- `.gitattributes`: Git attributes configuration.
- `cache.json`: Cache file.
- `cheatsheet.txt`: Cheatsheet for the project.
- `codebase_chunks.json`: JSON file containing codebase chunks.
- `consolidated_code.txt`: Consolidated code file.
- `consolidated_code2.txt`: Another consolidated code file.
- `data/`: Directory containing various data files.
- `documents/`: Directory containing document files.
- `metricsguide.txt`: Metrics guide for the project.
- `optimizerguide.txt`: Optimizer guide for the project.
- `output_20241021_203650/`: Directory containing output files.
- `query_results_evaluation.jsonl`: JSONL file containing query results for evaluation.
- `query_results_standard.jsonl`: JSONL file containing standard query results.
- `requirement.txt`: List of dependencies required for the project.

## Setup and Running the Project

### Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amanshresthaa/Example.git
   cd Example
   ```

2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirement.txt
   ```

### Running the Project

1. Set up the environment variables:
   - Create a `.env` file in the root directory with the following content:
     ```
     VOYAGE_API_KEY=your_voyage_api_key
     COHERE_API_KEY=your_cohere_api_key
     OPENAI_API_KEY=your_openai_api_key
     ANTHROPIC_API_KEY=your_anthropic_api_key
     ```

2. Run the main script:
   ```bash
   python src/main.py
   ```

This will load the data, process the queries, and generate the outputs as specified in the `main.py` script.
