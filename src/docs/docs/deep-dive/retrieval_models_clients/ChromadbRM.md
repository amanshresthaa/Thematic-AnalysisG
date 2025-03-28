# ChromadbRM

!!! note
    Adapted from documentation provided by https://github.com/animtel

ChromadbRM have the flexibility from a variety of embedding functions as outlined in the [chromadb embeddings documentation](https://docs.trychroma.com/embeddings). While different options are available, this example demonstrates how to utilize OpenAI embeddings specifically.


## Setting up the ChromadbRM Client

The constructor initializes an instance of the `ChromadbRM` class, with the option to use OpenAI's embeddings or any alternative supported by chromadb, as detailed in the official [chromadb embeddings documentation](https://docs.trychroma.com/guides/embeddings).

- `collection_name` (_str_): The name of the chromadb collection.
- `persist_directory` (_str_): Path to the directory where chromadb data is persisted.
- `embedding_function` (_Optional[EmbeddingFunction[Embeddable]]_, _optional_): The function used for embedding documents and queries. Defaults to `DefaultEmbeddingFunction()` if not specified.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 7.

Example of the ChromadbRM constructor: 

```python
ChromadbRM(
    collection_name: str,
    persist_directory: str,
    embedding_function: Optional[EmbeddingFunction[Embeddable]] = OpenAIEmbeddingFunction(),
    k: int = 7,
)
```

## Under the Hood

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**
- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**
- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Search the chromadb collection for the top `k` passages matching the given query or queries, using embeddings generated via the specified `embedding_function`.

## Sending Retrieval Requests via ChromadbRM Client

```python
from dspy.retrieve.chromadb_rm import ChromadbRM
import os
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name="text-embedding-3-small"
)

retriever_model = ChromadbRM(
    'your_collection_name',
    '/path/to/your/db',
    embedding_function=embedding_function,
    k=5
)

results = retriever_model("Explore the significance of quantum computing", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```