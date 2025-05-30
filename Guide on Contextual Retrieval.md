
Enhancing RAG with Contextual Retrieval
Note: For more background information on Contextual Retrieval, including additional performance evaluations on various datasets, we recommend reading our accompanying blog post.

Retrieval Augmented Generation (RAG) enables Claude to leverage your internal knowledge bases, codebases, or any other corpus of documents when providing a response. Enterprises are increasingly building RAG applications to improve workflows in customer support, Q&A over internal company documents, financial & legal analysis, code generation, and much more.

In a separate guide, we walked through setting up a basic retrieval system, demonstrated how to evaluate its performance, and then outlined a few techniques to improve performance. In this guide, we present a technique for improving retrieval performance: Contextual Embeddings.

In traditional RAG, documents are typically split into smaller chunks for efficient retrieval. While this approach works well for many applications, it can lead to problems when individual chunks lack sufficient context. Contextual Embeddings solve this problem by adding relevant context to each chunk before embedding. This method improves the quality of each embedded chunk, allowing for more accurate retrieval and thus better overall performance. Averaged across all data sources we tested, Contextual Embeddings reduced the top-20-chunk retrieval failure rate by 35%.

The same chunk-specific context can also be used with BM25 search to further improve retrieval performance. We introduce this technique in the “Contextual BM25” section.

In this guide, we'll demonstrate how to build and optimize a Contextual Retrieval system using a dataset of 9 codebases as our knowledge base. We'll walk through:

Setting up a basic retrieval pipeline to establish a baseline for performance.

Contextual Embeddings: what it is, why it works, and how prompt caching makes it practical for production use cases.

Implementing Contextual Embeddings and demonstrating performance improvements.

Contextual BM25: improving performance with contextual BM25 hybrid search.

Improving performance with reranking,

Evaluation Metrics & Dataset:
We use a pre-chunked dataset of 9 codebases - all of which have been chunked according to a basic character splitting mechanism. Our evaluation dataset contains 248 queries - each of which contains a 'golden chunk.' We'll use a metric called Pass@k to evaluate performance. Pass@k checks whether or not the 'golden document' was present in the first k documents retrieved for each query. Contextual Embeddings in this case helped us to improve Pass@10 performance from ~87% --> ~95%.

You can find the code files and their chunks in data/codebase_chunks.json and the evaluation dataset in data/evaluation_set.jsonl

Additional Notes:
Prompt caching is helpful in managing costs when using this retrieval method. This feature is currently available on Anthropic's 1P API, and is coming soon to our 3P partner environments in AWS Bedrock and GCP Vertex. We know that many of our customers leverage AWS Knowledge Bases and GCP Vertex AI APIs when building RAG solutions, and this method can be used on either platform with a bit of customization. Consider reaching out to Anthropic or your AWS/GCP account team for guidance on this!

To make it easier to use this method on Bedrock, the AWS team has provided us with code that you can use to implement a Lambda function that adds context to each document. If you deploy this Lambda function, you can select it as a custom chunking option when configuring a Bedrock Knowledge Base. You can find this code in contextual-rag-lambda-function. The main lambda function code is in lambda_function.py.

Table of Contents
Setup

Basic RAG

Contextual Embeddings

Contextual BM25

Reranking

Setup
We'll need a few libraries, including:

anthropic - to interact with Claude

voyageai - to generate high quality embeddings

cohere - for reranking

elasticsearch for performant BM25 search

pandas, numpy, matplotlib, and scikit-learn for data manipulation and visualization

You'll also need API keys from Anthropic, Voyage AI, and Cohere

!pip install anthropic
!pip install voyageai
!pip install cohere
!pip install elasticsearch
!pip install pandas
!pip install numpy
import os

os.environ['VOYAGE_API_KEY'] = "YOUR KEY HERE"
os.environ['ANTHROPIC_API_KEY'] = "YOUR KEY HERE"
os.environ['COHERE_API_KEY'] = "YOUR KEY HERE"
import anthropic

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
Initialize a Vector DB Class
In this example, we're using an in-memory vector DB, but for a production application, you may want to use a hosted solution.

import os
import pickle
import json
import numpy as np
import voyageai
from typing import List, Dict, Any
from tqdm import tqdm

class VectorDB:
    def __init__(self, name: str, api_key = None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, dataset: List[Dict[str, Any]]):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc['chunks']) for doc in dataset)
        
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for doc in dataset:
                for chunk in doc['chunks']:
                    texts_to_embed.append(chunk['content'])
                    metadata.append({
                        'doc_id': doc['doc_id'],
                        'original_uuid': doc['original_uuid'],
                        'chunk_id': chunk['chunk_id'],
                        'original_index': chunk['original_index'],
                        'content': chunk['content']
                    })
                    pbar.update(1)

        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()
        
        print(f"Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")

    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 128
        with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
            result = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_result = self.client.embed(batch, model="voyage-2").embeddings
                result.extend(batch_result)
                pbar.update(len(batch))
        
        self.embeddings = result
        self.metadata = data

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])

    def validate_embedded_chunks(self):
        unique_contents = set()
        for meta in self.metadata:
            unique_contents.add(meta['content'])
    
        print(f"Validation results:")
        print(f"Total embedded chunks: {len(self.metadata)}")
        print(f"Unique embedded contents: {len(unique_contents)}")
    
        if len(self.metadata) != len(unique_contents):
            print("Warning: There may be duplicate chunks in the embedded data.")
        else:
            print("All embedded chunks are unique.")
# Load your transformed dataset
with open('data/codebase_chunks.json', 'r') as f:
    transformed_dataset = json.load(f)

# Initialize the VectorDB
base_db = VectorDB("base_db")

# Load and process the data
base_db.load_data(transformed_dataset)
Basic RAG
To get started, we'll set up a basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

Chunk documents by heading - containing only the content from each subheading

Embed each document

Use Cosine similarity to retrieve documents in order to answer query

import json
from typing import List, Dict, Any, Callable, Union
from tqdm import tqdm

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def evaluate_retrieval(queries: List[Dict[str, Any]], retrieval_function: Callable, db, k: int = 20) -> Dict[str, float]:
    total_score = 0
    total_queries = len(queries)
    
    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = query_item['query']
        golden_chunk_uuids = query_item['golden_chunk_uuids']
        
        # Find all golden chunk contents
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)
            if not golden_doc:
                print(f"Warning: Golden document not found for UUID {doc_uuid}")
                continue
            
            golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index), None)
            if not golden_chunk:
                print(f"Warning: Golden chunk not found for index {chunk_index} in document {doc_uuid}")
                continue
            
            golden_contents.append(golden_chunk['content'].strip())
        
        if not golden_contents:
            print(f"Warning: No golden contents found for query: {query}")
            continue
        
        retrieved_docs = retrieval_function(query, db, k=k)
        
        # Count how many golden chunks are in the top k retrieved documents
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[:k]:
                retrieved_content = doc['metadata'].get('original_content', doc['metadata'].get('content', '')).strip()
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break
        
        query_score = chunks_found / len(golden_contents)
        total_score += query_score
    
    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {
        "pass_at_n": pass_at_n,
        "average_score": average_score,
        "total_queries": total_queries
    }

def retrieve_base(query: str, db, k: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using either VectorDB or ContextualVectorDB.
    
    :param query: The query string
    :param db: The VectorDB or ContextualVectorDB instance
    :param k: Number of top results to retrieve
    :return: List of retrieved documents
    """
    return db.search(query, k=k)

def evaluate_db(db, original_jsonl_path: str, k):
    # Load the original JSONL data for queries and ground truth
    original_data = load_jsonl(original_jsonl_path)
    
    # Evaluate retrieval
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Total Score: {results['average_score']}")
    print(f"Total queries: {results['total_queries']}")
results5 = evaluate_db(base_db, 'data/evaluation_set.jsonl', 5)
results10 = evaluate_db(base_db, 'data/evaluation_set.jsonl', 10)
results20 = evaluate_db(base_db, 'data/evaluation_set.jsonl', 20)
Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 40.70it/s]
Pass@5: 80.92%
Total Score: 0.8091877880184332
Total queries: 248
Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 39.50it/s]
Pass@10: 87.15%
Total Score: 0.8714957757296468
Total queries: 248
Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 39.43it/s]
Pass@20: 90.06%
Total Score: 0.9006336405529954
Total queries: 248
Contextual Embeddings
With basic RAG, each embedded chunk contains a potentially useful piece of information, but these chunks lack context. With Contextual Embeddings, we create a variation on the embedding itself by adding more context to each text chunk before embedding it. Specifically, we use Claude to create a concise context that explains the chunk using the context of the overall document. In the case of our codebases dataset, we can provide both the chunk and the full file that each chunk was found within to an LLM, then produce the context. Then, we will combine this 'context' and the raw text chunk together into a single text block prior to creating each embedding.

Additional Considerations: Cost and Latency
The extra work we're doing to 'situate' each document happens only at ingestion time: it's a cost you'll pay once when you store each document (and periodically in the future if you have a knowledge base that updates over time). There are many approaches like HyDE (hypothetical document embeddings) which involve performing steps to improve the representation of the query prior to executing a search. These techniques have shown to be moderately effective, but they add significant latency at runtime.

Prompt caching also makes this much more cost effective. Creating contextual embeddings requires us to pass the same document to the model for every chunk we want to generate extra context for. With prompt caching, we can write the overall doc to the cache once, and then because we're doing our ingestion job all in sequence, we can just read the document from cache as we generate context for each chunk within that document (the information you write to the cache has a 5 minute time to live). This means that the first time we pass a document to the model, we pay a bit more to write it to the cache, but for each subsequent API call that contains that doc, we receive a 90% discount on all of the input tokens read from the cache. Assuming 800 token chunks, 8k token documents, 50 token context instructions, and 100 tokens of context per chunk, the cost to generate contextualized chunks is $1.02 per million document tokens.

When you load data into your ContextualVectorDB below, you'll see in logs just how big this impact is.

Warning: some smaller embedding models have a fixed input token limit. Contextualizing the chunk makes it longer, so if you notice much worse performance from contextualized embeddings, the contextualized chunk is likely getting truncated

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

def situate_context(doc: str, chunk: str) -> str:
    response = client.beta.prompt_caching.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                        "cache_control": {"type": "ephemeral"} #we will make use of prompt caching for the full documents
                    },
                    {
                        "type": "text",
                        "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                    }
                ]
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )
    return response

jsonl_data = load_jsonl('data/evaluation_set.jsonl')
# Example usage
doc_content = jsonl_data[0]['golden_documents'][0]['content']
chunk_content = jsonl_data[0]['golden_chunks'][0]['content']

response = situate_context(doc_content, chunk_content)
print(f"Situated context: {response.content[0].text}")

# Print cache performance metrics
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Cache creation input tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read input tokens: {response.usage.cache_read_input_tokens}")
Situated context: This chunk describes the `DiffExecutor` struct, which is an executor for differential fuzzing. It wraps two executors that are run sequentially with the same input, and also runs the secondary executor in the `run_target` method.
Input tokens: 366
Output tokens: 55
Cache creation input tokens: 3046
Cache read input tokens: 0
import os
import pickle
import json
import numpy as np
import voyageai
from typing import List, Dict, Any
from tqdm import tqdm
import anthropic
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class ContextualVectorDB:
    def __init__(self, name: str, voyage_api_key=None, anthropic_api_key=None):
        if voyage_api_key is None:
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"

        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response = self.anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {"type": "ephemeral"} #we will make use of prompt caching for the full documents
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        },
                    ]
                },
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        return response.content[0].text, response.usage

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 1):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc['chunks']) for doc in dataset)

        def process_chunk(doc, chunk):
            #for each chunk, produce the context
            contextualized_text, usage = self.situate_context(doc['content'], chunk['content'])
            with self.token_lock:
                self.token_counts['input'] += usage.input_tokens
                self.token_counts['output'] += usage.output_tokens
                self.token_counts['cache_read'] += usage.cache_read_input_tokens
                self.token_counts['cache_creation'] += usage.cache_creation_input_tokens
            
            return {
                #append the context to the original text chunk
                'text_to_embed': f"{chunk['content']}\n\n{contextualized_text}",
                'metadata': {
                    'doc_id': doc['doc_id'],
                    'original_uuid': doc['original_uuid'],
                    'chunk_id': chunk['chunk_id'],
                    'original_index': chunk['original_index'],
                    'original_content': chunk['content'],
                    'contextualized_content': contextualized_text
                }
            }

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in dataset:
                for chunk in doc['chunks']:
                    futures.append(executor.submit(process_chunk, doc, chunk))
            
            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])

        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()

        #logging token usage
        print(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")
        print(f"Total input tokens without caching: {self.token_counts['input']}")
        print(f"Total output tokens: {self.token_counts['output']}")
        print(f"Total input tokens written to cache: {self.token_counts['cache_creation']}")
        print(f"Total input tokens read from cache: {self.token_counts['cache_read']}")
        
        total_tokens = self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation']
        savings_percentage = (self.token_counts['cache_read'] / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"Total input token savings from prompt caching: {savings_percentage:.2f}% of all input tokens used were read from cache.")
        print("Tokens read from cache come at a 90 percent discount!")

    #we use voyage AI here for embeddings. Read more here: https://docs.voyageai.com/docs/embeddings
    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 128
        result = [
            self.voyage_client.embed(
                texts[i : i + batch_size],
                model="voyage-2"
            ).embeddings
            for i in range(0, len(texts), batch_size)
        ]
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = data

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])
# Load the transformed dataset
with open('data/codebase_chunks.json', 'r') as f:
    transformed_dataset = json.load(f)

# Initialize the ContextualVectorDB
contextual_db = ContextualVectorDB("my_contextual_db")

# Load and process the data
#note: consider increasing the number of parallel threads to run this faster, or reducing the number of parallel threads if concerned about hitting your API rate limit
contextual_db.load_data(transformed_dataset, parallel_threads=5)
Processing 737 chunks with 5 threads
Processing chunks: 100%|██████████| 737/737 [02:37<00:00,  4.69it/s]
Contextual Vector database loaded and saved. Total chunks processed: 737
Total input tokens without caching: 500383
Total output tokens: 40318
Total input tokens written to cache: 341422
Total input tokens read from cache: 2825073
Total input token savings from prompt caching: 77.04% of all input tokens used were read from cache.
Tokens read from cache come at a 90 percent discount!
r5 = evaluate_db(contextual_db, 'data/evaluation_set.jsonl', 5)
r10 = evaluate_db(contextual_db, 'data/evaluation_set.jsonl', 10)
r20 = evaluate_db(contextual_db, 'data/evaluation_set.jsonl', 20)
Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 39.53it/s]
Pass@5: 86.37%
Total Score: 0.8637192780337941
Total queries: 248
Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 40.05it/s]
Pass@10: 92.81%
Total Score: 0.9280913978494625
Total queries: 248
Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 39.64it/s]
Pass@20: 93.78%
Total Score: 0.9378360215053763
Total queries: 248
Contextual BM25
Contextual embeddings is an improvement on traditional semantic search RAG, but we can improve performance further. In this section we'll show you how you can use contextual embeddings and contextual BM25 together. While you can see performance gains by pairing these techniques together without the context, adding context to these methods reduces the top-20-chunk retrieval failure rate by 42%.

BM25 is a probabilistic ranking function that improves upon TF-IDF. It scores documents based on query term frequency, while accounting for document length and term saturation. BM25 is widely used in modern search engines for its effectiveness in ranking relevant documents. For more details, see this blog post. We'll use elastic search for the BM25 portion of this section, which will require you to have the elasticsearch library installed and it will also require you to spin up an Elasticsearch server in the background. The easiest way to do this is to install docker and run the following docker command:

docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0

One difference between a typical BM25 search and what we'll do in this section is that, for each chunk, we'll run each BM25 search on both the chunk content and the additional context that we generated in the previous section. From there, we'll use a technique called reciprocal rank fusion to merge the results from our BM25 search with our semantic search results. This allows us to perform a hybrid search across both our BM25 corpus and vector DB to return the most optimal documents for a given query.

In the function below, we allow you the option to add weightings to the semantic search and BM25 search documents as you merge them with Reciprocal Rank Fusion. By default, we set these to 0.8 for the semantic search results and 0.2 to the BM25 results. We'd encourage you to experiment with different values here.

import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class ElasticsearchBM25:
    def __init__(self, index_name: str = "contextual_bm25_index"):
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False  # Disable query cache
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                }
            },
        }
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created index: {self.index_name}")

    def index_documents(self, documents: List[Dict[str, Any]]):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc["original_content"],
                    "contextualized_content": doc["contextualized_content"],
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "original_index": doc["original_index"],
                },
            }
            for doc in documents
        ]
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        self.es_client.indices.refresh(index=self.index_name)  # Force refresh before each search
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            "size": k,
        }
        response = self.es_client.search(index=self.index_name, body=search_body)
        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]
    
def create_elasticsearch_bm25_index(db: ContextualVectorDB):
    es_bm25 = ElasticsearchBM25()
    es_bm25.index_documents(db.metadata)
    return es_bm25

def retrieve_advanced(query: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int, semantic_weight: float = 0.8, bm25_weight: float = 0.2):
    num_chunks_to_recall = 150

    # Semantic search
    semantic_results = db.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids = [(result['metadata']['doc_id'], result['metadata']['original_index']) for result in semantic_results]

    # BM25 search using Elasticsearch
    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)
    ranked_bm25_chunk_ids = [(result['doc_id'], result['original_index']) for result in bm25_results]

    # Combine results
    chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
    chunk_id_to_score = {}

    # Initial scoring with weights
    for chunk_id in chunk_ids:
        score = 0
        if chunk_id in ranked_chunk_ids:
            index = ranked_chunk_ids.index(chunk_id)
            score += semantic_weight * (1 / (index + 1))  # Weighted 1/n scoring for semantic
        if chunk_id in ranked_bm25_chunk_ids:
            index = ranked_bm25_chunk_ids.index(chunk_id)
            score += bm25_weight * (1 / (index + 1))  # Weighted 1/n scoring for BM25
        chunk_id_to_score[chunk_id] = score

    # Sort chunk IDs by their scores in descending order
    sorted_chunk_ids = sorted(
        chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True
    )

    # Assign new scores based on the sorted order
    for index, chunk_id in enumerate(sorted_chunk_ids):
        chunk_id_to_score[chunk_id] = 1 / (index + 1)

    # Prepare the final results
    final_results = []
    semantic_count = 0
    bm25_count = 0
    for chunk_id in sorted_chunk_ids[:k]:
        chunk_metadata = next(chunk for chunk in db.metadata if chunk['doc_id'] == chunk_id[0] and chunk['original_index'] == chunk_id[1])
        is_from_semantic = chunk_id in ranked_chunk_ids
        is_from_bm25 = chunk_id in ranked_bm25_chunk_ids
        final_results.append({
            'chunk': chunk_metadata,
            'score': chunk_id_to_score[chunk_id],
            'from_semantic': is_from_semantic,
            'from_bm25': is_from_bm25
        })
        
        if is_from_semantic and not is_from_bm25:
            semantic_count += 1
        elif is_from_bm25 and not is_from_semantic:
            bm25_count += 1
        else:  # it's in both
            semantic_count += 0.5
            bm25_count += 0.5

    return final_results, semantic_count, bm25_count

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def evaluate_db_advanced(db: ContextualVectorDB, original_jsonl_path: str, k: int):
    original_data = load_jsonl(original_jsonl_path)
    es_bm25 = create_elasticsearch_bm25_index(db)
    
    try:
        # Warm-up queries
        warm_up_queries = original_data[:10]
        for query_item in warm_up_queries:
            _ = retrieve_advanced(query_item['query'], db, es_bm25, k)
        
        total_score = 0
        total_semantic_count = 0
        total_bm25_count = 0
        total_results = 0
        
        for query_item in tqdm(original_data, desc="Evaluating retrieval"):
            query = query_item['query']
            golden_chunk_uuids = query_item['golden_chunk_uuids']
            
            golden_contents = []
            for doc_uuid, chunk_index in golden_chunk_uuids:
                golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)
                if golden_doc:
                    golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index), None)
                    if golden_chunk:
                        golden_contents.append(golden_chunk['content'].strip())
            
            if not golden_contents:
                print(f"Warning: No golden contents found for query: {query}")
                continue
            
            retrieved_docs, semantic_count, bm25_count = retrieve_advanced(query, db, es_bm25, k)
            
            chunks_found = 0
            for golden_content in golden_contents:
                for doc in retrieved_docs[:k]:
                    retrieved_content = doc['chunk']['original_content'].strip()
                    if retrieved_content == golden_content:
                        chunks_found += 1
                        break
            
            query_score = chunks_found / len(golden_contents)
            total_score += query_score
            
            total_semantic_count += semantic_count
            total_bm25_count += bm25_count
            total_results += len(retrieved_docs)
        
        total_queries = len(original_data)
        average_score = total_score / total_queries
        pass_at_n = average_score * 100
        
        semantic_percentage = (total_semantic_count / total_results) * 100 if total_results > 0 else 0
        bm25_percentage = (total_bm25_count / total_results) * 100 if total_results > 0 else 0
        
        results = {
            "pass_at_n": pass_at_n,
            "average_score": average_score,
            "total_queries": total_queries
        }
        
        print(f"Pass@{k}: {pass_at_n:.2f}%")
        print(f"Average Score: {average_score:.2f}")
        print(f"Total queries: {total_queries}")
        print(f"Percentage of results from semantic search: {semantic_percentage:.2f}%")
        print(f"Percentage of results from BM25: {bm25_percentage:.2f}%")
        
        return results, {"semantic": semantic_percentage, "bm25": bm25_percentage}
    
    finally:
        # Delete the Elasticsearch index
        if es_bm25.es_client.indices.exists(index=es_bm25.index_name):
            es_bm25.es_client.indices.delete(index=es_bm25.index_name)
            print(f"Deleted Elasticsearch index: {es_bm25.index_name}")
results5 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 5)
results10 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 10)
results20 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 20)
Created index: contextual_bm25_index
Evaluating retrieval: 100%|██████████| 248/248 [00:08<00:00, 28.36it/s]
Pass@5: 86.43%
Average Score: 0.86
Total queries: 248
Percentage of results from semantic search: 55.12%
Percentage of results from BM25: 44.88%
Deleted Elasticsearch index: contextual_bm25_index
Created index: contextual_bm25_index
Evaluating retrieval: 100%|██████████| 248/248 [00:08<00:00, 28.02it/s]
Pass@10: 93.21%
Average Score: 0.93
Total queries: 248
Percentage of results from semantic search: 58.35%
Percentage of results from BM25: 41.65%
Deleted Elasticsearch index: contextual_bm25_index
Created index: contextual_bm25_index
Evaluating retrieval: 100%|██████████| 248/248 [00:08<00:00, 28.15it/s]
Pass@20: 94.99%
Average Score: 0.95
Total queries: 248
Percentage of results from semantic search: 61.94%
Percentage of results from BM25: 38.06%
Deleted Elasticsearch index: contextual_bm25_index
Adding a Reranking Step
If you want to improve performance further, we recommend adding a re-ranking step. When using a re-ranker, you can retrieve more documents initially from your vector store, then use your re-ranker to select a subset of these documents. One common technique is to use re-ranking as a way to implement high precision hybrid search. You can use a combination of semantic search and keyword based search in your initial retrieval step (as we have done earlier in this guide), then use a re-ranking step to choose only the k most relevant docs from a combined list of documents returned by your semantic search and keyword search systems.

Below, we'll demonstrate only the re-ranking step (skipping the hybrid search technique for now). You'll see that we retrieve 10x the number of documents than the number of final k documents we want to retrieve, then use a re-ranking model from Cohere to select the 10 most relevant results from that list. Adding the re-ranking step delivers a modest additional gain in performance. In our case, Pass@10 improves from 92.81% --> 94.79%.

import cohere
from typing import List, Dict, Any, Callable
import json
from tqdm import tqdm

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def chunk_to_content(chunk: Dict[str, Any]) -> str:
    original_content = chunk['metadata']['original_content']
    contextualized_content = chunk['metadata']['contextualized_content']
    return f"{original_content}\n\nContext: {contextualized_content}" 

def retrieve_rerank(query: str, db, k: int) -> List[Dict[str, Any]]:
    co = cohere.Client( os.getenv("COHERE_API_KEY"))
    
    # Retrieve more results than we normally would
    semantic_results = db.search(query, k=k*10)
    
    # Extract documents for reranking, using the contextualized content
    documents = [chunk_to_content(res) for res in semantic_results]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=k
    )
    time.sleep(0.1)
    
    final_results = []
    for r in response.results:
        original_result = semantic_results[r.index]
        final_results.append({
            "chunk": original_result['metadata'],
            "score": r.relevance_score
        })
    
    return final_results

def evaluate_retrieval_rerank(queries: List[Dict[str, Any]], retrieval_function: Callable, db, k: int = 20) -> Dict[str, float]:
    total_score = 0
    total_queries = len(queries)
    
    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = query_item['query']
        golden_chunk_uuids = query_item['golden_chunk_uuids']
        
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)
            if golden_doc:
                golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index), None)
                if golden_chunk:
                    golden_contents.append(golden_chunk['content'].strip())
        
        if not golden_contents:
            print(f"Warning: No golden contents found for query: {query}")
            continue
        
        retrieved_docs = retrieval_function(query, db, k)
        
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[:k]:
                retrieved_content = doc['chunk']['original_content'].strip()
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break
        
        query_score = chunks_found / len(golden_contents)
        total_score += query_score
    
    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {
        "pass_at_n": pass_at_n,
        "average_score": average_score,
        "total_queries": total_queries
    }

def evaluate_db_advanced(db, original_jsonl_path, k):
    original_data = load_jsonl(original_jsonl_path)
    
    def retrieval_function(query, db, k):
        return retrieve_rerank(query, db, k)
    
    results = evaluate_retrieval_rerank(original_data, retrieval_function, db, k)
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Average Score: {results['average_score']}")
    print(f"Total queries: {results['total_queries']}")
    return results
results5 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 5)
results10 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 10)
results20 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 20)
Evaluating retrieval: 100%|██████████| 248/248 [01:22<00:00,  2.99it/s]
Pass@5: 91.24%
Average Score: 0.912442396313364
Total queries: 248
Evaluating retrieval: 100%|██████████| 248/248 [01:34<00:00,  2.63it/s]
Pass@10: 94.79%
Average Score: 0.9479166666666667
Total queries: 248
Evaluating retrieval: 100%|██████████| 248/248 [02:08<00:00,  1.93it/s]
Pass@20: 96.30%
Average Score: 0.9630376344086022
Total queries: 248
Next Steps and Key Takeaways
We demonstrated how to use Contextual Embeddings to improve retrieval performance, then delivered additional improvements with Contextual BM25 and reranking.

This example used codebases, but these methods also apply to other data types such as internal company knowledge bases, financial & legal content, educational content, and much more.

If you are an AWS user, you can get started with the Lambda function in contextual-rag-lambda-function, and if you're a GCP user you can spin up your own Cloud Run instance and follow a similar pattern!