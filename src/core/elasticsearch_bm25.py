import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from elasticsearch.exceptions import NotFoundError, RequestError
import json
import os

from src.utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

es_host = "http://localhost:9200"
index_name = "contextual_bm25_index"
es_client = Elasticsearch(es_host)

if es_client.indices.exists(index=index_name):
    es_client.indices.delete(index=index_name)
    print(f"Index '{index_name}' deleted.")

class ElasticsearchBM25:
    
    def __init__(
        self, 
        index_name: str = "contextual_bm25_index",
        es_host: str = "http://localhost:9200",
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.index_name = index_name
        self.es_client = Elasticsearch(es_host)
        
        if not self.es_client.ping():
            raise ConnectionError(f"Failed to connect to Elasticsearch at {es_host}")
            
        self._create_index()
        
    def _create_index(self) -> None:
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "english"
                        },
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "porter_stem"]
                        }
                    }
                },
                "similarity": {
                    "custom_bm25": {
                        "type": "BM25",
                        "k1": 1.2,
                        "b": 0.75,
                    }
                },
                "index": {
                    "refresh_interval": "1s",
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_bm25"
                    },
                    "contextualized_content": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_bm25"
                    },
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "original_index": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": True}
                }
            }
        }
        
        index_data_dir = f"./data/{self.index_name}"
        os.makedirs(index_data_dir, exist_ok=True)
        self.logger.debug(f"Ensured existence of index data directory: {index_data_dir}")
        
        try:
            if self.es_client.indices.exists(index=self.index_name):
                self.es_client.indices.close(index=self.index_name)
                self.es_client.indices.put_settings(
                    index=self.index_name,
                    body=index_settings["settings"]
                )
                self.es_client.indices.open(index=self.index_name)
            else:
                self.es_client.indices.create(
                    index=self.index_name,
                    body=index_settings
                )
            self.logger.info(f"Successfully configured index: {self.index_name}")
        except Exception as e:
            self.logger.error(f"Failed to create/update index: {str(e)}")
            raise

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> Tuple[int, List[Dict[str, Any]]]:
        if not documents:
            self.logger.warning("No documents provided for indexing")
            return 0, []

        failed_docs = []
        success_count = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            actions = [{
                "_index": self.index_name,
                "_source": {
                    "content": doc.get("original_content", ""),
                    "contextualized_content": doc.get("contextualized_content", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "original_index": doc.get("original_index", 0),
                    "metadata": doc.get("metadata", {})
                }
            } for doc in batch]
            
            try:
                success, failed = bulk(
                    self.es_client,
                    actions,
                    raise_on_error=False,
                    raise_on_exception=False
                )
                success_count += success
                if failed:
                    for fail in failed:
                        failed_doc = batch[fail.get('index', 0)]
                        failed_docs.append(failed_doc)
                    self.logger.warning(f"Failed to index {len(failed)} documents in batch")
                    
            except Exception as e:
                self.logger.error(f"Batch indexing error: {str(e)}")
                failed_docs.extend(batch)
                
        self.es_client.indices.refresh(index=self.index_name)
        self.logger.info(f"Indexed {success_count}/{len(documents)} documents successfully")
        return success_count, failed_docs

    def search(
        self,
        query: str,
        k: int = 20,
        min_score: float = 0.1,
        fields: List[str] = None,
        operator: str = "or",
        minimum_should_match: str = "30%"
    ) -> List[Dict[str, Any]]:
        if not fields:
            fields = ["content^1", "contextualized_content^1.5"]
            
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "operator": operator,
                    "minimum_should_match": minimum_should_match,
                    "type": "best_fields",
                    "tie_breaker": 0.3
                }
            },
            "min_score": min_score,
            "size": k,
            "_source": True
        }

        try:
            search_body_json = json.dumps(search_body, indent=2)
            self.logger.debug(f"Elasticsearch search body: {search_body_json}")
        except Exception as e:
            self.logger.error(f"Error converting search body to JSON: {e}")
            search_body_json = str(search_body)
            self.logger.debug(f"Elasticsearch search body: {search_body_json}")

        try:
            response = self.es_client.search(
                index=self.index_name,
                body=search_body
            )
            
            hits = [{
                "doc_id": hit["_source"]["doc_id"],
                "chunk_id": hit["_source"]["chunk_id"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"].get("contextualized_content"),
                "score": hit["_score"],
                "metadata": hit["_source"].get("metadata", {})
            } for hit in response["hits"]["hits"]]
            
            self.logger.debug(
                f"Search for '{query}' returned {len(hits)} results "
                f"(max_score: {response['hits'].get('max_score', 0)})"
            )
            return hits
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            raise

    def search_content(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        self.logger.debug(f"Performing BM25 search on 'content' for query: '{query}'")
        return self.search(
            query=query,
            k=k,
            fields=["content^1"],
            operator="or",
            minimum_should_match="30%"
        )

    def search_contextualized(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        self.logger.debug(f"Performing BM25 search on 'contextualized_content' for query: '{query}'")
        return self.search(
            query=query,
            k=k,
            fields=["contextualized_content^1.5"],
            operator="or",
            minimum_should_match="30%"
        )
