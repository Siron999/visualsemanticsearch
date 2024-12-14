
from opensearchpy import OpenSearch


class OpensSearchService:
    def __init__(self, host, port, index_name):
        self.host = host
        self.port = port
        self.client = self.connect()
        self.index_name = index_name
        self.create_index_if_not_exists()

    def connect(self):
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            ssl_version=6,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )
        print("Connected to OpenSearch")
        return self.client

    def search(self, query):
        return self.client.search(index=self.index_name, body=query)

    def create_index_if_not_exists(self):
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.space_type": "cosinesimil"
                }
            },
            "mappings": {
                "properties": {
                    "text_vector_embeddings": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "image_vector_embeddings": {
                        "type": "knn_vector",
                        "dimension": 2048,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "metadata": {
                        "type": "object"
                    }
                }
            }
        }

        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                body=index_body
            )

    def index_product(self, product_id, vector, metadata=None, type="text"):
        """
        Index a product with its feature vector and metadata
        """
        self.create_index_if_not_exists()

        print(f"Indexing product [{type}] : {product_id} {vector.shape}")
        vector_field = f"{type}_vector_embeddings"
        doc = {
            vector_field: vector.tolist(),
            "metadata": metadata or {}
        }

        update_body = {
            "doc": doc,
            "doc_as_upsert": True  # Create if not exists
        }

        return self.client.update(
            index=self.index_name,
            id=product_id,
            body=update_body,
            retry_on_conflict=3
        )

    def is_index_empty(self):
        """
        Check if the index is empty
        """
        if not self.client.indices.exists(index=self.index_name):
            return True
        return self.client.count(index=self.index_name)['count'] == 0

    def search_similar_products(self, query_vector, top_k=10, type="text"):
        """
        Perform k-nearest neighbor search on vector
        """
        search_query = {}
        if type == "text":
            search_query = {
                "size": top_k,
                "query": {
                    "knn": {
                        "text_vector_embeddings": {
                            "vector": query_vector.tolist(),
                            "k": top_k
                        }
                    }
                },
                "_source": ["metadata"]
            }
        elif type == "image":
            search_query = {
                "size": top_k,
                "query": {
                    "knn": {
                        "image_vector_embeddings": {
                            "vector": query_vector.tolist(),
                            "k": top_k
                        }
                    }
                },
                "_source": ["metadata"]
            }
        results = self.client.search(
            index=self.index_name,
            body=search_query
        )
        return [
            {
                "product_id": hit['_id'],
                "score": hit['_score'],
                "metadata": hit['_source']['metadata']
            }
            for hit in results['hits']['hits']
        ]

    def delete_index(self):
        """
        Delete the index and recreate it with initial settings
        """
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

    def close(self):
        self.client.close()
