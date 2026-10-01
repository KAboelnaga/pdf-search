from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class VectorStore:
    def __init__(self, url : str, collection: str, vector_size: int) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
    
    def search(self, vector: list[float], limit: int) -> list[dict]:
        hits = self.client.query_points(
            self.collection,
            query=vector,
            limit=limit,
            with_payload=True
            ).points
        return[
            {
                "document": hit.payload["document"],
                "score": hit.score,
                "content": hit.payload["content"]
            }
            for hit in hits
        ]
