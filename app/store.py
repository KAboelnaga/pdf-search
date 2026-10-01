from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import uuid

class VectorStore:
    def __init__(self, url : str, collection: str, vector_size: int) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
    def upsert(self, vector: list[list[float]], payload: list[dict]) -> None:
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=pl) 
            for vec, pl in zip(vector, payload, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
    
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
