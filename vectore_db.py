from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 3072,
    ):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # fail fast if qdrant is unreachable
        self.client.get_collections()

        if not self.client.collection_exists(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        contexts = []
        sources = set()

        for point in response.points:
            payload = point.payload or {}
            if payload.get("text"):
                contexts.append(payload["text"])
            if payload.get("source"):
                sources.add(payload["source"])

        return {
            "contexts": contexts,
            "sources": list(sources),
        }
