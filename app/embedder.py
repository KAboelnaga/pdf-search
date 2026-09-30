from fastembed import TextEmbedding

class Embedder:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model = TextEmbedding(model_name, cache_dir=cache_dir)
    
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts)]
    
    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()