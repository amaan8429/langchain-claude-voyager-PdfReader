from typing import List
from langchain.embeddings.base import Embeddings
import voyageai
import os



class VoyageEmbeddings(Embeddings):
    def __init__(self, embeddings: List[List[float]]):
        self.embeddings = embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings

    def embed_query(self, text: str) -> List[float]:
        vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        result = vo.embed([text], model="voyage-2", input_type="query")
        return result.embeddings[0]
    

