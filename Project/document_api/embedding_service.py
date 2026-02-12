from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        """
        Initialize Vietnamese embedding model
        THAY ĐỔI: Embedding dimension = 768
        """
        self.device = "cpu"
        print(f"Loading embedding model on {self.device}")

        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )
        self.model = self.model.to(self.device)

        self.embedding_dim = 768
        print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")


    def get_embedding(self, text) -> List[float]:
        try:
            # 🔒 SAFETY: ensure string
            if isinstance(text, dict):
                # Ưu tiên field phổ biến
                text = text.get("content") or text.get("text") or str(text)

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()
            if not text:
                return [0.0] * self.embedding_dim

            with torch.no_grad():
                embedding = self.model.encode(text, convert_to_tensor=True)
                embedding = embedding.cpu().numpy()

            return embedding.tolist()

        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * self.embedding_dim

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text))
        return embeddings

    def is_ready(self) -> bool:
        """Check if model is ready"""
        try:
            test_embedding = self.get_embedding("test")
            return len(test_embedding) == self.embedding_dim
        except:
            return False