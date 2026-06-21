from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from src.config import settings

def get_embedding_function(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embeddings:
    """
    Returns an Embeddings instance based on the provided model name.
    
    Args:
        model_name: The name of the model to use.
        
    Returns:
        An instance of Embeddings (either HuggingFace or Ollama).
    """
    if "sentence-transformers" in model_name:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        return OllamaEmbeddings(
            model=model_name,
            base_url=settings.fastflowlm_base_url,
        )
