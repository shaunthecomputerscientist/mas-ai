import numpy as np
import os
from typing import List, Optional, Union, Callable

# Optional import for sentence transformers (heavy ML package)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Optional import for LangChain Document compatibility
# try:
from langchain.schema.document import Document
# except ImportError:
#     Document = None  # Fallback if LangChain isn’t installed

class InMemoryDocStore:
    def __init__(self, documents: Optional[List[Union[str, Document]]] = None, 
                 ids: Optional[List[str]] = None, 
                 embedding_model: Optional[Union[str, Callable, object]] = 'all-MiniLM-L6-v2'):
        """
        Initialize the custom document store.

        Args:
            documents: Optional list of strings, dicts, or Document objects representing document content. Defaults to None.
            ids: Optional list of unique identifiers for the documents. If None and documents are provided, generates sequential IDs. Defaults to None.
            embedding_model: Optional embedding model. Can be:
                - A string (e.g., "all-MiniLM-L6-v2") to load a SentenceTransformer model.
                - A callable (function) that takes a list of strings and returns embeddings (list or np.ndarray).
                - An object with an `embed_documents` method (e.g., LangChain Embeddings for compatibility).
                If None, no embeddings are computed until a model is provided.
        """
        self.doc_store = {}
        self.ids = []
        self.next_id = 0
        self.embedding_matrix = None

        # Set up embedding model
        self.embedding_model = None
        if embedding_model:
            if isinstance(embedding_model, str):
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError(
                        "sentence_transformers is not installed. "
                        "Install it with: pip install sentence-transformers"
                    )
                self.embedding_model = SentenceTransformer(embedding_model)
            elif callable(embedding_model):
                self.embedding_model = embedding_model
            elif hasattr(embedding_model, 'embed_documents'):
                self.embedding_model = embedding_model
            else:
                raise ValueError("embedding_model must be a string (model name), callable, or object with embed_documents method")

        # Handle optional documents and IDs
        if documents is not None:
            # Normalize documents to dict format
            documents = [self._normalize_doc(doc) for doc in documents]
            if ids is None:
                ids = [str(i) for i in range(len(documents))]
                self.next_id = len(documents)
            else:
                assert len(ids) == len(documents), "Number of IDs must match number of documents"
                assert len(set(ids)) == len(ids), "IDs must be unique"

            self.doc_store = dict(zip(ids, documents))
            self.ids = ids

            if self.embedding_model:
                self._update_embeddings(documents)

    def _normalize_doc(self, doc: Union[str, dict, Document]) -> dict:
        """Convert a document (str, dict, or Document) to a standard dict format."""
        if isinstance(doc, str):
            return {'page_content': doc}
        elif isinstance(doc, dict):
            return doc
        elif Document is not None and isinstance(doc, Document):
            return {'page_content': doc.page_content}
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    def _update_embeddings(self, documents: List[dict]) -> None:
        """Helper method to compute and update embeddings for given documents."""
        # Handle both dict and Document objects
        texts = [doc['page_content'] if isinstance(doc, dict) else doc.page_content for doc in documents]
        if isinstance(self.embedding_model, SentenceTransformer):
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        elif callable(self.embedding_model):
            embeddings = self.embedding_model(texts)
        elif hasattr(self.embedding_model, 'embed_documents'):
            embeddings = self.embedding_model.embed_documents(texts)
        else:
            return  # No embedding model, skip

        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        if self.embedding_matrix is None:
            self.embedding_matrix = embeddings
        else:
            self.embedding_matrix = np.vstack([self.embedding_matrix, embeddings])

    def add_documents(self, documents: List[Union[str, Document]], ids: Optional[List[str]] = None):
        """
        Add new documents to the store.

        Args:
            documents: List of strings, dicts, or Document objects to add.
            ids: Optional list of unique IDs. If None, generates new IDs starting from the next available integer.
        """
        documents = [self._normalize_doc(doc) for doc in documents]

        if ids is None:
            if not self.ids or all(id_.isdigit() for id_ in self.ids):
                start_id = self.next_id
                ids = [str(start_id + i) for i in range(len(documents))]
                self.next_id = start_id + len(documents)
            else:
                raise ValueError("Cannot auto-generate numeric IDs when existing IDs are not numeric.")
        else:
            assert len(ids) == len(documents), "Number of IDs must match number of documents"
            assert all(id_ not in self.doc_store for id_ in ids), "IDs must be unique and not already in store"

        for id_, doc in zip(ids, documents):
            self.doc_store[id_] = doc
        self.ids.extend(ids)

        if self.embedding_model:
            self._update_embeddings(documents)

    def get_document(self, id_: str) -> Optional[dict]:
        """
        Retrieve a document by its ID.

        Args:
            id_: The ID of the document to retrieve.

        Returns:
            A dict with 'page_content' key if found, else None.
        """
        return self.doc_store.get(id_)

    async def asearch(self, query: str, k: int = 5) -> List[dict]:
        """
        Search for the top-k most similar documents to the query.

        Args:
            query: The query string to search for.
            k: Number of documents to return (default is 5).

        Returns:
            List of top-k document dicts.
        """
        if not self.embedding_model or self.embedding_matrix is None:
            return [self.doc_store[id_] for id_ in self.ids[:min(k, len(self.ids))]]
        
        if isinstance(self.embedding_model, SentenceTransformer):
            query_emb = self.embedding_model.encode(query, convert_to_numpy=True)
        elif callable(self.embedding_model):
            query_emb = self.embedding_model([query])[0]
        elif hasattr(self.embedding_model, 'embed_documents'):
            query_emb = self.embedding_model.embed_documents([query])[0]
        else:
            raise ValueError("Invalid embedding model configuration")

        query_emb = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(self.embedding_matrix, query_emb)
        top_k_indices = np.argsort(similarities)[::-1][:min(k, len(self.ids))]
        top_k_ids = [self.ids[idx] for idx in top_k_indices]
        return [self.doc_store[id_] for id_ in top_k_ids]