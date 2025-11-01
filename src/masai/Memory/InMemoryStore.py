import numpy as np
import os
import warnings
from typing import List, Optional, Union, Callable

# Optional import for sentence transformers (heavy ML package)
# NOTE: sentence-transformers, torch, and transformers are NOT included in MASAI core dependencies
# Users must install them separately if they want to use SentenceTransformer embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Optional import for LangChain Document compatibility
# try:
from ..schema import Document
# except ImportError:
#     Document = None  # Fallback if LangChain isn’t installed

class InMemoryDocStore:
    def __init__(self, documents: Optional[List[Union[str, Document]]] = None,
                 ids: Optional[List[str]] = None,
                 embedding_model: Optional[Union[str, Callable, object]] = 'all-MiniLM-L6-v2',
                 max_documents: Optional[int] = None):
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
            max_documents: Optional maximum number of documents to retain in memory. If set,
                the store enforces a FIFO eviction policy to cap memory usage.
        """
        self.doc_store = {}
        self.ids = []
        self.next_id = 0
        self.max_documents = max_documents

        self.embedding_matrix = None

        # Set up embedding model
        self.embedding_model = None
        if embedding_model:
            if isinstance(embedding_model, str):
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError(
                        "\n\n"
                        "=" * 80 + "\n"
                        "ERROR: sentence-transformers is not installed.\n"
                        "=" * 80 + "\n\n"
                        "InMemoryDocStore requires sentence-transformers for embedding-based search.\n\n"
                        "MASAI does not include heavy ML dependencies (sentence-transformers, torch, transformers)\n"
                        "in its core installation to keep the framework lightweight.\n\n"
                        "To use InMemoryDocStore with SentenceTransformer embeddings, install:\n"
                        "  pip install sentence-transformers\n\n"
                        "This will also install torch and transformers (~2GB+ download).\n\n"
                        "ALTERNATIVES:\n"
                        "1. Use a custom embedding function:\n"
                        "   def my_embedder(texts): return embeddings\n"
                        "   store = InMemoryDocStore(embedding_model=my_embedder)\n\n"
                        "2. Use LangChain embeddings (e.g., OpenAI):\n"
                        "   from langchain.embeddings import OpenAIEmbeddings\n"
                        "   store = InMemoryDocStore(embedding_model=OpenAIEmbeddings())\n\n"
                        "3. Use no embeddings (keyword-based search only):\n"
                        "   store = InMemoryDocStore(embedding_model=None)\n"
                        "=" * 80 + "\n"
                    )
                self.embedding_model = SentenceTransformer(embedding_model)
                warnings.warn(
                    f"Using SentenceTransformer model '{embedding_model}'. "
                    "This requires sentence-transformers, torch, and transformers packages. "
                    "First-time usage will download the model (~100-500MB).",
                    UserWarning
                )
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
            return {'page_content': doc, 'metadata': {}}
        elif isinstance(doc, dict):
            # Ensure dict has both page_content and metadata keys
            if 'page_content' not in doc:
                raise ValueError("Dict document must have 'page_content' key")
            if 'metadata' not in doc:
                doc['metadata'] = {}
            return doc
        elif Document is not None and isinstance(doc, Document):
            # Preserve both page_content and metadata from Document objects
            return {'page_content': doc.page_content, 'metadata': doc.metadata}
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

        # Prune if exceeding capacity
        self._prune_to_limit()

    def _prune_to_limit(self) -> None:
        """Ensure the store does not exceed max_documents (FIFO eviction)."""
        if self.max_documents is None:
            return
        excess = len(self.ids) - self.max_documents
        if excess <= 0:
            return
        # Evict oldest entries
        old_ids = self.ids[:excess]
        for oid in old_ids:
            self.doc_store.pop(oid, None)
        self.ids = self.ids[excess:]
        if self.embedding_matrix is not None:
            if excess >= len(self.embedding_matrix):
                self.embedding_matrix = None
            else:
                self.embedding_matrix = self.embedding_matrix[excess:]

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

        # Ensure capacity after adding documents (handles non-embedding case too)
        self._prune_to_limit()


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

    def iter_documents(self):
        """Iterate over documents as dicts with keys: 'page_content', 'metadata'."""
        for id_ in self.ids:
            doc = self.doc_store.get(id_)
            if doc is not None:
                yield doc

    def export_documents(self) -> List[dict]:
        """Return a list copy of all documents in insertion (ID) order."""
        return [self.doc_store[id_] for id_ in self.ids if id_ in self.doc_store]
