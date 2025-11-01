from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import uuid
import hashlib

try:
    # Typing-only imports; runtime optional
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models as qmodels
    _QDRANT_AVAILABLE = True
except Exception:  # ImportError or others
    AsyncQdrantClient = None  # type: ignore
    qmodels = None  # type: ignore
    _QDRANT_AVAILABLE = False

try:
    # Redis imports; runtime optional
    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis
    from langchain_redis import RedisVectorStore
    _REDIS_AVAILABLE = True
except Exception:
    Redis = None  # type: ignore
    AsyncRedis = None  # type: ignore
    RedisVectorStore = None  # type: ignore
    _REDIS_AVAILABLE = False

from pydantic import BaseModel, Field
from ..schema import Document


class QdrantConfig(BaseModel):
    """Configuration for Qdrant persistent memory with embedded embedding model.

    This config is self-contained and includes the embedding model, so it can be
    passed directly to LongTermMemory without needing separate embedding_model parameters.
    """
    url: str = Field(..., description="Qdrant endpoint URL, e.g., http://localhost:6333 or a cloud endpoint")
    api_key: Optional[str] = Field(default=None, description="API key for Qdrant Cloud; not needed for local")
    collection_name: str = Field(default="masai_memories", description="Qdrant collection name to use/create")
    vector_size: int = Field(..., description="Embedding vector dimension. Must match embedder output.")
    distance: str = Field(default="cosine", description="Distance metric: cosine|dot|euclid")
    prefer_async: bool = Field(default=True, description="Use AsyncQdrantClient when available")
    timeout_sec: Optional[float] = Field(default=10.0, description="Client timeout")
    consistency: Optional[str] = Field(default=None, description="Optional consistency for distributed setups")
    dedup_mode: str = Field(default="similarity", description="Deduplication mode: 'none'|'similarity'|'hash'")
    dedup_similarity_threshold: float = Field(default=0.9, description="Similarity threshold [0,1] for dedup (similarity mode only)")
    embedding_model: Optional[Any] = Field(
        default=None,
        description="Embedding model (callable, LangChain embeddings, or custom class with embed_documents). "
                    "Required for persistence. Supports: callable, LangChain embeddings, or custom class with embed_documents(texts: List[str]) -> List[List[float]]"
    )

    class Config:
        arbitrary_types_allowed = True  # Allow Any type for embedding_model

    def validate_embedding_model(self) -> None:
        """Validate that embedding_model is properly configured.

        Raises:
            ValueError: If embedding_model is not callable or doesn't have embed_documents method
        """
        if self.embedding_model is None:
            raise ValueError(
                "embedding_model is required in QdrantConfig for persistent memory. "
                "Provide one of: callable function, LangChain embeddings, or custom class with embed_documents()"
            )

        model = self.embedding_model

        # Type 1: Direct callable
        if callable(model):
            return

        # Type 2: LangChain embeddings or custom class with embed_documents()
        if hasattr(model, 'embed_documents'):
            return

        # Type 3: Unsupported
        raise ValueError(
            f"embedding_model must be callable or have embed_documents() method, "
            f"got {type(model).__name__}. "
            f"Supported: callable, LangChain embeddings (HuggingFaceEmbeddings, OpenAIEmbeddings, etc.), "
            f"or custom class with embed_documents(texts: List[str]) -> List[List[float]]"
        )

    def distance_enum(self):
        """Convert distance metric string to Qdrant Distance enum.

        Supports multiple aliases for each distance metric:
        - Cosine: "cos", "cosine"
        - Dot product: "dot", "dotproduct", "ip"
        - Euclidean: "l2", "euclid", "euclidean"

        Returns:
            qmodels.Distance: Qdrant distance enum value, or None if Qdrant not available.
                             Defaults to COSINE if distance string not recognized.
        """
        if not _QDRANT_AVAILABLE:
            return None
        d = (self.distance or "cosine").lower()
        if d in ("cos", "cosine"):
            return qmodels.Distance.COSINE
        if d in ("dot", "dotproduct", "ip"):
            return qmodels.Distance.DOT
        if d in ("l2", "euclid", "euclidean"):
            return qmodels.Distance.EUCLID
        return qmodels.Distance.COSINE


class RedisConfig(BaseModel):
    """Configuration for Redis persistent memory with embedded embedding model.

    This config is self-contained and includes the embedding model, so it can be
    passed directly to LongTermMemory without needing separate embedding_model parameters.
    Supports both sync and async Redis operations with comprehensive configuration options.
    """
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL, e.g., redis://localhost:6379 or redis://user:password@host:port")
    index_name: str = Field(default="masai_vectors", description="Redis index name for vector storage")
    vector_size: int = Field(default=384, description="Embedding vector dimension. Must match embedder output.")
    distance_metric: str = Field(default="cosine", description="Distance metric: cosine|l2|ip")
    embedding_model: Optional[Any] = Field(
        default=None,
        description="Embedding model (callable, LangChain embeddings, or custom class with embed_documents). "
                    "Required for persistence. Supports: callable, LangChain embeddings, or custom class with embed_documents(texts: List[str]) -> List[List[float]]"
    )
    dedup_mode: str = Field(default="similarity", description="Deduplication mode: 'none'|'similarity'|'hash'")
    dedup_similarity_threshold: float = Field(default=0.95, description="Similarity threshold [0,1] for dedup (similarity mode only)")
    ttl_seconds: Optional[int] = Field(default=None, description="Optional TTL in seconds for auto-expiration of documents")
    connection_pool_size: int = Field(default=10, description="Redis connection pool size")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(default=5.0, description="Socket connection timeout in seconds")
    socket_keepalive: bool = Field(default=True, description="Enable TCP keepalive")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    batch_size: int = Field(default=100, description="Batch size for bulk operations")
    use_async: bool = Field(default=True, description="Use async Redis client when available")

    class Config:
        arbitrary_types_allowed = True  # Allow Any type for embedding_model

    def validate_embedding_model(self) -> None:
        """Validate that embedding_model is properly configured.

        Raises:
            ValueError: If embedding_model is not callable or doesn't have embed_documents method
        """
        if self.embedding_model is None:
            raise ValueError(
                "embedding_model is required in RedisConfig for persistent memory. "
                "Provide one of: callable function, LangChain embeddings, or custom class with embed_documents()"
            )

        model = self.embedding_model

        # Type 1: Direct callable
        if callable(model):
            return

        # Type 2: LangChain embeddings or custom class with embed_documents()
        if hasattr(model, 'embed_documents'):
            return

        # Type 3: Unsupported
        raise ValueError(
            f"embedding_model must be callable or have embed_documents() method, "
            f"got {type(model).__name__}. "
            f"Supported: callable, LangChain embeddings (HuggingFaceEmbeddings, OpenAIEmbeddings, etc.), "
            f"or custom class with embed_documents(texts: List[str]) -> List[List[float]]"
        )


class QdrantAdapter:
    """Light wrapper around AsyncQdrantClient for MASAI persistent memory.

    This class is intentionally minimal and focused only on capabilities needed by
    MASAI's long-term memory orchestration: ensure collection, upsert, search, delete.
    """

    def __init__(self, cfg: QdrantConfig):
        """Initialize QdrantAdapter with configuration.

        Args:
            cfg: QdrantConfig instance with Qdrant connection details

        Raises:
            ImportError: If qdrant-client is not installed

        Side Effects:
            - Creates AsyncQdrantClient connection
            - Stores config for later use
        """
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Install with: pip install qdrant-client"
            )
        self.cfg = cfg
        self.client = AsyncQdrantClient(url=cfg.url, api_key=cfg.api_key, timeout=cfg.timeout_sec)

    async def ensure_collection(self):
        """Ensure collection exists and required payload indexes are created.

        Creates the Qdrant collection if it doesn't exist with configured
        vector size and distance metric. Also creates payload indexes for
        user_id and categories fields to enable efficient filtering.

        Returns:
            None. Creates collection and indexes asynchronously.

        Side Effects:
            - Creates collection if not exists
            - Creates payload indexes for user_id and categories
            - Silently ignores errors if indexes already exist
        """
        assert _QDRANT_AVAILABLE
        collections = await self.client.get_collections()
        names = {c.name for c in collections.collections}
        if self.cfg.collection_name not in names:
            await self.client.create_collection(
                collection_name=self.cfg.collection_name,
                vectors_config=qmodels.VectorParams(size=self.cfg.vector_size, distance=self.cfg.distance_enum()),
            )
        # Ensure payload indexes for filters used in queries
        try:
            await self.client.create_payload_index(
                collection_name=self.cfg.collection_name,
                field_name="user_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
        try:
            await self.client.create_payload_index(
                collection_name=self.cfg.collection_name,
                field_name="categories",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    async def upsert_documents(
        self,
        user_id: Union[str, int],
        documents: Sequence[Union[Document, str, Dict[str, Any]]],
        embed_fn: Callable[[List[str]], Union[List[List[float]], Any]],
        categories_resolver: Optional[Callable[[Document], List[str]]] = None,
    ) -> None:
        """Upsert documents into Qdrant with user_id and categories in payload.

        - Documents may be MASAI Document, dict with page_content/metadata, or plain string
        - Embeddings computed via embed_fn(texts)
        - Categories taken from doc.metadata["categories"] or from categories_resolver(doc)
        - Deduplication: if dedup_mode=="similarity", searches for similar existing docs and merges
        """
        await self.ensure_collection()

        # Normalize docs and collect texts
        norm_docs: List[Document] = []
        texts: List[str] = []
        for d in documents:
            if isinstance(d, Document):
                doc = d
            elif isinstance(d, str):
                doc = Document(page_content=d, metadata={})
            elif isinstance(d, dict):
                page = d.get("page_content")
                meta = d.get("metadata") or {}
                if page is None:
                    continue
                doc = Document(page_content=page, metadata=meta)
            else:
                continue

            # Derive categories
            cats = doc.metadata.get("categories") or []
            if not cats and categories_resolver is not None:
                try:
                    cats = categories_resolver(doc) or []
                except Exception:
                    cats = []
            # Normalize to list[str]
            if isinstance(cats, str):
                cats = [cats]
            elif not isinstance(cats, list):
                cats = []

            # Ensure user_id in metadata for completeness (payload will carry explicit field)
            doc.metadata = {**doc.metadata, "user_id": user_id, "categories": cats}
            norm_docs.append(doc)
            texts.append(doc.page_content)

        if not norm_docs:
            return

        # Compute embeddings
        embeds = embed_fn(texts)
        # Convert to plain list of list[float]
        if hasattr(embeds, "tolist"):
            embeds = embeds.tolist()

        points: list = []

        for i, doc in enumerate(norm_docs):
            payload = {
                "user_id": user_id,
                "categories": doc.metadata.get("categories", []),
                "metadata": {k: v for k, v in doc.metadata.items() if k not in ("categories", "user_id")},
                "page_content": doc.page_content,
            }

            # Determine point ID based on dedup mode
            point_id = await self._get_point_id_with_dedup(
                user_id=user_id,
                doc=doc,
                vector=embeds[i],
                payload=payload,
                embed_fn=embed_fn,
            )

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=embeds[i],
                    payload=payload,
                )
            )

        await self.client.upsert(collection_name=self.cfg.collection_name, points=points)

    async def _get_point_id_with_dedup(
        self,
        user_id: Union[str, int],
        doc: Document,
        vector: List[float],
        payload: Dict[str, Any],
        embed_fn: Callable,
    ) -> Union[str, int]:
        """Determine point ID with optional deduplication.

        - "similarity" mode: search top-1, merge if score >= threshold, else new UUID
        - "hash" mode: use hash of text to generate UUID (deterministic)
        - "none" mode: always new UUID
        """
        if self.cfg.dedup_mode == "hash":
            import hashlib
            normalized = doc.page_content.lower().strip()
            # Convert hash to UUID for Qdrant compatibility
            hash_bytes = hashlib.sha256(normalized.encode()).digest()
            # Create UUID from first 16 bytes of hash
            hash_uuid = str(uuid.UUID(bytes=hash_bytes[:16]))
            return hash_uuid

        elif self.cfg.dedup_mode == "similarity":
            # Search for similar existing doc (same user)
            try:
                similar = await self.search(
                    user_id=user_id,
                    query=doc.page_content,
                    k=1,
                    categories=payload.get("categories"),
                    embed_fn=embed_fn,
                )
                if similar:
                    # ✅ FIX: Check actual similarity score against threshold
                    similarity_score = similar[0].metadata.get("_similarity_score", 0)
                    if similarity_score >= self.cfg.dedup_similarity_threshold:
                        # Similar enough - reuse existing point ID
                        return similar[0].metadata.get("_point_id", uuid.uuid4().hex)
            except Exception:
                pass

            # No similar doc found (or score below threshold), use new UUID
            return uuid.uuid4().hex

        else:  # "none" mode
            return uuid.uuid4().hex

    async def search(
        self,
        user_id: Union[str, int],
        query: str,
        k: int = 5,
        categories: Optional[List[str]] = None,
        embed_fn: Optional[Callable[[List[str]], Union[List[List[float]], Any]]] = None,
    ) -> List[Document]:
        """Search top-k by similarity filtered by user_id and optional categories.

        Returns documents with _point_id and _similarity_score in metadata for dedup reference.

        The _similarity_score is the cosine/dot/euclidean similarity returned by Qdrant,
        which can be used to enforce deduplication thresholds.
        """
        await self.ensure_collection()
        if embed_fn is None:
            raise ValueError("embed_fn is required for search")

        query_vec = embed_fn([query])
        if hasattr(query_vec, "tolist"):
            query_vec = query_vec.tolist()
        if isinstance(query_vec, list) and query_vec and isinstance(query_vec[0], list):
            query_vec = query_vec[0]

        must_conditions = [qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
        if categories:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="categories",
                    match=qmodels.MatchAny(any=categories),
                )
            )

        result = await self.client.search(
            collection_name=self.cfg.collection_name,
            query_vector=query_vec,
            limit=max(1, k),
            query_filter=qmodels.Filter(must=must_conditions),
        )

        docs: List[Document] = []
        for pt in result:
            payload = pt.payload or {}
            docs.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata={
                        "user_id": payload.get("user_id"),
                        "categories": payload.get("categories", []),
                        "_point_id": str(pt.id),  # Store point ID for dedup reference
                        "_similarity_score": pt.score,  # ✅ NEW: Store similarity score for threshold checking
                        **(payload.get("metadata") or {}),
                    },
                )
            )
        return docs

    async def delete_by_doc_id(self, user_id: Union[str, int], doc_id: Union[str, int]) -> None:
        """Delete document from Qdrant by point ID with user isolation.

        Deletes a specific document using both point ID and user_id filter
        for safety. Ensures users can only delete their own documents.

        Args:
            user_id: User identifier for isolation
            doc_id: Qdrant point ID to delete

        Returns:
            None. Deletes document asynchronously from Qdrant.

        Raises:
            Logs errors but does not raise exception.
        """
        await self.ensure_collection()
        await self.client.delete(
            collection_name=self.cfg.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[
                    qmodels.HasIdCondition(has_id=[doc_id]),
                    qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
                ])
            ),
        )


class RedisAdapter:
    """Light wrapper around Redis for MASAI persistent memory using RedisVectorStore.

    This class provides vector storage and retrieval capabilities using Redis with
    RediSearch module. Supports user isolation, deduplication, TTL, and comprehensive
    configuration options.
    """

    def __init__(self, cfg: RedisConfig):
        """Initialize RedisAdapter with configuration.

        Args:
            cfg: RedisConfig instance with Redis connection details

        Raises:
            ImportError: If redis or langchain-redis is not installed

        Side Effects:
            - Creates Redis connection
            - Stores config for later use
        """
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "redis and langchain-redis are not installed. "
                "Install with: pip install redis langchain-redis"
            )
        self.cfg = cfg
        self.redis_url = cfg.redis_url
        self.index_name = cfg.index_name

    def _get_redis_client(self):
        """Get Redis client (sync or async based on config).

        Returns:
            Redis or AsyncRedis client instance
        """
        if self.cfg.use_async:
            return AsyncRedis.from_url(
                self.redis_url,
                socket_timeout=self.cfg.socket_timeout,
                socket_connect_timeout=self.cfg.socket_connect_timeout,
                socket_keepalive=self.cfg.socket_keepalive,
                retry_on_timeout=self.cfg.retry_on_timeout,
                health_check_interval=self.cfg.health_check_interval,
            )
        else:
            return Redis.from_url(
                self.redis_url,
                socket_timeout=self.cfg.socket_timeout,
                socket_connect_timeout=self.cfg.socket_connect_timeout,
                socket_keepalive=self.cfg.socket_keepalive,
                retry_on_timeout=self.cfg.retry_on_timeout,
                health_check_interval=self.cfg.health_check_interval,
            )

    def _get_doc_id_with_dedup(
        self,
        doc: Document,
        content_hash: Optional[str] = None,
    ) -> str:
        """Determine document ID with optional deduplication.

        - "hash" mode: use hash of text to generate deterministic ID
        - "similarity" mode: use UUID (similarity check done during search)
        - "none" mode: always new UUID

        Args:
            doc: Document to process
            content_hash: Pre-computed content hash (optional)

        Returns:
            Document ID string
        """
        if self.cfg.dedup_mode == "hash":
            if content_hash is None:
                normalized = doc.page_content.lower().strip()
                content_hash = hashlib.md5(normalized.encode()).hexdigest()
            return content_hash

        elif self.cfg.dedup_mode == "similarity":
            # For similarity mode, use UUID; actual dedup happens during search
            return str(uuid.uuid4())

        else:  # "none" mode
            return str(uuid.uuid4())

    async def upsert_documents(
        self,
        user_id: Union[str, int],
        documents: Sequence[Union[Document, str, Dict[str, Any]]],
        embed_fn: Callable[[List[str]], Union[List[List[float]], Any]],
        categories_resolver: Optional[Callable[[Document], List[str]]] = None,
    ) -> None:
        """Upsert documents into Redis with user_id and categories in metadata.

        - Documents may be MASAI Document, dict with page_content/metadata, or plain string
        - Embeddings computed via embed_fn(texts)
        - Categories taken from doc.metadata["categories"] or from categories_resolver(doc)
        - Deduplication: if dedup_mode=="similarity", searches for similar existing docs
        - TTL: if configured, documents auto-expire after ttl_seconds

        Args:
            user_id: User identifier for isolation
            documents: Documents to upsert
            embed_fn: Function to compute embeddings
            categories_resolver: Optional function to extract categories

        Returns:
            None. Upserts documents asynchronously to Redis.
        """
        # Normalize docs and collect texts
        norm_docs: List[Document] = []
        texts: List[str] = []
        for d in documents:
            if isinstance(d, Document):
                doc = d
            elif isinstance(d, str):
                doc = Document(page_content=d, metadata={})
            elif isinstance(d, dict):
                page = d.get("page_content")
                meta = d.get("metadata") or {}
                if page is None:
                    continue
                doc = Document(page_content=page, metadata=meta)
            else:
                continue

            # Derive categories
            cats = doc.metadata.get("categories") or []
            if not cats and categories_resolver is not None:
                try:
                    cats = categories_resolver(doc) or []
                except Exception:
                    cats = []
            # Normalize to list[str]
            if isinstance(cats, str):
                cats = [cats]
            elif not isinstance(cats, list):
                cats = []

            # Ensure user_id in metadata for completeness
            doc.metadata = {**doc.metadata, "user_id": user_id, "categories": cats}
            norm_docs.append(doc)
            texts.append(doc.page_content)

        if not norm_docs:
            return

        # Compute embeddings
        embeds = embed_fn(texts)
        # Convert to plain list of list[float]
        if hasattr(embeds, "tolist"):
            embeds = embeds.tolist()

        # Prepare documents for RedisVectorStore
        docs_to_add: List[Document] = []
        doc_ids: List[str] = []

        for i, doc in enumerate(norm_docs):
            # Compute content hash for hash-based dedup
            normalized = doc.page_content.lower().strip()
            content_hash = hashlib.md5(normalized.encode()).hexdigest()

            # Get document ID with dedup logic
            doc_id = self._get_doc_id_with_dedup(doc, content_hash)

            # Add metadata for Redis storage
            doc.metadata["_doc_id"] = doc_id
            doc.metadata["_content_hash"] = content_hash
            doc.metadata["_ttl"] = self.cfg.ttl_seconds

            docs_to_add.append(doc)
            doc_ids.append(doc_id)

        # Add to Redis in batches
        for i in range(0, len(docs_to_add), self.cfg.batch_size):
            batch_docs = docs_to_add[i:i + self.cfg.batch_size]
            batch_ids = doc_ids[i:i + self.cfg.batch_size]

            try:
                # Create RedisVectorStore and add documents
                vector_store = await self._get_vector_store()
                await vector_store.aadd_documents(batch_docs, ids=batch_ids)
            except Exception:
                # Log but don't fail entire operation
                pass

    async def _get_vector_store(self):
        """Get or create RedisVectorStore instance.

        Returns:
            RedisVectorStore instance configured with embedding model and index
        """
        # Import here to avoid circular imports
        from langchain_redis import RedisVectorStore
        from langchain_redis.config import RedisConfig as LangChainRedisConfig

        # Resolve embedding model
        embeddings = self._resolve_embeddings()

        # Create LangChain RedisConfig object (not our RedisConfig)
        config = LangChainRedisConfig(
            redis_url=self.redis_url,
            index_name=self.index_name,
            distance_metric=self.cfg.distance_metric,
        )

        return RedisVectorStore(
            embeddings=embeddings,
            config=config,
        )

    def _resolve_embeddings(self):
        """Resolve embedding model from RedisConfig.embedding_model.

        Supports three types of embedding models:
        1. Callable function: embed_fn(texts: List[str]) -> List[List[float]]
        2. LangChain embeddings: has embed_documents(texts) method
        3. Custom class: has embed_documents(texts) method

        Returns:
            Embedding model instance

        Raises:
            ValueError: If embedding_model is not found or not supported
        """
        model = self.cfg.embedding_model

        if not model:
            raise ValueError(
                "No embedding_model provided in RedisConfig. "
                "Supported: callable, LangChain embeddings, or custom class with embed_documents()"
            )

        # Type 1: Direct callable - wrap it
        if callable(model) and not hasattr(model, 'embed_documents'):
            # Create a wrapper class for callable
            class CallableEmbeddings:
                def __init__(self, fn):
                    self.fn = fn

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return self.fn(texts)

                def embed_query(self, text: str) -> List[float]:
                    result = self.fn([text])
                    return result[0] if result else []

            return CallableEmbeddings(model)

        # Type 2: LangChain embeddings or custom class with embed_documents()
        if hasattr(model, 'embed_documents'):
            return model

        # Type 3: Unsupported
        raise ValueError(
            f"embedding_model must be callable or have embed_documents() method, "
            f"got {type(model).__name__}. "
            f"Supported: callable, LangChain embeddings (HuggingFaceEmbeddings, OpenAIEmbeddings, etc.), "
            f"or custom class with embed_documents(texts: List[str]) -> List[List[float]]"
        )

    async def search(
        self,
        user_id: Union[str, int],
        query: str,
        k: int = 5,
        categories: Optional[List[str]] = None,
        embed_fn: Optional[Callable[[List[str]], Union[List[List[float]], Any]]] = None,
    ) -> List[Document]:
        """Search top-k by similarity filtered by user_id and optional categories.

        Args:
            user_id: User identifier for isolation
            query: Query string
            k: Number of results to return
            categories: Optional categories to filter by
            embed_fn: Embedding function (optional, uses model from config if not provided)

        Returns:
            List of similar documents with metadata
        """
        try:
            vector_store = await self._get_vector_store()

            # Build filter string for RediSearch
            # Since user_id is stored in _metadata_json field, we search in that field
            # Use simple text search format: @field:value
            filter_str = f'@_metadata_json:{user_id}'

            # If categories provided, add to filter
            if categories:
                # Build category filter - search for each category in _metadata_json
                category_filters = " | ".join([f'@_metadata_json:{cat}' for cat in categories])
                filter_str = f'({filter_str}) & ({category_filters})'

            results = await vector_store.asimilarity_search(
                query,
                k=k,
                filter=filter_str,
            )

            return results
        except Exception:
            # Return empty list on error
            return []

    async def delete_by_doc_id(self, user_id: Union[str, int], doc_id: str) -> None:
        """Delete document from Redis by doc ID with user isolation.

        Args:
            user_id: User identifier for isolation
            doc_id: Document ID to delete

        Returns:
            None. Deletes document asynchronously from Redis.
        """
        try:
            vector_store = await self._get_vector_store()
            await vector_store.adelete(ids=[doc_id])
        except Exception:
            # Silently ignore errors
            pass

    async def flush(self) -> None:
        """Flush all documents from Redis index.

        Returns:
            None. Clears all documents asynchronously from Redis.
        """
        try:
            client = self._get_redis_client()
            if self.cfg.use_async:
                await client.flushdb()
                await client.close()
            else:
                client.flushdb()
                client.close()
        except Exception:
            pass

    async def close(self) -> None:
        """Close Redis connection.

        Returns:
            None. Closes connection asynchronously.
        """
        try:
            client = self._get_redis_client()
            if self.cfg.use_async:
                await client.close()
            else:
                client.close()
        except Exception:
            pass


class LongTermMemory:
    """Orchestrator for persistent memory with support for multiple backends (Qdrant, Redis).

    This class:
    1. Uses embedding_model from backend config (self-contained)
    2. Handles all backend operations (upsert, search, delete) via adapter pattern
    3. Provides clean interface to MASGenerativeModel
    4. Supports both Qdrant and Redis backends via configuration

    The embedding_model is part of the backend config, making configuration self-contained.
    """

    def __init__(
        self,
        backend_config: Union[QdrantConfig, RedisConfig, Dict[str, Any]],
        categories_resolver: Optional[Callable[[Document], List[str]]] = None,
    ) -> None:
        """Initialize LongTermMemory with self-contained backend configuration.

        Args:
            backend_config: QdrantConfig, RedisConfig instance, or dict with backend settings (must include embedding_model)
            categories_resolver: Optional function to extract categories from documents

        Raises:
            ValueError: If backend_config is invalid or embedding_model is not provided/supported
            TypeError: If backend_config type is not recognized
        """
        # Determine backend type and create appropriate config
        if isinstance(backend_config, dict):
            # Infer backend type from dict keys
            if "url" in backend_config and "collection_name" in backend_config:
                self.cfg = QdrantConfig(**backend_config)
                self.backend_type = "qdrant"
            elif "redis_url" in backend_config or "index_name" in backend_config:
                self.cfg = RedisConfig(**backend_config)
                self.backend_type = "redis"
            else:
                # Default to Qdrant for backward compatibility
                self.cfg = QdrantConfig(**backend_config)
                self.backend_type = "qdrant"
        elif isinstance(backend_config, QdrantConfig):
            self.cfg = backend_config
            self.backend_type = "qdrant"
        elif isinstance(backend_config, RedisConfig):
            self.cfg = backend_config
            self.backend_type = "redis"
        else:
            raise TypeError(
                f"backend_config must be QdrantConfig, RedisConfig, or dict, "
                f"got {type(backend_config).__name__}"
            )

        # Validate that embedding_model is provided and supported
        self.cfg.validate_embedding_model()

        self.categories_resolver = categories_resolver

        # Create appropriate adapter based on backend type
        if self.backend_type == "qdrant":
            self.adapter = QdrantAdapter(self.cfg)
        elif self.backend_type == "redis":
            self.adapter = RedisAdapter(self.cfg)
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")

    def _resolve_embed_fn(self) -> Callable[[List[str]], Union[List[List[float]], Any]]:
        """Resolve embedding function from backend config.embedding_model.

        Supports three types of embedding models:
        1. Callable function: embed_fn(texts: List[str]) -> List[List[float]]
        2. LangChain embeddings: has embed_documents(texts) method
        3. Custom class: has embed_documents(texts) method

        Returns:
            Callable that takes List[str] and returns List[List[float]]

        Raises:
            ValueError: If embedding_model is not found or not supported
        """
        model = self.cfg.embedding_model

        if not model:
            raise ValueError(
                f"No embedding_model provided in {self.backend_type.upper()}Config. "
                "Supported: callable, LangChain embeddings, or custom class with embed_documents()"
            )

        # Type 1: Direct callable
        if callable(model):
            return model

        # Type 2: LangChain embeddings or custom class with embed_documents()
        if hasattr(model, 'embed_documents'):
            def _wrap(texts: List[str]):
                return model.embed_documents(texts)
            return _wrap

        # Type 3: Unsupported
        raise ValueError(
            f"embedding_model must be callable or have embed_documents() method, "
            f"got {type(model).__name__}. "
            f"Supported: callable, LangChain embeddings (HuggingFaceEmbeddings, OpenAIEmbeddings, etc.), "
            f"or custom class with embed_documents(texts: List[str]) -> List[List[float]]"
        )

    async def save(self, user_id: Union[str, int], documents: Sequence[Union[Document, str, Dict[str, Any]]]) -> None:
        """Save documents to backend (Qdrant or Redis) with automatic embedding.

        Args:
            user_id: User identifier for isolation
            documents: Documents to save (Document, str, or dict with page_content/metadata)

        Returns:
            None. Saves documents asynchronously to backend.

        Raises:
            Logs errors but does not raise exception.

        Side Effects:
            - Resolves embedding function from backend config
            - Normalizes documents to Document objects
            - Computes embeddings
            - Applies deduplication based on dedup_mode
            - Upserts to backend with user_id isolation
        """
        embedder = self._resolve_embed_fn()
        await self.adapter.upsert_documents(user_id=user_id, documents=documents, embed_fn=embedder, categories_resolver=self.categories_resolver)

    async def search(
        self,
        user_id: Union[str, int],
        query: str,
        k: int = 5,
        categories: Optional[List[str]] = None,
    ) -> List[Document]:
        """Search for similar documents in backend (Qdrant or Redis).

        Args:
            user_id: User identifier for isolation
            query: Query string
            k: Number of results to return
            categories: Optional categories to filter by

        Returns:
            List of similar documents
        """
        embedder = self._resolve_embed_fn()
        return await self.adapter.search(user_id=user_id, query=query, k=k, categories=categories, embed_fn=embedder)

