# Embeddings Technical Reference

## Internal Implementation Details

### How MASAI Validates Embeddings

Located in `src/masai/Memory/LongTermMemory.py`:

```python
def validate_embedding_model(self) -> None:
    """Validate that embedding_model is properly configured."""
    if self.embedding_model is None:
        raise ValueError("embedding_model is required...")
    
    model = self.embedding_model
    
    # Type 1: Direct callable
    if callable(model):
        return  # ✅ Valid
    
    # Type 2: LangChain embeddings or custom class with embed_documents()
    if hasattr(model, 'embed_documents'):
        return  # ✅ Valid
    
    # Type 3: Unsupported
    raise ValueError(f"embedding_model must be callable or have embed_documents()...")
```

**Validation happens at:**
1. Config creation time (when you create RedisConfig/QdrantConfig)
2. LongTermMemory initialization
3. Before any search/save operations

---

### How MASAI Resolves Embeddings at Runtime

#### For LongTermMemory (lines 884-922):

```python
def _resolve_embed_fn(self) -> Callable[[List[str]], Union[List[List[float]], Any]]:
    """Resolve embedding function from backend config.embedding_model."""
    model = self.cfg.embedding_model
    
    if not model:
        raise ValueError("No embedding_model provided...")
    
    # Type 1: Direct callable
    if callable(model):
        return model  # Use directly
    
    # Type 2: LangChain embeddings or custom class with embed_documents()
    if hasattr(model, 'embed_documents'):
        def _wrap(texts: List[str]):
            return model.embed_documents(texts)
        return _wrap  # Wrap and return
    
    # Type 3: Unsupported
    raise ValueError(f"embedding_model must be callable...")
```

#### For RedisAdapter (lines 674-722):

```python
def _resolve_embeddings(self):
    """Resolve embedding model from RedisConfig.embedding_model."""
    model = self.cfg.embedding_model
    
    if not model:
        raise ValueError("No embedding_model provided...")
    
    # Type 1: Direct callable - wrap it
    if callable(model) and not hasattr(model, 'embed_documents'):
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
    raise ValueError(f"embedding_model must be callable...")
```

---

## Function Signatures

### Required Signature for Callable

```python
def embedder(texts: List[str]) -> List[List[float]]:
    """
    Args:
        texts: List of strings to embed
        
    Returns:
        List of embeddings, where each embedding is List[float]
        
    Example:
        Input:  ["Hello", "World"]
        Output: [[0.1, 0.2, ..., 0.384], [0.5, 0.6, ..., 0.384]]
    """
    pass
```

### Required Method for Custom Class

```python
class CustomEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embeddings, where each embedding is List[float]
        """
        pass
    
    # Optional but recommended
    def embed_query(self, text: str) -> List[float]:
        """
        Args:
            text: Single string to embed
            
        Returns:
            Single embedding as List[float]
        """
        pass
```

### LangChain Embeddings Interface

```python
from langchain_core.embeddings import Embeddings

class Embeddings(BaseModel):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        pass
```

---

## Data Flow: How Embeddings Are Used

### During Save/Upsert

```
1. User calls: memory.save(user_id, documents)
   ↓
2. LongTermMemory._resolve_embed_fn() is called
   ↓
3. Returns embedding function: List[str] -> List[List[float]]
   ↓
4. Adapter.upsert_documents() is called with embed_fn
   ↓
5. For each document:
   - Extract page_content (text)
   - Call embed_fn([text])
   - Get embedding vector
   - Store in Redis/Qdrant with metadata
```

### During Search

```
1. User calls: memory.search(user_id, query)
   ↓
2. LongTermMemory._resolve_embed_fn() is called
   ↓
3. Returns embedding function: List[str] -> List[List[float]]
   ↓
4. Adapter.search() is called with embed_fn
   ↓
5. Call embed_fn([query])
   ↓
6. Get query embedding vector
   ↓
7. Search Redis/Qdrant for similar vectors
   ↓
8. Return top-k documents with similarity scores
```

---

## Vector Dimension Requirements

### How to Determine Vector Size

```python
# Method 1: Check embedder documentation
from langchain_openai import OpenAIEmbeddings
# OpenAI text-embedding-3-small: 1536 dims
# OpenAI text-embedding-3-large: 3072 dims

# Method 2: Test the embedder
def get_vector_size(embedder):
    """Get vector dimension from embedder"""
    test_embedding = embedder.embed_documents(["test"])
    return len(test_embedding[0])

# Method 3: Check model card
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(model.get_sentence_embedding_dimension())  # 384
```

### Common Vector Sizes

| Model | Dimension | Provider |
|-------|-----------|----------|
| text-embedding-3-small | 1536 | OpenAI |
| text-embedding-3-large | 3072 | OpenAI |
| all-MiniLM-L6-v2 | 384 | Sentence Transformers |
| all-mpnet-base-v2 | 768 | Sentence Transformers |
| all-roberta-large-v1 | 1024 | Sentence Transformers |
| embed-english-v3.0 | 1024 | Cohere |
| Vertex AI | 768 | Google |

---

## Error Handling

### Common Errors and Solutions

#### Error 1: embedding_model is None

```python
# ❌ WRONG
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    embedding_model=None  # Error!
)

# ✅ CORRECT
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    embedding_model=OpenAIEmbeddings()
)
```

#### Error 2: Wrong Vector Dimension

```python
# ❌ WRONG
embeddings = OpenAIEmbeddings()  # 1536 dims
redis_config = RedisConfig(
    vector_size=384,  # Mismatch!
    embedding_model=embeddings
)

# ✅ CORRECT
redis_config = RedisConfig(
    vector_size=1536,  # Matches OpenAI
    embedding_model=embeddings
)
```

#### Error 3: Callable Returns Wrong Type

```python
# ❌ WRONG
def bad_embedder(texts):
    return "embedding"  # String, not List[List[float]]

# ✅ CORRECT
def good_embedder(texts):
    return [[0.1, 0.2, ..., 0.384], ...]  # List[List[float]]
```

#### Error 4: Missing embed_documents() Method

```python
# ❌ WRONG
class BadEmbedder:
    def encode(self, texts):  # Wrong method name
        return embeddings

# ✅ CORRECT
class GoodEmbedder:
    def embed_documents(self, texts):  # Correct method name
        return embeddings
```

---

## Performance Considerations

### Batch Embedding

```python
# ✅ GOOD - Batch multiple texts
embeddings = embedder.embed_documents([
    "Text 1",
    "Text 2",
    "Text 3"
])  # Single API call for 3 texts

# ❌ BAD - Individual embeddings
for text in texts:
    embedding = embedder.embed_documents([text])  # 3 API calls
```

### Caching Embeddings

```python
# For expensive embedders, cache results
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedder(text: str):
    """Cache embeddings for repeated texts"""
    embedding = embedder.embed_documents([text])
    return tuple(embedding[0])  # Convert to tuple for hashing
```

### Async Embedders

```python
# MASAI supports async embedders
async def async_embedder(texts: list[str]) -> list[list[float]]:
    """Async embedding function"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/embed",
            json={"texts": texts}
        )
        return response.json()["embeddings"]

redis_config = RedisConfig(
    embedding_model=async_embedder
)
```

---

## Advanced: Custom Embedder with State

```python
class StatefulEmbedder:
    """Embedder with internal state (e.g., model, cache)"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = load_model(model_name)
        self.cache = {}
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed with caching"""
        embeddings = []
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                embedding = self.model.encode(text)
                self.cache[text] = embedding
                embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]

# Use with MASAI
embedder = StatefulEmbedder("all-MiniLM-L6-v2")
redis_config = RedisConfig(
    vector_size=384,
    embedding_model=embedder
)
```

---

## Summary

✅ **MASAI accepts any embedding function/class**
✅ **Only requirement: takes List[str], returns List[List[float]]**
✅ **Validation at config time, resolution at runtime**
✅ **Automatic wrapping of callables**
✅ **Vector dimension must match config**
✅ **No decoding - only encoding**
✅ **Supports sync, async, stateful embedders**

