# Embeddings Configuration Guide for LongTermMemory

## Overview

MASAI's `LongTermMemory` class is **extremely flexible** with embeddings. It supports:

1. **LangChain embeddings** (OpenAI, HuggingFace, Cohere, etc.)
2. **Sentence Transformers** (local models)
3. **Custom callable functions** (any function that encodes text)
4. **Custom classes** with `embed_documents()` method
5. **API-based embedders** (external services)

---

## Supported Embedding Types

### Type 1: Direct Callable Function ✅

**Most Flexible Option** - Any function that takes `List[str]` and returns `List[List[float]]`

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
import numpy as np

# Simple custom embedder
def my_embedder(texts: list[str]) -> list[list[float]]:
    """Custom embedding function - just encode text"""
    embeddings = []
    for text in texts:
        # Your embedding logic here
        embedding = np.random.rand(384)  # Example: 384-dim vector
        embeddings.append(embedding.tolist())
    return embeddings

# Use with LongTermMemory
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=384,  # Must match embedder output dimension
    embedding_model=my_embedder  # Pass the function directly
)

memory = LongTermMemory(backend_config=redis_config)
```

**Key Points:**
- ✅ Function receives `List[str]` (list of texts)
- ✅ Function returns `List[List[float]]` (list of embeddings)
- ✅ Each embedding is a list of floats
- ✅ Vector dimension must match `vector_size` in config
- ✅ No decoding needed - just encoding

---

### Type 2: Sentence Transformers ✅

**Local, Lightweight Models**

```python
from sentence_transformers import SentenceTransformer
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims

# Create wrapper function
def sentence_transformer_embedder(texts: list[str]) -> list[list[float]]:
    """Wrapper for Sentence Transformers"""
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()

# Use with LongTermMemory
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=384,  # all-MiniLM-L6-v2 outputs 384 dims
    embedding_model=sentence_transformer_embedder
)

memory = LongTermMemory(backend_config=redis_config)
```

**Popular Sentence Transformer Models:**
- `all-MiniLM-L6-v2` - 384 dims (fast, lightweight)
- `all-mpnet-base-v2` - 768 dims (better quality)
- `all-roberta-large-v1` - 1024 dims (highest quality)

---

### Type 3: LangChain Embeddings ✅

**OpenAI, HuggingFace, Cohere, etc.**

```python
from langchain_openai import OpenAIEmbeddings
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory

# OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dims

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=embeddings  # LangChain object with embed_documents()
)

memory = LongTermMemory(backend_config=redis_config)
```

**Supported LangChain Embeddings:**
- `OpenAIEmbeddings` - 1536 or 3072 dims
- `HuggingFaceEmbeddings` - 384-1024 dims
- `CohereEmbeddings` - 4096 dims
- `VertexAIEmbeddings` - 768 dims
- `BedrockEmbeddings` - varies

---

### Type 4: Custom Class with embed_documents() ✅

**For Complex Embedding Logic**

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
import numpy as np

class CustomEmbedder:
    """Custom embedder class with embed_documents() method"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Load your model here
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Required method - takes texts, returns embeddings"""
        embeddings = []
        for text in texts:
            # Your embedding logic
            embedding = np.random.rand(384)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Optional method - for single query embedding"""
        result = self.embed_documents([text])
        return result[0] if result else []

# Use with LongTermMemory
embedder = CustomEmbedder("my-model")

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=384,
    embedding_model=embedder  # Custom class instance
)

memory = LongTermMemory(backend_config=redis_config)
```

**Key Points:**
- ✅ Must have `embed_documents(texts: List[str]) -> List[List[float]]` method
- ✅ Optional `embed_query(text: str) -> List[float]` method
- ✅ MASAI automatically wraps it and calls `embed_documents()`

---

### Type 5: API-Based Embedder ✅

**External Embedding Services**

```python
import httpx
import numpy as np
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory

async def api_embedder(texts: list[str]) -> list[list[float]]:
    """Embedding via external API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/embed",
            json={"texts": texts}
        )
        embeddings = response.json()["embeddings"]
        return embeddings

# Use with LongTermMemory
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=api_embedder
)

memory = LongTermMemory(backend_config=redis_config)
```

---

## How MASAI Handles Embeddings

### Validation (at config time)

```python
# MASAI checks if embedding_model is:
# 1. Callable (function)
# 2. Has embed_documents() method (LangChain or custom class)

# If neither, raises ValueError
```

### Resolution (at runtime)

```python
# When you call memory.search() or memory.save():
# 1. MASAI calls _resolve_embed_fn()
# 2. If callable: uses it directly
# 3. If has embed_documents(): wraps it
# 4. Returns a function: List[str] -> List[List[float]]
```

### Wrapping Callables

```python
# MASAI automatically wraps plain callables:
class CallableEmbeddings:
    def __init__(self, fn):
        self.fn = fn
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.fn(texts)
    
    def embed_query(self, text: str) -> List[float]:
        result = self.fn([text])
        return result[0] if result else []
```

---

## Important: Vector Dimension Matching

**CRITICAL**: `vector_size` in config must match embedder output dimension

```python
# ✅ CORRECT
embeddings = OpenAIEmbeddings()  # 1536 dims
redis_config = RedisConfig(
    vector_size=1536,  # Matches OpenAI output
    embedding_model=embeddings
)

# ❌ WRONG - Will cause errors
redis_config = RedisConfig(
    vector_size=384,  # Doesn't match OpenAI's 1536
    embedding_model=embeddings
)
```

---

## Encoding vs Decoding

**MASAI Only Handles Encoding**

```python
# ✅ Your function only needs to ENCODE
def my_embedder(texts: list[str]) -> list[list[float]]:
    # Input: ["Hello world", "How are you"]
    # Output: [[0.1, 0.2, ..., 0.384], [0.5, 0.6, ..., 0.384]]
    # Just encoding - no decoding needed
    return embeddings

# ❌ MASAI doesn't handle decoding
# You cannot decode embeddings back to text
# Embeddings are lossy - information is lost
```

---

## Comparison Table

| Type | Pros | Cons | Use Case |
|------|------|------|----------|
| **Callable** | Most flexible, lightweight | Need to implement | Custom logic |
| **Sentence Transformers** | Local, fast, no API calls | Requires download | Production, offline |
| **LangChain** | Well-tested, many options | Dependency | Quick setup |
| **Custom Class** | Full control, reusable | More code | Complex logic |
| **API-Based** | No local resources | Network dependent | Cloud-only |

---

## Complete Example: Multi-Option Setup

```python
from masai.Memory.LongTermMemory import RedisConfig, QdrantConfig, LongTermMemory
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# Option 1: OpenAI (API-based)
openai_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

# Option 2: Sentence Transformers (local)
st_model = SentenceTransformer('all-MiniLM-L6-v2')
st_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=384,
    embedding_model=lambda texts: st_model.encode(texts).tolist()
)

# Option 3: Custom function
def custom_embedder(texts):
    # Your logic
    return embeddings

custom_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=384,
    embedding_model=custom_embedder
)

# All work the same way
memory1 = LongTermMemory(backend_config=openai_config)
memory2 = LongTermMemory(backend_config=st_config)
memory3 = LongTermMemory(backend_config=custom_config)
```

---

## Summary

✅ **MASAI is extremely flexible with embeddings**
✅ **Supports any function that encodes text to vectors**
✅ **No decoding needed - just encoding**
✅ **Vector dimension must match config**
✅ **Works with LangChain, Sentence Transformers, custom functions, APIs**
✅ **No vendor lock-in - use any embedder**

**Choose based on your needs:**
- **Speed & offline**: Sentence Transformers
- **Quality & API**: OpenAI/LangChain
- **Custom logic**: Callable function
- **Complex setup**: Custom class
- **External service**: API-based

