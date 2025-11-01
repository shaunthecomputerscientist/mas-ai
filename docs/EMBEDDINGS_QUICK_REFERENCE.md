# Embeddings Quick Reference

## One-Liner Setup for Each Type

### Type 1: OpenAI (API)
```python
from langchain_openai import OpenAIEmbeddings
from masai.Memory.LongTermMemory import RedisConfig

config = RedisConfig(vector_size=1536, embedding_model=OpenAIEmbeddings())
```

### Type 2: Sentence Transformers (Local)
```python
from sentence_transformers import SentenceTransformer
from masai.Memory.LongTermMemory import RedisConfig

model = SentenceTransformer('all-MiniLM-L6-v2')
config = RedisConfig(vector_size=384, embedding_model=lambda texts: model.encode(texts).tolist())
```

### Type 3: HuggingFace (LangChain)
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from masai.Memory.LongTermMemory import RedisConfig

config = RedisConfig(vector_size=384, embedding_model=HuggingFaceEmbeddings())
```

### Type 4: Custom Function
```python
from masai.Memory.LongTermMemory import RedisConfig

def my_embedder(texts): return [[0.1]*384 for _ in texts]
config = RedisConfig(vector_size=384, embedding_model=my_embedder)
```

### Type 5: Custom Class
```python
from masai.Memory.LongTermMemory import RedisConfig

class MyEmbedder:
    def embed_documents(self, texts): return [[0.1]*384 for _ in texts]

config = RedisConfig(vector_size=384, embedding_model=MyEmbedder())
```

---

## Vector Sizes Cheat Sheet

| Model | Size | Type |
|-------|------|------|
| text-embedding-3-small | 1536 | OpenAI |
| text-embedding-3-large | 3072 | OpenAI |
| all-MiniLM-L6-v2 | 384 | Sentence Transformers |
| all-mpnet-base-v2 | 768 | Sentence Transformers |
| all-roberta-large-v1 | 1024 | Sentence Transformers |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Multilingual |
| embed-english-v3.0 | 1024 | Cohere |

---

## Validation Rules

```python
# ✅ VALID
embedding_model = OpenAIEmbeddings()  # Has embed_documents()
embedding_model = lambda texts: embeddings  # Callable
embedding_model = MyEmbedder()  # Has embed_documents()

# ❌ INVALID
embedding_model = None  # Must provide
embedding_model = "model_name"  # Must be callable or have embed_documents()
embedding_model = 123  # Must be callable or have embed_documents()
```

---

## Function Signature

```python
# Required signature
def embedder(texts: list[str]) -> list[list[float]]:
    """
    Args:
        texts: ["text1", "text2", ...]
    
    Returns:
        [[0.1, 0.2, ..., 0.384], [0.5, 0.6, ..., 0.384], ...]
    """
    pass
```

---

## Common Errors & Fixes

### Error 1: Vector Dimension Mismatch
```python
# ❌ Error
embeddings = OpenAIEmbeddings()  # 1536 dims
config = RedisConfig(vector_size=384, embedding_model=embeddings)

# ✅ Fix
config = RedisConfig(vector_size=1536, embedding_model=embeddings)
```

### Error 2: Missing embedding_model
```python
# ❌ Error
config = RedisConfig(redis_url="...", vector_size=384)

# ✅ Fix
config = RedisConfig(
    redis_url="...",
    vector_size=384,
    embedding_model=OpenAIEmbeddings()
)
```

### Error 3: Wrong Function Signature
```python
# ❌ Error
def bad_embedder(text):  # Single string, not list
    return [0.1, 0.2, ...]  # Single embedding, not list of embeddings

# ✅ Fix
def good_embedder(texts):  # List of strings
    return [[0.1, 0.2, ...], ...]  # List of embeddings
```

### Error 4: Missing embed_documents() Method
```python
# ❌ Error
class BadEmbedder:
    def encode(self, texts):  # Wrong method name
        return embeddings

# ✅ Fix
class GoodEmbedder:
    def embed_documents(self, texts):  # Correct method name
        return embeddings
```

---

## Usage Examples

### Save Documents
```python
from masai.schema import Document

docs = [
    Document(page_content="Text 1"),
    Document(page_content="Text 2")
]

await memory.save(user_id="user_1", documents=docs)
```

### Search Documents
```python
results = await memory.search(
    user_id="user_1",
    query="search query",
    k=5
)

for doc in results:
    print(doc.page_content)
```

---

## Configuration Options

```python
from masai.Memory.LongTermMemory import RedisConfig

config = RedisConfig(
    # Connection
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    
    # Embeddings (REQUIRED)
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    
    # Search
    distance_metric="cosine",  # cosine|l2|ip
    
    # Deduplication
    dedup_mode="similarity",  # none|similarity|hash
    dedup_similarity_threshold=0.95,
    
    # TTL (optional)
    ttl_seconds=86400,
    
    # Connection pool
    connection_pool_size=10,
    socket_timeout=5.0,
    socket_connect_timeout=5.0,
    socket_keepalive=True
)
```

---

## Qdrant Configuration

```python
from masai.Memory.LongTermMemory import QdrantConfig

config = QdrantConfig(
    # Connection
    url="http://localhost:6333",
    api_key=None,  # For Qdrant Cloud
    collection_name="masai_memories",
    
    # Embeddings (REQUIRED)
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    
    # Search
    distance="cosine",  # cosine|dot|euclid
    
    # Deduplication
    dedup_mode="similarity",  # none|similarity|hash
    dedup_similarity_threshold=0.9,
    
    # Performance
    prefer_async=True,
    timeout_sec=10.0,
    consistency=None
)
```

---

## Complete Example

```python
import asyncio
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from masai.schema import Document
from langchain_openai import OpenAIEmbeddings

async def main():
    # Setup
    config = RedisConfig(
        redis_url="redis://localhost:6379",
        index_name="research",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings(),
        dedup_mode="similarity"
    )
    
    memory = LongTermMemory(backend_config=config)
    
    # Save
    docs = [
        Document(page_content="Machine learning basics"),
        Document(page_content="Deep learning overview")
    ]
    await memory.save(user_id="user_1", documents=docs)
    
    # Search
    results = await memory.search(
        user_id="user_1",
        query="What is machine learning?",
        k=5
    )
    
    for doc in results:
        print(f"Content: {doc.page_content}")

asyncio.run(main())
```

---

## Decision Tree

```
Do you have embeddings?
├─ Yes, from LangChain (OpenAI, HuggingFace, etc.)
│  └─ Use directly: embedding_model=OpenAIEmbeddings()
├─ Yes, from Sentence Transformers
│  └─ Wrap in lambda: embedding_model=lambda texts: model.encode(texts).tolist()
├─ Yes, custom function
│  └─ Use directly: embedding_model=my_embedder
├─ Yes, custom class with embed_documents()
│  └─ Use directly: embedding_model=MyEmbedder()
└─ Yes, from external API
   └─ Use async function: embedding_model=async_api_embedder
```

---

## Performance Tips

1. **Batch embeddings**: Process multiple texts at once
2. **Cache results**: Store embeddings for repeated texts
3. **Use local models**: Sentence Transformers for offline
4. **Use API models**: OpenAI for best quality
5. **Async operations**: Use async/await for I/O
6. **Connection pooling**: Configure pool_size for Redis

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dimension mismatch | Check vector_size matches embedder output |
| Slow embeddings | Use local model or increase batch size |
| API errors | Check API key and rate limits |
| Memory errors | Reduce batch size or use streaming |
| Connection errors | Check Redis/Qdrant is running |

---

## Summary

✅ **5 embedding types supported**
✅ **Flexible and extensible**
✅ **No vendor lock-in**
✅ **Simple validation**
✅ **Easy to use**

**Choose your embedder and go!** 🚀

