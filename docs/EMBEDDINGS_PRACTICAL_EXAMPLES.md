# Embeddings Practical Examples

## Real-World Scenarios

### Scenario 1: Using OpenAI Embeddings (API-Based)

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from masai.schema import Document
from langchain_openai import OpenAIEmbeddings
import asyncio

# Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="research_papers",
    vector_size=1536,  # text-embedding-3-small outputs 1536 dims
    embedding_model=embeddings,
    dedup_mode="similarity"
)

memory = LongTermMemory(backend_config=redis_config)

# Create documents
docs = [
    Document(
        page_content="Machine learning is a subset of AI",
        metadata={"source": "textbook", "year": 2024}
    ),
    Document(
        page_content="Deep learning uses neural networks",
        metadata={"source": "paper", "year": 2023}
    )
]

# Save documents
async def save_docs():
    await memory.save(user_id="researcher_1", documents=docs)

# Search
async def search_docs():
    results = await memory.search(
        user_id="researcher_1",
        query="What is machine learning?",
        k=5
    )
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

# Run
asyncio.run(save_docs())
asyncio.run(search_docs())
```

---

### Scenario 2: Using Sentence Transformers (Local)

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from masai.schema import Document
from sentence_transformers import SentenceTransformer
import asyncio

# Load local model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embedder function
def sentence_transformer_embedder(texts: list[str]) -> list[list[float]]:
    """Wrapper for Sentence Transformers"""
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()

# Setup
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="local_embeddings",
    vector_size=384,  # all-MiniLM-L6-v2 outputs 384 dims
    embedding_model=sentence_transformer_embedder,
    dedup_mode="similarity"
)

memory = LongTermMemory(backend_config=redis_config)

# Usage
async def main():
    docs = [
        Document(page_content="Python is a programming language"),
        Document(page_content="JavaScript runs in browsers")
    ]
    
    await memory.save(user_id="user_1", documents=docs)
    
    results = await memory.search(
        user_id="user_1",
        query="programming languages",
        k=2
    )
    
    for doc in results:
        print(doc.page_content)

asyncio.run(main())
```

---

### Scenario 3: Custom Embedder with Caching

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from masai.schema import Document
import asyncio
import hashlib

class CachedEmbedder:
    """Custom embedder with caching to reduce API calls"""
    
    def __init__(self, base_embedder):
        self.base_embedder = base_embedder
        self.cache = {}
    
    def _hash_text(self, text: str) -> str:
        """Create hash of text for caching"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed with caching"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self.base_embedder.embed_documents(uncached_texts)
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = self._hash_text(text)
                self.cache[text_hash] = embedding
        
        # Reconstruct in original order
        result = [None] * len(texts)
        cache_idx = 0
        uncached_idx = 0
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if text_hash in self.cache:
                result[i] = self.cache[text_hash]
        
        return result

# Usage
from langchain_openai import OpenAIEmbeddings

base_embeddings = OpenAIEmbeddings()
cached_embedder = CachedEmbedder(base_embeddings)

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=1536,
    embedding_model=cached_embedder
)

memory = LongTermMemory(backend_config=redis_config)
```

---

### Scenario 4: API-Based Custom Embedder

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from masai.schema import Document
import httpx
import asyncio

async def custom_api_embedder(texts: list[str]) -> list[list[float]]:
    """Embedding via custom API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/embed",
            json={"texts": texts},
            timeout=30.0
        )
        data = response.json()
        return data["embeddings"]

# Setup
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=1024,  # Your API's output dimension
    embedding_model=custom_api_embedder
)

memory = LongTermMemory(backend_config=redis_config)

# Usage
async def main():
    docs = [
        Document(page_content="Text 1"),
        Document(page_content="Text 2")
    ]
    
    await memory.save(user_id="user_1", documents=docs)
    results = await memory.search(user_id="user_1", query="search", k=2)

asyncio.run(main())
```

---

### Scenario 5: Multi-Language Embedder

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from sentence_transformers import SentenceTransformer
import asyncio

# Multi-language model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def multilingual_embedder(texts: list[str]) -> list[list[float]]:
    """Embedder supporting multiple languages"""
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=384,
    embedding_model=multilingual_embedder
)

memory = LongTermMemory(backend_config=redis_config)

# Usage with multiple languages
async def main():
    from masai.schema import Document
    
    docs = [
        Document(page_content="Hello world", metadata={"lang": "en"}),
        Document(page_content="Hola mundo", metadata={"lang": "es"}),
        Document(page_content="Bonjour le monde", metadata={"lang": "fr"})
    ]
    
    await memory.save(user_id="user_1", documents=docs)
    
    # Search in English
    results = await memory.search(
        user_id="user_1",
        query="greeting",
        k=3
    )
    
    for doc in results:
        print(f"{doc.metadata['lang']}: {doc.page_content}")

asyncio.run(main())
```

---

### Scenario 6: Hybrid Embedder (Multiple Models)

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridEmbedder:
    """Combine multiple embedders for better quality"""
    
    def __init__(self):
        self.model1 = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims
        self.model2 = SentenceTransformer('all-mpnet-base-v2')  # 768 dims
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Combine embeddings from multiple models"""
        emb1 = self.model1.encode(texts, convert_to_tensor=False)
        emb2 = self.model2.encode(texts, convert_to_tensor=False)
        
        # Concatenate embeddings
        combined = np.concatenate([emb1, emb2], axis=1)
        return combined.tolist()

# Setup
hybrid_embedder = HybridEmbedder()

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=1152,  # 384 + 768
    embedding_model=hybrid_embedder
)

memory = LongTermMemory(backend_config=redis_config)
```

---

### Scenario 7: Domain-Specific Embedder

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from sentence_transformers import SentenceTransformer
import asyncio

# Fine-tuned model for specific domain
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class DomainEmbedder:
    """Domain-specific embedder with preprocessing"""
    
    def __init__(self, model, domain: str):
        self.model = model
        self.domain = domain
    
    def preprocess(self, text: str) -> str:
        """Domain-specific preprocessing"""
        if self.domain == "medical":
            # Medical text preprocessing
            text = text.lower()
            text = text.replace("dr.", "doctor")
            text = text.replace("pt.", "patient")
        return text
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed with preprocessing"""
        processed = [self.preprocess(text) for text in texts]
        embeddings = self.model.encode(processed, convert_to_tensor=False)
        return embeddings.tolist()

# Setup for medical domain
medical_embedder = DomainEmbedder(model, domain="medical")

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=384,
    embedding_model=medical_embedder
)

memory = LongTermMemory(backend_config=redis_config)
```

---

### Scenario 8: Batch Processing with Progress

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from masai.schema import Document
from langchain_openai import OpenAIEmbeddings
import asyncio
from tqdm import tqdm

class ProgressEmbedder:
    """Embedder with progress tracking"""
    
    def __init__(self, base_embedder):
        self.base_embedder = base_embedder
        self.total_texts = 0
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed with progress bar"""
        self.total_texts += len(texts)
        
        # Process in batches
        batch_size = 10
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.base_embedder.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings

# Usage
embeddings = OpenAIEmbeddings()
progress_embedder = ProgressEmbedder(embeddings)

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    vector_size=1536,
    embedding_model=progress_embedder
)

memory = LongTermMemory(backend_config=redis_config)
```

---

## Comparison: Which Embedder to Use?

| Scenario | Recommendation | Reason |
|----------|---|---|
| **Production, high quality** | OpenAI | Best quality, reliable |
| **Offline, fast** | Sentence Transformers | Local, no API calls |
| **Cost-sensitive** | Sentence Transformers | Free, no API costs |
| **Custom logic** | Custom function | Full control |
| **Multi-language** | Multilingual model | Supports many languages |
| **Domain-specific** | Fine-tuned model | Better for domain |
| **Hybrid quality** | Multiple models | Best quality |
| **External service** | API-based | Outsourced |

---

## Summary

✅ **OpenAI**: Best quality, API-based
✅ **Sentence Transformers**: Fast, local, free
✅ **Custom functions**: Maximum flexibility
✅ **Caching**: Reduce API calls
✅ **Preprocessing**: Domain-specific optimization
✅ **Batch processing**: Handle large datasets
✅ **Progress tracking**: Monitor long operations

