# Memory System

MASAI provides a sophisticated memory system with persistent storage, semantic search, and automatic management.

## Overview

### Memory Types

1. **Chat History**: Recent messages (in-memory)
2. **Context Summaries**: Summarized older messages (in-memory)
3. **Long-Term Memory**: Persistent storage (Redis/Qdrant)

### Memory Flow

```
New Message
    ↓
Add to chat_history
    ↓
chat_history > memory_order?
    ├─ YES: Summarize older messages
    └─ NO: Continue
    ↓
context_summaries > long_context_order?
    ├─ YES: Flush to long-term memory
    └─ NO: Continue
    ↓
On Query: Retrieve from long-term memory (if overflow)
```

## Configuration

### Basic Setup

```python
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
)

agent = manager.create_agent(
    agent_name="assistant",
    agent_details=agent_details,
    memory_config=redis_config,
    persist_memory=True,
    long_context=True,
    memory_order=5,
    long_context_order=10
)
```

---

## 🔌 **Embedding Model Configuration**

MASAI's LongTermMemory supports **three types of embedding models** with automatic adapter resolution:

### **Supported Embedding Types**

| Type | Description | Signature | Example |
|------|-------------|-----------|---------|
| **LangChain Embeddings** | Any LangChain embedding class with `embed_documents()` method | `embed_documents(texts: List[str]) -> List[List[float]]` | `OpenAIEmbeddings`, `HuggingFaceEmbeddings` |
| **SentenceTransformers** | Direct SentenceTransformer model instance | `encode(sentences: List[str]) -> np.ndarray` | `SentenceTransformer('all-MiniLM-L6-v2')` |
| **Custom Callable** | Any callable function that takes list of strings | `fn(texts: List[str]) -> List[List[float]]` | Custom embedding function |

---

### **1️⃣ LangChain Embeddings (Recommended)**

LangChain embeddings are the **recommended approach** because they provide:
- ✅ Standardized interface across providers
- ✅ Built-in error handling and retries
- ✅ Automatic batching for large datasets
- ✅ Support for async operations

#### **OpenAI Embeddings**

```python
from langchain_openai import OpenAIEmbeddings
from masai.Memory.LongTermMemory import QdrantConfig, RedisConfig

# OpenAI text-embedding-3-small (1536 dimensions)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="your-api-key"  # Optional if set in environment
)

# For Qdrant
qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=1536,  # Must match model output
    embedding_model=embeddings
)

# For Redis
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,  # Must match model output
    embedding_model=embeddings
)
```

**Available OpenAI Models:**
- `text-embedding-3-small` (1536 dims) - Fast, cost-effective
- `text-embedding-3-large` (3072 dims) - Higher quality
- `text-embedding-ada-002` (1536 dims) - Legacy model

#### **HuggingFace Embeddings**

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from masai.Memory.LongTermMemory import QdrantConfig

# HuggingFace sentence-transformers model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # 384 dimensions
    model_kwargs={'device': 'cpu'},  # or 'cuda' for GPU
    encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
)

qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=384,  # Must match model output
    embedding_model=embeddings
)
```

**Popular HuggingFace Models:**
- `all-MiniLM-L6-v2` (384 dims) - Fast, lightweight
- `all-mpnet-base-v2` (768 dims) - Better quality
- `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) - Multilingual

---

### **2️⃣ SentenceTransformers (Direct)**

**⚠️ Important:** SentenceTransformer models do **NOT** have an `embed_documents()` method. They use `encode()` which accepts **both single strings and lists of strings**.

MASAI automatically wraps SentenceTransformer models to work with the LongTermMemory interface.

```python
from sentence_transformers import SentenceTransformer
from masai.Memory.LongTermMemory import RedisConfig

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# MASAI automatically detects it doesn't have embed_documents()
# and wraps it internally to work with the memory system
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=384,  # all-MiniLM-L6-v2 outputs 384 dimensions
    embedding_model=model  # Will be auto-wrapped
)
```

**How MASAI Handles SentenceTransformers:**

```python
# Internal adapter logic (you don't need to write this)
# MASAI checks: hasattr(model, 'embed_documents')
# If False but model is callable, it wraps it:

class CallableEmbeddings:
    def __init__(self, fn):
        self.fn = fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # SentenceTransformer.encode() accepts List[str]
        return self.fn(texts).tolist()  # Convert numpy to list

    def embed_query(self, text: str) -> List[float]:
        result = self.fn([text])  # Wrap single string in list
        return result[0].tolist()
```

---

### **3️⃣ Custom Callable Function**

For maximum flexibility, provide your own embedding function:

```python
from masai.Memory.LongTermMemory import QdrantConfig
import numpy as np

def custom_embedder(texts: list[str]) -> list[list[float]]:
    """
    Custom embedding function.

    Args:
        texts: List of strings to embed (ALWAYS a list, never a single string)

    Returns:
        List of embeddings, where each embedding is a list of floats
        Shape: [len(texts), embedding_dim]
    """
    # Example: Use your own model
    embeddings = []
    for text in texts:
        # Your custom logic here
        embedding = your_model.encode(text)  # Returns numpy array or list
        embeddings.append(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)

    return embeddings  # Must return List[List[float]]

qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=768,  # Must match your model's output dimension
    embedding_model=custom_embedder
)
```

**Custom Function Requirements:**
- ✅ **Input:** `texts: List[str]` (always a list, even for single query)
- ✅ **Output:** `List[List[float]]` (list of embeddings)
- ✅ **Shape:** `[len(texts), embedding_dim]`
- ❌ **Never:** Single string input or single embedding output

---

### **🔍 How MASAI Resolves Embedding Models**

MASAI uses a **3-tier adapter system** to support any embedding model:

```python
# Tier 1: LangChain Embeddings (has embed_documents method)
if hasattr(model, 'embed_documents'):
    return model  # Use directly

# Tier 2: Callable (function or SentenceTransformer)
elif callable(model):
    # Wrap it to provide embed_documents interface
    class CallableEmbeddings:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return model(texts)  # Call the function
    return CallableEmbeddings(model)

# Tier 3: Unsupported
else:
    raise ValueError("embedding_model must be callable or have embed_documents()")
```

**Key Points:**
1. **All embedding functions receive `List[str]`** - Never a single string
2. **All embedding functions return `List[List[float]]`** - List of embeddings
3. **MASAI handles the wrapping** - You don't need to write adapters
4. **Vector size must match** - Set `vector_size` parameter correctly

---

### **📊 Embedding Model Comparison**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `text-embedding-3-small` | 1536 | ⚡⚡⚡ | ⭐⭐⭐ | General purpose, cost-effective |
| `text-embedding-3-large` | 3072 | ⚡⚡ | ⭐⭐⭐⭐⭐ | High-quality retrieval |
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡⚡ | ⭐⭐⭐ | Fast, local, lightweight |
| `all-mpnet-base-v2` | 768 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Better quality, local |
| Custom | Variable | Variable | Variable | Specialized domains |

---

### **⚠️ Common Mistakes**

#### **❌ Wrong: Passing single string to custom function**
```python
def wrong_embedder(text: str):  # ❌ Takes single string
    return model.encode(text)
```

#### **✅ Correct: Always accept list of strings**
```python
def correct_embedder(texts: list[str]):  # ✅ Takes list of strings
    return [model.encode(text).tolist() for text in texts]
```

#### **❌ Wrong: Mismatched vector size**
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dims
redis_config = RedisConfig(
    vector_size=384,  # ❌ Wrong! Should be 1536
    embedding_model=embeddings
)
```

#### **✅ Correct: Matching vector size**
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dims
redis_config = RedisConfig(
    vector_size=1536,  # ✅ Correct!
    embedding_model=embeddings
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_order` | int | 5 | Messages before summarization |
| `long_context_order` | int | 10 | Summaries before flush |
| `persist_memory` | bool | False | Enable persistent storage |
| `memory_config` | Config | None | Redis/Qdrant config |

## Redis Backend

### Setup

```bash
# Start Redis
redis-server

# Or with Docker
docker run -d -p 6379:6379 redis:latest
```

### Configuration

```python
from masai.Memory.LongTermMemory import RedisConfig

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=embeddings,
    ttl_seconds=86400,  # 24 hours
    dedup_mode="similarity",
    dedup_similarity_threshold=0.95
)
```

### Features

- **Fast**: In-memory vector search
- **Scalable**: Handles millions of vectors
- **Automatic Indexing**: RediSearch integration
- **TTL Support**: Auto-expiring documents
- **Deduplication**: Automatic duplicate merging

## Qdrant Backend

### Setup

```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

### Configuration

```python
from masai.Memory.LongTermMemory import QdrantConfig

qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=1536,
    embedding_model=embeddings,
    distance="cosine"
)
```

### Features

- **Distributed**: Multi-node support
- **Persistent**: Disk-backed storage
- **Advanced Filtering**: Complex queries
- **Scalable**: Handles large datasets

## Memory Operations

### Save Memories

```python
from langchain_core.documents import Document
# or from masai.schema import Document

documents = [
    Document(
        page_content="User likes Python programming",
        metadata={
            "category": "preferences",
            "confidence": 0.9
        }
    ),
    Document(
        page_content="User works in AI/ML",
        metadata={
            "category": "profession",
            "confidence": 0.95
        }
    )
]

await agent.long_term_memory.save(
    user_id="user_123",
    documents=documents
)
```

### Search Memories

```python
memories = await agent.long_term_memory.search(
    user_id="user_123",
    query="What does user do?",
    k=5,
    categories=["profession"]
)

for memory in memories:
    print(memory.page_content)
    print(memory.metadata)
```

### Delete Memories

```python
await agent.long_term_memory.delete(
    user_id="user_123",
    doc_id="memory_id"
)
```

## Automatic Memory Management

### Context Summarization

When `chat_history > memory_order`:

```
Original Messages:
1. "Hello"
2. "How are you?"
3. "Tell me about AI"
4. "What is machine learning?"
5. "Explain neural networks"

↓ Summarize messages 1-4

Summary: "User greeted and asked about AI and machine learning concepts"
Message 5: "Explain neural networks"
```

### Memory Flushing

When `context_summaries > long_context_order`:

```
context_summaries:
1. Summary 1
2. Summary 2
3. Summary 3
4. Summary 4
5. Summary 5
6. Summary 6

↓ Flush summaries 1-5 to Redis/Qdrant

Remaining in memory:
- Summary 6 (most recent)

In persistent storage:
- Summaries 1-5
```

## User Isolation

All memories are automatically filtered by user_id:

```python
# User A's memories
await agent.long_term_memory.save(
    user_id="user_a",
    documents=[doc1, doc2]
)

# User B's memories
await agent.long_term_memory.save(
    user_id="user_b",
    documents=[doc3, doc4]
)

# Search only returns user_a's memories
memories = await agent.long_term_memory.search(
    user_id="user_a",
    query="...",
    k=5
)
# Returns: [doc1, doc2] (not doc3, doc4)
```

## Deduplication

### Modes

1. **none**: No deduplication
2. **hash**: Hash-based deduplication
3. **similarity**: Semantic similarity-based

### Configuration

```python
redis_config = RedisConfig(
    ...,
    dedup_mode="similarity",
    dedup_similarity_threshold=0.95  # 95% similarity = duplicate
)
```

## Performance Tuning

### Optimize Memory Order

```python
# For short conversations
memory_order=3
long_context_order=5

# For long conversations
memory_order=10
long_context_order=20

# For very long conversations
memory_order=20
long_context_order=50
```

### Optimize Embedding Model

```python
# Fast (smaller)
embedding_model=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # 384 dims
)

# Accurate (larger)
embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-large"  # 3072 dims
)
```

## Troubleshooting

### Memory Not Being Retrieved

Check if overflow occurred:
```python
print(f"Summaries: {len(agent.context_summaries)}")
print(f"Order: {agent.long_context_order}")
# Should have: len(context_summaries) > long_context_order
```

### Redis Connection Error

```bash
# Check Redis is running
redis-cli ping
# Should return: PONG

# Start Redis if needed
redis-server
```

### Slow Memory Search

- Reduce vector size
- Use faster embedding model
- Increase Redis memory
- Use Qdrant for distributed setup

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more help.

