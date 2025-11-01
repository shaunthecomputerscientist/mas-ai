# MASAI Document Class Guide

## Overview

MASAI provides its own **Document** class to eliminate LangChain dependency. It's a drop-in replacement that maintains API compatibility while being lightweight and focused.

---

## Why MASAI Has Its Own Document Class

### Problem with LangChain Dependency

```python
# ❌ LangChain approach (heavy dependency)
from langchain.schema import Document
# Requires: langchain, langchain-core, and many transitive deps
```

### MASAI Solution

```python
# ✅ MASAI approach (lightweight)
from masai.schema import Document
# Requires: Only Pydantic (already a dependency)
```

### Benefits

| Aspect | MASAI Document | LangChain Document |
|--------|---|---|
| Dependencies | Pydantic only | LangChain + transitive |
| User Isolation | Built-in | Not built-in |
| Metadata | Full support | Limited |
| Serialization | to_dict(), from_dict() | Limited |
| Size | ~100 lines | ~1000+ lines |
| Flexibility | Customizable | Fixed |

---

## Document Class

### Constructor

```python
from masai.schema import Document

# Basic usage
doc = Document(
    page_content="Hello world",
    metadata={"source": "test"}
)

# With empty metadata
doc = Document(page_content="Hello world")

# Metadata is optional
doc = Document(
    page_content="Content here",
    metadata=None  # Defaults to {}
)
```

### Attributes

```python
doc = Document(
    page_content="Machine learning is...",
    metadata={
        "source": "wikipedia",
        "page": 1,
        "user_id": "user_123",
        "categories": ["AI", "ML"]
    }
)

# Access attributes
print(doc.page_content)  # "Machine learning is..."
print(doc.metadata)      # {"source": "wikipedia", ...}
print(doc.metadata["source"])  # "wikipedia"
```

---

## Methods

### to_dict()

```python
doc = Document(
    page_content="Hello",
    metadata={"source": "test"}
)

# Convert to dictionary
doc_dict = doc.to_dict()
print(doc_dict)
# Output: {
#     "page_content": "Hello",
#     "metadata": {"source": "test"}
# }
```

### from_dict()

```python
# Create from dictionary
doc_dict = {
    "page_content": "Hello",
    "metadata": {"source": "test"}
}

doc = Document.from_dict(doc_dict)
print(doc.page_content)  # "Hello"
```

### String Representations

```python
doc = Document(
    page_content="This is a long document content...",
    metadata={"source": "test"}
)

# Short representation
print(str(doc))
# Output: Document(page_content='This is a long docum...', metadata={'source': 'test'})

# Full representation
print(repr(doc))
# Output: Document(page_content='This is a long document content...', metadata={'source': 'test'})
```

---

## Usage in Memory Systems

### With Redis

```python
from masai.schema import Document
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from langchain_openai import OpenAIEmbeddings

# Create documents
docs = [
    Document(
        page_content="Machine learning basics",
        metadata={"source": "tutorial", "level": "beginner"}
    ),
    Document(
        page_content="Deep learning advanced",
        metadata={"source": "research", "level": "advanced"}
    )
]

# Setup Redis
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

memory = LongTermMemory(backend_config=redis_config)

# Upsert documents
await memory.upsert_documents(
    user_id="user_123",
    documents=docs,
    embed_fn=lambda texts: embeddings_model.embed_documents(texts)
)

# Search
results = await memory.search(
    query="machine learning",
    user_id="user_123",
    top_k=5
)

for doc in results:
    print(f"Found: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### With Qdrant

```python
from masai.schema import Document
from masai.Memory.LongTermMemory import QdrantConfig, LongTermMemory

# Create documents
docs = [
    Document(
        page_content="Quantum computing overview",
        metadata={"source": "arxiv", "year": 2024}
    ),
    Document(
        page_content="Quantum algorithms",
        metadata={"source": "textbook", "year": 2023}
    )
]

# Setup Qdrant
qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="research",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

memory = LongTermMemory(backend_config=qdrant_config)

# Upsert documents
await memory.upsert_documents(
    user_id="researcher_1",
    documents=docs,
    embed_fn=lambda texts: embeddings_model.embed_documents(texts)
)

# Search
results = await memory.search(
    query="quantum algorithms",
    user_id="researcher_1",
    top_k=3
)
```

---

## Metadata Handling

### Standard Metadata Fields

```python
doc = Document(
    page_content="Content",
    metadata={
        "source": "file.txt",           # Where it came from
        "page": 1,                      # Page number
        "user_id": "user_123",          # User identifier
        "categories": ["AI", "ML"],     # Categories
        "timestamp": "2024-01-01",      # When it was created
        "relevance": 0.95               # Relevance score
    }
)
```

### Custom Metadata

```python
# Add any custom fields
doc = Document(
    page_content="Research paper",
    metadata={
        "authors": ["Alice", "Bob"],
        "citations": 150,
        "doi": "10.1234/example",
        "custom_field": "custom_value"
    }
)

# Access custom fields
print(doc.metadata["authors"])
print(doc.metadata["citations"])
```

### User Isolation

```python
# Documents are isolated by user_id
doc1 = Document(
    page_content="User 1 data",
    metadata={"user_id": "user_1"}
)

doc2 = Document(
    page_content="User 2 data",
    metadata={"user_id": "user_2"}
)

# When searching, only user_1's documents are returned
results = await memory.search(
    query="data",
    user_id="user_1",
    top_k=10
)
# Only returns doc1, not doc2
```

---

## Integration with Agent Memory

### Adding to Agent Memory

```python
from masai.schema import Document

# Create document
doc = Document(
    page_content="Important research finding",
    metadata={"source": "research", "importance": "high"}
)

# Add to agent's long-term memory
await agent.llm_router.long_term_memory.upsert_documents(
    user_id="user_123",
    documents=[doc],
    embed_fn=embedding_function
)
```

### Retrieving from Agent Memory

```python
# Search agent's memory
results = await agent.llm_router.long_term_memory.search(
    query="research findings",
    user_id="user_123",
    top_k=5
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Source: {doc.metadata.get('source')}")
```

---

## Conversion Examples

### String to Document

```python
text = "This is a document"
doc = Document(page_content=text)
```

### Dict to Document

```python
data = {
    "page_content": "Content",
    "metadata": {"source": "api"}
}
doc = Document.from_dict(data)
```

### Document to Dict

```python
doc = Document(page_content="Content", metadata={"source": "api"})
data = doc.to_dict()
# Can be serialized to JSON
import json
json_str = json.dumps(data)
```

### List of Documents

```python
docs = [
    Document(page_content=f"Document {i}", metadata={"index": i})
    for i in range(10)
]

# Convert all to dicts
docs_dicts = [doc.to_dict() for doc in docs]

# Convert back
docs_restored = [Document.from_dict(d) for d in docs_dicts]
```

---

## Best Practices

### 1. Always Include Metadata

```python
# ✅ GOOD - Include relevant metadata
doc = Document(
    page_content="Content",
    metadata={
        "source": "file.txt",
        "user_id": "user_123",
        "timestamp": "2024-01-01"
    }
)

# ❌ BAD - No metadata
doc = Document(page_content="Content")
```

### 2. Use Consistent Metadata Keys

```python
# ✅ GOOD - Consistent keys
docs = [
    Document(page_content="...", metadata={"source": "file1", "user_id": "user_1"}),
    Document(page_content="...", metadata={"source": "file2", "user_id": "user_1"})
]

# ❌ BAD - Inconsistent keys
docs = [
    Document(page_content="...", metadata={"source": "file1"}),
    Document(page_content="...", metadata={"origin": "file2"})
]
```

### 3. Include User ID for Isolation

```python
# ✅ GOOD - User ID included
doc = Document(
    page_content="User data",
    metadata={"user_id": "user_123"}
)

# ❌ BAD - No user ID
doc = Document(page_content="User data")
```

### 4. Keep page_content Concise

```python
# ✅ GOOD - Focused content
doc = Document(
    page_content="Machine learning is a subset of AI",
    metadata={"source": "definition"}
)

# ❌ BAD - Too long
doc = Document(
    page_content="Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on tasks through experience. It is a key component of modern AI systems..."
)
```

---

## Troubleshooting

### Issue: "Document not found"

```python
# Make sure to use MASAI Document
from masai.schema import Document  # ✅ CORRECT
# from langchain.schema import Document  # ❌ WRONG
```

### Issue: "Metadata not preserved"

```python
# Ensure metadata is passed correctly
doc = Document(
    page_content="Content",
    metadata={"key": "value"}  # ✅ Must be dict
)
```

### Issue: "User isolation not working"

```python
# Always include user_id in metadata
doc = Document(
    page_content="Content",
    metadata={"user_id": "user_123"}  # ✅ Required for isolation
)
```

---

## See Also

- [LANGCHAIN_AGNOSTIC_GUIDE.md](LANGCHAIN_AGNOSTIC_GUIDE.md) - LangChain independence
- [REDIS_QDRANT_CRUD_GUIDE.md](REDIS_QDRANT_CRUD_GUIDE.md) - Memory operations
- [MEMORY_ARCHITECTURE_DEEP_DIVE.md](MEMORY_ARCHITECTURE_DEEP_DIVE.md) - Memory system

