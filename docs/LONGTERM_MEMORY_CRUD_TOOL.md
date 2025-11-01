# LongTermMemory CRUD Tool

## Overview

The **LongTermMemoryCRUDTool** is a comprehensive wrapper around MASAI's LongTermMemory module that provides a unified interface for all persistent memory operations: **Create, Read (Search), Update, Delete**.

**Location**: `src/masai/Tools/tools/LongTermMemoryCRUDTool.py`

---

## Features

✅ **Unified CRUD Interface**: Single tool for all memory operations
✅ **Backend Agnostic**: Works with both Qdrant and Redis
✅ **User Isolation**: Built-in user_id filtering for multi-tenant scenarios
✅ **Category Filtering**: Optional category-based document filtering
✅ **Flexible Document Format**: Accepts strings, dicts, or Document objects
✅ **Async-First**: Designed for async/await patterns
✅ **Tool Compatible**: Works with MASAI's Tool framework
✅ **Error Handling**: Comprehensive error handling with detailed messages

---

## Installation & Setup

### 1. Import the Tool

```python
from masai.Tools.tools.LongTermMemoryCRUDTool import (
    LongTermMemoryCRUDTool,
    create_longterm_memory_tool
)
from masai.Memory.LongTermMemory import QdrantConfig, RedisConfig
```

### 2. Configure Backend

#### Option A: Qdrant Backend

```python
from langchain_openai import OpenAIEmbeddings

qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    distance="cosine",
    dedup_mode="similarity"
)

tool = LongTermMemoryCRUDTool(memory_config=qdrant_config)
```

#### Option B: Redis Backend

```python
from langchain_openai import OpenAIEmbeddings

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="similarity"
)

tool = LongTermMemoryCRUDTool(memory_config=redis_config)
```

#### Option C: Dict Configuration

```python
config = {
    "url": "http://localhost:6333",
    "collection_name": "masai_memories",
    "vector_size": 1536,
    "embedding_model": OpenAIEmbeddings(),
    "distance": "cosine"
}

tool = LongTermMemoryCRUDTool(memory_config=config)
```

---

## Operations

### 1. CREATE - Save Documents

Save new documents to persistent memory.

```python
result = await tool.ainvoke({
    "mode": "create",
    "user_id": "user_123",
    "documents": [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ],
    "categories": ["ai", "ml"]
})

# Response
{
    "status": "success",
    "operation": "create",
    "message": "Successfully saved 2 document(s) for user user_123",
    "count": 2,
    "backend": "qdrant"
}
```

**Parameters**:
- `mode`: "create" (required)
- `user_id`: User identifier (required)
- `documents`: List of strings, dicts, or Document objects (required)
- `categories`: Optional list of categories to tag documents

**Document Formats**:

```python
# Format 1: Plain strings
documents = ["Document 1", "Document 2"]

# Format 2: Dicts with metadata
documents = [
    {
        "page_content": "Document 1",
        "metadata": {"source": "book", "year": 2024}
    }
]

# Format 3: Document objects
from masai.schema import Document
documents = [
    Document(page_content="Document 1", metadata={"source": "book"})
]
```

---

### 2. SEARCH - Query Documents

Search for similar documents using semantic similarity.

```python
result = await tool.ainvoke({
    "mode": "search",
    "user_id": "user_123",
    "query": "What is machine learning?",
    "k": 5,
    "categories": ["ai", "ml"]
})

# Response
{
    "status": "success",
    "operation": "search",
    "message": "Found 2 document(s) matching query",
    "query": "What is machine learning?",
    "results": [
        {
            "content": "Machine learning is a subset of AI",
            "metadata": {"categories": ["ai", "ml"], "user_id": "user_123"},
            "doc_id": "point_id_123"
        },
        {
            "content": "Deep learning uses neural networks",
            "metadata": {"categories": ["ai", "ml"], "user_id": "user_123"},
            "doc_id": "point_id_124"
        }
    ],
    "count": 2,
    "backend": "qdrant"
}
```

**Parameters**:
- `mode`: "search" (required)
- `user_id`: User identifier (required)
- `query`: Search query string (required)
- `k`: Number of results to return (default: 5)
- `categories`: Optional categories to filter by

---

### 3. UPDATE - Upsert Documents

Update existing documents or create new ones (upsert operation).

```python
result = await tool.ainvoke({
    "mode": "update",
    "user_id": "user_123",
    "documents": [
        "Updated machine learning definition",
        "New deep learning content"
    ],
    "categories": ["ai", "ml"]
})

# Response
{
    "status": "success",
    "operation": "update",
    "message": "Successfully updated 2 document(s) for user user_123",
    "count": 2,
    "backend": "qdrant"
}
```

**Parameters**:
- `mode`: "update" (required)
- `user_id`: User identifier (required)
- `documents`: List of documents to update (required)
- `categories`: Optional categories to tag documents

**Note**: Update uses upsert semantics - if document exists (based on dedup_mode), it's updated; otherwise, it's created.

---

### 4. DELETE - Remove Documents

Delete a specific document from persistent memory.

```python
result = await tool.ainvoke({
    "mode": "delete",
    "user_id": "user_123",
    "doc_id": "point_id_123"
})

# Response
{
    "status": "success",
    "operation": "delete",
    "message": "Successfully deleted document point_id_123 for user user_123",
    "doc_id": "point_id_123",
    "backend": "qdrant"
}
```

**Parameters**:
- `mode`: "delete" (required)
- `user_id`: User identifier (required)
- `doc_id`: Document ID to delete (required)

**Getting doc_id**: The `doc_id` is returned in search results as `doc_id` field in metadata.

---

## Advanced Usage

### With Categories Resolver

```python
def extract_categories(doc):
    """Extract categories from document metadata"""
    return doc.metadata.get("tags", [])

tool = LongTermMemoryCRUDTool(
    memory_config=config,
    categories_resolver=extract_categories
)
```

### As MASAI Tool

```python
from masai.Tools.tools.LongTermMemoryCRUDTool import create_longterm_memory_tool

# Create Tool wrapper
tool = create_longterm_memory_tool(memory_config=config)

# Use with agents
result = await tool.ainvoke({
    "mode": "search",
    "user_id": "user_123",
    "query": "search query"
})
```

### Batch Operations

```python
# Create multiple documents
documents = [
    f"Document {i}: Content about topic {i}"
    for i in range(100)
]

result = await tool.ainvoke({
    "mode": "create",
    "user_id": "user_123",
    "documents": documents,
    "categories": ["batch", "import"]
})
```

---

## Error Handling

All operations return structured responses with status and error information:

```python
# Error response
{
    "status": "error",
    "operation": "search",
    "message": "Error searching documents: ...",
    "error": "Detailed error message"
}
```

**Common Errors**:

| Error | Cause | Solution |
|-------|-------|----------|
| Missing required parameters | mode or user_id not provided | Provide both mode and user_id |
| Unknown mode | Invalid mode value | Use: create, search, update, delete |
| Connection error | Backend not running | Start Qdrant/Redis server |
| Embedding error | Embedding model not configured | Provide valid embedding_model in config |

---

## Configuration Options

### Qdrant Config

```python
QdrantConfig(
    url="http://localhost:6333",
    api_key=None,  # For Qdrant Cloud
    collection_name="masai_memories",
    vector_size=1536,
    distance="cosine",  # cosine|dot|euclid
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="similarity",  # none|similarity|hash
    dedup_similarity_threshold=0.9
)
```

### Redis Config

```python
RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="similarity",  # none|similarity|hash
    ttl_seconds=86400  # Optional TTL
)
```

---

## Best Practices

✅ **Always provide user_id** for multi-tenant isolation
✅ **Use categories** to organize documents logically
✅ **Batch operations** for better performance
✅ **Handle errors** gracefully in production
✅ **Use appropriate k** value in search (5-10 usually sufficient)
✅ **Configure dedup_mode** based on your use case
✅ **Test with small datasets** before scaling

---

## Complete Example

```python
import asyncio
from masai.Tools.tools.LongTermMemoryCRUDTool import LongTermMemoryCRUDTool
from masai.Memory.LongTermMemory import QdrantConfig
from langchain_openai import OpenAIEmbeddings

async def main():
    # Setup
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="research",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Create
    await tool.ainvoke({
        "mode": "create",
        "user_id": "researcher_1",
        "documents": [
            "Machine learning basics",
            "Deep learning overview"
        ],
        "categories": ["ml"]
    })
    
    # Search
    results = await tool.ainvoke({
        "mode": "search",
        "user_id": "researcher_1",
        "query": "What is machine learning?",
        "k": 5
    })
    
    print(f"Found {results['count']} results")
    for result in results['results']:
        print(f"- {result['content']}")
    
    # Update
    await tool.ainvoke({
        "mode": "update",
        "user_id": "researcher_1",
        "documents": ["Updated ML content"]
    })
    
    # Delete
    if results['results']:
        doc_id = results['results'][0]['doc_id']
        await tool.ainvoke({
            "mode": "delete",
            "user_id": "researcher_1",
            "doc_id": doc_id
        })

asyncio.run(main())
```

---

## Summary

The **LongTermMemoryCRUDTool** provides:
- ✅ Unified CRUD interface for persistent memory
- ✅ Support for Qdrant and Redis backends
- ✅ User isolation and category filtering
- ✅ Flexible document formats
- ✅ Comprehensive error handling
- ✅ Production-ready implementation

**Start using it today!** 🚀

