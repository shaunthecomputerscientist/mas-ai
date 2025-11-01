# Redis & Qdrant CRUD Guide

## Overview

MASAI supports both **Redis** and **Qdrant** as persistent memory backends. This guide shows how to use them as CRUD tools for agents and external data storage.

---

## Setup

### Redis Configuration

```python
from masai.Memory.LongTermMemory import RedisConfig, LongTermMemory
from langchain_openai import OpenAIEmbeddings

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="similarity",  # "exact", "similarity", "none"
    ttl_seconds=None  # Optional: auto-expire after N seconds
)

memory = LongTermMemory(backend_config=redis_config)
```

### Qdrant Configuration

```python
from masai.Memory.LongTermMemory import QdrantConfig, LongTermMemory
from langchain_openai import OpenAIEmbeddings

qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="similarity"
)

memory = LongTermMemory(backend_config=qdrant_config)
```

---

## CRUD Operations

### CREATE: Upsert Documents

```python
from masai.schema import Document

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

# Upsert to memory
await memory.upsert_documents(
    user_id="user_123",
    documents=docs,
    embed_fn=lambda texts: embeddings_model.embed_documents(texts)
)

print("✅ Documents created/updated")
```

### READ: Search Documents

```python
# Search by query
results = await memory.search(
    query="machine learning",
    user_id="user_123",
    top_k=5
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {doc.metadata.get('score', 'N/A')}")
```

### UPDATE: Upsert with New Data

```python
# Update existing document (same ID)
updated_doc = Document(
    page_content="Updated machine learning content",
    metadata={"source": "tutorial", "level": "intermediate", "updated": True}
)

await memory.upsert_documents(
    user_id="user_123",
    documents=[updated_doc],
    embed_fn=lambda texts: embeddings_model.embed_documents(texts)
)

print("✅ Document updated")
```

### DELETE: Remove Documents

```python
# Delete by ID
await memory.delete_documents(
    user_id="user_123",
    doc_ids=["doc_id_1", "doc_id_2"]
)

print("✅ Documents deleted")

# Delete all for user
await memory.delete_documents(
    user_id="user_123",
    doc_ids=None  # Delete all
)

print("✅ All documents deleted for user")
```

---

## Deduplication Modes

### Mode 1: Exact Deduplication

```python
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="exact"  # Exact match only
)

# Only identical documents are deduplicated
```

### Mode 2: Similarity Deduplication

```python
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="similarity"  # Merge similar documents
)

# Similar documents are merged (recommended)
```

### Mode 3: No Deduplication

```python
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    dedup_mode="none"  # Keep all documents
)

# All documents are kept (may have duplicates)
```

---

## Wrapping in Separate Functions

### CRUD Wrapper Class

```python
from masai.schema import Document
from typing import List, Optional

class MemoryCRUD:
    def __init__(self, memory):
        self.memory = memory
    
    async def create(
        self,
        user_id: str,
        content: str,
        metadata: dict = None,
        embed_fn = None
    ) -> None:
        """Create a new document"""
        doc = Document(
            page_content=content,
            metadata=metadata or {}
        )
        await self.memory.upsert_documents(
            user_id=user_id,
            documents=[doc],
            embed_fn=embed_fn
        )
    
    async def read(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Document]:
        """Read documents by search"""
        return await self.memory.search(
            query=query,
            user_id=user_id,
            top_k=top_k
        )
    
    async def update(
        self,
        user_id: str,
        content: str,
        metadata: dict = None,
        embed_fn = None
    ) -> None:
        """Update a document"""
        doc = Document(
            page_content=content,
            metadata=metadata or {}
        )
        await self.memory.upsert_documents(
            user_id=user_id,
            documents=[doc],
            embed_fn=embed_fn
        )
    
    async def delete(
        self,
        user_id: str,
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """Delete documents"""
        await self.memory.delete_documents(
            user_id=user_id,
            doc_ids=doc_ids
        )

# Usage
crud = MemoryCRUD(memory)

await crud.create(
    user_id="user_123",
    content="New document",
    metadata={"source": "api"}
)

results = await crud.read(
    user_id="user_123",
    query="document"
)

await crud.delete(
    user_id="user_123",
    doc_ids=["doc_1"]
)
```

---

## Using as Agent Memory Backend

### Setup

```python
from masai.AgentManager import AgentManager
from masai.Memory.LongTermMemory import RedisConfig

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="agent_memories",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

manager = AgentManager(
    model_config_path="model_config.json",
    user_id="user_123",
    memory_config=redis_config
)

agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=...,
    persist_memory=True
)
```

### Accessing Agent Memory

```python
# Search agent's memory
results = await agent.llm_router.long_term_memory.search(
    query="previous conversation",
    user_id="user_123",
    top_k=5
)

# Add to agent's memory
from masai.schema import Document

doc = Document(
    page_content="Important fact",
    metadata={"source": "user_input"}
)

await agent.llm_router.long_term_memory.upsert_documents(
    user_id="user_123",
    documents=[doc],
    embed_fn=embedding_function
)
```

---

## Using as External Data Store

### Knowledge Base

```python
class KnowledgeBase:
    def __init__(self, memory):
        self.memory = memory
    
    async def add_knowledge(self, content: str, category: str):
        """Add knowledge to base"""
        doc = Document(
            page_content=content,
            metadata={"category": category, "type": "knowledge"}
        )
        await self.memory.upsert_documents(
            user_id="knowledge_base",
            documents=[doc],
            embed_fn=embedding_function
        )
    
    async def query_knowledge(self, query: str, top_k: int = 5):
        """Query knowledge base"""
        return await self.memory.search(
            query=query,
            user_id="knowledge_base",
            top_k=top_k
        )

# Usage
kb = KnowledgeBase(memory)

await kb.add_knowledge(
    content="Python is a programming language",
    category="programming"
)

results = await kb.query_knowledge("programming languages")
```

### Document Store

```python
class DocumentStore:
    def __init__(self, memory):
        self.memory = memory
    
    async def store_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict
    ):
        """Store document"""
        doc = Document(
            page_content=content,
            metadata={**metadata, "doc_id": doc_id}
        )
        await self.memory.upsert_documents(
            user_id="documents",
            documents=[doc],
            embed_fn=embedding_function
        )
    
    async def retrieve_document(self, query: str):
        """Retrieve documents"""
        return await self.memory.search(
            query=query,
            user_id="documents",
            top_k=10
        )

# Usage
store = DocumentStore(memory)

await store.store_document(
    doc_id="doc_001",
    content="Document content",
    metadata={"author": "Alice", "date": "2024-01-01"}
)

results = await store.retrieve_document("content")
```

---

## TTL and Expiration (Redis Only)

```python
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(),
    ttl_seconds=86400  # Expire after 24 hours
)

memory = LongTermMemory(backend_config=redis_config)

# Documents automatically expire after 24 hours
await memory.upsert_documents(
    user_id="user_123",
    documents=[doc],
    embed_fn=embedding_function
)
```

---

## Performance Tuning

### Batch Operations

```python
# ✅ GOOD - Batch upsert
docs = [Document(...) for _ in range(100)]
await memory.upsert_documents(
    user_id="user_123",
    documents=docs,
    embed_fn=embedding_function
)

# ❌ BAD - Individual upserts
for doc in docs:
    await memory.upsert_documents(
        user_id="user_123",
        documents=[doc],
        embed_fn=embedding_function
    )
```

### Optimize top_k

```python
# ✅ GOOD - Reasonable top_k
results = await memory.search(
    query="...",
    user_id="user_123",
    top_k=5  # Get top 5
)

# ❌ BAD - Too large
results = await memory.search(
    query="...",
    user_id="user_123",
    top_k=1000  # Slow
)
```

---

## Real-World Example

### Multi-User Research System

```python
class ResearchMemory:
    def __init__(self, memory):
        self.memory = memory
    
    async def save_research(
        self,
        user_id: str,
        title: str,
        content: str,
        tags: List[str]
    ):
        """Save research paper"""
        doc = Document(
            page_content=content,
            metadata={
                "title": title,
                "tags": tags,
                "type": "research"
            }
        )
        await self.memory.upsert_documents(
            user_id=user_id,
            documents=[doc],
            embed_fn=embedding_function
        )
    
    async def find_research(
        self,
        user_id: str,
        query: str
    ):
        """Find research papers"""
        return await self.memory.search(
            query=query,
            user_id=user_id,
            top_k=10
        )

# Usage
research = ResearchMemory(memory)

await research.save_research(
    user_id="researcher_1",
    title="AI Advances 2024",
    content="Recent advances in AI...",
    tags=["AI", "ML", "2024"]
)

results = await research.find_research(
    user_id="researcher_1",
    query="machine learning"
)
```

---

## See Also

- [DOCUMENT_CLASS_GUIDE.md](DOCUMENT_CLASS_GUIDE.md) - Document class
- [MEMORY_ARCHITECTURE_DEEP_DIVE.md](MEMORY_ARCHITECTURE_DEEP_DIVE.md) - Memory system
- [AGENTMANAGER_DETAILED.md](AGENTMANAGER_DETAILED.md) - AgentManager setup

