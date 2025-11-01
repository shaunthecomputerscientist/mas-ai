# LongTermMemory CRUD Tool - Practical Examples

## Example 1: Basic Setup with Qdrant

```python
import asyncio
from masai.Tools.tools.LongTermMemoryCRUDTool import LongTermMemoryCRUDTool
from masai.Memory.LongTermMemory import QdrantConfig
from langchain_openai import OpenAIEmbeddings

async def example_basic():
    # Configure Qdrant backend
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="documents",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    # Create tool
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Save documents
    result = await tool.ainvoke({
        "mode": "create",
        "user_id": "user_1",
        "documents": ["Python is a programming language", "JavaScript runs in browsers"]
    })
    print(result)

asyncio.run(example_basic())
```

---

## Example 2: Multi-User Isolation

```python
async def example_multi_user():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="shared_memory",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # User 1 saves documents
    await tool.ainvoke({
        "mode": "create",
        "user_id": "alice",
        "documents": ["Alice's research on AI"],
        "categories": ["research"]
    })
    
    # User 2 saves documents
    await tool.ainvoke({
        "mode": "create",
        "user_id": "bob",
        "documents": ["Bob's notes on ML"],
        "categories": ["notes"]
    })
    
    # User 1 searches - only sees their documents
    alice_results = await tool.ainvoke({
        "mode": "search",
        "user_id": "alice",
        "query": "AI research",
        "k": 5
    })
    
    # User 2 searches - only sees their documents
    bob_results = await tool.ainvoke({
        "mode": "search",
        "user_id": "bob",
        "query": "machine learning",
        "k": 5
    })
    
    print(f"Alice found: {alice_results['count']} results")
    print(f"Bob found: {bob_results['count']} results")

asyncio.run(example_multi_user())
```

---

## Example 3: Category-Based Organization

```python
async def example_categories():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="organized_docs",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Save documents with categories
    await tool.ainvoke({
        "mode": "create",
        "user_id": "researcher",
        "documents": [
            "Neural networks are inspired by biology",
            "Backpropagation is a training algorithm",
            "Convolutional networks process images"
        ],
        "categories": ["deep_learning", "neural_networks"]
    })
    
    # Search within specific categories
    results = await tool.ainvoke({
        "mode": "search",
        "user_id": "researcher",
        "query": "How do neural networks work?",
        "k": 5,
        "categories": ["neural_networks"]
    })
    
    print(f"Found {results['count']} results in neural_networks category")

asyncio.run(example_categories())
```

---

## Example 4: Document Metadata

```python
from masai.schema import Document

async def example_metadata():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="metadata_docs",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Create documents with rich metadata
    documents = [
        Document(
            page_content="Machine learning fundamentals",
            metadata={
                "source": "textbook",
                "year": 2024,
                "author": "John Doe",
                "categories": ["ml", "fundamentals"]
            }
        ),
        Document(
            page_content="Advanced deep learning techniques",
            metadata={
                "source": "paper",
                "year": 2023,
                "author": "Jane Smith",
                "categories": ["dl", "advanced"]
            }
        )
    ]
    
    result = await tool.ainvoke({
        "mode": "create",
        "user_id": "student",
        "documents": documents
    })
    
    # Search and get metadata
    search_result = await tool.ainvoke({
        "mode": "search",
        "user_id": "student",
        "query": "machine learning",
        "k": 5
    })
    
    for result in search_result['results']:
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")

asyncio.run(example_metadata())
```

---

## Example 5: Batch Import

```python
async def example_batch_import():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="batch_docs",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Prepare batch of documents
    documents = [
        f"Document {i}: Content about topic {i % 5}"
        for i in range(100)
    ]
    
    # Import all at once
    result = await tool.ainvoke({
        "mode": "create",
        "user_id": "importer",
        "documents": documents,
        "categories": ["batch_import"]
    })
    
    print(f"Imported {result['count']} documents")

asyncio.run(example_batch_import())
```

---

## Example 6: Update and Delete

```python
async def example_update_delete():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="crud_docs",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Create initial documents
    await tool.ainvoke({
        "mode": "create",
        "user_id": "editor",
        "documents": ["Original content"]
    })
    
    # Search to get doc_id
    search_result = await tool.ainvoke({
        "mode": "search",
        "user_id": "editor",
        "query": "content",
        "k": 1
    })
    
    doc_id = search_result['results'][0]['doc_id']
    
    # Update document
    await tool.ainvoke({
        "mode": "update",
        "user_id": "editor",
        "documents": ["Updated content with new information"]
    })
    
    # Delete document
    delete_result = await tool.ainvoke({
        "mode": "delete",
        "user_id": "editor",
        "doc_id": doc_id
    })
    
    print(f"Deleted: {delete_result['message']}")

asyncio.run(example_update_delete())
```

---

## Example 7: Redis Backend

```python
from masai.Memory.LongTermMemory import RedisConfig

async def example_redis():
    config = RedisConfig(
        redis_url="redis://localhost:6379",
        index_name="masai_vectors",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings(),
        ttl_seconds=86400  # 24 hours
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Same operations work with Redis backend
    result = await tool.ainvoke({
        "mode": "create",
        "user_id": "redis_user",
        "documents": ["Redis-backed document"]
    })
    
    print(f"Backend: {result['backend']}")

asyncio.run(example_redis())
```

---

## Example 8: Error Handling

```python
async def example_error_handling():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="error_docs",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Missing required parameters
    result = await tool.ainvoke({
        "mode": "create"
        # Missing user_id
    })
    print(f"Error: {result['message']}")
    
    # Invalid mode
    result = await tool.ainvoke({
        "mode": "invalid_mode",
        "user_id": "user_1"
    })
    print(f"Error: {result['message']}")
    
    # Connection error (if backend not running)
    try:
        result = await tool.ainvoke({
            "mode": "search",
            "user_id": "user_1",
            "query": "test"
        })
        if result['status'] == 'error':
            print(f"Operation error: {result['error']}")
    except Exception as e:
        print(f"Exception: {e}")

asyncio.run(example_error_handling())
```

---

## Example 9: With MASAI Agent

```python
from masai.Tools.tools.LongTermMemoryCRUDTool import create_longterm_memory_tool

async def example_with_agent():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="agent_memory",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    # Create Tool wrapper
    memory_tool = create_longterm_memory_tool(memory_config=config)
    
    # Use with agent
    result = await memory_tool.ainvoke({
        "mode": "create",
        "user_id": "agent_user",
        "documents": ["Agent-created document"]
    })
    
    print(f"Tool result: {result}")

asyncio.run(example_with_agent())
```

---

## Example 10: Complete Workflow

```python
async def example_complete_workflow():
    config = QdrantConfig(
        url="http://localhost:6333",
        collection_name="workflow",
        vector_size=1536,
        embedding_model=OpenAIEmbeddings()
    )
    
    tool = LongTermMemoryCRUDTool(memory_config=config)
    user_id = "workflow_user"
    
    # 1. Create knowledge base
    print("1. Creating knowledge base...")
    await tool.ainvoke({
        "mode": "create",
        "user_id": user_id,
        "documents": [
            "Python is a high-level programming language",
            "Python supports multiple programming paradigms",
            "Python has a large standard library"
        ],
        "categories": ["python", "programming"]
    })
    
    # 2. Search for information
    print("2. Searching for information...")
    search_result = await tool.ainvoke({
        "mode": "search",
        "user_id": user_id,
        "query": "What is Python?",
        "k": 3,
        "categories": ["python"]
    })
    print(f"Found {search_result['count']} results")
    
    # 3. Update knowledge
    print("3. Updating knowledge...")
    await tool.ainvoke({
        "mode": "update",
        "user_id": user_id,
        "documents": ["Python 3.12 is the latest version"],
        "categories": ["python", "versions"]
    })
    
    # 4. Search again
    print("4. Searching updated knowledge...")
    search_result = await tool.ainvoke({
        "mode": "search",
        "user_id": user_id,
        "query": "Python version",
        "k": 5
    })
    
    # 5. Delete old information
    if search_result['results']:
        print("5. Cleaning up...")
        doc_id = search_result['results'][0]['doc_id']
        await tool.ainvoke({
            "mode": "delete",
            "user_id": user_id,
            "doc_id": doc_id
        })
    
    print("Workflow complete!")

asyncio.run(example_complete_workflow())
```

---

## Quick Reference

| Operation | Mode | Required Params |
|-----------|------|-----------------|
| Create | "create" | user_id, documents |
| Search | "search" | user_id, query |
| Update | "update" | user_id, documents |
| Delete | "delete" | user_id, doc_id |

---

## Tips & Tricks

✅ **Get doc_id from search results** for deletion
✅ **Use categories** to organize documents
✅ **Batch operations** for better performance
✅ **Handle errors** with status checks
✅ **Use appropriate k** value (5-10 usually)
✅ **Test locally** before production

**Happy coding!** 🚀

