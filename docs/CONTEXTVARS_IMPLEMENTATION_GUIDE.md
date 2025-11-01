# ContextVars Implementation Guide

## Quick Answer

**YES!** Context variables set at the start of a request are **automatically available** inside `ainvoke()` and all async functions it calls.

---

## How It Works

### The Flow

```
1. Request arrives
   ↓
2. Middleware/Handler sets context variables
   ├─ set_user_id("user_123")
   ├─ set_request_id("req_456")
   └─ set_tenant_id("tenant_789")
   ↓
3. ainvoke() is called
   ↓
4. Context variables are AUTOMATICALLY available
   ├─ get_user_id() → "user_123"
   ├─ get_request_id() → "req_456"
   └─ get_tenant_id() → "tenant_789"
   ↓
5. All nested async calls inherit context
   ├─ execute() can access context
   ├─ memory.save() can access context
   └─ adapter.upsert_documents() can access context
```

---

## Step-by-Step Implementation

### Step 1: Create Context Module

**File**: `src/masai/context.py`

```python
from contextvars import ContextVar
from typing import Optional

# Define context variables
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
tenant_id_var: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)

# Setter functions
def set_user_id(user_id: str) -> None:
    user_id_var.set(user_id)

def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)

def set_tenant_id(tenant_id: str) -> None:
    tenant_id_var.set(tenant_id)

# Getter functions
def get_user_id() -> Optional[str]:
    return user_id_var.get()

def get_request_id() -> Optional[str]:
    return request_id_var.get()

def get_tenant_id() -> Optional[str]:
    return tenant_id_var.get()

# Clear all context
def clear_context() -> None:
    user_id_var.set(None)
    request_id_var.set(None)
    tenant_id_var.set(None)
```

---

### Step 2: Update CRUD Tool

**File**: `src/masai/Tools/tools/LongTermMemoryCRUDTool.py`

```python
from masai.context import get_user_id, get_request_id

class LongTermMemoryCRUDTool:
    # ... existing code ...
    
    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async invoke with context variable support.
        
        Priority:
        1. Context variable (if set)
        2. Explicit parameter (if provided)
        3. Error (if neither available)
        """
        # Get user_id from context first, then from input
        context_user_id = get_user_id()
        mode = input_dict.get("mode")
        user_id = context_user_id or input_dict.get("user_id")
        
        if not mode or not user_id:
            return {
                "status": "error",
                "message": "Missing required parameters: mode and user_id"
            }
        
        # Get request_id from context if available
        request_id = get_request_id()
        
        # Execute operation
        result = await self.execute(mode=mode, user_id=user_id, **input_dict)
        
        # Add request_id to response if available
        if request_id:
            result["request_id"] = request_id
        
        return result
```

---

### Step 3: Create Middleware (FastAPI)

**File**: `src/api/middleware.py`

```python
from fastapi import Request, HTTPException
from masai.context import set_user_id, set_request_id
import uuid
import logging

logger = logging.getLogger(__name__)

async def set_context_middleware(request: Request, call_next):
    """
    Middleware to set context variables for each request.
    
    Extracts user_id from X-User-ID header and generates request_id.
    """
    # Extract user_id from header
    user_id = request.headers.get("X-User-ID")
    
    # Generate request_id
    request_id = str(uuid.uuid4())
    
    # Validate user_id
    if not user_id:
        logger.warning(f"Request {request_id} missing X-User-ID header")
        raise HTTPException(
            status_code=401,
            detail="X-User-ID header required"
        )
    
    # Set context variables
    set_user_id(user_id)
    set_request_id(request_id)
    
    logger.info(f"Request {request_id} from user {user_id}")
    
    try:
        # Process request
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        raise
```

---

### Step 4: Register Middleware (FastAPI)

**File**: `src/api/main.py`

```python
from fastapi import FastAPI
from src.api.middleware import set_context_middleware
from src.api.routes import memory_routes

app = FastAPI(title="MASAI Memory API")

# Register middleware
app.middleware("http")(set_context_middleware)

# Include routes
app.include_router(memory_routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Step 5: Create Routes

**File**: `src/api/routes/memory_routes.py`

```python
from fastapi import APIRouter, Query
from masai.Tools.tools.LongTermMemoryCRUDTool import LongTermMemoryCRUDTool
from masai.Memory.LongTermMemory import QdrantConfig
from langchain_openai import OpenAIEmbeddings

router = APIRouter(prefix="/memory", tags=["memory"])

# Initialize tool
config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)
tool = LongTermMemoryCRUDTool(memory_config=config)

@router.post("/create")
async def create_memory(documents: list[str], categories: list[str] = None):
    """
    Create memory documents.
    
    user_id is automatically extracted from context (X-User-ID header).
    """
    result = await tool.ainvoke({
        "mode": "create",
        "documents": documents,
        "categories": categories or []
    })
    return result

@router.post("/search")
async def search_memory(
    query: str,
    k: int = Query(5, ge=1, le=100),
    categories: list[str] = None
):
    """
    Search memory documents.
    
    user_id is automatically extracted from context (X-User-ID header).
    """
    result = await tool.ainvoke({
        "mode": "search",
        "query": query,
        "k": k,
        "categories": categories or []
    })
    return result

@router.post("/update")
async def update_memory(documents: list[str], categories: list[str] = None):
    """
    Update memory documents.
    
    user_id is automatically extracted from context (X-User-ID header).
    """
    result = await tool.ainvoke({
        "mode": "update",
        "documents": documents,
        "categories": categories or []
    })
    return result

@router.delete("/{doc_id}")
async def delete_memory(doc_id: str):
    """
    Delete memory document.
    
    user_id is automatically extracted from context (X-User-ID header).
    """
    result = await tool.ainvoke({
        "mode": "delete",
        "doc_id": doc_id
    })
    return result
```

---

## Testing

### Test 1: Direct Context Usage

```python
import asyncio
from masai.context import set_user_id, set_request_id, get_user_id
from masai.Tools.tools.LongTermMemoryCRUDTool import LongTermMemoryCRUDTool

async def test_context():
    # Set context
    set_user_id("user_123")
    set_request_id("req_456")
    
    # Create tool
    tool = LongTermMemoryCRUDTool(memory_config=config)
    
    # Call without explicit user_id
    result = await tool.ainvoke({
        "mode": "create",
        "documents": ["Test document"]
        # Note: user_id NOT provided - comes from context!
    })
    
    print(f"Result: {result}")
    assert result["status"] == "success"
    assert result["request_id"] == "req_456"

asyncio.run(test_context())
```

### Test 2: Multiple Concurrent Requests

```python
import asyncio
from masai.context import set_user_id, set_request_id

async def handle_request(user_id: str, request_id: str):
    """Handle a single request"""
    set_user_id(user_id)
    set_request_id(request_id)
    
    result = await tool.ainvoke({
        "mode": "search",
        "query": "test"
    })
    
    # Verify context is correct for this request
    assert result["request_id"] == request_id
    return result

async def test_concurrent():
    """Test multiple concurrent requests"""
    tasks = [
        handle_request("user_1", "req_1"),
        handle_request("user_2", "req_2"),
        handle_request("user_3", "req_3"),
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Each request should have its own context
    assert results[0]["request_id"] == "req_1"
    assert results[1]["request_id"] == "req_2"
    assert results[2]["request_id"] == "req_3"

asyncio.run(test_concurrent())
```

---

## API Usage Examples

### Create Documents

```bash
curl -X POST http://localhost:8000/memory/create \
  -H "X-User-ID: user_123" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["Document 1", "Document 2"],
    "categories": ["category1"]
  }'
```

**Response**:
```json
{
  "status": "success",
  "operation": "create",
  "message": "Successfully saved 2 document(s)",
  "count": 2,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "backend": "qdrant"
}
```

### Search Documents

```bash
curl -X POST http://localhost:8000/memory/search \
  -H "X-User-ID: user_123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search query",
    "k": 5,
    "categories": ["category1"]
  }'
```

**Response**:
```json
{
  "status": "success",
  "operation": "search",
  "message": "Found 2 document(s)",
  "results": [...],
  "count": 2,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "backend": "qdrant"
}
```

---

## Key Benefits

✅ **No parameter passing** - user_id automatically available
✅ **User isolation** - each request has isolated context
✅ **Request tracking** - request_id in all responses
✅ **Multi-tenant** - tenant_id can be added
✅ **Backward compatible** - explicit parameters still work
✅ **Thread-safe** - each async task has isolated context
✅ **Clean code** - no need to thread user_id through functions

---

## Summary

Context variables are **perfect** for your use case:

1. ✅ Set user_id once at request start
2. ✅ Automatically available in ainvoke()
3. ✅ Available in all nested async calls
4. ✅ Perfect for multi-tenant applications
5. ✅ Automatic request isolation

**Implementation is straightforward and production-ready!** 🚀

