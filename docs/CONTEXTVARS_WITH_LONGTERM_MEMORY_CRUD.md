# Using ContextVars with LongTermMemory CRUD Tool

## Overview

**Context variables (contextvars)** are perfect for storing request-scoped data like `user_id`, `request_id`, or `tenant_id` that should be automatically available throughout an async call chain without passing them as parameters.

---

## How ContextVars Work

### Key Concept
Context variables are **automatically inherited** by all async tasks spawned within an async context. They're perfect for:
- ✅ User isolation
- ✅ Request tracking
- ✅ Tenant isolation
- ✅ Automatic parameter propagation

### Availability
Once set, context variables are available in:
- ✅ The current async function
- ✅ All functions called from it
- ✅ All async tasks spawned from it
- ✅ All nested async calls

---

## Basic Setup

### Step 1: Define Context Variables

```python
# src/masai/context.py
from contextvars import ContextVar
from typing import Optional

# Define context variables
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
tenant_id_var: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)

def set_user_id(user_id: str) -> None:
    """Set user_id for current request context"""
    user_id_var.set(user_id)

def get_user_id() -> Optional[str]:
    """Get user_id from current request context"""
    return user_id_var.get()

def set_request_id(request_id: str) -> None:
    """Set request_id for current request context"""
    request_id_var.set(request_id)

def get_request_id() -> Optional[str]:
    """Get request_id from current request context"""
    return request_id_var.get()

def set_tenant_id(tenant_id: str) -> None:
    """Set tenant_id for current request context"""
    tenant_id_var.set(tenant_id)

def get_tenant_id() -> Optional[str]:
    """Get tenant_id from current request context"""
    return tenant_id_var.get()
```

---

## Integration with CRUD Tool

### Modified CRUD Tool

```python
# src/masai/Tools/tools/LongTermMemoryCRUDTool.py
from masai.context import get_user_id, get_request_id

class LongTermMemoryCRUDTool:
    # ... existing code ...
    
    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async invoke for Tool compatibility.
        
        Now supports both explicit user_id parameter AND context variable.
        Context variable takes precedence if both are provided.
        """
        # Get user_id from context variable first, then from input
        context_user_id = get_user_id()
        mode = input_dict.get("mode")
        user_id = context_user_id or input_dict.get("user_id")
        
        if not mode or not user_id:
            return {
                "status": "error",
                "message": "Missing required parameters: mode and user_id"
            }
        
        # Add request_id to response if available
        request_id = get_request_id()
        result = await self.execute(mode=mode, user_id=user_id, **input_dict)
        
        if request_id:
            result["request_id"] = request_id
        
        return result
```

---

## Usage Patterns

### Pattern 1: Web Framework Integration (FastAPI)

```python
# main.py
from fastapi import FastAPI, Request
from masai.context import set_user_id, set_request_id
from masai.Tools.tools.LongTermMemoryCRUDTool import LongTermMemoryCRUDTool
import uuid

app = FastAPI()
tool = LongTermMemoryCRUDTool(memory_config=config)

@app.middleware("http")
async def set_context_middleware(request: Request, call_next):
    """Middleware to set context variables for each request"""
    # Extract user_id from request (e.g., from JWT token)
    user_id = request.headers.get("X-User-ID")
    request_id = str(uuid.uuid4())
    
    # Set context variables
    if user_id:
        set_user_id(user_id)
    set_request_id(request_id)
    
    # Process request
    response = await call_next(request)
    return response

@app.post("/memory/search")
async def search_memory(query: str):
    """Search memory - user_id comes from context"""
    result = await tool.ainvoke({
        "mode": "search",
        "query": query,
        "k": 5
        # Note: user_id NOT needed here - comes from context!
    })
    return result

@app.post("/memory/create")
async def create_memory(documents: List[str]):
    """Create memory - user_id comes from context"""
    result = await tool.ainvoke({
        "mode": "create",
        "documents": documents
        # Note: user_id NOT needed here - comes from context!
    })
    return result
```

---

### Pattern 2: Agent Integration

```python
# agent_with_context.py
from masai.context import set_user_id, set_request_id
from masai.Agents import Agent
import uuid

async def process_user_request(user_id: str, query: str):
    """Process request with automatic user isolation"""
    # Set context for this request
    set_user_id(user_id)
    set_request_id(str(uuid.uuid4()))
    
    # Create agent
    agent = Agent(...)
    
    # Agent can use CRUD tool without passing user_id
    result = await agent.initiate_agent(query=query)
    
    return result
```

---

### Pattern 3: Multi-Tenant Application

```python
# multi_tenant.py
from masai.context import set_user_id, set_tenant_id, set_request_id
import uuid

async def handle_tenant_request(tenant_id: str, user_id: str, operation: Dict):
    """Handle request in multi-tenant context"""
    # Set context variables
    set_tenant_id(tenant_id)
    set_user_id(user_id)
    set_request_id(str(uuid.uuid4()))
    
    # Tool automatically uses tenant_id and user_id from context
    result = await tool.ainvoke(operation)
    
    return result
```

---

## Advanced: Custom Context Manager

### Context Manager for Automatic Setup/Cleanup

```python
# src/masai/context.py
from contextlib import asynccontextmanager
from typing import Optional

@asynccontextmanager
async def request_context(
    user_id: str,
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None
):
    """
    Async context manager for setting up request context.
    
    Usage:
        async with request_context(user_id="user_123"):
            # user_id is available here
            result = await tool.ainvoke({"mode": "search", ...})
    """
    # Save previous values
    prev_user_id = user_id_var.get()
    prev_request_id = request_id_var.get()
    prev_tenant_id = tenant_id_var.get()
    
    try:
        # Set new values
        set_user_id(user_id)
        if request_id:
            set_request_id(request_id)
        if tenant_id:
            set_tenant_id(tenant_id)
        
        yield
    finally:
        # Restore previous values
        if prev_user_id:
            set_user_id(prev_user_id)
        else:
            user_id_var.set(None)
        
        if prev_request_id:
            set_request_id(prev_request_id)
        else:
            request_id_var.set(None)
        
        if prev_tenant_id:
            set_tenant_id(prev_tenant_id)
        else:
            tenant_id_var.set(None)
```

### Usage

```python
async def main():
    async with request_context(user_id="user_123", request_id="req_456"):
        # Both user_id and request_id are available
        result = await tool.ainvoke({
            "mode": "search",
            "query": "test"
        })
        print(result)

asyncio.run(main())
```

---

## Complete Example: FastAPI with CRUD Tool

```python
# main.py
from fastapi import FastAPI, Request, HTTPException
from masai.context import set_user_id, set_request_id, get_user_id
from masai.Tools.tools.LongTermMemoryCRUDTool import LongTermMemoryCRUDTool
from masai.Memory.LongTermMemory import QdrantConfig
from langchain_openai import OpenAIEmbeddings
import uuid
import json

app = FastAPI()

# Initialize tool
config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)
tool = LongTermMemoryCRUDTool(memory_config=config)

@app.middleware("http")
async def set_context_middleware(request: Request, call_next):
    """Set context variables for each request"""
    user_id = request.headers.get("X-User-ID")
    request_id = str(uuid.uuid4())
    
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")
    
    set_user_id(user_id)
    set_request_id(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.post("/memory/create")
async def create_memory(documents: list[str], categories: list[str] = None):
    """Create memory documents"""
    result = await tool.ainvoke({
        "mode": "create",
        "documents": documents,
        "categories": categories or []
    })
    return result

@app.post("/memory/search")
async def search_memory(query: str, k: int = 5, categories: list[str] = None):
    """Search memory documents"""
    result = await tool.ainvoke({
        "mode": "search",
        "query": query,
        "k": k,
        "categories": categories or []
    })
    return result

@app.post("/memory/update")
async def update_memory(documents: list[str], categories: list[str] = None):
    """Update memory documents"""
    result = await tool.ainvoke({
        "mode": "update",
        "documents": documents,
        "categories": categories or []
    })
    return result

@app.delete("/memory/{doc_id}")
async def delete_memory(doc_id: str):
    """Delete memory document"""
    result = await tool.ainvoke({
        "mode": "delete",
        "doc_id": doc_id
    })
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Usage

```bash
# Create
curl -X POST http://localhost:8000/memory/create \
  -H "X-User-ID: user_123" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Document 1", "Document 2"]}'

# Search
curl -X POST http://localhost:8000/memory/search \
  -H "X-User-ID: user_123" \
  -H "Content-Type: application/json" \
  -d '{"query": "search query", "k": 5}'

# Update
curl -X POST http://localhost:8000/memory/update \
  -H "X-User-ID: user_123" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Updated doc"]}'

# Delete
curl -X DELETE http://localhost:8000/memory/doc_id_123 \
  -H "X-User-ID: user_123"
```

---

## Key Points

✅ **Context variables are inherited** by all async functions in the call chain
✅ **No parameter passing needed** - just set once at request start
✅ **Thread-safe and async-safe** - each request has isolated context
✅ **Perfect for user isolation** - automatic per-request user_id
✅ **Works with frameworks** - FastAPI, Starlette, etc.
✅ **Backward compatible** - explicit parameters still work

---

## Best Practices

✅ Set context variables at the **entry point** (middleware, handler)
✅ Use **context managers** for automatic cleanup
✅ Provide **fallback** to explicit parameters
✅ Document which variables are **context-dependent**
✅ Test with **multiple concurrent requests**
✅ Use **request_id** for tracing

---

## Summary

Context variables are **perfect** for your CRUD tool because:
1. ✅ User_id is automatically available everywhere
2. ✅ No need to pass user_id through multiple function calls
3. ✅ Perfect for multi-tenant applications
4. ✅ Automatic request isolation
5. ✅ Works seamlessly with async/await

**Start using contextvars today!** 🚀

