# Tools Guide

Complete guide for defining, using, and caching tools in MASAI framework.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Tool Basics](#tool-basics)
3. [Defining Tools](#defining-tools)
4. [Tool Parameters](#tool-parameters)
5. [Synchronous vs Asynchronous Tools](#synchronous-vs-asynchronous-tools)
6. [Tool Execution Methods](#tool-execution-methods)
7. [Redis Caching](#redis-caching)
8. [Integration with MASAI Agents](#integration-with-masai-agents)
9. [Best Practices](#best-practices)
10. [Complete Examples](#complete-examples)

---

## Introduction

Tools in MASAI are functions that agents can call to perform specific tasks. They provide a standardized interface for:
- **Function execution** with automatic parameter validation
- **Type safety** using Pydantic schemas
- **Async support** for I/O-bound operations
- **Redis caching** for expensive operations
- **Seamless integration** with MASAI agents

---

## Tool Basics

### What is a Tool?

A tool is a Python function wrapped with the `@tool` decorator that:
1. Automatically generates a Pydantic schema from function signature
2. Validates input parameters
3. Provides metadata for agent systems
4. Supports both sync and async execution

### Tool Class

The `Tool` class provides:
- **Automatic schema generation** from function signature
- **Input validation** using Pydantic
- **Multiple execution methods** (invoke, run, ainvoke, arun)
- **Metadata extraction** for agent integration

---

## Defining Tools

### Basic Tool Definition

Use the `@tool` decorator to create a tool:

```python
from masai.Tools.Tool import tool

@tool("calculator")
def calculator(operation: str, x: float, y: float) -> float:
    """
    Performs basic arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        x: First number
        y: Second number
    
    Returns:
        Result of the operation
    """
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        return x / y if y != 0 else float('inf')
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

### Tool with Optional Parameters

```python
@tool("search")
def search(query: str, max_results: int = 10, filter_type: Optional[str] = None) -> List[str]:
    """
    Searches for information with optional filtering.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 10)
        filter_type: Optional filter type (news, images, videos)
    
    Returns:
        List of search results
    """
    results = perform_search(query, max_results)
    if filter_type:
        results = filter_results(results, filter_type)
    return results
```

### Tool with Complex Types

```python
from typing import List, Dict, Optional

@tool("process_data")
def process_data(
    items: List[str],
    config: Dict[str, Any],
    options: Optional[Dict[str, int]] = None
) -> Dict[str, List[str]]:
    """
    Processes a list of items with configuration.
    
    Args:
        items: List of items to process
        config: Configuration dictionary
        options: Optional processing options
    
    Returns:
        Dictionary with processed results
    """
    processed = []
    for item in items:
        processed.append(process_item(item, config, options))
    return {"processed": processed, "count": len(processed)}
```

### Tool with return_direct

Use `return_direct=True` to return raw output without string conversion:

```python
@tool("get_user_data", return_direct=True)
def get_user_data(user_id: int) -> Dict[str, Any]:
    """
    Retrieves user data as a dictionary.
    
    Args:
        user_id: User identifier
    
    Returns:
        User data dictionary
    """
    return {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com"
    }
```

**Note**: Without `return_direct=True`, the output is automatically converted to string.

---

## Tool Parameters

### Required Parameters

Parameters without default values are required:

```python
@tool("greet")
def greet(name: str) -> str:
    """Greets a user by name."""
    return f"Hello, {name}!"
```

### Optional Parameters

Parameters with default values are optional:

```python
@tool("greet_with_title")
def greet_with_title(name: str, title: str = "Mr.") -> str:
    """Greets a user with an optional title."""
    return f"Hello, {title} {name}!"
```

### Type Annotations

Always use type annotations for better validation:

```python
@tool("calculate_age")
def calculate_age(birth_year: int, current_year: int = 2025) -> int:
    """Calculates age from birth year."""
    return current_year - birth_year
```

**Supported Types**:
- Basic: `str`, `int`, `float`, `bool`
- Collections: `List[T]`, `Dict[K, V]`, `Set[T]`, `Tuple[T, ...]`
- Optional: `Optional[T]` or `Union[T, None]`
- Any: `Any` (no validation)

---

## Synchronous vs Asynchronous Tools

### Synchronous Tools

For CPU-bound or fast operations:

```python
@tool("add_numbers")
def add_numbers(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y
```

### Asynchronous Tools

For I/O-bound operations (API calls, database queries, file I/O):

```python
import asyncio

@tool("fetch_data")
async def fetch_data(url: str) -> str:
    """Fetches data from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

**When to use async**:
- ✅ API calls
- ✅ Database queries
- ✅ File I/O
- ✅ Network operations
- ❌ CPU-intensive calculations (use sync)

---

## Tool Execution Methods

### 1. invoke() - Synchronous with dict/JSON input

```python
# With dict input
result = calculator.invoke({"operation": "add", "x": 5, "y": 3})

# With JSON string input
result = calculator.invoke('{"operation": "add", "x": 5, "y": 3}')
```

### 2. ainvoke() - Asynchronous with dict/JSON input

```python
# With dict input
result = await fetch_data.ainvoke({"url": "https://api.example.com"})

# With JSON string input
result = await fetch_data.ainvoke('{"url": "https://api.example.com"}')
```

### 3. run() - Synchronous with string input/output

```python
result = calculator.run('{"operation": "add", "x": 5, "y": 3}')
# Returns string output
```

### 4. arun() - Asynchronous with string input/output

```python
result = await fetch_data.arun('{"url": "https://api.example.com"}')
# Returns string output
```

### 5. get_metadata() - Get tool metadata

```python
metadata = calculator.get_metadata()
# Returns: {
#     "name": "calculator",
#     "description": "Performs basic arithmetic operations.",
#     "args_schema": CalculatorSchema  # Pydantic BaseModel
# }
```

---

## Redis Caching

### Why Cache Tools?

Caching is essential for:
- **Expensive API calls** (avoid rate limits and costs)
- **Database queries** (reduce load)
- **Slow computations** (improve response time)
- **Repeated requests** (same input → same output)

### Setting Up Redis Cache

```python
from masai.Tools.utilities.cache import ToolCache

# Initialize cache
cache = ToolCache(
    host='localhost',
    port=6379,
    db=0,
    password=None,  # Optional
    timeout=30  # Cache timeout in minutes
)
```

### Caching a Tool

```python
from masai.Tools.Tool import tool
from masai.Tools.utilities.cache import ToolCache

cache = ToolCache(host='localhost', port=6379, timeout=60)

@tool("expensive_api_call")
@cache.masai_cache
def expensive_api_call(query: str) -> str:
    """
    Makes an expensive API call (cached for 60 minutes).
    
    Args:
        query: Search query
    
    Returns:
        API response
    """
    # This will only execute if not in cache
    response = requests.get(f"https://api.example.com/search?q={query}")
    return response.text
```

### How Caching Works

1. **First call**: Function executes, result stored in Redis
2. **Subsequent calls** (within timeout): Result retrieved from Redis
3. **After timeout**: Cache expires, function executes again

**Cache Key Format**: `{function_name}:{args}:{kwargs}`

### Cache Configuration

```python
cache = ToolCache(
    host='localhost',      # Redis host
    port=6379,             # Redis port
    db=0,                  # Redis database number
    password='secret',     # Redis password (if required)
    timeout=30             # Cache timeout in MINUTES
)
```

**Timeout Examples**:
- `timeout=5`: 5 minutes
- `timeout=60`: 1 hour
- `timeout=1440`: 24 hours

---

## Integration with MASAI Agents

### Adding Tools to Agents

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.Tools.Tool import tool

# Define tools
@tool("calculator")
def calculator(operation: str, x: float, y: float) -> float:
    """Performs arithmetic operations."""
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    return 0.0

@tool("search")
def search(query: str) -> str:
    """Searches for information."""
    return f"Search results for: {query}"

# Create agent with tools
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json"
)

agent = manager.create_agent(
    agent_name="assistant",
    tools=[calculator, search],  # Pass tool instances
    agent_details=AgentDetails(
        capabilities=["calculation", "search"],
        description="Assistant with calculator and search",
        style="helpful"
    )
)
```

### Tool Usage in Agent Workflow

When an agent receives a query:
1. **Router** decides if a tool is needed
2. **Evaluator** selects the appropriate tool
3. **Tool executes** with validated parameters
4. **Reflector** processes the tool output
5. **Response** is generated using tool results

---

## Best Practices

### 1. Write Clear Docstrings

```python
@tool("search_database")
def search_database(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Searches the database for matching records.
    
    This tool performs a full-text search across all indexed fields
    and returns the top matching results.
    
    Args:
        query: Search query string (supports wildcards)
        limit: Maximum number of results to return (default: 10, max: 100)
    
    Returns:
        List of matching records with id, title, and score
    
    Raises:
        ValueError: If query is empty or limit is invalid
    """
    # Implementation
```

### 2. Use Type Annotations

```python
# ✅ Good: Clear types
@tool("process")
def process(items: List[str], count: int) -> Dict[str, int]:
    pass

# ❌ Bad: No types
@tool("process")
def process(items, count):
    pass
```

### 3. Handle Errors Gracefully

```python
@tool("divide")
def divide(x: float, y: float) -> float:
    """Divides two numbers safely."""
    try:
        return x / y
    except ZeroDivisionError:
        return float('inf')
    except Exception as e:
        raise ValueError(f"Division error: {str(e)}")
```

### 4. Cache Expensive Operations

```python
# ✅ Good: Cache expensive API calls
@tool("fetch_weather")
@cache.masai_cache
async def fetch_weather(city: str) -> str:
    """Fetches weather data (cached)."""
    return await api_call(city)

# ❌ Bad: No caching for repeated calls
@tool("fetch_weather")
async def fetch_weather(city: str) -> str:
    """Fetches weather data (not cached)."""
    return await api_call(city)
```

### 5. Use Async for I/O Operations

```python
# ✅ Good: Async for API calls
@tool("fetch_data")
async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# ❌ Bad: Sync for I/O (blocks event loop)
@tool("fetch_data")
def fetch_data(url: str) -> str:
    return requests.get(url).text
```

### 6. Validate Input Parameters

```python
@tool("search")
def search(query: str, limit: int = 10) -> List[str]:
    """Searches with validation."""
    if not query or len(query) < 3:
        raise ValueError("Query must be at least 3 characters")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")
    return perform_search(query, limit)
```

---

## Complete Examples

### Example 1: Simple Calculator Tool

```python
from masai.Tools.Tool import tool

@tool("calculator")
def calculator(operation: str, x: float, y: float) -> float:
    """
    Performs basic arithmetic operations.
    
    Args:
        operation: add, subtract, multiply, or divide
        x: First number
        y: Second number
    
    Returns:
        Result of the operation
    """
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else float('inf')
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    return operations[operation](x, y)

# Usage
result = calculator.invoke({"operation": "add", "x": 10, "y": 5})
print(result)  # 15.0
```

### Example 2: Async API Tool with Caching

```python
from masai.Tools.Tool import tool
from masai.Tools.utilities.cache import ToolCache
import aiohttp

# Setup cache
cache = ToolCache(host='localhost', port=6379, timeout=30)

@tool("fetch_weather")
@cache.masai_cache
async def fetch_weather(city: str, units: str = "metric") -> str:
    """
    Fetches weather data for a city (cached for 30 minutes).

    Args:
        city: City name
        units: Temperature units (metric or imperial)

    Returns:
        Weather information as JSON string
    """
    api_key = "your_api_key"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid={api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return f"Temperature in {city}: {data['main']['temp']}°"

# Usage
result = await fetch_weather.ainvoke({"city": "London", "units": "metric"})
print(result)  # Temperature in London: 15.5°
```

### Example 3: Database Query Tool

```python
from masai.Tools.Tool import tool
from masai.Tools.utilities.cache import ToolCache
from typing import List, Dict, Any
import asyncpg

cache = ToolCache(host='localhost', port=6379, timeout=10)

@tool("search_users")
@cache.masai_cache
async def search_users(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Searches users in the database (cached for 10 minutes).

    Args:
        query: Search query (name or email)
        limit: Maximum results (default: 10)

    Returns:
        List of matching user records
    """
    conn = await asyncpg.connect('postgresql://user:password@localhost/db')

    try:
        rows = await conn.fetch(
            "SELECT id, name, email FROM users WHERE name ILIKE $1 OR email ILIKE $1 LIMIT $2",
            f"%{query}%", limit
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()

# Usage
users = await search_users.ainvoke({"query": "john", "limit": 5})
print(users)  # [{"id": 1, "name": "John Doe", "email": "john@example.com"}, ...]
```

### Example 4: File Processing Tool

```python
from masai.Tools.Tool import tool
from typing import List
import os

@tool("list_files")
def list_files(directory: str, extension: str = "") -> List[str]:
    """
    Lists files in a directory with optional extension filter.

    Args:
        directory: Directory path
        extension: File extension filter (e.g., ".txt", ".py")

    Returns:
        List of file names
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")

    files = os.listdir(directory)

    if extension:
        files = [f for f in files if f.endswith(extension)]

    return files

# Usage
files = list_files.invoke({"directory": "/home/user/docs", "extension": ".pdf"})
print(files)  # ["document1.pdf", "document2.pdf", ...]
```

### Example 5: Complete Agent with Multiple Tools

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.Tools.Tool import tool
from masai.Tools.utilities.cache import ToolCache
import asyncio

# Setup cache
cache = ToolCache(host='localhost', port=6379, timeout=60)

# Define tools
@tool("calculator")
def calculator(operation: str, x: float, y: float) -> float:
    """Performs arithmetic operations."""
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else float('inf')
    }
    return operations.get(operation, lambda a, b: 0)(x, y)

@tool("search_web")
@cache.masai_cache
async def search_web(query: str, max_results: int = 5) -> str:
    """Searches the web (cached for 60 minutes)."""
    # Simulated search
    return f"Top {max_results} results for '{query}': [result1, result2, ...]"

@tool("get_time")
def get_time(timezone: str = "UTC") -> str:
    """Gets current time in specified timezone."""
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

@tool("translate")
@cache.masai_cache
async def translate(text: str, target_lang: str = "en") -> str:
    """Translates text to target language (cached)."""
    # Simulated translation
    return f"Translated '{text}' to {target_lang}"

# Create agent with all tools
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json",
    logging=True
)

agent = manager.create_agent(
    agent_name="multi_tool_assistant",
    tools=[calculator, search_web, get_time, translate],
    agent_details=AgentDetails(
        capabilities=["calculation", "search", "time", "translation"],
        description="Multi-purpose assistant with various tools",
        style="helpful and efficient"
    ),
    memory_order=10,
    long_context=True
)

# Use the agent
async def main():
    # Agent can now use any of the tools
    result = await agent.initiate_agent(
        query="What is 25 * 4, and what time is it in Tokyo?",
        passed_from="user"
    )
    print(result["answer"])

asyncio.run(main())
```

### Example 6: Tool with Complex Return Type

```python
from masai.Tools.Tool import tool
from typing import Dict, List, Any
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float

@tool("advanced_search", return_direct=True)
def advanced_search(
    query: str,
    filters: Dict[str, Any],
    limit: int = 10
) -> Dict[str, Any]:
    """
    Performs advanced search with filters.

    Args:
        query: Search query
        filters: Filter criteria (e.g., {"date": "2024", "type": "article"})
        limit: Maximum results

    Returns:
        Dictionary with results and metadata
    """
    # Simulated search
    results = [
        {
            "title": f"Result {i}",
            "url": f"https://example.com/{i}",
            "snippet": f"Snippet for result {i}",
            "score": 0.9 - (i * 0.1)
        }
        for i in range(1, limit + 1)
    ]

    return {
        "query": query,
        "filters": filters,
        "total_results": len(results),
        "results": results
    }

# Usage
result = advanced_search.invoke({
    "query": "machine learning",
    "filters": {"date": "2024", "type": "article"},
    "limit": 5
})
print(result)  # Returns dict with results
```

---

## Troubleshooting

### Issue: "Tool cannot be called directly"

**Cause**: Trying to call tool like a regular function.

**Solution**: Use execution methods:
```python
# ❌ Wrong
result = calculator("add", 5, 3)

# ✅ Correct
result = calculator.invoke({"operation": "add", "x": 5, "y": 3})
```

### Issue: "Tool is asynchronous; use ainvoke()"

**Cause**: Using sync method on async tool.

**Solution**: Use async methods:
```python
# ❌ Wrong
result = async_tool.invoke({"param": "value"})

# ✅ Correct
result = await async_tool.ainvoke({"param": "value"})
```

### Issue: Redis connection error

**Cause**: Redis server not running or wrong configuration.

**Solution**:
1. Start Redis: `redis-server`
2. Check connection: `redis-cli ping`
3. Verify host/port in ToolCache configuration

### Issue: Cache not working

**Cause**: Different argument formats create different cache keys.

**Solution**: Use consistent argument order and types:
```python
# These create DIFFERENT cache keys:
tool.invoke({"x": 5, "y": 3})
tool.invoke({"y": 3, "x": 5})

# Solution: Use consistent order or normalize in tool
```

---

## Related Documentation

- [Quick Start Guide](QUICK_START.md) - Get started with MASAI
- [Agent Manager Detailed](AGENTMANAGER_DETAILED.md) - AgentManager API
- [Model Parameters](MODEL_PARAMETERS.md) - Model configuration
- [Multi-Agent System Guide](MULTIAGENT_SYSTEM_GUIDE.md) - Multi-agent coordination
- [API Reference](API_REFERENCE.md) - Complete API documentation

---

## Summary

MASAI tools provide:
- ✅ **Simple decorator-based** tool definition
- ✅ **Automatic schema generation** from function signatures
- ✅ **Type validation** using Pydantic
- ✅ **Sync and async support** for different use cases
- ✅ **Redis caching** for expensive operations
- ✅ **Seamless integration** with MASAI agents
- ✅ **Multiple execution methods** (invoke, run, ainvoke, arun)

Use this guide to create powerful, efficient tools for your MASAI agents!

