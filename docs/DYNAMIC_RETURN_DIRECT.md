# Dynamic `return_direct` Feature

## Overview

MAS-AI now supports **dynamic `return_direct` control** for tools, allowing you to decide at runtime whether a tool's output should be returned directly to the user or passed through the evaluation pipeline.

## How It Works

### Three Levels of `return_direct` Control

1. **Decorator Level (Static)**: Set in the `@tool` decorator
2. **Parameter Level (Dynamic)**: Passed as a tool argument at runtime
3. **Return Value Level (Runtime)**: Decided internally during tool execution

**Priority:** Return Value > Parameter > Decorator

---

## Usage Examples

### Example 1: Static `return_direct` (Decorator)

```python
from langchain.tools import tool

@tool("MongoDB Query Executor", return_direct=True)
async def execute_mongodb_query(
    collection_name: str,
    query: Union[str, Dict, List, bool, None],
    limit: Optional[int] = 20
) -> str:
    """Execute a MongoDB query and return results."""
    # ... implementation ...
    return json.dumps(results)
```

**Behavior:**
- Tool output is **always** returned directly to the user
- Skips evaluator and reflector nodes
- Goes straight to END state

---

### Example 2: Dynamic `return_direct` (Parameter)

```python
from langchain.tools import tool

@tool("MongoDB Query Executor", return_direct=False)
async def execute_mongodb_query(
    collection_name: str,
    query: Union[str, Dict, List, bool, None],
    brokerageId: Optional[str] = None,
    *,
    query_type: str = "find",
    limit: Optional[int] = 20,
    skip: Optional[int] = 0,
    sort: Optional[Dict[str, int]] = None,
    projection: Optional[Dict[str, int]] = None,
    distinct_field: Optional[str] = None,
    return_direct: bool = False,  # ✅ Dynamic parameter
    output_format: str = "summary"
) -> Union[str, Dict[str, Any]]:
    """
    Execute a MongoDB query with optional return_direct control.
    
    Args:
        collection_name: Name of the MongoDB collection
        query: Query to execute
        return_direct: If True, return results directly without evaluation
        output_format: Format of output ("summary", "detailed", "raw")
    """
    # ... implementation ...
    
    # The return_direct parameter is automatically handled by the framework
    # You don't need to do anything special in the tool implementation
    return results
```

**Behavior:**
- When LLM calls tool with `return_direct=True` in tool_input:
  ```json
  {
      "tool": "MongoDB Query Executor",
      "tool_input": {
          "collection_name": "users",
          "query": {"status": "active"},
          "return_direct": true
      }
  }
  ```
  → Output returned directly to user

- When LLM calls tool with `return_direct=False` or omits it:
  ```json
  {
      "tool": "MongoDB Query Executor",
      "tool_input": {
          "collection_name": "users",
          "query": {"status": "active"}
      }
  }
  ```
  → Output goes through evaluator/reflector pipeline

---

### Example 3: Runtime Decision (Return Value)

```python
from langchain.tools import tool

@tool("Smart Query Executor", return_direct=False)
async def smart_query_executor(
    query: str,
    collection: str,
    auto_decide: bool = True
) -> Union[str, Dict[str, Any]]:
    """
    Execute query and decide internally whether to return directly.

    Args:
        query: Query to execute
        collection: Collection name
        auto_decide: If True, tool decides based on query complexity
    """
    results = db.query(collection, query)

    if auto_decide:
        # Tool decides internally based on logic
        is_simple_query = len(results) < 10 and not requires_analysis(query)

        if is_simple_query:
            # Simple query - return directly
            return {
                "data": results,
                "return_direct": True  # ✅ Tool decides to return directly
            }
        else:
            # Complex query - needs evaluation
            return {
                "data": results,
                "return_direct": False  # ✅ Tool decides to evaluate
            }
    else:
        # Just return data, use decorator/parameter setting
        return {"data": results}
```

**Behavior:**
- Tool analyzes query complexity internally
- Simple queries → `return_direct=True` in return value
- Complex queries → `return_direct=False` in return value
- Framework extracts `return_direct` from result and acts accordingly

---

### Example 4: Override Decorator with Dynamic Parameter

```python
@tool("Data Fetcher", return_direct=True)  # Decorator says True
async def fetch_data(
    source: str,
    return_direct: bool = False  # But parameter can override
) -> str:
    """Fetch data from source."""
    return f"Data from {source}"
```

**Behavior:**

| Tool Input | Return Value | Effective `return_direct` | Reason |
|------------|--------------|---------------------------|--------|
| `{"source": "api"}` | No `return_direct` | `True` | Uses decorator default |
| `{"source": "api", "return_direct": true}` | No `return_direct` | `True` | Parameter matches decorator |
| `{"source": "api", "return_direct": false}` | No `return_direct` | `False` | **Parameter overrides decorator** |
| `{"source": "api"}` | `{"data": "...", "return_direct": true}` | `True` | **Return value overrides decorator** |
| `{"source": "api", "return_direct": false}` | `{"data": "...", "return_direct": true}` | `True` | **Return value overrides parameter** |

---

## Implementation Details

### Framework Logic

```python
# In base_agent.py execute_tool method:

# 1. Check decorator's return_direct (lowest priority)
tool_decorator_return_direct = hasattr(tool, 'return_direct') and tool.return_direct

# 2. Check dynamic return_direct from tool_input (medium priority)
input_return_direct = False
if isinstance(tool_input, dict) and 'return_direct' in tool_input:
    input_return_direct = tool_input.get('return_direct', False)

# 3. Check return_direct from tool result (highest priority)
result_return_direct = False
actual_result = result
if isinstance(result, dict) and 'return_direct' in result:
    result_return_direct = result.get('return_direct', False)
    # Extract actual result (remove return_direct metadata)
    actual_result = {k: v for k, v in result.items() if k != 'return_direct'}

# 4. Priority: result > input > decorator
should_return_direct = result_return_direct or input_return_direct or tool_decorator_return_direct

# 5. If return_direct is True, set final answer
if should_return_direct:
    state.update({
        'current_tool': None,
        'tool_input': None,
        'answer': state['tool_output'],
        'reasoning': f"Result directly provided by tool '{tool_name}'.",
        'satisfied': True,
        'delegate_to_agent': None,
    })
```

---

## Use Cases

### Use Case 1: Database Queries

**Scenario:** User wants raw query results without AI interpretation

```python
@tool("SQL Query", return_direct=False)
async def execute_sql(
    query: str,
    return_direct: bool = False
) -> str:
    """Execute SQL query."""
    results = db.execute(query)
    return json.dumps(results)
```

**Agent Behavior:**
- User: "Get all active users, return raw data"
- LLM calls: `execute_sql(query="SELECT * FROM users WHERE active=1", return_direct=True)`
- Framework returns raw JSON directly to user

---

### Use Case 2: File Operations

**Scenario:** File content should be returned directly, but file metadata needs evaluation

```python
@tool("File Reader", return_direct=False)
async def read_file(
    filepath: str,
    return_direct: bool = False,
    include_metadata: bool = False
) -> str:
    """Read file content."""
    content = open(filepath).read()
    
    if include_metadata:
        # Metadata needs evaluation
        return json.dumps({
            "content": content,
            "size": len(content),
            "lines": content.count('\n')
        })
    else:
        # Raw content can be returned directly
        return content
```

**Agent Behavior:**
- User: "Show me config.json"
- LLM calls: `read_file(filepath="config.json", return_direct=True)`
- Framework returns file content directly

- User: "Analyze config.json"
- LLM calls: `read_file(filepath="config.json", include_metadata=True, return_direct=False)`
- Framework passes to evaluator for analysis

---

### Use Case 3: API Calls

**Scenario:** Some API responses need processing, others don't

```python
@tool("API Caller", return_direct=False)
async def call_api(
    endpoint: str,
    method: str = "GET",
    return_direct: bool = False,
    parse_response: bool = True
) -> str:
    """Call external API."""
    response = requests.request(method, endpoint)
    
    if parse_response:
        # Parsed response needs evaluation
        return json.dumps(response.json())
    else:
        # Raw response can be returned directly
        return response.text
```

---

## Benefits

### 1. **Flexibility**
- Single tool can handle both direct and evaluated outputs
- No need to create separate tools for different behaviors

### 2. **LLM Control**
- LLM can decide based on user intent whether to return directly
- User can explicitly request raw output: "give me raw data"

### 3. **Efficiency**
- Skip unnecessary evaluation for simple queries
- Reduce token usage and latency

### 4. **Backward Compatibility**
- Existing tools with decorator `return_direct` still work
- New parameter is optional

---

## Best Practices

### 1. **Default to `False`**
```python
@tool("My Tool", return_direct=False)  # Let evaluation happen by default
async def my_tool(
    param: str,
    return_direct: bool = False  # Optional override
) -> str:
    pass
```

### 2. **Document the Parameter**
```python
async def my_tool(
    param: str,
    return_direct: bool = False
) -> str:
    """
    Execute tool.
    
    Args:
        param: Main parameter
        return_direct: If True, return output directly without evaluation.
                      Useful when user wants raw data.
    """
    pass
```

### 3. **Use Keyword-Only Arguments**
```python
async def my_tool(
    required_param: str,
    *,  # Everything after this is keyword-only
    return_direct: bool = False,
    optional_param: str = "default"
) -> str:
    pass
```

This prevents accidental positional argument issues.

---

## Testing

### Test Case 1: Decorator Only
```python
@tool("Test Tool", return_direct=True)
def test_tool(x: int) -> str:
    return f"Result: {x}"

# Expected: Always returns directly
```

### Test Case 2: Dynamic Override
```python
@tool("Test Tool", return_direct=True)
def test_tool(x: int, return_direct: bool = True) -> str:
    return f"Result: {x}"

# Call with return_direct=False
# Expected: Goes through evaluation (dynamic overrides decorator)
```

### Test Case 3: No Decorator, Dynamic Only
```python
@tool("Test Tool")  # No return_direct in decorator
def test_tool(x: int, return_direct: bool = False) -> str:
    return f"Result: {x}"

# Call with return_direct=True
# Expected: Returns directly (dynamic parameter works)
```

---

## Logging

The framework logs the source of `return_direct`:

```
Tool 'MongoDB Query Executor' has dynamic return_direct=True from tool_input
Tool 'MongoDB Query Executor' requested return_direct (dynamic parameter). Setting final answer.
```

Or:

```
Tool 'File Reader' requested return_direct (decorator). Setting final answer.
```

---

## Summary

| Feature | Description |
|---------|-------------|
| **Decorator `return_direct`** | Static setting in `@tool()` decorator |
| **Parameter `return_direct`** | Dynamic setting in tool arguments |
| **Priority** | Parameter > Decorator |
| **Default Behavior** | `False` (goes through evaluation) |
| **Use Cases** | Raw data queries, file operations, API calls |
| **Benefits** | Flexibility, efficiency, LLM control |

---

## Version

- **Added in:** v0.2.5
- **Status:** Production Ready ✅

