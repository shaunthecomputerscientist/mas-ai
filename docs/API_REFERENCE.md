# API Reference

Complete API documentation for MASAI.

## AgentManager

Main orchestrator for agent creation and management.

### Constructor

```python
class AgentManager:
    def __init__(
        self,
        user_id: str,
        model_config_path: Optional[str] = None,
        memory_config: Optional[Config] = None,
        logging: bool = True
    )
```

**Parameters**:
- `user_id` (str): Unique user identifier
- `model_config_path` (str, optional): Path to JSON config file
- `memory_config` (Config, optional): Redis/Qdrant configuration
- `logging` (bool): Enable logging

### Methods

#### create_agent

```python
def create_agent(
    self,
    agent_name: str,
    agent_details: AgentDetails,
    tools: list = [],
    memory_order: int = 5,
    long_context: bool = False,
    long_context_order: int = 10,
    persist_memory: bool = False,
    memory_config: Optional[Config] = None,
    config_dict: Optional[dict] = None,
    **kwargs
) -> Agent
```

**Parameters**:
- `agent_name` (str): Unique agent identifier
- `agent_details` (AgentDetails): Agent capabilities
- `tools` (list): LangChain tools
- `memory_order` (int): Messages before summarization
- `long_context` (bool): Enable long-context mode
- `long_context_order` (int): Summaries before flush
- `persist_memory` (bool): Enable persistence
- `memory_config` (Config): Memory backend config
- `config_dict` (dict): Model configuration

**Returns**: Agent instance

#### get_agent

```python
def get_agent(self, agent_name: str) -> Optional[Agent]
```

**Returns**: Agent or None if not found

#### list_agents

```python
def list_agents(self) -> list[str]
```

**Returns**: List of agent names

---

## Agent

Router-Evaluator-Reflector agent with LangGraph-based workflow.

### Methods

#### initiate_agent

```python
async def initiate_agent(
    self,
    query: str,
    passed_from: Optional[str] = None,
    previous_node: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters**:
- `query` (str): User query to process
- `passed_from` (str, optional): Identifier of component that passed query
- `previous_node` (str, optional): Previous node in workflow

**Returns**: Dict with keys:
- `answer` (str): Final answer
- `messages` (List): Conversation history
- `reasoning` (str): Agent reasoning
- `satisfied` (bool): Whether agent is satisfied
- `current_node` (str): Final node
- `tool_output` (str): Tool execution output
- `delegate_to_agent` (Optional[str]): Delegated agent name

**Example**:
```python
result = await agent.initiate_agent(
    query="What is machine learning?",
    passed_from="user"
)
print(result["answer"])
```

#### initiate_agent_astream

```python
async def initiate_agent_astream(
    self,
    query: str,
    passed_from: Optional[str] = None,
    previous_node: Optional[str] = None
) -> AsyncGenerator[Tuple[str, Dict], None]
```

**Parameters**:
- `query` (str): User query to process
- `passed_from` (str, optional): Identifier of component that passed query
- `previous_node` (str, optional): Previous node in workflow

**Yields**: Tuple[str, Dict] with state updates

**Example**:
```python
async for state in agent.initiate_agent_astream(
    query="What is machine learning?",
    passed_from="user"
):
    node_name, state_dict = state
    state_value = [v for k, v in state_dict.items()][0]
    print(f"Node: {state_value['current_node']}")
```

#### set_context

```python
def set_context(
    self,
    context: Optional[Dict] = None,
    mode: str = "set"
) -> None
```

**Parameters**:
- `context` (dict, optional): Context data
- `mode` (str): "set" (replace) or "update" (merge)

**Example**:
```python
agent.set_context({
    "user_id": "user_123",
    "project": "research"
}, mode="set")
```

---

## MultiAgentSystem

Orchestrates multiple agents with different patterns.

### Constructor

```python
class MultiAgentSystem:
    def __init__(
        self,
        agentManager: AgentManager,
        supervisor_config: Optional[SupervisorConfig] = None,
        heirarchical_mas_result_callback: Optional[Callable] = None,
        agent_return_direct: bool = False
    )
```

### Methods

#### initiate_sequential_mas

```python
async def initiate_sequential_mas(
    self,
    query: str,
    agent_sequence: List[str],
    memory_order: int = 3
) -> str
```

**Parameters**:
- `query` (str): Initial query
- `agent_sequence` (List[str]): Agent names in order
- `memory_order` (int): Shared memory size

**Returns**: Final result string

#### initiate_hierarchical_mas

```python
async def initiate_hierarchical_mas(
    self,
    query: str
) -> Dict[str, Any]
```

**Parameters**:
- `query` (str): Query to process

**Returns**: Dict with `status`, `answer`, `task_id`

#### initiate_decentralized_mas

```python
async def initiate_decentralized_mas(
    self,
    query: str,
    set_entry_agent: Agent,
    memory_order: int = 3
) -> Dict[str, Any]
```

**Parameters**:
- `query` (str): Query to process
- `set_entry_agent` (Agent): Starting agent
- `memory_order` (int): Shared memory size

**Returns**: Dict with `answer`, `messages`, `reasoning`

---

## LongTermMemory

Persistent memory interface.

### Methods

#### save

```python
async def save(
    self,
    user_id: Union[str, int],
    documents: Sequence[Union[Document, str, Dict]]
) -> None
```

**Parameters**:
- `user_id`: User identifier
- `documents`: Documents to save

#### search

```python
async def search(
    self,
    user_id: Union[str, int],
    query: str,
    k: int = 5,
    categories: Optional[List[str]] = None
) -> List[Document]
```

**Parameters**:
- `user_id`: User identifier
- `query`: Search query
- `k`: Number of results
- `categories`: Filter by categories

**Returns**: List of matching documents

#### delete

```python
async def delete(
    self,
    user_id: Union[str, int],
    doc_id: str
) -> None
```

**Parameters**:
- `user_id`: User identifier
- `doc_id`: Document ID to delete

---

## RedisConfig

Redis backend configuration.

### Constructor

```python
class RedisConfig:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "masai_vectors",
        vector_size: int = 1536,
        embedding_model: Optional[Embeddings] = None,
        ttl_seconds: Optional[int] = None,
        dedup_mode: str = "similarity",
        dedup_similarity_threshold: float = 0.95
    )
```

**Parameters**:
- `redis_url` (str): Redis connection URL
- `index_name` (str): Index name
- `vector_size` (int): Embedding dimension
- `embedding_model` (Embeddings): Embedding model
- `ttl_seconds` (int, optional): Document TTL
- `dedup_mode` (str): Deduplication mode
- `dedup_similarity_threshold` (float): Similarity threshold

---

## QdrantConfig

Qdrant backend configuration.

### Constructor

```python
class QdrantConfig:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "masai_memories",
        vector_size: int = 1536,
        embedding_model: Optional[Embeddings] = None,
        distance: str = "cosine"
    )
```

**Parameters**:
- `url` (str): Qdrant URL
- `collection_name` (str): Collection name
- `vector_size` (int): Embedding dimension
- `embedding_model` (Embeddings): Embedding model
- `distance` (str): Distance metric

---

## AgentDetails

Agent configuration details.

### Constructor

```python
class AgentDetails:
    def __init__(
        self,
        capabilities: list[str],
        description: str,
        style: str = "concise"
    )
```

**Parameters**:
- `capabilities` (list[str]): Agent capabilities
- `description` (str): Agent description
- `style` (str): Response style

---

## Document

LangChain document wrapper.

### Constructor

```python
class Document:
    def __init__(
        self,
        page_content: str,
        metadata: Optional[dict] = None
    )
```

**Parameters**:
- `page_content` (str): Document content
- `metadata` (dict, optional): Document metadata

### Properties

- `page_content` (str): Document text
- `metadata` (dict): Document metadata

---

## Embeddings

Base embedding interface.

### Methods

#### embed_documents

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]
```

**Parameters**:
- `texts` (list[str]): Texts to embed

**Returns**: List of embeddings

#### embed_query

```python
def embed_query(self, text: str) -> list[float]
```

**Parameters**:
- `text` (str): Text to embed

**Returns**: Embedding vector

---

## Tool

LangChain tool decorator.

### Usage

```python
from langchain.tools import tool

@tool
def my_tool(input: str) -> str:
    """Tool description"""
    return "result"
```

**Parameters**:
- Function name: Tool name
- Docstring: Tool description
- Parameters: Tool inputs
- Return type: Tool output

---

## BaseModel

Pydantic model for structured output.

### Usage

```python
from pydantic import BaseModel

class MyModel(BaseModel):
    field1: str
    field2: int
    field3: list[str]
```

**Features**:
- Type validation
- JSON serialization
- Schema generation

---

## Exceptions

### MASAIError

Base exception for MASAI errors.

### ConfigError

Configuration error.

### MemoryError

Memory operation error.

### ToolError

Tool execution error.

---

## Type Hints

### Common Types

```python
from typing import Optional, List, Dict, Union, AsyncGenerator

# Optional parameter
param: Optional[str] = None

# List of items
items: List[str]

# Dictionary
config: Dict[str, Any]

# Union type
result: Union[str, dict]

# Async generator
async def stream() -> AsyncGenerator[str, None]:
    yield "chunk"
```

---

## Examples

### Basic Usage

```python
from masai.AgentManager import AgentManager, AgentDetails

manager = AgentManager(user_id="user_123")
agent = manager.create_agent(
    agent_name="assistant",
    agent_details=AgentDetails(
        capabilities=["analysis"],
        description="Assistant"
    )
)

result = await agent.initiate_agent(
    query="Hello!",
    passed_from="user"
)
print(result["answer"])
```

### With Memory

```python
from masai.AgentManager import AgentManager
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

# Configure memory backend
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

# Create manager with memory config
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json",
    memory_config=redis_config  # Pass to AgentManager
)

# Create agent with persistent memory enabled
agent = manager.create_agent(
    agent_name="assistant",
    agent_details=agent_details,
    persist_memory=True  # Enable persistence
)
```

### With Tools

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add numbers"""
    return a + b

agent = manager.create_agent(
    agent_name="calculator",
    agent_details=agent_details,
    tools=[add]
)
```

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for more examples.

