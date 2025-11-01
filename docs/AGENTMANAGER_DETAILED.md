# AgentManager - Comprehensive Guide

## Overview

**AgentManager** is the central registry and coordinator for all agents in MASAI. It manages:
- Agent creation and lifecycle
- Shared memory (LongTermMemory)
- Model configuration loading
- User isolation
- Streaming callbacks

---

## Constructor

### Signature

```python
AgentManager(
    logging: bool = True,
    context: dict = None,
    model_config_path: str = None,
    chat_log: str = None,
    streaming: bool = False,
    streaming_callback: Optional[Callable] = None,
    user_id: Optional[Union[str, int]] = None,
    memory_config: Optional[Union[QdrantConfig, RedisConfig, Dict]] = None,
    categories_resolver: Optional[Callable] = None
)
```

### Parameters

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logging` | bool | True | Enable logging output |
| `model_config_path` | str | **Required** | Path to model_config.json |
| `context` | dict | None | Shared context for all agents |
| `chat_log` | str | None | Path to save chat logs |
| `streaming` | bool | False | Enable streaming responses |
| `streaming_callback` | Callable | None | Async callback for streaming chunks |

#### Memory Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | str/int | None | User identifier for isolation |
| `memory_config` | QdrantConfig/RedisConfig | None | Persistent memory backend |
| `categories_resolver` | Callable | None | Custom categorization function |

### Example

```python
from masai.AgentManager import AgentManager
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

# Basic setup
manager = AgentManager(
    model_config_path="model_config.json",
    user_id="user_123",
    logging=True
)

# With persistent memory
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

manager = AgentManager(
    model_config_path="model_config.json",
    user_id="user_123",
    memory_config=redis_config,
    logging=True
)

# With streaming
async def stream_callback(chunk: str):
    print(chunk, end="", flush=True)

manager = AgentManager(
    model_config_path="model_config.json",
    streaming=True,
    streaming_callback=stream_callback
)
```

---

## Memory Configuration

### memory_config vs persist_memory

**memory_config** (AgentManager level):
- Defines WHERE to store persistent memory (Redis/Qdrant)
- Set once at AgentManager initialization
- Shared across all agents

**persist_memory** (Agent level):
- Enables/disables persistence for specific agent
- Can be True/False per agent
- Requires memory_config in AgentManager

### Example

```python
# Setup AgentManager with memory backend
manager = AgentManager(
    model_config_path="model_config.json",
    memory_config=redis_config,  # Define backend
    user_id="user_123"
)

# Create agent with persistence enabled
agent1 = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=...,
    persist_memory=True  # Use AgentManager's memory_config
)

# Create agent without persistence
agent2 = manager.create_agent(
    agent_name="analyzer",
    tools=[],
    agent_details=...,
    persist_memory=False  # No persistence
)
```

---

## create_agent() Method

### Signature

```python
def create_agent(
    agent_name: str,
    tools: List[object],
    agent_details: AgentDetails,
    memory_order: int = 10,
    long_context: bool = True,
    long_context_order: int = 20,
    shared_memory_order: int = 10,
    plan: bool = False,
    temperature: float = 0.2,
    context_callable: Optional[Callable] = None,
    retain_messages_order: int = 10,
    max_tool_output_words: int = 3000,
    persist_memory: Optional[bool] = None,
    **kwargs
) -> Agent
```

### Memory Order Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_order` | 10 | Short-term memory size (chat_history) |
| `long_context_order` | 20 | Summaries before flush to persistent |
| `shared_memory_order` | 10 | Component shared memory size |
| `retain_messages_order` | 10 | Internal state retention |

### Example

```python
agent = manager.create_agent(
    agent_name="research_assistant",
    tools=[search_tool, web_tool],
    agent_details=AgentDetails(
        capabilities=["research", "analysis"],
        description="Research assistant",
        style="detailed and thorough"
    ),
    memory_order=15,           # Keep 15 recent messages
    long_context_order=30,     # Summarize after 30 summaries
    shared_memory_order=10,    # Component shared memory
    retain_messages_order=20,  # Retain 20 messages internally
    plan=True,                 # Include planner
    temperature=0.3,           # Lower randomness
    persist_memory=True        # Enable persistence
)
```

---

## config_dict for Component Overrides

### Structure

```python
config_dict = {
    # Router overrides
    "router_temperature": 0.5,
    "router_memory_order": 15,
    "router_long_context_order": 25,
    "router_max_output_tokens": 512,
    
    # Evaluator overrides
    "evaluator_temperature": 0.3,
    "evaluator_memory_order": 12,
    "evaluator_long_context_order": 20,
    
    # Reflector overrides
    "reflector_temperature": 0.2,
    "reflector_memory_order": 10,
    "reflector_long_context_order": 30,
    "reflector_max_output_tokens": 2048,
    
    # Planner overrides (if plan=True)
    "planner_temperature": 0.4,
    "planner_memory_order": 8,
    "planner_long_context_order": 15
}

agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=...,
    config_dict=config_dict
)
```

### Precedence

1. **config_dict** (highest priority)
2. **Agent-specific model_config.json**
3. **"all" in model_config.json** (lowest priority)

---

## User Isolation

### user_id Parameter

```python
# User 1
manager1 = AgentManager(
    model_config_path="model_config.json",
    user_id="user_alice",
    memory_config=redis_config
)

# User 2
manager2 = AgentManager(
    model_config_path="model_config.json",
    user_id="user_bob",
    memory_config=redis_config
)

# Each user's memories are isolated
# agent1.llm_router.long_term_memory only sees user_alice's data
# agent2.llm_router.long_term_memory only sees user_bob's data
```

### categories_resolver

```python
def my_categorizer(doc):
    """Custom categorization function"""
    if "urgent" in doc.page_content.lower():
        return ["urgent"]
    elif "research" in doc.page_content.lower():
        return ["research"]
    return ["general"]

manager = AgentManager(
    model_config_path="model_config.json",
    user_id="user_123",
    memory_config=redis_config,
    categories_resolver=my_categorizer
)
```

---

## Agent Registry Methods

### get_agent()

```python
agent = manager.get_agent("assistant")
if agent:
    result = await agent.initiate_agent(query="Hello")
else:
    print("Agent not found")
```

### list_agents()

```python
agents = manager.list_agents()
print(f"Available agents: {agents}")
# Output: ['assistant', 'analyzer', 'researcher']
```

### delete_agent()

```python
manager.delete_agent("assistant")
print("Agent deleted")
```

---

## Real-World Example

### Multi-User Research System

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

# Setup shared memory backend
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="research_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

# Create manager for user
manager = AgentManager(
    model_config_path="model_config.json",
    user_id="researcher_alice",
    memory_config=redis_config,
    logging=True
)

# Create research agent
research_agent = manager.create_agent(
    agent_name="researcher",
    tools=[search_tool, web_tool, pdf_tool],
    agent_details=AgentDetails(
        capabilities=["research", "analysis", "synthesis"],
        description="Research assistant",
        style="detailed and academic"
    ),
    memory_order=20,
    long_context_order=40,
    plan=True,
    persist_memory=True,
    config_dict={
        "router_temperature": 0.3,
        "reflector_temperature": 0.2,
        "reflector_max_output_tokens": 4096
    }
)

# Execute research query
result = await research_agent.initiate_agent(
    query="Research the latest advances in quantum computing",
    passed_from="user"
)

print(result["answer"])
```

---

## Troubleshooting

### Issue: "model_config_path must be provided"

```
ValueError: model_config_path must be provided
```

**Solution**: Pass model_config_path to AgentManager:
```python
manager = AgentManager(model_config_path="model_config.json")
```

### Issue: "persist_memory=True requires memory_config"

```
ValueError: persist_memory=True requires memory_config
```

**Solution**: Either set memory_config or persist_memory=False:
```python
manager = AgentManager(
    model_config_path="model_config.json",
    memory_config=redis_config
)
```

### Issue: "Streaming callback needs to be provided"

```
ValueError: Streaming callback needs to be provided
```

**Solution**: Provide streaming_callback if streaming=True:
```python
async def callback(chunk):
    print(chunk, end="", flush=True)

manager = AgentManager(
    model_config_path="model_config.json",
    streaming=True,
    streaming_callback=callback
)
```

---

## See Also

- [FRAMEWORK_OVERVIEW.md](FRAMEWORK_OVERVIEW.md) - Architecture overview
- [AGENT_DETAILED.md](AGENT_DETAILED.md) - Agent API
- [MODEL_CONFIG_GUIDE.md](MODEL_CONFIG_GUIDE.md) - Model configuration
- [MEMORY_ARCHITECTURE_DEEP_DIVE.md](MEMORY_ARCHITECTURE_DEEP_DIVE.md) - Memory system

