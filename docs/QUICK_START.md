# Quick Start Guide

Get started with MASAI in 5 minutes.

## Installation

```bash
pip install masai-framework
```

## Basic Setup

### 1. Create Agent Manager

```python
from masai.AgentManager import AgentManager, AgentDetails

manager = AgentManager(
    user_id="user_123",
    logging=True
)
```

### 2. Define Agent Details

```python
agent_details = AgentDetails(
    capabilities=["analysis", "reasoning", "problem-solving"],
    description="A helpful AI assistant",
    style="concise"
)
```

### 3. Create Agent

```python
agent = manager.create_agent(
    agent_name="assistant",
    agent_details=agent_details,
    tools=[]  # Add LangChain tools here
)
```

### 4. Execute Agent

```python
# Full execution - waits for complete response
result = await agent.initiate_agent(
    query="What is machine learning?",
    passed_from="user"
)
print(result["answer"])
```

## Streaming Responses

Stream responses in real-time with state updates:

```python
# Streaming - get real-time updates
async for state in agent.initiate_agent_astream(
    query="Tell me a story",
    passed_from="user"
):
    # state is a tuple: (node_name, state_dict)
    node_name, state_dict = state

    # Extract the actual state
    state_value = [v for k, v in state_dict.items()][0]

    # Access state fields
    current_node = state_value.get("current_node")
    answer = state_value.get("answer")
    tool = state_value.get("current_tool")

    if tool:
        print(f"Executing tool: {tool}")
    if answer:
        print(f"Answer: {answer}")
```

## Structured Output

Use Pydantic models for structured responses:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    explanation: str
    examples: list[str]
    key_concepts: list[str]

# Create agent with structured output
agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=agent_details,
    AnswerFormat=Answer  # Specify output structure
)

result = await agent.initiate_agent(
    query="Explain machine learning",
    passed_from="user"
)

# Result is already structured
print(result["answer"])  # Contains Answer model
```

## With Tools

Integrate LangChain tools:

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

agent = manager.create_agent(
    agent_name="calculator",
    agent_details=agent_details,
    tools=[add]
)
```

## With Persistent Memory

Enable Redis-backed persistent memory:

```python
from masai.AgentManager import AgentManager
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

# Configure memory backend
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
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
    persist_memory=True,  # Enable persistence
    long_context=True,
    long_context_order=5
)
```

## Multi-Agent System

Coordinate multiple agents:

```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem

# Create specialized agents
agent1 = manager.create_agent(
    agent_name="researcher",
    agent_details=AgentDetails(
        capabilities=["research"],
        description="Research specialist"
    ),
    tools=[]
)

agent2 = manager.create_agent(
    agent_name="writer",
    agent_details=AgentDetails(
        capabilities=["writing"],
        description="Writing specialist"
    ),
    tools=[]
)

# Decentralized MAS (peer-to-peer collaboration)
mas = MultiAgentSystem(agentManager=manager)

result = await mas.initiate_decentralized_mas(
    query="Research and write about AI",
    set_entry_agent=agent1,  # Start with researcher
    memory_order=3
)

print(result["answer"])
```

## Configuration File

Use JSON config for model settings:

```json
{
  "router": {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.3,
    "max_output_tokens": 2048
  },
  "evaluator": {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.5
  },
  "reflector": {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.7
  }
}
```

Load config:

```python
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json"
)
```

## Environment Variables

Set up your API keys:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export REDIS_URL="redis://localhost:6379"
```

## Next Steps

- [Full Usage Guide](USAGE_GUIDE.md)
- [Memory System](MEMORY_SYSTEM.md)
- [Configuration](CONFIGURATION.md)
- [Advanced Topics](ADVANCED.md)
- [API Reference](API_REFERENCE.md)

## Common Issues

**Redis Connection Error**: Ensure Redis is running
```bash
redis-server
```

**API Key Not Found**: Set environment variables
```bash
export OPENAI_API_KEY="your-key"
```

**Memory Not Working**: Verify long-context is enabled
```python
agent = manager.create_agent(
    ...,
    long_context=True,
    persist_memory=True
)
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more help.

