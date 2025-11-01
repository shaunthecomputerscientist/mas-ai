# Configuration Guide

Complete reference for all MASAI configuration options.

## Configuration Methods

### 1. JSON Config File

Create `model_config.json`:

```json
{
  "router": {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.3,
    "max_output_tokens": 2048,
    "top_p": 0.9
  },
  "evaluator": {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.5,
    "max_output_tokens": 1024
  },
  "reflector": {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_output_tokens": 1024
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

### 2. Dictionary Config

```python
config_dict = {
    "router": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.3,
        "max_output_tokens": 2048
    },
    "evaluator": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.5
    },
    "reflector": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.7
    }
}

agent = manager.create_agent(
    agent_name="assistant",
    agent_details=agent_details,
    config_dict=config_dict
)
```

### 3. Environment Variables

```bash
export ROUTER_MODEL="gemini-2.5-flash"
export ROUTER_TEMPERATURE="0.3"
export EVALUATOR_MODEL="gemini-2.5-flash"
export REFLECTOR_MODEL="gemini-2.5-flash"
```

## Agent Configuration

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | str | Required | Unique agent identifier |
| `agent_details` | AgentDetails | Required | Agent capabilities |
| `tools` | list | [] | LangChain tools |
| `memory_order` | int | 5 | Messages before summarization |
| `long_context` | bool | False | Enable long-context |
| `long_context_order` | int | 10 | Summaries before flush |
| `persist_memory` | bool | False | Enable persistence |
| `memory_config` | Config | None | Redis/Qdrant config |

### Example

```python
agent = manager.create_agent(
    agent_name="research_agent",
    agent_details=AgentDetails(
        capabilities=["research", "analysis"],
        description="Research specialist",
        style="detailed"
    ),
    tools=[search_tool, summarize_tool],
    memory_order=10,
    long_context=True,
    long_context_order=20,
    persist_memory=True,
    memory_config=redis_config
)
```

## Model Parameters

### Supported Models

**OpenAI**:
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-3.5-turbo

**Google Gemini**:
- gemini-2.5-flash
- gemini-2.5-pro
- gemini-1.5-flash
- gemini-1.5-pro

**Anthropic Claude**:
- claude-3-5-sonnet
- claude-3-opus
- claude-3-haiku

### Common Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `temperature` | float | 0.0-2.0 | Randomness (0=deterministic) |
| `top_p` | float | 0.0-1.0 | Nucleus sampling |
| `top_k` | int | 1+ | Top-k sampling |
| `max_output_tokens` | int | 1+ | Max response length |
| `frequency_penalty` | float | -2.0-2.0 | Reduce repetition |
| `presence_penalty` | float | -2.0-2.0 | Encourage new topics |

### Model-Specific Parameters

**OpenAI**:
```python
{
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

**Google Gemini**:
```python
{
    "model_name": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.9,
    "top_k": 40,
    "thinking_budget": 10000  # For thinking models
}
```

**Anthropic Claude**:
```python
{
    "model_name": "claude-3-5-sonnet",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "top_k": 40
}
```

## Memory Configuration

### Redis Config

```python
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(
        model="text-embedding-3-small"
    ),
    ttl_seconds=86400,
    dedup_mode="similarity",
    dedup_similarity_threshold=0.95
)
```

### Qdrant Config

```python
from masai.Memory.LongTermMemory import QdrantConfig

qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=1536,
    embedding_model=embeddings,
    distance="cosine"
)
```

## Embedding Models

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 dims
    # model="text-embedding-3-large",  # 3072 dims
)
```

### HuggingFace Embeddings

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # 384 dims
    # model_name="all-mpnet-base-v2",  # 768 dims
)
```

### Custom Embeddings

```python
from langchain_core.embeddings import Embeddings

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Your embedding logic
        return embeddings
    
    def embed_query(self, text):
        # Your embedding logic
        return embedding
```

## Advanced Configuration

### Router-Evaluator-Reflector Settings

```python
config = {
    "router": {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.3,  # Low for consistent routing
        "max_output_tokens": 2048
    },
    "evaluator": {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.5,  # Medium for evaluation
        "max_output_tokens": 1024
    },
    "reflector": {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.7,  # Higher for creativity
        "max_output_tokens": 1024
    }
}
```

### Long-Context Settings

```python
# For short conversations
agent = manager.create_agent(
    ...,
    memory_order=3,
    long_context_order=5
)

# For medium conversations
agent = manager.create_agent(
    ...,
    memory_order=10,
    long_context_order=20
)

# For long conversations
agent = manager.create_agent(
    ...,
    memory_order=20,
    long_context_order=50
)
```

## Configuration Validation

### Check Configuration

```python
# Print agent config
print(agent.config)

# Check memory settings
print(f"Memory order: {agent.memory_order}")
print(f"Long context order: {agent.long_context_order}")
print(f"Persist memory: {agent.persist_memory}")

# Check model settings
print(f"Router model: {agent.router_model}")
print(f"Temperature: {agent.temperature}")
```

## Best Practices

1. **Use JSON config** for production deployments
2. **Set environment variables** for sensitive data
3. **Test configuration** before deployment
4. **Document custom settings** for team
5. **Version control** configuration files
6. **Monitor model costs** with expensive models
7. **Use appropriate temperatures** for each component

## Troubleshooting

### Issue: "Unknown parameter"

**Solution**: Check parameter name and model support
```python
# Wrong
config = {"max_tokens": 2048}

# Right (for OpenAI)
config = {"max_output_tokens": 2048}
```

### Issue: "Invalid temperature"

**Solution**: Use valid range
```python
# Wrong
temperature = 3.0  # Out of range

# Right
temperature = 0.7  # 0.0-2.0
```

### Issue: "Model not found"

**Solution**: Use supported model name
```python
# Check available models
from langchain_openai import ChatOpenAI
# See documentation for available models
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more help.

