# Troubleshooting Guide

Solutions for common issues and problems.

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'masai'"

**Cause**: MASAI not installed or wrong Python environment

**Solutions**:
```bash
# Reinstall MASAI
pip uninstall masai-framework
pip install masai-framework

# Check installation
python -c "import masai; print(masai.__version__)"

# Use correct Python environment
which python
python --version  # Should be 3.8+
```

### Issue: "Dependency conflicts"

**Cause**: Incompatible package versions

**Solutions**:
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Issue: "LangChain import error"

**Cause**: LangChain not installed

**Solutions**:
```bash
pip install langchain langchain-core langchain-openai
pip install langchain-community  # For additional integrations
```

## Connection Issues

### Issue: "Redis connection refused"

**Cause**: Redis not running

**Solutions**:
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Start Redis
redis-server

# Or with Docker
docker run -d -p 6379:6379 redis:latest

# Check Redis logs
redis-cli INFO
```

### Issue: "Qdrant connection refused"

**Cause**: Qdrant not running

**Solutions**:
```bash
# Start Qdrant with Docker
docker run -d -p 6333:6333 qdrant/qdrant

# Check Qdrant health
curl http://localhost:6333/health

# Check logs
docker logs <container_id>
```

### Issue: "OpenAI API key not found"

**Cause**: API key not set

**Solutions**:
```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or in .env file
echo "OPENAI_API_KEY=sk-..." >> .env

# Verify
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Issue: "Google API key not found"

**Cause**: Google API key not set

**Solutions**:
```bash
export GOOGLE_API_KEY="..."
# Or
export GOOGLE_GENERATIVE_AI_API_KEY="..."
```

## Memory Issues

### Issue: "Memory not being retrieved"

**Cause**: Context overflow not triggered

**Solutions**:
```python
# Check if overflow occurred
print(f"Summaries: {len(agent.context_summaries)}")
print(f"Order: {agent.long_context_order}")

# Should have: len(context_summaries) > long_context_order

# Reduce thresholds to trigger overflow
agent = manager.create_agent(
    ...,
    memory_order=2,
    long_context_order=3
)
```

### Issue: "Slow memory search"

**Cause**: Large vector database or slow embeddings

**Solutions**:
```python
# Use faster embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Faster than OpenAI
)

# Reduce vector size
redis_config = RedisConfig(
    ...,
    vector_size=384  # Smaller than 1536
)

# Increase Redis memory
redis-cli CONFIG SET maxmemory 2gb
```

### Issue: "Memory persistence not working"

**Cause**: persist_memory not enabled or config missing

**Solutions**:
```python
# Enable persistence
agent = manager.create_agent(
    ...,
    persist_memory=True,
    long_context=True,
    memory_config=redis_config
)

# Verify config
print(f"Persist memory: {agent.persist_memory}")
print(f"Memory config: {agent.memory_config}")
```

## Model Issues

### Issue: "Model not found"

**Cause**: Invalid model name

**Solutions**:
```python
# Check available models
from langchain_openai import ChatOpenAI

# Valid OpenAI models
models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

# Valid Gemini models
models = ["gemini-2.5-flash", "gemini-2.5-pro"]

# Use correct model name
agent = manager.create_agent(
    ...,
    config_dict={
        "router": {"model_name": "gpt-4o-mini"}
    }
)
```

### Issue: "Invalid temperature"

**Cause**: Temperature out of range

**Solutions**:
```python
# Temperature must be 0.0-2.0
config = {
    "router": {
        "temperature": 0.7  # Valid: 0.0-2.0
    }
}

# Not valid
config = {
    "router": {
        "temperature": 3.0  # Invalid: > 2.0
    }
}
```

### Issue: "Rate limit exceeded"

**Cause**: Too many API calls

**Solutions**:
```python
# Add delay between requests
import asyncio

for query in queries:
    response = await agent.generate_response_mas(prompt=query)
    await asyncio.sleep(1)  # Wait 1 second

# Use batch processing with delays
async def batch_with_delay(queries, delay=1):
    results = []
    for query in queries:
        result = await agent.generate_response_mas(prompt=query)
        results.append(result)
        await asyncio.sleep(delay)
    return results
```

## Tool Issues

### Issue: "Tool not found"

**Cause**: Tool not registered

**Solutions**:
```python
# Define tool
from langchain.tools import tool

@tool
def my_tool(input: str) -> str:
    """Tool description"""
    return "result"

# Register with agent
agent = manager.create_agent(
    ...,
    tools=[my_tool]
)

# Verify
print(agent.tools)
```

### Issue: "Tool execution error"

**Cause**: Tool implementation error

**Solutions**:
```python
# Add error handling
@tool
def safe_tool(input: str) -> str:
    """Tool with error handling"""
    try:
        result = do_something(input)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Test tool independently
result = my_tool("test input")
print(result)
```

## Output Issues

### Issue: "Invalid structured output"

**Cause**: Response doesn't match schema

**Solutions**:
```python
from pydantic import BaseModel, ValidationError

class Answer(BaseModel):
    text: str
    confidence: float

try:
    response = await agent.generate_response_mas(
        prompt="...",
        output_structure=Answer
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    # Adjust schema or prompt
```

### Issue: "Streaming not working"

**Cause**: Model doesn't support streaming

**Solutions**:
```python
# Use streaming-compatible model
async for chunk in agent.astream_response_mas(
    prompt="...",
    output_structure=dict
):
    print(chunk, end="", flush=True)

# Check if model supports streaming
# Most modern models support it
```

## Performance Issues

### Issue: "Slow response generation"

**Cause**: Large context or slow model

**Solutions**:
```python
# Use faster model
config = {
    "router": {"model_name": "gpt-4o-mini"}  # Faster
}

# Reduce context size
agent = manager.create_agent(
    ...,
    memory_order=3,  # Smaller
    long_context_order=5
)

# Use streaming
async for chunk in agent.astream_response_mas(...):
    print(chunk, end="", flush=True)
```

### Issue: "High memory usage"

**Cause**: Large chat history or embeddings

**Solutions**:
```python
# Reduce memory order
agent = manager.create_agent(
    ...,
    memory_order=3,
    long_context_order=5
)

# Use smaller embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # 384 dims
)

# Clear old memories
await agent.long_term_memory.delete(user_id, doc_id)
```

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("masai")

# Now you'll see detailed logs
```

### Check Agent State

```python
print(f"Agent: {agent.agent_name}")
print(f"Chat history: {len(agent.chat_history)}")
print(f"Context summaries: {len(agent.context_summaries)}")
print(f"Memory config: {agent.memory_config}")
```

### Test Components Individually

```python
# Test LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello")

# Test embeddings
embeddings = OpenAIEmbeddings()
embedding = embeddings.embed_query("test")

# Test Redis
import redis
r = redis.Redis()
r.ping()
```

## Getting Help

- 📖 [Documentation](../docs/)
- 🐛 [Issues](https://github.com/shaunthecomputerscientist/mas-ai/issues)
- 💬 [Discussions](https://github.com/shaunthecomputerscientist/mas-ai/discussions)

