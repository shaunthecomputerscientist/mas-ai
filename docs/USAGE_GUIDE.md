# Usage Guide

Comprehensive guide to using MASAI for various tasks.

## Basic Agent Usage

### Create and Use Agent


```python
import asyncio
from masai.AgentManager import AgentManager, AgentDetails

async def main():
    # Create manager
    manager = AgentManager(user_id="user_123")

    # Create agent
    agent = manager.create_agent(
        agent_name="assistant",
        tools=[],  # Add LangChain tools here
        agent_details=AgentDetails(
            capabilities=["analysis", "reasoning"],
            description="Helpful assistant",
            style="concise"
        )
    )

    # Execute agent - full execution
    result = await agent.initiate_agent(
        query="What is machine learning?",
        passed_from="user"
    )

    print(result["answer"])

asyncio.run(main())
```

## Structured Output

### Using Pydantic Models

```python
from pydantic import BaseModel
from masai.AgentManager import AgentManager, AgentDetails

class Article(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    word_count: int

# Create agent with structured output
agent = manager.create_agent(
    agent_name="writer",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["writing", "analysis"],
        description="Article writer"
    ),
    AnswerFormat=Article  # Specify output structure
)

result = await agent.initiate_agent(
    query="Write an article about AI",
    passed_from="user"
)

# Result contains structured data
print(f"Title: {result['answer'].title}")
print(f"Summary: {result['answer'].summary}")
print(f"Key Points: {result['answer'].key_points}")
```

### Nested Structures

```python
class Author(BaseModel):
    name: str
    expertise: str

class BlogPost(BaseModel):
    title: str
    author: Author
    content: str
    tags: list[str]

# Create agent with nested structure
agent = manager.create_agent(
    agent_name="blogger",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["writing"],
        description="Blog writer"
    ),
    AnswerFormat=BlogPost
)

result = await agent.initiate_agent(
    query="Create a blog post about Python",
    passed_from="user"
)

print(f"Author: {result['answer'].author.name}")
print(f"Expertise: {result['answer'].author.expertise}")
```

## Streaming Responses

### Stream Agent Execution

```python
# Stream agent execution with real-time state updates
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
    reasoning = state_value.get("reasoning")

    # Print updates
    if tool:
        print(f"[{current_node}] Executing: {tool}")
    if answer:
        print(f"[{current_node}] Answer: {answer}")
    if reasoning:
        print(f"[{current_node}] Reasoning: {reasoning}")
```

### Stream with Structured Output

```python
class Story(BaseModel):
    title: str
    content: str

async for state in agent.initiate_agent_astream(
    query="Tell me a story",
    passed_from="user"
):
    node_name, state_dict = state
    state_value = [v for k, v in state_dict.items()][0]
    if state_value.get("answer"):
        print(state_value["answer"], end="", flush=True)
```

## Tool Integration

### Define Tools

```python
from masai.Tools.Tool import tool

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    # Implement web search
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    # Implement weather API call
    return f"Weather in {location}: ..."
```

### Use Tools in Agent

```python
agent = manager.create_agent(
    agent_name="assistant",
    tools=[calculate, search_web, get_weather],
    agent_details=agent_details
)

result = await agent.initiate_agent(
    query="What is 2+2? Also search for AI news",
    passed_from="user"
)

print(result["answer"])
```

## Multi-Agent Systems

### Create Multiple Agents

```python
researcher = manager.create_agent(
    agent_name="researcher",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["research", "analysis"],
        description="Research specialist"
    )
)

writer = manager.create_agent(
    agent_name="writer",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["writing", "editing"],
        description="Writing specialist"
    )
)

editor = manager.create_agent(
    agent_name="editor",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["editing", "review"],
        description="Editor"
    )
)
```

### Sequential Workflow

```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem

mas = MultiAgentSystem(agentManager=manager)

result = await mas.initiate_sequential_mas(
    query="Research and write an article about AI",
    agent_sequence=["researcher", "writer", "editor"],
    memory_order=3
)

print(result)
```

### Hierarchical Workflow

```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig

supervisor_config = SupervisorConfig(
    model_name="gpt-4",
    temperature=0.7,
    model_category="openai",
    memory_order=20,
    memory=True
)

mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config
)

result = await mas.initiate_hierarchical_mas(
    query="Research and write an article about AI"
)

if result["status"] == "completed":
    print(result["answer"])
```

### Decentralized Workflow

```python
mas = MultiAgentSystem(agentManager=manager)

result = await mas.initiate_decentralized_mas(
    query="Investigate market trends",
    set_entry_agent=manager.get_agent("researcher"),
    memory_order=3
)

print(result["answer"])
```

## Memory Management

### Enable Persistent Memory

```python
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

# Configure Redis backend
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
)

# Set memory config in manager
manager.memory_config = redis_config

# Create agent with persistent memory
agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=agent_details,
    persist_memory=True,  # Enable persistent memory
    long_context=True,    # Enable long-context summarization
    memory_order=5,       # Short-term memory size
    long_context_order=10 # Summaries before flush
)
```

### Access Long-Term Memory

```python
# Access memory through agent's LLM router
long_term_memory = agent.llm_router.long_term_memory

# Save memories
from langchain_core.documents import Document

memories = [
    Document(
        page_content="User likes Python",
        metadata={"category": "preferences"}
    ),
    Document(
        page_content="User works in AI",
        metadata={"category": "profession"}
    )
]

await long_term_memory.save(
    user_id="user_123",
    documents=memories
)

# Search memories
results = await long_term_memory.search(
    user_id="user_123",
    query="What does user like?",
    k=5,
    categories=["preferences"]
)

for doc in results:
    print(doc.page_content)
```

## Advanced Patterns

### Conditional Tool Use

```python
result = await agent.initiate_agent(
    query="Calculate 2+2 if needed, otherwise just say hello",
    passed_from="user"
)
print(result["answer"])
```

### Multi-Turn Conversation

```python
messages = [
    "Hello, what's your name?",
    "Tell me about yourself",
    "What can you help me with?"
]

for message in messages:
    result = await agent.initiate_agent(
        query=message,
        passed_from="user"
    )
    print(f"User: {message}")
    print(f"Agent: {result['answer']}\n")
```

### Batch Processing

```python
queries = [
    "What is AI?",
    "What is ML?",
    "What is DL?"
]

responses = []
for query in queries:
    result = await agent.initiate_agent(
        query=query,
        passed_from="user"
    )
    responses.append(result["answer"])

for query, response in zip(queries, responses):
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

### Error Handling

```python
try:
    result = await agent.initiate_agent(
        query="Your query",
        passed_from="user"
    )
    print(result["answer"])
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Optimization

### Use Streaming for Long Responses

```python
# Full execution - waits for complete response
result = await agent.initiate_agent(
    query="Write a long article",
    passed_from="user"
)
print(result["answer"])

# Streaming - get real-time updates
async for state in agent.initiate_agent_astream(
    query="Write a long article",
    passed_from="user"
):
    node_name, state_dict = state
    state_value = [v for k, v in state_dict.items()][0]
    if state_value.get("answer"):
        print(state_value["answer"], end="", flush=True)
```

### Batch Similar Queries

```python
import asyncio

# Instead of individual queries
for query in queries:
    result = await agent.initiate_agent(query=query, passed_from="user")

# Use batch processing
responses = await asyncio.gather(*[
    agent.initiate_agent(query=q, passed_from="user")
    for q in queries
])
```

### Optimize Memory Settings

```python
# For short conversations
agent = manager.create_agent(
    ...,
    memory_order=3,
    long_context_order=5
)

# For long conversations
agent = manager.create_agent(
    ...,
    memory_order=20,
    long_context_order=50
)
```

## Debugging

### Enable Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("masai")

manager = AgentManager(
    user_id="user_123",
    logging=True
)
```

### Check Agent State

```python
print(f"Agent name: {agent.agent_name}")
# Access memory state through LLM components
print(f"Chat history: {len(agent.llm_router.chat_history)}")
print(f"Context summaries: {len(agent.llm_router.context_summaries)}")
print(f"Persist memory: {agent.llm_router.persist_memory}")
print(f"Long context: {agent.llm_router.long_context}")
```

### Monitor Memory

```python
print(f"Memory order: {agent.memory_order}")
print(f"Long context order: {agent.long_context_order}")
print(f"Persist memory: {agent.persist_memory}")
print(f"Long context: {agent.long_context}")
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

