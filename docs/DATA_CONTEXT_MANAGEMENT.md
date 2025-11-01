# Data and Context Management Guide

## Overview

This guide covers how MASAI manages data flow, context sharing, and information isolation across agents and multi-agent systems.

---

## Table of Contents

1. [Context Management](#context-management)
2. [Data Flow Patterns](#data-flow-patterns)
3. [Memory Hierarchy](#memory-hierarchy)
4. [Context Isolation](#context-isolation)
5. [Cross-Agent Communication](#cross-agent-communication)
6. [Best Practices](#best-practices)

---

## Context Management

### What is Context?

Context is metadata and information shared across agents. It includes:
- User information (user_id, session_id)
- Request information (request_id, timestamp)
- Domain information (project, environment)
- Custom data (business rules, constraints)

### Setting Context

#### Global Context (All Agents)

```python
from masai.AgentManager import AgentManager

manager = AgentManager(user_id="user_123")

# Set context for all agents
global_context = {
    "user_id": "user_123",
    "session_id": "session_456",
    "environment": "production",
    "company": "TechCorp",
    "department": "Engineering"
}

manager.set_context(global_context)
```

#### Agent-Specific Context

```python
# Get specific agent
researcher = manager.get_agent("researcher")

# Set context for this agent only
agent_context = {
    "research_domain": "AI",
    "max_sources": 10,
    "language": "English"
}

researcher.set_context(agent_context, mode="set")
```

#### Context Modes

```python
# Mode 1: SET (replace entire context)
agent.set_context(new_context, mode="set")

# Mode 2: UPDATE (merge with existing context)
agent.set_context(additional_context, mode="update")
```

### Accessing Context

```python
# Access context from agent
context = agent.llm_router.info  # Context stored in LLM component

print(context["user_id"])
print(context["environment"])
```

---

## Data Flow Patterns

### 1. Sequential Data Flow

Data flows from one agent to the next in a pipeline.

```python
# Sequential workflow
result = await mas.initiate_sequential_mas(
    query="Analyze → Process → Report",
    agent_sequence=["analyzer", "processor", "reporter"],
    memory_order=3
)

# Data flow:
# Query → Analyzer (output_1)
#       → Processor (output_1 + output_2)
#       → Reporter (output_1 + output_2 + output_3)
```

**Data Structure**:
```python
# Each agent receives:
{
    "messages": [
        {"role": "user", "content": "original_query"},
        {"role": "assistant", "content": "analyzer_output"},
        {"role": "assistant", "content": "processor_output"}
    ],
    "context": {
        "user_id": "user_123",
        "current_agent": "reporter"
    }
}
```

### 2. Hierarchical Data Flow

Supervisor routes data to appropriate agents.

```python
# Hierarchical workflow
result = await mas.initiate_hierarchical_mas(
    query="Complex analysis needed"
)

# Data flow:
# Query → Supervisor (analyzes)
#       ├─ Agent_1 (processes)
#       ├─ Agent_2 (processes)
#       └─ Agent_3 (processes)
#       → Supervisor (aggregates)
```

### 3. Decentralized Data Flow

Agents pass data peer-to-peer.

```python
# Decentralized workflow
result = await mas.initiate_decentralized_mas(
    query="Collaborative analysis",
    set_entry_agent=entry_agent,
    memory_order=3
)

# Data flow:
# Query → Agent_A (decides next)
#       → Agent_B (decides next)
#       → Agent_C (decides next)
#       → Agent_A (final)
```

---

## Memory Hierarchy

### Level 1: Short-Term Memory (chat_history)

Stores recent messages in each agent.

```python
# Configuration
agent = manager.create_agent(
    agent_name="analyzer",
    tools=[...],
    agent_details=...,
    memory_order=10  # Keep last 10 messages
)

# Access short-term memory
short_term = agent.llm_router.chat_history
print(f"Messages in memory: {len(short_term)}")
```

### Level 2: Context Summaries (context_summaries)

Stores summarized conversations.

```python
# Configuration
agent = manager.create_agent(
    agent_name="analyzer",
    tools=[...],
    agent_details=...,
    long_context=True,
    long_context_order=20  # Keep 20 summaries
)

# Access context summaries
summaries = agent.llm_router.context_summaries
print(f"Summaries: {len(summaries)}")
```

### Level 3: Long-Term Memory (persistent)

Stores memories in Redis/Qdrant.

```python
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

# Configuration
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings()
)

manager = AgentManager(
    user_id="user_123",
    memory_config=redis_config
)

agent = manager.create_agent(
    agent_name="analyzer",
    tools=[...],
    agent_details=...,
    persist_memory=True
)

# Access long-term memory
results = await agent.llm_router.long_term_memory.search(
    query="previous analysis",
    k=5,
    user_id="user_123"
)
```

### Memory Flow

```
New Message
    ↓
chat_history (memory_order=10)
    ↓
When len(chat_history) > memory_order:
    ↓
Summarize → context_summaries (long_context_order=20)
    ↓
When len(context_summaries) > long_context_order:
    ↓
Flush → Long-Term Memory (Redis/Qdrant)
    ↓
Keep only last summary in context_summaries
```

---

## Context Isolation

### User Isolation

```python
# Each user has isolated context
user1_manager = AgentManager(user_id="user_1")
user2_manager = AgentManager(user_id="user_2")

# Create agents for each user
user1_agent = user1_manager.create_agent(...)
user2_agent = user2_manager.create_agent(...)

# Memories are isolated by user_id
# user1_agent cannot access user2_agent's memories
```

### Session Isolation

```python
# Different sessions for same user
session1_context = {
    "user_id": "user_123",
    "session_id": "session_1",
    "project": "ProjectA"
}

session2_context = {
    "user_id": "user_123",
    "session_id": "session_2",
    "project": "ProjectB"
}

# Set context for each session
agent.set_context(session1_context, mode="set")
# ... process session 1 ...

agent.set_context(session2_context, mode="set")
# ... process session 2 ...
```

### MAS Isolation

```python
# Each MAS has isolated memory
mas1 = MultiAgentSystem(agentManager=manager1)
mas2 = MultiAgentSystem(agentManager=manager2)

# Memories don't cross between MAS instances
```

---

## Cross-Agent Communication

### Shared Memory Pattern

```python
# All agents in MAS share memory
result = await mas.initiate_sequential_mas(
    query="Process data",
    agent_sequence=["analyzer", "processor", "reporter"],
    memory_order=5  # Shared memory size
)

# Each agent can access previous agent's output
# through shared memory
```

### Context Passing Pattern

```python
# Pass context through agents
context = {
    "analysis_type": "financial",
    "fiscal_year": 2024,
    "currency": "USD"
}

# Set context before MAS execution
mas.agentManager.set_context(context)

# All agents access this context
result = await mas.initiate_sequential_mas(
    query="Analyze financial data",
    agent_sequence=["analyzer", "processor"]
)
```

### Delegation Pattern

```python
# Hierarchical delegation with context
result = await mas.initiate_hierarchical_mas(
    query="Complex task"
)

# Supervisor passes context to delegated agents
# Each agent receives:
# - Original query
# - Supervisor's analysis
# - Shared context
```

---

## Best Practices

### 1. Context Design

```python
# Good: Minimal, focused context
context = {
    "user_id": "user_123",
    "request_id": "req_456",
    "domain": "finance"
}

# Bad: Too much context
context = {
    "user_id": "user_123",
    "request_id": "req_456",
    "domain": "finance",
    "all_user_data": {...},  # Too much
    "entire_database": {...}  # Too much
}
```

### 2. Memory Configuration

```python
# Optimal settings
memory_order = 5-10              # Short-term
long_context_order = 20-30       # Summaries
shared_memory_order = 3-5        # Shared between components
retain_messages_order = 10-20    # Retained state
```

### 3. Data Validation

```python
# Validate context before setting
def validate_context(context):
    required_fields = ["user_id", "request_id"]
    for field in required_fields:
        if field not in context:
            raise ValueError(f"Missing required field: {field}")
    return True

if validate_context(context):
    agent.set_context(context)
```

### 4. Error Handling

```python
try:
    result = await mas.initiate_sequential_mas(
        query="Process data",
        agent_sequence=["analyzer", "processor"]
    )
except Exception as e:
    print(f"Data flow error: {e}")
    # Implement recovery logic
```

---

## See Also

- [MULTI_AGENT_ORCHESTRATION.md](MULTI_AGENT_ORCHESTRATION.md) - Orchestration patterns
- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) - Memory management
- [ADVANCED.md](ADVANCED.md) - Advanced patterns
- [EXAMPLES.md](EXAMPLES.md) - Real-world examples

