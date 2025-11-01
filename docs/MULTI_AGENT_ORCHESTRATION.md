# Multi-Agent Orchestration Guide

## Overview

MASAI provides powerful multi-agent orchestration capabilities for building complex AI systems. This guide covers all orchestration patterns, data management, and context sharing strategies.

---

## Table of Contents

1. [Multi-Agent System (MAS)](#multi-agent-system)
2. [Orchestration Patterns](#orchestration-patterns)
3. [Data and Context Management](#data-and-context-management)
4. [Orchestrated Multi-Agent Network (OMAN)](#orchestrated-multi-agent-network)
5. [Best Practices](#best-practices)

---

## Multi-Agent System

### What is MAS?

A Multi-Agent System (MAS) coordinates multiple agents to work together. Each agent has specialized capabilities and can communicate with other agents.

### Creating a MAS

```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem
from masai.AgentManager import AgentManager, AgentDetails

# Create agent manager
manager = AgentManager(user_id="user_123")

# Create specialized agents
researcher = manager.create_agent(
    agent_name="researcher",
    tools=[search_tool, web_scraper],
    agent_details=AgentDetails(
        capabilities=["research", "information_gathering"],
        description="Researches topics and gathers information"
    )
)

writer = manager.create_agent(
    agent_name="writer",
    tools=[document_tool, formatting_tool],
    agent_details=AgentDetails(
        capabilities=["writing", "content_creation"],
        description="Writes and formats content"
    )
)

editor = manager.create_agent(
    agent_name="editor",
    tools=[review_tool, grammar_tool],
    agent_details=AgentDetails(
        capabilities=["editing", "quality_assurance"],
        description="Reviews and edits content"
    )
)

# Create MAS
mas = MultiAgentSystem(agentManager=manager)
```

---

## Orchestration Patterns

### 1. Sequential Workflow

Agents execute in a fixed order, each receiving output from the previous agent.

**Use Case**: Document processing pipeline (research → write → edit)

```python
result = await mas.initiate_sequential_mas(
    query="Write an article about AI",
    agent_sequence=["researcher", "writer", "editor"],
    memory_order=3  # Shared memory between agents
)

print(result["answer"])
```

**Data Flow**:
```
Query
  ↓
Researcher (research_output)
  ↓
Writer (research_output + written_content)
  ↓
Editor (all_previous_outputs + final_review)
  ↓
Final Result
```

**Key Features**:
- Fixed execution order
- Each agent sees previous outputs
- Shared memory across agents
- Deterministic workflow

---

### 2. Hierarchical Workflow

A supervisor agent delegates tasks to specialized agents based on the query.

**Use Case**: Complex problem solving with task decomposition

```python
# Create supervisor
supervisor = manager.create_agent(
    agent_name="supervisor",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["task_delegation", "coordination"],
        description="Coordinates agent tasks"
    )
)

# Create MAS with hierarchical mode
mas = MultiAgentSystem(
    agentManager=manager,
    mode="hierarchical",
    supervisor_agent=supervisor
)

result = await mas.initiate_hierarchical_mas(
    query="Analyze sales data and create a report"
)

print(result["answer"])
```

**Data Flow**:
```
Query
  ↓
Supervisor (analyzes query)
  ↓
├─ Analyst Agent (data analysis)
├─ Writer Agent (report creation)
└─ Reviewer Agent (quality check)
  ↓
Supervisor (aggregates results)
  ↓
Final Result
```

**Key Features**:
- Dynamic task delegation
- Supervisor makes routing decisions
- Parallel task execution possible
- Flexible workflow

---

### 3. Decentralized Workflow

Agents collaborate peer-to-peer, with control passing between agents dynamically.

**Use Case**: Collaborative problem solving, multi-perspective analysis

```python
# Create MAS with decentralized mode
mas = MultiAgentSystem(
    agentManager=manager,
    mode="decentralized"
)

# Start with entry agent
entry_agent = manager.get_agent("researcher")

result = await mas.initiate_decentralized_mas(
    query="Investigate and report on market trends",
    set_entry_agent=entry_agent,
    memory_order=3
)

print(result["answer"])
```

**Data Flow**:
```
Query
  ↓
Agent A (processes, decides next agent)
  ↓
Agent B (processes, decides next agent)
  ↓
Agent C (processes, decides next agent)
  ↓
Agent A (final processing)
  ↓
Final Result
```

**Key Features**:
- Dynamic agent selection
- Agents decide next agent
- Peer-to-peer collaboration
- Flexible routing

---

## Data and Context Management

### Shared Memory

All agents in a MAS share memory through `memory_order` parameter.

```python
# Configure shared memory
mas = MultiAgentSystem(
    agentManager=manager,
    mode="sequential"
)

result = await mas.initiate_sequential_mas(
    query="Process document",
    agent_sequence=["analyzer", "processor", "finalizer"],
    memory_order=5  # Keep last 5 messages in shared memory
)
```

**Memory Structure**:
```
Shared Memory (memory_order=5)
├─ Message 1 (oldest)
├─ Message 2
├─ Message 3
├─ Message 4
└─ Message 5 (newest)

Each agent can access all messages
```

### Context Propagation

Pass context between agents using `set_context()`:

```python
# Set context for all agents
context = {
    "user_id": "user_123",
    "project": "market_analysis",
    "deadline": "2024-12-31",
    "budget": 10000
}

mas.agentManager.set_context(context)

# Or set context for specific agent
researcher = manager.get_agent("researcher")
researcher.set_context(context, mode="set")

# Update context
researcher.set_context({"status": "in_progress"}, mode="update")
```

### Context Isolation

Isolate context between different MAS instances:

```python
# MAS 1 - Finance team
finance_manager = AgentManager(user_id="finance_team")
finance_mas = MultiAgentSystem(agentManager=finance_manager)

# MAS 2 - Research team
research_manager = AgentManager(user_id="research_team")
research_mas = MultiAgentSystem(agentManager=research_manager)

# Each MAS has isolated context and memory
```

---

## Orchestrated Multi-Agent Network (OMAN)

### What is OMAN?

OMAN coordinates multiple MAS instances, enabling large-scale multi-agent systems.

### Creating OMAN

```python
from masai.OMAN.oman import OrchestratedMultiAgentNetwork

# Create multiple MAS instances
finance_mas = MultiAgentSystem(agentManager=finance_manager)
research_mas = MultiAgentSystem(agentManager=research_manager)
support_mas = MultiAgentSystem(agentManager=support_manager)

# Create OMAN
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[finance_mas, research_mas, support_mas],
    network_memory_order=5,
    oman_llm_config={
        "model_name": "gpt-4",
        "category": "openai",
        "temperature": 0.2,
        "memory_order": 5
    },
    extra_context={
        "environment": "production",
        "company": "MyCompany"
    }
)
```

### OMAN Data Flow

```
Query
  ↓
OMAN Router (analyzes query)
  ↓
├─ Finance MAS (financial queries)
├─ Research MAS (research queries)
└─ Support MAS (support queries)
  ↓
OMAN Aggregator (combines results)
  ↓
Final Result
```

### Delegating Tasks in OMAN

```python
# Delegate task to appropriate network
result = oman.delegate_task(
    query="What is our Q3 revenue and market analysis?"
)

print(result)
```

### Shared Memory in OMAN

```python
# Access OMAN shared memory
shared_memory = oman.shared_memory

print(shared_memory["tasks"])      # Completed tasks
print(shared_memory["outcomes"])   # Task outcomes
print(shared_memory["capabilities"])  # Network capabilities
```

---

## Best Practices

### 1. Choose Right Pattern

| Pattern | Use Case | Complexity |
|---------|----------|-----------|
| Sequential | Fixed pipeline | Low |
| Hierarchical | Task decomposition | Medium |
| Decentralized | Collaboration | Medium |
| OMAN | Large-scale systems | High |

### 2. Memory Management

```python
# Optimal memory_order values
memory_order = 3-5  # For sequential workflows
memory_order = 5-10  # For hierarchical workflows
memory_order = 3-5  # For decentralized workflows
```

### 3. Context Sharing

```python
# Share context efficiently
context = {
    "user_id": "user_123",
    "session_id": "session_456",
    "request_id": "req_789"
}

mas.agentManager.set_context(context)
```

### 4. Error Handling

```python
try:
    result = await mas.initiate_sequential_mas(
        query="Process data",
        agent_sequence=["analyzer", "processor"]
    )
except Exception as e:
    print(f"MAS execution failed: {e}")
    # Implement fallback logic
```

---

## See Also

- [ADVANCED.md](ADVANCED.md) - Advanced patterns and customization
- [EXAMPLES.md](EXAMPLES.md) - Real-world multi-agent examples
- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) - Memory management details
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture

