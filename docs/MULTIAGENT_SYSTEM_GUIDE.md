# Multi-Agent System Guide

Complete guide for using MASAI's Multi-Agent System (MAS) for coordinating multiple agents.

---

## Table of Contents

1. [Introduction](#introduction)
2. [MAS Modes](#mas-modes)
3. [Decentralized MAS](#decentralized-mas)
4. [Hierarchical MAS](#hierarchical-mas)
5. [SupervisorConfig](#supervisorconfig)
6. [State Management](#state-management)
7. [Streaming Support](#streaming-support)
8. [Complete Examples](#complete-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

The Multi-Agent System (MAS) in MASAI enables coordination between multiple agents to solve complex tasks. It supports two modes:

1. **Decentralized MAS**: Peer-to-peer agent collaboration with dynamic delegation
2. **Hierarchical MAS**: Supervisor-based task management with concurrent execution

### When to Use MAS

Use MAS when:
- ✅ Tasks require multiple specialized agents
- ✅ Agents need to delegate work to each other
- ✅ Complex workflows require coordination
- ✅ Parallel task execution is beneficial

Use single agent when:
- ❌ Task is simple and doesn't require specialization
- ❌ No delegation or coordination needed
- ❌ Single agent has all required capabilities

---

## MAS Modes

### Decentralized MAS

**Characteristics**:
- Peer-to-peer agent collaboration
- Dynamic delegation between agents
- Agents decide which agent to delegate to
- State maintained across delegations
- Entry agent can be specified

**Use Cases**:
- Research workflows (researcher → writer → reviewer)
- Customer support (triage → specialist → resolution)
- Content creation (planner → writer → editor)

### Hierarchical MAS

**Characteristics**:
- Supervisor assigns tasks to agents
- Concurrent task execution
- Task queue management
- Automatic agent selection by supervisor
- Result callbacks for completed tasks

**Use Cases**:
- Parallel data processing
- Multi-step workflows with dependencies
- High-throughput task processing
- Complex orchestration scenarios

---

## Decentralized MAS

### Basic Setup

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem

# Create AgentManager
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json",
    logging=True
)

# Create multiple agents
researcher = manager.create_agent(
    agent_name="researcher",
    tools=[search_tool, analysis_tool],
    agent_details=AgentDetails(
        capabilities=["research", "analysis", "data gathering"],
        description="Specializes in research and data analysis",
        style="thorough and analytical"
    )
)

writer = manager.create_agent(
    agent_name="writer",
    tools=[document_tool],
    agent_details=AgentDetails(
        capabilities=["writing", "content creation", "summarization"],
        description="Specializes in content creation and writing",
        style="clear and engaging"
    )
)

# Create decentralized MAS
mas = MultiAgentSystem(agentManager=manager)
```

### Executing Decentralized MAS

```python
import asyncio

async def run_decentralized():
    # Execute with entry agent
    result = await mas.initiate_decentralized_mas(
        query="Research AI trends and write a summary",
        set_entry_agent=researcher,  # Entry point
        memory_order=3,  # Keep last 3 messages from each agent
        passed_from="user"
    )
    
    print(result["answer"])
    print(f"Last agent: {result['last_agent']}")
    print(f"Reasoning: {result['reasoning']}")

asyncio.run(run_decentralized())
```

### How Decentralized MAS Works

1. **Entry**: Query sent to entry agent (e.g., researcher)
2. **Processing**: Entry agent processes query
3. **Delegation**: Agent decides to delegate to another agent (e.g., writer)
4. **Continuation**: Delegated agent processes and may delegate further
5. **Completion**: Final agent returns answer
6. **State**: System remembers last agent for next query

### State Persistence

The MAS maintains state across queries:

```python
# First query
result1 = await mas.initiate_decentralized_mas(
    query="Research AI trends",
    set_entry_agent=researcher
)
# Last agent: researcher

# Second query (continues from researcher)
result2 = await mas.initiate_decentralized_mas(
    query="Now write a summary",
    set_entry_agent=researcher  # Ignored, uses last agent
)
# Last agent: writer (if researcher delegated to writer)
```

### Memory Order

The `memory_order` parameter controls context sharing:

```python
result = await mas.initiate_decentralized_mas(
    query="Complex multi-step task",
    set_entry_agent=researcher,
    memory_order=5  # Pass last 5 messages to next agent
)
```

**Guidelines**:
- `memory_order=1`: Minimal context (fast, less coherent)
- `memory_order=3`: Balanced (recommended)
- `memory_order=5+`: Rich context (slower, more coherent)

---

## Hierarchical MAS

### Basic Setup

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig

# Create AgentManager
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json",
    logging=True
)

# Create multiple agents
agent1 = manager.create_agent(
    agent_name="data_processor",
    tools=[data_tools],
    agent_details=AgentDetails(
        capabilities=["data processing", "analysis"],
        description="Processes and analyzes data",
        style="efficient and accurate"
    )
)

agent2 = manager.create_agent(
    agent_name="report_generator",
    tools=[report_tools],
    agent_details=AgentDetails(
        capabilities=["report generation", "visualization"],
        description="Generates reports and visualizations",
        style="clear and informative"
    )
)

# Create supervisor configuration
supervisor_config = SupervisorConfig(
    model_name="gpt-4o",
    temperature=0.7,
    model_category="openai",
    memory_order=20,
    memory=True,
    extra_context={"user_name": "John"},
    supervisor_system_prompt=None  # Use default
)

# Create hierarchical MAS
mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    heirarchical_mas_result_callback=handle_result,  # Optional callback
    agent_return_direct=True  # Skip supervisor evaluation
)
```

### Executing Hierarchical MAS

```python
async def run_hierarchical():
    result = await mas.initiate_hierarchical_mas(
        query="Process data and generate report"
    )
    
    if result["status"] == "completed":
        print(result["answer"])
    elif result["status"] == "queued":
        print(f"Task {result['task_id']} queued")
    elif result["status"] == "failed":
        print(f"Task failed: {result['error']}")

asyncio.run(run_hierarchical())
```

### Result Callback

Define a callback to handle completed tasks:

```python
def handle_task_result(task):
    """Callback for completed tasks."""
    print(f"Task completed: {task['task_id']}")
    print(f"Answer: {task['answer']}")
    print(f"Agent: {task['agent']}")
    
    # Save to file
    with open('results.txt', 'a') as f:
        f.write(f"{task}\n")

# Use callback in MAS
mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    heirarchical_mas_result_callback=handle_task_result
)
```

### How Hierarchical MAS Works

1. **Task Submission**: Query submitted to supervisor
2. **Agent Selection**: Supervisor selects appropriate agent
3. **Task Assignment**: Task assigned to agent
4. **Execution**: Agent executes task (possibly concurrently)
5. **Completion**: Result returned via callback
6. **Evaluation**: Supervisor evaluates result (if `agent_return_direct=False`)

### Concurrent Task Execution

Hierarchical MAS supports concurrent execution:

```python
async def run_multiple_tasks():
    tasks = [
        "Process dataset A",
        "Process dataset B",
        "Process dataset C"
    ]
    
    results = []
    for task in tasks:
        result = await mas.initiate_hierarchical_mas(task)
        results.append(result)
    
    # Tasks execute concurrently
    return results
```

---

## SupervisorConfig

### Configuration Parameters

```python
supervisor_config = SupervisorConfig(
    model_name="gpt-4o",           # Supervisor model
    temperature=0.7,                # Sampling temperature
    model_category="openai",        # Model provider
    memory_order=20,                # Context window size
    memory=True,                    # Enable memory
    extra_context={"key": "value"}, # Additional context
    supervisor_system_prompt=None   # Custom prompt (optional)
)
```

### Parameter Details

#### `model_name`
- **Type**: `str`
- **Description**: Model for supervisor
- **Examples**: `"gpt-4o"`, `"gemini-2.5-flash"`, `"claude-3.5-sonnet"`

#### `temperature`
- **Type**: `float`
- **Range**: `0.0` - `2.0`
- **Description**: Controls supervisor's decision randomness
- **Recommended**: `0.7` for balanced decisions

#### `model_category`
- **Type**: `str`
- **Options**: `"openai"`, `"gemini"`, `"anthropic"`
- **Description**: Model provider category

#### `memory_order`
- **Type**: `int`
- **Description**: Number of past interactions to keep
- **Recommended**: `20` for good context

#### `memory`
- **Type**: `bool`
- **Description**: Enable/disable memory tracking
- **Recommended**: `True`

#### `extra_context`
- **Type**: `dict`
- **Description**: Additional context for supervisor
- **Example**: `{"user_name": "John", "preferences": "detailed"}`

#### `supervisor_system_prompt`
- **Type**: `Optional[str]`
- **Description**: Custom system prompt for supervisor
- **Default**: Uses built-in prompt if `None`

### Custom Supervisor Prompt

```python
custom_prompt = """
You are a task coordinator for a team of specialized agents.
Your role is to:
1. Analyze incoming tasks
2. Select the most appropriate agent
3. Provide clear instructions
4. Evaluate results

Available agents: {agents}
User context: {context}
"""

supervisor_config = SupervisorConfig(
    model_name="gpt-4o",
    temperature=0.7,
    model_category="openai",
    memory_order=20,
    memory=True,
    extra_context={"user_name": "John"},
    supervisor_system_prompt=custom_prompt
)
```

---

## State Management

### Decentralized MAS State

The MAS maintains state with:
- `last_agent`: Last agent that processed query
- `last_agent_answers`: History of agent answers
- `last_agent_input`: Last input to agent
- `agent_reasoning`: Reasoning from last agent

### Accessing State

```python
# After execution
result = await mas.initiate_decentralized_mas(
    query="Task",
    set_entry_agent=agent
)

# Access state
print(mas.state['last_agent'])  # Last agent name
print(mas.state['last_agent_answers'])  # Answer history
print(mas.state['agent_reasoning'])  # Reasoning
```

### Resetting State

```python
# Reset state manually
mas.state = {
    'last_agent_answers': [],
    'last_agent': None,
    'last_agent_input': None,
    'agent_reasoning': None,
    'pending_tasks': {}
}
```

---

## Streaming Support

### Decentralized MAS Streaming

```python
async def run_streaming():
    async for state in mas.initiate_decentralized_mas_astream(
        query="Complex task",
        set_entry_agent=researcher,
        memory_order=3
    ):
        # state is a tuple: (node_name, state_dict)
        node_name, state_dict = state
        
        # Extract state value
        state_value = list(state_dict.values())[0]
        
        print(f"Node: {state_value['current_node']}")
        print(f"Agent: {state_value.get('current_agent', 'N/A')}")
        
        # Check if final answer
        if 'answer' in state_value:
            print(f"Answer: {state_value['answer']}")

asyncio.run(run_streaming())
```

### Benefits of Streaming

- ✅ Real-time progress updates
- ✅ Better user experience
- ✅ Early error detection
- ✅ Intermediate results visibility

---

## Complete Examples

### Example 1: Research Workflow (Decentralized)

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem
from masai.Tools.Tool import tool
import asyncio

# Define tools
@tool("search")
async def search(query: str) -> str:
    """Searches for information."""
    return f"Search results for: {query}"

@tool("write_document")
def write_document(content: str, title: str) -> str:
    """Writes a document."""
    with open(f"{title}.txt", 'w') as f:
        f.write(content)
    return f"Document '{title}' created"

# Create manager
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json"
)

# Create agents
researcher = manager.create_agent(
    agent_name="researcher",
    tools=[search],
    agent_details=AgentDetails(
        capabilities=["research", "analysis"],
        description="Researches topics thoroughly",
        style="analytical"
    )
)

writer = manager.create_agent(
    agent_name="writer",
    tools=[write_document],
    agent_details=AgentDetails(
        capabilities=["writing", "documentation"],
        description="Creates well-written documents",
        style="clear and engaging"
    )
)

# Create MAS
mas = MultiAgentSystem(agentManager=manager)

# Execute workflow
async def research_workflow():
    result = await mas.initiate_decentralized_mas(
        query="Research AI trends in 2024 and create a summary document",
        set_entry_agent=researcher,
        memory_order=3
    )
    print(result["answer"])

asyncio.run(research_workflow())
```

### Example 2: Data Processing Pipeline (Hierarchical)

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig
from masai.Tools.Tool import tool
import asyncio

# Define tools
@tool("process_data")
async def process_data(dataset: str) -> str:
    """Processes a dataset."""
    # Simulated processing
    return f"Processed {dataset}: 1000 records"

@tool("generate_report")
def generate_report(data: str, format: str = "pdf") -> str:
    """Generates a report."""
    return f"Report generated in {format} format"

# Create manager
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json"
)

# Create agents
processor = manager.create_agent(
    agent_name="data_processor",
    tools=[process_data],
    agent_details=AgentDetails(
        capabilities=["data processing", "analysis"],
        description="Processes large datasets efficiently",
        style="fast and accurate"
    )
)

reporter = manager.create_agent(
    agent_name="report_generator",
    tools=[generate_report],
    agent_details=AgentDetails(
        capabilities=["report generation", "visualization"],
        description="Creates comprehensive reports",
        style="detailed and visual"
    )
)

# Supervisor config
supervisor_config = SupervisorConfig(
    model_name="gemini-2.5-flash",
    temperature=0.7,
    model_category="gemini",
    memory_order=20,
    memory=True,
    extra_context={"project": "Data Analysis Pipeline"},
    supervisor_system_prompt=None
)

# Result callback
def handle_result(task):
    print(f"✅ Task completed: {task['answer']}")

# Create hierarchical MAS
mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    heirarchical_mas_result_callback=handle_result,
    agent_return_direct=True
)

# Execute pipeline
async def data_pipeline():
    tasks = [
        "Process customer_data.csv",
        "Process sales_data.csv",
        "Generate monthly report"
    ]

    for task in tasks:
        result = await mas.initiate_hierarchical_mas(task)
        if result["status"] == "completed":
            print(f"Task completed: {result['answer']}")
        elif result["status"] == "queued":
            print(f"Task queued: {result['task_id']}")

asyncio.run(data_pipeline())
```

### Example 3: Customer Support System (Decentralized)

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem
import asyncio

# Create manager
manager = AgentManager(
    user_id="customer_123",
    model_config_path="model_config.json"
)

# Create support agents
triage = manager.create_agent(
    agent_name="triage_agent",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["issue classification", "routing"],
        description="Classifies customer issues and routes to specialists",
        style="efficient and empathetic"
    )
)

technical = manager.create_agent(
    agent_name="technical_support",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["technical troubleshooting", "debugging"],
        description="Resolves technical issues",
        style="detailed and patient"
    )
)

billing = manager.create_agent(
    agent_name="billing_support",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["billing inquiries", "payment processing"],
        description="Handles billing and payment issues",
        style="clear and helpful"
    )
)

# Create MAS
mas = MultiAgentSystem(agentManager=manager)

# Support workflow
async def handle_customer_query():
    while True:
        query = input("Customer query (or 'exit'): ")
        if query.lower() == 'exit':
            break

        result = await mas.initiate_decentralized_mas(
            query=query,
            set_entry_agent=triage,  # Always start with triage
            memory_order=5  # Rich context for better support
        )

        print(f"\nResponse: {result['answer']}")
        print(f"Handled by: {result['last_agent']}\n")

asyncio.run(handle_customer_query())
```

### Example 4: Complete Production Setup

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig
from masai.Memory.LongTermMemory import QdrantConfig
from langchain_openai import OpenAIEmbeddings
import asyncio
import os

# Setup persistent memory
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_config = QdrantConfig(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection_name="production_memories",
    vector_size=1536,
    embedding_model=embedding_model,
    dedup_mode="similarity",
    dedup_similarity_threshold=0.75
)

# Create manager with persistent memory
manager = AgentManager(
    user_id="production_user",
    model_config_path="model_config.json",
    memory_config=qdrant_config,
    logging=True
)

# Create agents with persistent memory
agent1 = manager.create_agent(
    agent_name="agent1",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["task1"],
        description="Agent 1",
        style="efficient"
    ),
    memory_order=15,
    long_context=True,
    long_context_order=25,
    persist_memory=True  # Enable persistent memory
)

agent2 = manager.create_agent(
    agent_name="agent2",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["task2"],
        description="Agent 2",
        style="thorough"
    ),
    memory_order=15,
    long_context=True,
    long_context_order=25,
    persist_memory=True
)

# Supervisor config
supervisor_config = SupervisorConfig(
    model_name="gpt-4o",
    temperature=0.7,
    model_category="openai",
    memory_order=20,
    memory=True,
    extra_context={"environment": "production"},
    supervisor_system_prompt=None
)

# Create hierarchical MAS
mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    agent_return_direct=True
)

# Production workflow
async def production_workflow():
    try:
        result = await mas.initiate_hierarchical_mas(
            query="Production task"
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "failed", "error": str(e)}

asyncio.run(production_workflow())
```

---

## Best Practices

### 1. Choose the Right Mode

**Use Decentralized when**:
- Agents need to make delegation decisions
- Workflow is dynamic and context-dependent
- Sequential processing is required
- Agent expertise determines next steps

**Use Hierarchical when**:
- Supervisor should control task assignment
- Parallel execution is beneficial
- Task queue management is needed
- Centralized coordination is preferred

### 2. Configure Memory Appropriately

```python
# Decentralized: Balance context and performance
result = await mas.initiate_decentralized_mas(
    query="Task",
    set_entry_agent=agent,
    memory_order=3  # 3-5 recommended
)

# Hierarchical: Supervisor needs more context
supervisor_config = SupervisorConfig(
    memory_order=20,  # 15-25 recommended
    memory=True
)
```

### 3. Use Persistent Memory for Long Sessions

```python
# Enable persistent memory for agents
agent = manager.create_agent(
    agent_name="agent",
    tools=[],
    agent_details=AgentDetails(...),
    persist_memory=True,  # Enable
    long_context=True,
    long_context_order=25
)
```

### 4. Implement Result Callbacks

```python
def handle_result(task):
    """Log and process completed tasks."""
    # Log to file
    with open('task_log.txt', 'a') as f:
        f.write(f"{task}\n")

    # Send notification
    send_notification(task['answer'])

    # Update database
    update_task_status(task['task_id'], 'completed')

mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    heirarchical_mas_result_callback=handle_result
)
```

### 5. Handle Errors Gracefully

```python
async def safe_execution():
    try:
        result = await mas.initiate_hierarchical_mas(query)
        if result["status"] == "failed":
            print(f"Task failed: {result['error']}")
            # Implement retry logic
        return result
    except Exception as e:
        print(f"Execution error: {e}")
        return {"status": "error", "error": str(e)}
```

### 6. Monitor Task Status

```python
async def monitor_tasks():
    result = await mas.initiate_hierarchical_mas(query)

    if result["status"] == "queued":
        task_id = result["task_id"]
        print(f"Task {task_id} queued, monitoring...")

        # Monitor via callback
        # Callback will be triggered when complete
```

### 7. Use Streaming for Long Tasks

```python
async def stream_progress():
    async for state in mas.initiate_decentralized_mas_astream(
        query="Long task",
        set_entry_agent=agent
    ):
        # Show progress to user
        node_name, state_dict = state
        print(f"Processing: {node_name}")
```

---

## Troubleshooting

### Issue: Agent not found in decentralized MAS

**Cause**: Agent name mismatch or agent not created.

**Solution**:
```python
# Check available agents
print(manager.agents.keys())

# Ensure agent exists
if "agent_name" in manager.agents:
    agent = manager.agents["agent_name"]
```

### Issue: Supervisor not selecting correct agent

**Cause**: Agent capabilities not clear or supervisor prompt unclear.

**Solution**:
```python
# Improve agent descriptions
agent_details = AgentDetails(
    capabilities=["specific", "clear", "capabilities"],
    description="Very clear description of what this agent does",
    style="specific style"
)

# Use custom supervisor prompt
custom_prompt = """
Select agent based on:
1. Task type
2. Agent capabilities
3. Current context
"""
```

### Issue: Tasks not executing in hierarchical MAS

**Cause**: TaskManager not properly initialized or stopped.

**Solution**:
```python
# Ensure supervisor_config is provided
supervisor_config = SupervisorConfig(...)

mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config  # Required for hierarchical
)

# Properly stop TaskManager
try:
    result = await mas.initiate_hierarchical_mas(query)
finally:
    await mas.task_manager.stop()
```

### Issue: Memory not persisting across queries

**Cause**: Persistent memory not enabled or configured.

**Solution**:
```python
# Enable persistent memory
agent = manager.create_agent(
    agent_name="agent",
    tools=[],
    agent_details=AgentDetails(...),
    persist_memory=True,  # Must be True
    long_context=True
)

# Ensure memory_config is set
manager.memory_config = qdrant_config
```

---

## Related Documentation

- [Quick Start Guide](QUICK_START.md) - Get started with MASAI
- [Agent Manager Detailed](AGENTMANAGER_DETAILED.md) - AgentManager API
- [Memory System Guide](MEMORY_SYSTEM.md) - Persistent memory
- [Tools Guide](TOOLS_GUIDE.md) - Tool definition and usage
- [Model Parameters](MODEL_PARAMETERS.md) - Model configuration
- [API Reference](API_REFERENCE.md) - Complete API documentation

---

## Summary

MASAI's Multi-Agent System provides:
- ✅ **Two coordination modes**: Decentralized and Hierarchical
- ✅ **Flexible delegation**: Agents decide or supervisor assigns
- ✅ **State management**: Maintains context across interactions
- ✅ **Concurrent execution**: Parallel task processing (hierarchical)
- ✅ **Streaming support**: Real-time progress updates
- ✅ **Persistent memory**: Long-term context retention
- ✅ **Result callbacks**: Handle completed tasks

Use this guide to build sophisticated multi-agent workflows for complex tasks!

