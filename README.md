# MAS-AI Framework: Complete Implementation Guide

+ PLEASE STAR THE PROJECT IF YO LIKE IT, BEFORE CLONING. A LOT OF PEOPLE CLONE WITHOUT STARRING

# MAS AI: Multi-Agent System Framework

|                                     |
|:-----------------------------------:|
|![MAS AI](/MAS/Logo/image.png) |

## MAS AI is a powerful framework for building scalable, intelligent multi-agent systems with advanced memory management and flexible collaboration patterns built using langgraph.

- Each agent in mas-ai has different components that work together to achieve the goal.

- Combine such agents in Multi-Agent Systems to achieve more complex goals.

- Combine such Multi-Agent Systems in an Orchestrated Multi-Agent Network (OMAN) to achieve even more complex goals.


## Featured

- ### [MAS AI was featred by sageflow community as a part of their newsletter. Check them out.](https://sageflow.ai/)

|                                     |
|:-----------------------------------:|

![SageFlow Shoutout](/MAS/Logo/sageflow_masai.jpg)|## Agent Architecture

MAS AI introduces two agent architectures optimized for different use cases:

### 1. Router, Reflector, Evaluator
A reactive architecture for dynamic task routing and output validation:

![Router, Reflector, Evaluator](./MAS/Architecture/general.png)


- **Router:** Analyzes queries and directs them to appropriate processing components
- **Evaluator:** Reviews outputs to ensure quality and relevance
- **Reflector:** Updates memory and improves routing strategies based on outcomes

**Workflow:**
1. Query received → Router analyzes
2. Router delegates to appropriate components
3. Components process the query
4. Evaluator validates the output
5. Reflector updates memory and strategies
6. Return final output

### 2. Planner, Executor, Reflector

![Router, Reflector, Planner](./MAS/Architecture/planner.png)

A proactive architecture for task planning and dependency management:

- **Planner:** Breaks queries into structured task plans
- **Executor:** Assigns tasks to appropriate components or agents
- **Reflector:** Assesses results and adjusts plans as needed



## Table of Contents
1. [Introduction](#introduction)
2. [Framework Architecture](#framework-architecture)
3. [Agent Components](#agent-components)
4. [Memory System](#memory-system)
5. [Multi-Agent Workflows](#multi-agent-workflows)
6. [Parameter Reference](#parameter-reference)
7. [Use Cases & Best Practices](#use-cases--best-practices)
8. [Advanced Features](#advanced-features)

---

## Introduction

**MAS-AI** (Multi-Agent System AI) is a modular framework for building intelligent agent systems using LangGraph. It provides:

- **Modular Agent Architecture**: Router, Evaluator, Reflector, Planner components
- **Hierarchical Memory System**: Short-term, component-shared, long-term, and vector store memory
- **Multiple Collaboration Patterns**: Sequential, Hierarchical, Decentralized workflows
- **LLM Flexibility**: Support for OpenAI, Gemini, Anthropic, Groq, Ollama, HuggingFace
- **Tool Integration**: LangChain-compatible tools with Redis caching support

### Why MAS-AI?

MAS-AI stands apart from conventional multi-agent frameworks by offering:

1. **Explicit Node Separation**: Unlike monolithic LLM systems, MAS-AI distributes responsibilities across specialized components (Router, Evaluator, Reflector, Planner)
2. **State-Machine Orchestration**: LangGraph-based state machine allows dynamic transitions between nodes based on satisfaction criteria
3. **Granular Memory Integration**: Multi-layered memory system (short-term, component-shared, long-term, vector store)
4. **Optimized for Complex Workflows**: Particularly well-suited for research-intensive tasks, multi-step decision processes, and tool-augmented execution

---

## Framework Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MAS-AI FRAMEWORK                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           AGENT MANAGER                              │  │
│  │  - Creates and manages agents                        │  │
│  │  - Configures tools and memory                       │  │
│  │  - Handles model configuration                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           SINGULAR AGENT                             │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │   Router   │→ │ Evaluator  │→ │ Reflector  │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  │         OR                                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │  Planner   │→ │  Executor  │→ │ Reflector  │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      MULTI-AGENT SYSTEM (MAS)                        │  │
│  │  - Sequential: Fixed pipeline                        │  │
│  │  - Hierarchical: Supervisor-based                    │  │
│  │  - Decentralized: Peer-to-peer                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ORCHESTRATED MULTI-AGENT NETWORK (OMAN)             │  │
│  │  - Coordinates multiple MAS instances                │  │
│  │  - Network-level memory and routing                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Reference Table

| Component | Purpose | Import Path |
|-----------|---------|-------------|
| **AgentManager** | Central registry for creating and managing agents | `from masai.AgentManager.AgentManager import AgentManager, AgentDetails` |
| **Agent** | Singular agent with Router-Evaluator-Reflector architecture | `from masai.Agents.singular_agent import Agent` |
| **BaseAgent** | Base class with common agent functionality | `from masai.Agents.base_agent import BaseAgent` |
| **MASGenerativeModel** | LLM wrapper with memory and context management | `from masai.GenerativeModel.generativeModels import MASGenerativeModel` |
| **BaseGenerativeModel** | Simple LLM wrapper without agent architecture | `from masai.GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel` |
| **MultiAgentSystem** | Coordinates multiple agents in workflows | `from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig` |
| **TaskManager** | Manages concurrent tasks in hierarchical MAS | `from masai.MultiAgents.TaskManager import TaskManager` |
| **OMAN** | Orchestrates multiple MAS instances | `from masai.OMAN.oman import OrchestratedMultiAgentNetwork` |
| **InMemoryDocStore** | Vector store for semantic search | `from masai.Memory.InMemoryStore import InMemoryDocStore` |
| **ToolCache** | Redis-based caching for tools | `from masai.Tools.utilities.cache import ToolCache` |
| **Config** | Global configuration parameters | `from masai.Config import config` |

### Framework Levels Explained

#### Level 1: Singular Agent
- **Single agent** with internal components (Router, Evaluator, Reflector, optional Planner)
- Handles queries independently using tools
- Can delegate to other agents in decentralized mode
- **Use when**: Single domain, tool-heavy tasks, simple queries

#### Level 2: Multi-Agent System (MAS)
- **Multiple agents** working together in coordinated workflows
- Three workflow types: Sequential, Hierarchical, Decentralized
- Shared memory and context across agents
- **Use when**: Multi-domain tasks, complex workflows, quality control needed

#### Level 3: Orchestrated Multi-Agent Network (OMAN)
- **Multiple MAS instances** coordinated by OMAN supervisor
- Each MAS specializes in different domains
- Network-level routing and memory
- **Use when**: Enterprise-scale, multiple specialized systems, cross-domain coordination

---

## Agent Components

MAS AI introduces two agent architectures optimized for different use cases:

### 1. Router, Reflector, Evaluator
A reactive architecture for dynamic task routing and output validation:

![Router, Reflector, Evaluator](./MAS/Architecture/general.png)


- **Router:** Analyzes queries and directs them to appropriate processing components
- **Evaluator:** Reviews outputs to ensure quality and relevance
- **Reflector:** Updates memory and improves routing strategies based on outcomes

### 2. Planner, Executor, Reflector

![Router, Reflector, Planner](./MAS/Architecture/planner.png)

A proactive architecture for task planning and dependency management:

- **Planner:** Breaks queries into structured task plans
- **Executor:** Assigns tasks to appropriate components or agents
- **Reflector:** Assesses results and adjusts plans as needed


### 1. Router-Evaluator-Reflector Architecture

**Purpose**: Reactive architecture for dynamic task routing and validation

#### Router
- **Function**: Analyzes queries and routes to appropriate tools/agents
- **Input**: User query + chat history + context
- **Output**: Tool selection OR agent delegation OR direct answer
- **Model Config**: `model_config.json → router`

#### Evaluator
- **Function**: Validates tool outputs and agent responses
- **Input**: Tool output + original query + context
- **Output**: Satisfaction status (satisfied/not_satisfied) + reasoning
- **Model Config**: `model_config.json → evaluator`

#### Reflector
- **Function**: Updates memory and refines strategies
- **Input**: Conversation history + outcomes
- **Output**: Updated memory + insights
- **Model Config**: `model_config.json → reflector`

**Workflow**:
```
Query → Router → Tool/Agent → Evaluator → [Satisfied?]
                                              ↓ No
                                          Reflector → Router (retry)
                                              ↓ Yes
                                          Final Answer
```

### 2. Planner-Executor-Reflector Architecture

**Purpose**: Proactive architecture for complex task decomposition

#### Planner
- **Function**: Decomposes queries into structured task plans
- **Input**: User query + context
- **Output**: Task list with dependencies
- **Model Config**: `model_config.json → planner`

#### Executor
- **Function**: Executes tasks using tools/agents
- **Input**: Task plan + tools
- **Output**: Task results
- **Model Config**: Uses router model

#### Reflector
- **Function**: Evaluates results and adjusts plans
- **Input**: Task results + original plan
- **Output**: Re-planning decisions
- **Model Config**: `model_config.json → reflector`

**Workflow**:
```
Query → Planner → Task List → Executor → Results → Reflector
                                                      ↓
                                              [Complete?]
                                                ↓ No
                                            Planner (re-plan)
                                                ↓ Yes
                                            Final Answer
```

### 3. Agent State Machine

MAS-AI agents use a **LangGraph state machine** to manage workflow execution. Understanding the state is crucial for debugging and optimization.

#### State Structure

```python
class State(TypedDict):
    messages: List[Dict[str, str]]           # Chat history
    current_tool: str                        # Currently selected tool
    tool_input: Any                          # Input for the tool
    tool_output: Any                         # Output from the tool
    answer: str                              # Current answer
    satisfied: bool                          # Satisfaction flag
    reasoning: str                           # LLM reasoning
    delegate_to_agent: Optional[str]         # Agent to delegate to
    current_node: str                        # Current node in workflow
    previous_node: Optional[str]             # Previous node
    plan: Optional[dict]                     # Task plan (if using planner)
    passed_from: Optional[str]               # Source of delegation
    reflection_counter: int                  # Number of reflections
    tool_loop_counter: int                   # Number of tool loops
```

#### State Transitions

```
┌─────────────────────────────────────────────────────────────┐
│                  AGENT STATE MACHINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  START                                                      │
│    ↓                                                        │
│  [plan=True?] ──Yes──→ PLANNER ──→ EXECUTOR                │
│    ↓ No                              ↓                      │
│  ROUTER ─────────────────────────────┘                      │
│    ↓                                                        │
│  [tool selected?]                                           │
│    ↓ Yes                                                    │
│  EXECUTE_TOOL                                               │
│    ↓                                                        │
│  EVALUATOR                                                  │
│    ↓                                                        │
│  [satisfied=True?]                                          │
│    ↓ No                                                     │
│  [tool_loop_counter > MAX?] ──Yes──→ REFLECTOR             │
│    ↓ No                                ↓                    │
│  ROUTER (retry)                        ↓                    │
│    ↓ Yes                               ↓                    │
│  [delegate_to_agent?] ──Yes──→ DELEGATE_TO_AGENT           │
│    ↓ No                                ↓                    │
│  END (Final Answer)                    END                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Loop Prevention Mechanisms

**1. Tool Loop Counter**
- Tracks consecutive tool uses
- Max limit: `config.max_tool_loops` (default: 3)
- When exceeded: Triggers warning prompt and forces reflection

**2. Reflection Counter**
- Tracks number of reflection cycles
- Max limit: `config.MAX_REFLECTION_COUNT` (default: 3)
- When exceeded: Forces final answer or delegation

**3. Recursion Limit**
- Overall workflow recursion limit
- Max limit: `config.MAX_RECURSION_LIMIT` (default: 100)
- Prevents infinite loops in complex workflows

#### Configuration Parameters

```python
from masai.Config import config

# Modify global config
config.max_tool_loops = 5              # Default: 3
config.MAX_REFLECTION_COUNT = 5        # Default: 3
config.MAX_RECURSION_LIMIT = 150       # Default: 100
config.stream_mode = "updates"         # LangGraph stream mode
config.truncated_response_length = 500 # Logging truncation
```

---

## Memory System

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: AGENT SHORT-TERM MEMORY                          │
│  ├─ Chat history (last N messages)                         │
│  ├─ Current context                                        │
│  └─ Parameter: memory_order (default: 5)                   │
│                                                             │
│  Level 2: COMPONENT MEMORY                                 │
│  ├─ Component short-term (per Router/Evaluator/etc.)       │
│  ├─ Component shared (between components)                  │
│  └─ Component long-term (summarized history)               │
│      └─ Parameter: long_context_order (default: 10)        │
│                                                             │
│  Level 3: MULTI-AGENT SYSTEM MEMORY                        │
│  ├─ Shared across all agents in MAS                        │
│  └─ Parameter: shared_memory_order (default: 3)            │
│                                                             │
│  Level 4: NETWORK MEMORY                                   │
│  ├─ Spans all MAS instances in OMAN                        │
│  └─ Parameter: network_memory_order                        │
│                                                             │
│  Level 5: EXTENDED MEMORY STORE                            │
│  ├─ Vector store for semantic search                       │
│  ├─ InMemoryDocStore (sentence-transformers)               │
│  └─ Parameter: in_memory_store, top_k                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Memory Parameters

| Parameter | Scope | Default | Purpose |
|-----------|-------|---------|---------|
| `memory` | Agent | `True` | Enable/disable memory |
| `memory_order` | Agent | `5` | Number of recent messages to keep |
| `long_context` | Agent | `False` | Enable long-term memory summarization |
| `long_context_order` | Agent | `10` | Number of old messages to summarize |
| `shared_memory_order` | MAS | `3` | Shared memory size across agents |
| `network_memory_order` | OMAN | N/A | Network-level memory size |
| `in_memory_store` | Agent | `None` | Vector store instance |
| `top_k` | Agent | `3` | Number of vector search results |
| `chat_log` | Agent | `None` | File path to save chat history |

### Memory Flow Detailed

#### 1. Chat History Management

```
┌─────────────────────────────────────────────────────────────┐
│              CHAT HISTORY LIFECYCLE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  New Message                                                │
│    ↓                                                        │
│  Append to chat_history[]                                   │
│    ↓                                                        │
│  [len(chat_history) > memory_order?]                        │
│    ↓ Yes                                                    │
│  [long_context=True?]                                       │
│    ↓ Yes                          ↓ No                      │
│  Summarize old messages      Truncate to memory_order/2     │
│    ↓                              ↓                         │
│  Add to context_summaries[]  [chat_log set?]                │
│    ↓                              ↓ Yes                     │
│  [len(summaries) > long_context_order?]  Save to file      │
│    ↓ Yes                          ↓                         │
│  [LTIMStore set?]            Keep recent messages           │
│    ↓ Yes                                                    │
│  Move old summaries to vector store                         │
│    ↓                                                        │
│  Keep recent summaries                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
- `chat_history` stores raw messages as `{'role': 'user/assistant', 'content': '...'}`
- When `len(chat_history) > memory_order`, old messages are processed
- If `long_context=True`, old messages are summarized using a separate LLM
- Summaries are stored in `context_summaries[]` as LangChain `Document` objects
- When summaries exceed `long_context_order`, oldest are moved to `LTIMStore` (if configured)
- If `chat_log` is set, truncated messages are saved to file before removal

#### 2. Context Summaries

**Purpose**: Compress old conversations while retaining key information

**Process**:
1. When `chat_history` exceeds `memory_order`, oldest messages are selected
2. A separate LLM (GenerativeModel with `temperature=0.5`) summarizes them
3. Summary prompt focuses on: main topics, key information, specific keywords, conclusions
4. Summary is stored as a `Document` with `page_content` field
5. Summaries are included in prompts under `<EXTENDED CONTEXT>` section

**Example Summary**:
```
"The user asked about implementing a caching system for API calls.
The assistant recommended Redis with a TTL of 30 minutes.
Key points: Use pickle for serialization, handle connection errors,
implement cache invalidation strategy. User confirmed implementation."
```

#### 3. InMemoryDocStore (LTIMStore)

**Purpose**: Semantic search over very old conversation history

**How It Works**:
```
┌─────────────────────────────────────────────────────────────┐
│           LTIMSTORE WORKFLOW                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Old Summaries (beyond long_context_order)                  │
│    ↓                                                        │
│  Convert to Document objects                                │
│    ↓                                                        │
│  Embed using SentenceTransformer                            │
│    ↓                                                        │
│  Store in InMemoryDocStore                                  │
│    ↓                                                        │
│  On New Query:                                              │
│    ↓                                                        │
│  Embed query                                                │
│    ↓                                                        │
│  Cosine similarity search                                   │
│    ↓                                                        │
│  Return top_k most relevant summaries                       │
│    ↓                                                        │
│  Include in prompt under <EXTENDED CONTEXT>                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Setup**:
```python
from masai.Memory.InMemoryStore import InMemoryDocStore

# Create vector store
memory_store = InMemoryDocStore(
    embedding_model="all-MiniLM-L6-v2"  # SentenceTransformer model
)

# Or use custom embedding function
def custom_embedder(texts: List[str]) -> np.ndarray:
    # Your embedding logic
    return embeddings

memory_store = InMemoryDocStore(embedding_model=custom_embedder)

# Or use LangChain embeddings
from langchain.embeddings import OpenAIEmbeddings
memory_store = InMemoryDocStore(embedding_model=OpenAIEmbeddings())

# Pass to agent
manager.create_agent(
    agent_name="research_agent",
    tools=tools,
    agent_details=details,
    long_context=True,
    long_context_order=20,
    in_memory_store=memory_store,
    top_k=3  # Return top 3 relevant memories
)
```

#### 4. Component Context

**Purpose**: Share information between agent components (Router → Evaluator → Reflector)

**Mechanism**:
- Each component can add messages to `component_context[]`
- These messages are passed to the next component
- Controlled by `shared_memory_order` parameter
- Allows components to communicate reasoning and intermediate results

**Example**:
```python
# Router adds context
component_context = [
    {'role': 'router', 'content': 'Selected database_tool because query mentions "users"'}
]

# Evaluator receives this context and adds its own
component_context.append(
    {'role': 'evaluator', 'content': 'Tool returned 150 users, satisfies query'}
)

# Reflector receives both contexts
```

#### 5. Chat Log Persistence

**Purpose**: Save conversation history to file for later analysis or resumption

**Setup**:
```python
manager = AgentManager(
    context={},
    logging=True,
    model_config_path="model_config.json",
    chat_log="./logs/agent_chat.json"  # File path
)
```

**Behavior**:
- When `chat_history` is truncated, removed messages are saved to file
- File format: JSON array of message objects
- Useful for debugging, analysis, or resuming conversations
- Automatically creates directory if it doesn't exist

---

## Multi-Agent Workflows

### 1. Sequential Workflow

**Use Case**: Fixed pipeline processing (ETL, document processing)

**Architecture**:
```
Agent 1 → Agent 2 → Agent 3 → Final Output
```

**Implementation**:
```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem

mas = MultiAgentSystem(agentManager=manager)

result = mas.initiate_sequential_mas(
    query="Process this document",
    agent_sequence=["research_agent", "analysis_agent", "summary_agent"],
    memory_order=3  # Shared memory across agents
)
```

**Parameters**:
- `query` (str): Input query
- `agent_sequence` (List[str]): Ordered list of agent names
- `memory_order` (int): Shared memory size

**Data Flow**:
```
Query → Agent 1 (output_1) → Agent 2 (output_1 + output_2) → Agent 3 (final)
         ↓                      ↓                              ↓
      Memory[0]              Memory[1]                     Memory[2]
```

### 2. Hierarchical Workflow

**Use Case**: Complex tasks requiring supervision and quality control

**Architecture**:
```
                    Supervisor LLM
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    Agent 1          Agent 2          Agent 3
        │                │                │
        └────────────────┴────────────────┘
                         │
                    Task Manager
                         │
                  Result Callback
```

**Implementation**:
```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig

def handle_task_result(task):
    print(f"Task completed: {task['answer']}")

supervisor_config = SupervisorConfig(
    model_name="gpt-4",
    temperature=0.2,
    model_category="openai",
    memory_order=20,
    memory=True,
    extra_context={"organization": "Benosphere"}
)

mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    heirarchical_mas_result_callback=handle_task_result,
    agent_return_direct=True
)

result = await mas.initiate_hierarchical_mas("Complex research task")
```

**Parameters**:
- `supervisor_config` (SupervisorConfig): Supervisor LLM configuration
- `heirarchical_mas_result_callback` (Callable): Callback for task completion
- `agent_return_direct` (bool): Return agent output directly without supervisor review

**Data Flow**:
```
Query → Supervisor → Task Queue → Agent → Result → Supervisor Review
                                                         ↓
                                                  [Satisfied?]
                                                    ↓ No
                                            Revision Request → Agent
                                                    ↓ Yes
                                                Callback → Final Output
```

### 3. Decentralized Workflow

**Use Case**: Peer-to-peer collaboration, adaptive workflows

**Architecture**:
```
Entry Agent ⇄ Agent 2 ⇄ Agent 3
     ↕           ↕         ↕
  Agent 4 ⇄  Agent 5 ⇄ Agent 6
```

**Implementation**:
```python
mas = MultiAgentSystem(agentManager=manager)

result = await mas.initiate_decentralized_mas(
    query="Research AI trends and schedule meeting",
    set_entry_agent=manager.get_agent("personal_assistant")
)
```

**Parameters**:
- `query` (str): Input query
- `set_entry_agent` (Agent): Initial agent to handle query

**Data Flow**:
```
Query → Entry Agent → [Delegate?] → Agent 2 → [Delegate?] → Agent 3
                         ↓ No                    ↓ No
                      Answer                  Answer
```

---

## Parameter Reference

### 1. AgentManager.__init__()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `logging` | `bool` | No | `True` | Enable/disable logging of agent activities |
| `context` | `dict` | No | `None` | Static context shared with all agents (e.g., `{"org": "MyCompany"}`) |
| `model_config_path` | `str` | **Yes** | N/A | Path to model configuration JSON file |
| `chat_log` | `str` | No | `None` | File path to save chat history when truncated |
| `streaming` | `bool` | No | `False` | Enable streaming responses from LLMs |
| `streaming_callback` | `Callable` | No | `None` | Async callback function for streaming chunks (required if `streaming=True`) |

**Example**:
```python
from masai.AgentManager.AgentManager import AgentManager

manager = AgentManager(
    context={"organization": "MyCompany", "environment": "production"},
    logging=True,
    model_config_path="./config/model_config.json",
    chat_log="./logs/chat_history.json",
    streaming=True,
    streaming_callback=async_streaming_handler
)
```

### 2. AgentManager.create_agent()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_name` | `str` | **Yes** | N/A | Unique identifier for the agent (converted to lowercase) |
| `tools` | `List[Tool]` | **Yes** | N/A | List of LangChain tools the agent can use |
| `agent_details` | `AgentDetails` | **Yes** | N/A | Agent configuration (capabilities, description, style) |
| `memory_order` | `int` | No | `10` | Number of recent messages to keep in short-term memory |
| `long_context` | `bool` | No | `True` | Enable long-term memory with summarization |
| `long_context_order` | `int` | No | `20` | Number of old message summaries to keep |
| `shared_memory_order` | `int` | No | `10` | Shared memory size between components |
| `plan` | `bool` | No | `False` | Use Planner-Executor-Reflector architecture (vs Router-Evaluator-Reflector) |
| `temperature` | `float` | No | `0.2` | Default temperature for all LLMs (can be overridden per component) |
| `context_callable` | `Callable` | No | `None` | Function to fetch dynamic context on each query |
| `in_memory_store` | `InMemoryDocStore` | No | `None` | Vector store for semantic search over old conversations |
| `top_k` | `int` | No | `3` | Number of results to retrieve from vector store |
| `config_dict` | `dict` | No | `None` | Per-component configuration overrides (see below) |

**config_dict Structure**:
```python
config_dict = {
    "router_memory_order": 10,
    "router_long_context_order": 15,
    "router_temperature": 0.3,
    "evaluator_memory_order": 5,
    "evaluator_long_context_order": 10,
    "evaluator_temperature": 0.1,
    "reflector_memory_order": 15,
    "reflector_long_context_order": 20,
    "reflector_temperature": 0.7,
    "planner_memory_order": 10,  # Only if plan=True
    "planner_long_context_order": 15,
    "planner_temperature": 0.2
}
```

**Example**:
```python
from masai.AgentManager.AgentManager import AgentDetails
from masai.Memory.InMemoryStore import InMemoryDocStore

# Create vector store
memory_store = InMemoryDocStore(embedding_model="all-MiniLM-L6-v2")

# Define agent details
agent_details = AgentDetails(
    capabilities=["database queries", "data analysis", "report generation"],
    description="Analyzes database data and generates insights",
    style="concise and data-focused"
)

# Create agent with all options
manager.create_agent(
    agent_name="data_analyst",
    tools=[database_tool, chart_tool],
    agent_details=agent_details,
    memory_order=15,
    long_context=True,
    long_context_order=30,
    shared_memory_order=10,
    plan=False,
    temperature=0.3,
    context_callable=fetch_user_permissions,
    in_memory_store=memory_store,
    top_k=5,
    config_dict={
        "router_temperature": 0.2,
        "evaluator_temperature": 0.1,
        "reflector_temperature": 0.5
    }
)
```

### 3. AgentDetails

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `capabilities` | `List[str]` | **Yes** | N/A | List of agent capabilities (e.g., `["reasoning", "coding", "science"]`) |
| `description` | `str` | No | `""` | Detailed description of agent's purpose and behavior |
| `style` | `str` | No | `"gives very elaborate answers"` | Communication style for responses |

**Example**:
```python
from masai.AgentManager.AgentManager import AgentDetails

agent_details = AgentDetails(
    capabilities=["database queries", "data analysis", "visualization"],
    description="Specializes in analyzing database data and creating visual reports",
    style="concise and data-focused, uses bullet points"
)
```

### 4. MASGenerativeModel.__init__()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | `str` | **Yes** | N/A | LLM model name (e.g., `"gpt-4"`, `"gemini-2.0-flash"`) |
| `temperature` | `float` | **Yes** | N/A | Temperature for randomness (0.0-1.0) |
| `category` | `str` | **Yes** | N/A | Model category: `"openai"`, `"gemini"`, `"anthropic"`, `"groq"`, `"ollama"`, `"huggingface"` |
| `prompt_template` | `ChatPromptTemplate` | No | `None` | LangChain prompt template |
| `memory_order` | `int` | No | `5` | Number of recent messages to keep |
| `extra_context` | `dict` | No | `None` | Static context (e.g., `{"user_id": "123"}`) |
| `long_context` | `bool` | No | `False` | Enable long-term memory summarization |
| `long_context_order` | `int` | No | `10` | Number of summaries to keep |
| `chat_log` | `str` | No | `None` | File path to save chat history |
| `streaming` | `bool` | No | `False` | Enable streaming responses |
| `streaming_callback` | `Callable` | No | `None` | Async callback for streaming chunks |
| `context_callable` | `Callable` | No | `None` | Function to fetch dynamic context per query |
| `memory_store` | `InMemoryDocStore` | No | `None` | Vector store for semantic search (kwarg) |
| `k` | `int` | No | `3` | Number of vector search results (kwarg) |

**Example**:
```python
from masai.GenerativeModel.generativeModels import MASGenerativeModel

llm = MASGenerativeModel(
    model_name="gemini-2.0-flash",
    temperature=0.3,
    category="gemini",
    memory_order=10,
    extra_context={"user_role": "admin"},
    long_context=True,
    long_context_order=20,
    context_callable=fetch_dynamic_context
)
```

### 5. MultiAgentSystem.__init__()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agentManager` | `AgentManager` | **Yes** | N/A | AgentManager instance with created agents |
| `supervisor_config` | `SupervisorConfig` | No | `None` | Supervisor configuration (required for hierarchical MAS) |
| `heirarchical_mas_result_callback` | `Callable` | No | `None` | Callback function for task completion in hierarchical MAS |
| `agent_return_direct` | `bool` | No | `False` | If `False`, supervisor evaluates agent responses before returning |

**Example**:
```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig

supervisor_config = SupervisorConfig(
    model_name="gpt-4",
    temperature=0.2,
    model_category="openai",
    memory_order=20,
    memory=True,
    extra_context={"organization": "MyCompany"},
    supervisor_system_prompt="You are an efficient task coordinator"
)

mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    heirarchical_mas_result_callback=handle_result,
    agent_return_direct=False
)
```

### 6. SupervisorConfig

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | `str` | **Yes** | N/A | LLM model name for supervisor |
| `temperature` | `float` | **Yes** | N/A | Temperature (0.0-1.0) |
| `model_category` | `str` | **Yes** | N/A | Model category |
| `memory_order` | `int` | **Yes** | N/A | Supervisor memory size |
| `memory` | `bool` | **Yes** | N/A | Enable supervisor memory |
| `extra_context` | `dict` | **Yes** | N/A | Additional context for supervisor |
| `supervisor_system_prompt` | `str` | No | `None` | Custom system prompt (uses default if not provided) |

### 7. InMemoryDocStore.__init__()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `documents` | `List[Union[str, Document]]` | No | `None` | Initial documents to store |
| `ids` | `List[str]` | No | `None` | Document IDs (auto-generated if not provided) |
| `embedding_model` | `Union[str, Callable, object]` | No | `"all-MiniLM-L6-v2"` | Embedding model: string (SentenceTransformer name), callable, or object with `embed_documents` method |

**Example**:
```python
from masai.Memory.InMemoryStore import InMemoryDocStore

# Option 1: SentenceTransformer model name
store = InMemoryDocStore(embedding_model="all-MiniLM-L6-v2")

# Option 2: Custom embedding function
def my_embedder(texts: List[str]) -> np.ndarray:
    # Your embedding logic
    return embeddings

store = InMemoryDocStore(embedding_model=my_embedder)

# Option 3: LangChain embeddings
from langchain.embeddings import OpenAIEmbeddings
store = InMemoryDocStore(embedding_model=OpenAIEmbeddings())
```

### 8. ToolCache.__init__()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `host` | `str` | No | `"localhost"` | Redis server host |
| `port` | `int` | No | `6379` | Redis server port |
| `db` | `int` | No | `0` | Redis database number |
| `password` | `str` | No | `None` | Redis password (if required) |
| `timeout` | `int` | No | `30` | Cache timeout in minutes |

**Example**:
```python
from masai.Tools.utilities.cache import ToolCache
from langchain.tools import tool

# Initialize cache
cache = ToolCache(
    host="localhost",
    port=6379,
    db=0,
    timeout=60  # 60 minutes
)

# Use as decorator
@tool
@cache.masai_cache
def expensive_api_call(query: str) -> dict:
    """Makes an expensive API call. Results are cached."""
    return api_response
```

### 9. OrchestratedMultiAgentNetwork.__init__()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mas_instances` | `List[MultiAgentSystem]` | **Yes** | N/A | List of MAS instances to orchestrate |
| `network_memory_order` | `int` | No | `3` | Network-level memory size |
| `oman_llm_config` | `dict` | No | `None` | OMAN supervisor LLM configuration |
| `extra_context` | `dict` | No | `None` | Additional context for OMAN supervisor |

**oman_llm_config Structure**:
```python
oman_llm_config = {
    "model_name": "gemini-2.0-flash-001",
    "category": "gemini",
    "temperature": 0.2,
    "memory_order": 3
}
```

**Example**:
```python
from masai.OMAN.oman import OrchestratedMultiAgentNetwork

# Create multiple MAS instances
mas1 = MultiAgentSystem(agentManager=manager1)
mas2 = MultiAgentSystem(agentManager=manager2)

# Create OMAN
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[mas1, mas2],
    network_memory_order=5,
    oman_llm_config={
        "model_name": "gpt-4",
        "category": "openai",
        "temperature": 0.2,
        "memory_order": 5
    },
    extra_context={"environment": "production"}
)
```

---

## Use Cases & Best Practices

### When to Use Each Architecture

| Architecture | Use Case | Example |
|--------------|----------|---------|
| **Router-Evaluator-Reflector** | Dynamic routing, tool-heavy tasks | Database queries, API integrations, data retrieval |
| **Planner-Executor-Reflector** | Complex multi-step tasks | Research projects, data analysis, report generation |

### When to Use Each Workflow

| Workflow | Use Case | Example |
|----------|----------|---------|
| **Sequential** | Fixed pipeline, deterministic flow | ETL, document processing, data transformation |
| **Hierarchical** | Quality control, supervision needed | Research with review, complex analysis, content creation |
| **Decentralized** | Adaptive collaboration, peer tasks | Personal assistant systems, multi-domain problem solving |

### Memory Configuration Guidelines

| Scenario | memory_order | long_context | long_context_order | in_memory_store |
|----------|--------------|--------------|-------------------|-----------------|
| **Short conversations** | 5 | False | N/A | No |
| **Medium conversations** | 10 | True | 10 | No |
| **Long conversations** | 10 | True | 20 | Yes |
| **Research tasks** | 5 | True | 30 | Yes |

### Model Selection Guidelines

| Component | Recommended Model | Reasoning |
|-----------|------------------|-----------|
| **Router** | Fast model (Gemini Flash, GPT-3.5) | Frequent calls, simple routing |
| **Evaluator** | Fast model (Gemini Flash Lite) | Binary decision (satisfied/not) |
| **Reflector** | Medium model (Gemini Flash) | Summarization and insights |
| **Planner** | Strong model (GPT-4, Gemini Pro) | Complex task decomposition |
| **Supervisor** | Strong model (GPT-4, Gemini Pro) | High-level coordination |

---

## Advanced Features

### 1. Using BaseGenerativeModel (Without Agent Architecture)

For simple conversational AI without tools or routing, use `BaseGenerativeModel`:

**Use Case**: Simple chatbots, Q&A systems, content generation

**Configuration**:
```python
from masai.GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel

model = BaseGenerativeModel(
    model_name="gemini-2.0-flash",
    category='gemini',
    temperature=1,
    memory=True,
    memory_order=20,
    info={'USER_NAME': 'John', 'CONTEXT': 'Customer support'},
    system_prompt="You are a helpful assistant"
)

# Streaming response
async for chunk in model.astream_response(prompt):
    print(chunk, end='', flush=True)

# Non-streaming response
response = await model.generate_response(prompt)
print(response)
```

**Key Features**:
- No tools, no routing - pure LLM interaction
- Simple memory management
- Streaming and non-streaming support
- Custom system prompts
- Lightweight and fast

**When to Use**:
- Simple Q&A without external data
- Content generation tasks
- Conversational interfaces without tool needs
- Prototyping before adding agent architecture

### 2. Redis Caching for Tools

Cache tool outputs to improve performance for frequently used queries:

**Setup**:
```python
from langchain.tools import tool
from masai.Tools.utilities.cache import ToolCache

# Initialize Redis cache
redis_cache = ToolCache(host='localhost', port=6379, db=0)

@tool
@redis_cache.masai_cache
def expensive_api_call(query: str) -> str:
    """
    Makes an expensive API call. Results are cached.

    Args:
        query: Search query

    Returns:
        API response
    """
    # Expensive operation here
    return api_response
```

**Benefits**:
- Faster response times for repeated queries
- Reduced API costs
- Automatic cache invalidation
- Monitor cache with Redis CLI

**Requirements**:
- Redis server running
- `redis` Python package installed

### 3. In-Memory Vector Store

For long conversations with semantic search capabilities:

**Setup**:
```python
from masai.Memory.InMemoryStore import InMemoryDocStore

# Create vector store with embedding model
memory_store = InMemoryDocStore(embedding_model="all-MiniLM-L6-v2")

# Create agent with vector store
manager.create_agent(
    agent_name="research_agent",
    tools=tools,
    agent_details=details,
    long_context=True,
    long_context_order=20,
    in_memory_store=memory_store,
    top_k=3  # Retrieve top 3 relevant memories
)
```

**How It Works**:
1. Old messages (beyond `memory_order`) are summarized
2. Summaries are embedded and stored in vector store
3. On new queries, semantic search retrieves relevant past context
4. Retrieved context is added to prompt

**Benefits**:
- Semantic search over conversation history
- Better context retention for long conversations
- Efficient memory usage

**Supported Embedding Models**:
- Sentence Transformers (default: `all-MiniLM-L6-v2`)
- LangChain embedding models (OpenAI, Cohere, etc.)

### 4. Streaming Callbacks

For real-time response streaming:

**Setup**:
```python
async def streaming_callback(chunk):
    """Handle streaming chunks"""
    if isinstance(chunk, dict):
        if "answer" in chunk:
            print(f"Answer: {chunk['answer']}")
    else:
        print(chunk, end="", flush=True)

manager = AgentManager(
    context={},
    logging=True,
    model_config_path="model_config.json",
    streaming=True,
    streaming_callback=streaming_callback
)
```

**Use Cases**:
- Real-time UI updates
- Progress indicators
- Debugging and monitoring

### 5. Per-Component Configuration

Override model settings for specific components:

**Setup**:
```python
config_dict = {
    "router": {
        "memory_order": 10,
        "temperature": 0.3
    },
    "evaluator": {
        "memory_order": 5,
        "temperature": 0.1
    },
    "reflector": {
        "memory_order": 15,
        "temperature": 0.7
    }
}

manager.create_agent(
    agent_name="custom_agent",
    tools=tools,
    agent_details=details,
    config_dict=config_dict
)
```

**Benefits**:
- Fine-tune each component independently
- Optimize for specific use cases
- Balance cost vs. performance

### 6. Context Callable (Dynamic Context)

Fetch dynamic context on each query:

**Setup**:
```python
async def get_user_context(query: str) -> dict:
    """Fetch user-specific context dynamically"""
    # Fetch from database, API, etc.
    user_data = await fetch_user_data()
    return {
        "user_name": user_data["name"],
        "user_role": user_data["role"],
        "permissions": user_data["permissions"]
    }

manager.create_agent(
    agent_name="personalized_agent",
    tools=tools,
    agent_details=details,
    context_callable=get_user_context
)
```

**Use Cases**:
- User-specific personalization
- Real-time data fetching
- Dynamic permission checks

**Note**: Currently only called for user queries, not agent-to-agent delegation.

**Context Callable Flow**:
```
┌─────────────────────────────────────────────────────────────┐
│           CONTEXT CALLABLE WORKFLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query Received                                        │
│    ↓                                                        │
│  [context_callable set?]                                    │
│    ↓ Yes                                                    │
│  [role == "user"?]                                          │
│    ↓ Yes                                                    │
│  Call context_callable(query)                               │
│    ↓                                                        │
│  [Is coroutine?]                                            │
│    ↓ Yes              ↓ No                                  │
│  await callable()   callable()                              │
│    ↓                  ↓                                     │
│  [Result is dict?]                                          │
│    ↓ Yes              ↓ No                                  │
│  Update info dict   Add as 'USEFUL DATA' key                │
│    ↓                                                        │
│  Include in prompt under <INFO>                             │
│    ↓                                                        │
│  After LLM response:                                        │
│    ↓                                                        │
│  Remove 'USEFUL DATA' from info                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7. TaskManager (Hierarchical MAS Internals)

The TaskManager handles concurrent task execution in hierarchical MAS:

**Key Features**:
- **Concurrent Execution**: Uses ThreadPoolExecutor for parallel task processing
- **Task Queue**: Manages pending tasks with unique IDs
- **Result Callbacks**: Async callbacks for task completion
- **Completed Task Context**: Maintains history of completed tasks for supervisor context

**Configuration**:
```python
# TaskManager is automatically created by MultiAgentSystem
# You can configure it through SupervisorConfig

supervisor_config = SupervisorConfig(
    model_name="gpt-4",
    temperature=0.2,
    model_category="openai",
    memory_order=20,  # Supervisor memory
    memory=True,
    extra_context={"organization": "MyCompany"},
    supervisor_system_prompt="Custom supervisor prompt"
)

mas = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config,
    heirarchical_mas_result_callback=my_callback,
    agent_return_direct=False  # Supervisor reviews agent output
)
```

**TaskManager Workflow**:
```
┌─────────────────────────────────────────────────────────────┐
│              TASKMANAGER WORKFLOW                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query → Supervisor                                         │
│    ↓                                                        │
│  Supervisor Decision:                                       │
│    ├─ Direct Answer → Return                                │
│    └─ Delegate to Agent                                     │
│         ↓                                                   │
│  Create Task (unique ID)                                    │
│    ↓                                                        │
│  Add to pending_tasks{}                                     │
│    ↓                                                        │
│  Submit to ThreadPoolExecutor                               │
│    ↓                                                        │
│  Agent Executes (async)                                     │
│    ↓                                                        │
│  Result → Result Queue                                      │
│    ↓                                                        │
│  Listener Task picks up result                              │
│    ↓                                                        │
│  [agent_return_direct=True?]                                │
│    ↓ Yes              ↓ No                                  │
│  Return result    Supervisor reviews                        │
│                       ↓                                     │
│                   [Satisfied?]                              │
│                     ↓ No                                    │
│                   Request revision                          │
│                     ↓ Yes                                   │
│  Add to completed_tasks[]                                   │
│    ↓                                                        │
│  Call result_callback (if set)                              │
│    ↓                                                        │
│  Return final result                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Parameters**:
- `max_thread_workers`: Maximum concurrent tasks (default: 5)
- `check_interval`: Interval for checking completed tasks (default: 1.0s)
- `_last_n_completed_tasks`: Number of completed tasks to keep in context (default: 10)

### 8. OMAN (Orchestrated Multi-Agent Network) Detailed

OMAN coordinates multiple MAS instances, each specializing in different domains:

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                    OMAN ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           OMAN SUPERVISOR                            │  │
│  │  - Routes queries to appropriate MAS                 │  │
│  │  - Maintains network-level memory                    │  │
│  │  - Coordinates cross-MAS communication               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ↓                ↓                ↓                │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │  MAS 1   │     │  MAS 2   │     │  MAS 3   │          │
│  │ (Finance)│     │(Research)│     │(Customer)│          │
│  │          │     │          │     │ Support  │          │
│  │ Agent A  │     │ Agent D  │     │ Agent G  │          │
│  │ Agent B  │     │ Agent E  │     │ Agent H  │          │
│  │ Agent C  │     │ Agent F  │     │ Agent I  │          │
│  └──────────┘     └──────────┘     └──────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Setup Example**:
```python
from masai.OMAN.oman import OrchestratedMultiAgentNetwork
from masai.MultiAgents.MultiAgent import MultiAgentSystem

# Create specialized MAS instances
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
    extra_context={"environment": "production", "company": "MyCompany"}
)

# Use OMAN
result = oman.delegate_task("Analyze Q4 financial performance")
```

**OMAN Routing Process**:
1. OMAN supervisor receives query
2. Analyzes query against each MAS's agent capabilities
3. Selects most appropriate MAS based on specialization
4. Delegates query to selected MAS
5. MAS processes query using its agents
6. Result returned through OMAN supervisor
7. Network memory updated with task outcome

**Use Cases**:
- **Enterprise Systems**: Multiple departments with specialized agents
- **Multi-Domain Applications**: Finance + Research + Support in one system
- **Scalable Architecture**: Add new MAS instances without modifying existing ones

---

## Troubleshooting & Best Practices

### 1. Tool Loop Prevention

**Problem**: Agent repeatedly calls the same tool without making progress

**Causes**:
- Tool returns insufficient information
- LLM doesn't recognize task completion
- Tool errors not properly handled

**Solutions**:
```python
# 1. Increase max_tool_loops
from masai.Config import config
config.max_tool_loops = 5  # Default: 3

# 2. Improve tool descriptions
@tool
def search_database(query: str) -> dict:
    """
    Searches database for records.

    Args:
        query: Search query string

    Returns:
        dict with 'count' (int) and 'results' (list) keys.
        Returns empty list if no results found.
    """
    # Implementation

# 3. Add error handling in tools
@tool
def api_call(endpoint: str) -> dict:
    """Makes API call with error handling"""
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 2. Reflection Counter Management

**Problem**: Agent gets stuck in reflection loops

**Solution**:
```python
# Adjust reflection limit
config.MAX_REFLECTION_COUNT = 5  # Default: 3

# Improve reflector temperature for more decisive answers
config_dict = {
    "reflector_temperature": 0.3  # Lower = more deterministic
}
```

### 3. Memory Optimization Strategies

**Strategy 1: Short Conversations**
```python
manager.create_agent(
    agent_name="quick_agent",
    tools=tools,
    agent_details=details,
    memory_order=5,
    long_context=False  # No summarization needed
)
```

**Strategy 2: Long Conversations with Summarization**
```python
manager.create_agent(
    agent_name="research_agent",
    tools=tools,
    agent_details=details,
    memory_order=10,
    long_context=True,
    long_context_order=30  # Keep 30 summaries
)
```

**Strategy 3: Very Long Conversations with Vector Store**
```python
from masai.Memory.InMemoryStore import InMemoryDocStore

memory_store = InMemoryDocStore(embedding_model="all-MiniLM-L6-v2")

manager.create_agent(
    agent_name="longterm_agent",
    tools=tools,
    agent_details=details,
    memory_order=10,
    long_context=True,
    long_context_order=20,
    in_memory_store=memory_store,
    top_k=5  # Retrieve 5 most relevant memories
)
```

### 4. When to Use long_context vs LTIMStore

| Feature | long_context | LTIMStore |
|---------|--------------|-----------|
| **Purpose** | Sequential conversation history | Semantic search over old conversations |
| **Storage** | List of summaries | Vector embeddings |
| **Retrieval** | All recent summaries | Top-k most relevant |
| **Best For** | Conversations with clear progression | Research tasks with topic jumps |
| **Overhead** | Low (summarization only) | Medium (embedding computation) |
| **Use When** | Conversation length < 100 messages | Conversation length > 100 messages |

### 5. Performance Tuning

**Reduce Latency**:
```python
# Use faster models for frequent operations
config_dict = {
    "router_model_name": "gemini-2.0-flash",  # Fast routing
    "evaluator_model_name": "gemini-2.0-flash",  # Fast evaluation
    "reflector_model_name": "gemini-pro"  # Quality reflection
}

# Reduce memory_order for faster context processing
memory_order=5  # Instead of 20

# Enable Redis caching for expensive tools
cache = ToolCache(timeout=60)
@tool
@cache.masai_cache
def expensive_tool(query: str) -> dict:
    # Implementation
```

**Reduce Costs**:
```python
# Use cheaper models where possible
config_dict = {
    "router_model_name": "gpt-3.5-turbo",
    "evaluator_model_name": "gpt-3.5-turbo",
    "reflector_model_name": "gpt-4"  # Only use expensive model where needed
}

# Reduce memory_order to minimize token usage
memory_order=3
long_context_order=10
```

### 6. Error Handling Patterns

**Pattern 1: Tool Error Handling**
```python
@tool
def robust_tool(query: str) -> dict:
    """Tool with comprehensive error handling"""
    try:
        result = perform_operation(query)
        return {"success": True, "data": result}
    except ValueError as e:
        return {"success": False, "error": f"Invalid input: {e}"}
    except ConnectionError as e:
        return {"success": False, "error": f"Connection failed: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
```

**Pattern 2: Agent Error Handling**
```python
try:
    agent = manager.get_agent("my_agent")
    result = await agent.initiate_agent("Query")

    if "error" in result:
        # Handle agent-level errors
        logger.error(f"Agent error: {result['error']}")
    else:
        # Process successful result
        print(result['answer'])

except ValueError as e:
    # Handle agent not found
    logger.error(f"Agent not found: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
```

---

## Summary

MAS-AI provides a comprehensive framework for building intelligent multi-agent systems with:

✅ **Modular Architecture**: Router, Evaluator, Reflector, Planner components
✅ **Flexible Memory**: 5-level hierarchy from short-term to vector store
✅ **Multiple Workflows**: Sequential, Hierarchical, Decentralized
✅ **LLM Agnostic**: Support for OpenAI, Gemini, Anthropic, Groq, Ollama, HuggingFace
✅ **Advanced Features**: Redis caching, streaming, dynamic context, per-component config
✅ **Scalable**: From single agents to OMAN networks

**Next Steps**:
1. Read [MASAI_PROMPT_TEMPLATES_AND_DATA_FLOW.md](./MASAI_PROMPT_TEMPLATES_AND_DATA_FLOW.md) for prompt details
2. Review [README.md](./README.md) for quick start examples
3. Check [MASAI_CONTEXT_MANAGEMENT_ANALYSIS.md](./MASAI_CONTEXT_MANAGEMENT_ANALYSIS.md) for context deep dive

---

**Last Updated**: 2025-01-04
**Framework Version**: Based on MAS-AI v0.1.24
**Documentation Version**: 2.0 (Comprehensive)