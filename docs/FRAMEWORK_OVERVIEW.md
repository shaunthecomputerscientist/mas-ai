# MASAI Framework Overview

## What is MASAI?

**MASAI** (Multi-Agent System AI) is a sophisticated, production-ready framework for building intelligent multi-agent systems with advanced memory management, state orchestration, and tool integration.

Unlike traditional LLM frameworks, MASAI is **LangChain-agnostic** and provides explicit separation of concerns through specialized components.

---

## Key Principle: LangChain Agnosticism

### Where LangChain is Used

MASAI **only uses LangChain for embeddings** (optional). This means:

✅ **Embeddings**: Optional LangChain embeddings (OpenAI, Hugging Face, etc.)
✅ **Custom Embeddings**: Use any embedding function
✅ **No LangChain Dependency**: Everything else is MASAI-native

### What MASAI Provides

- ✅ **Own Document Class**: `masai.schema.Document` (not LangChain's)
- ✅ **Own Memory System**: Redis/Qdrant backends
- ✅ **Own Agent Architecture**: LangGraph-based state machine
- ✅ **Own Tool System**: Tool integration without LangChain
- ✅ **Own Prompt System**: Custom prompt templates

---

## Architecture Overview

### Component Hierarchy

```
AgentManager (Central Registry)
    ├── Agent (LangGraph StateGraph Wrapper)
    │   ├── llm_router (MASGenerativeModel)
    │   ├── llm_evaluator (MASGenerativeModel)
    │   ├── llm_reflector (MASGenerativeModel)
    │   └── llm_planner (MASGenerativeModel)
    │
    └── LongTermMemory (Shared)
        ├── Redis Backend
        └── Qdrant Backend
```

### Component Roles

#### 1. **AgentManager**
- Central registry for all agents
- Manages shared memory (LongTermMemory)
- Loads model configurations
- Creates and coordinates agents

#### 2. **Agent**
- LangGraph StateGraph wrapper (NOT an LLM)
- Orchestrates Router → Evaluator → Reflector → Planner flow
- Manages tool execution
- Handles state transitions

#### 3. **MASGenerativeModel**
- Actual LLM wrapper with memory
- Manages chat_history, context_summaries, long_term_memory
- Handles streaming and callbacks
- Implements memory truncation and summarization

#### 4. **Router**
- First decision point
- Routes query to appropriate tools/agents
- Fast and lightweight

#### 5. **Evaluator**
- Evaluates router's decision
- Checks if answer is satisfactory
- Decides whether to continue or escalate

#### 6. **Reflector**
- Deep reasoning and reflection
- Provides comprehensive analysis
- Handles complex reasoning tasks

#### 7. **Planner** (Optional)
- Plans multi-step workflows
- Coordinates complex tasks
- Manages task dependencies

---

## Memory Hierarchy

### Level 1: Chat History (In-Memory)
```python
agent.llm_router.chat_history  # Recent messages
# Size: memory_order (default: 10)
```

### Level 2: Context Summaries (In-Memory)
```python
agent.llm_router.context_summaries  # Summarized older messages
# Size: long_context_order (default: 20)
```

### Level 3: Long-Term Memory (Persistent)
```python
agent.llm_router.long_term_memory.search(...)  # Redis/Qdrant
# Unlimited size, persistent across sessions
```

---

## MASAI's Document Class

### Why MASAI Has Its Own Document Class

```python
# ✅ MASAI Document (masai.schema.Document)
from masai.schema import Document

doc = Document(
    page_content="Hello world",
    metadata={"source": "test", "user_id": "user_123"}
)

# ❌ LangChain Document (not used in MASAI)
from langchain.schema import Document  # Don't use this
```

### Key Differences

| Feature | MASAI Document | LangChain Document |
|---------|---|---|
| Import | `masai.schema.Document` | `langchain.schema.Document` |
| Dependency | None (Pydantic only) | Requires LangChain |
| Metadata | Full support | Limited support |
| User Isolation | Built-in | Not built-in |
| Serialization | to_dict(), from_dict() | Limited |

---

## Tool Integration

### MASAI Tools

```python
from masai.Tools import Tool

class SearchTool(Tool):
    name = "search"
    description = "Search the web"
    
    def execute(self, query: str) -> str:
        # Your implementation
        return results
```

### No LangChain Required

Tools work independently without LangChain dependencies.

---

## Multi-Agent Patterns

### 1. Sequential
```
Agent1 → Agent2 → Agent3
```
Fixed pipeline, each agent processes output of previous.

### 2. Hierarchical
```
Supervisor
├── Specialist1
├── Specialist2
└── Specialist3
```
Supervisor delegates to specialists.

### 3. Decentralized
```
Agent1 ↔ Agent2
  ↕       ↕
Agent3 ↔ Agent4
```
Peer-to-peer collaboration.

### 4. OMAN (Orchestrated Multi-Agent Network)
Large-scale orchestration of multiple MAS instances.

---

## When to Use MASAI

### ✅ Use MASAI When You Need

- Multi-agent systems with explicit component separation
- Advanced memory management (short-term + long-term)
- State machine orchestration
- Tool-augmented agents
- Persistent memory across sessions
- User isolation and multi-tenancy
- Complex reasoning workflows
- Production-grade reliability

### ❌ Don't Use MASAI When You Need

- Simple single-agent chatbot (use LangChain directly)
- No memory requirements
- Minimal dependencies
- Quick prototyping only

---

## Quick Start

### 1. Installation

```bash
pip install masai-framework
```

### 2. Basic Setup

```python
from masai.AgentManager import AgentManager, AgentDetails

manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json"
)

agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["reasoning"],
        description="General assistant"
    )
)
```

### 3. Execute Query

```python
result = await agent.initiate_agent(
    query="What is AI?",
    passed_from="user"
)
print(result["answer"])
```

---

## Key Concepts

### Memory Order Parameters

- **memory_order**: Short-term memory size (default: 10)
- **long_context_order**: Summaries before flush (default: 20)
- **shared_memory_order**: Component shared memory (default: 10)
- **retain_messages_order**: Internal state retention (default: 10)

### Persistence

- **persist_memory**: Enable persistent storage (requires memory_config)
- **memory_config**: Redis or Qdrant configuration

### User Isolation

- **user_id**: Isolate memories per user
- **categories_resolver**: Custom categorization

---

## Next Steps

1. **[MODEL_CONFIG_GUIDE.md](MODEL_CONFIG_GUIDE.md)** - Configure LLM models
2. **[AGENTMANAGER_DETAILED.md](AGENTMANAGER_DETAILED.md)** - Understand AgentManager
3. **[AGENT_DETAILED.md](AGENT_DETAILED.md)** - Learn Agent API
4. **[MEMORY_ARCHITECTURE_DEEP_DIVE.md](MEMORY_ARCHITECTURE_DEEP_DIVE.md)** - Master memory
5. **[REDIS_QDRANT_CRUD_GUIDE.md](REDIS_QDRANT_CRUD_GUIDE.md)** - Use persistent storage

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    AgentManager                         │
│  (Central Registry, Config Loading, Memory Management)  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼────────────┐    ┌──────▼──────────┐
    │     Agent      │    │ LongTermMemory  │
    │  (LangGraph)   │    │ (Redis/Qdrant)  │
    └───┬────────────┘    └─────────────────┘
        │
    ┌───┴──────────────────────────────────┐
    │                                      │
┌───▼──────┐ ┌──────────┐ ┌──────────┐ ┌─▼────────┐
│  Router  │ │Evaluator │ │Reflector │ │ Planner  │
│(MASGenM) │ │(MASGenM) │ │(MASGenM) │ │(MASGenM) │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
              │
         ┌────▼─────┐
         │   Tools  │
         └──────────┘
```

---

## See Also

- [QUICK_START.md](QUICK_START.md) - Get started in 5 minutes
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Common usage patterns
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [LANGCHAIN_AGNOSTIC_GUIDE.md](LANGCHAIN_AGNOSTIC_GUIDE.md) - LangChain independence

