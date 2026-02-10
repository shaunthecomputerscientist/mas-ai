# MASAI - Multi-Agent System AI Framework

A powerful, production-ready framework for building multi-agent AI systems with advanced features like persistent memory, long-context management, and sophisticated agent orchestration.

|                                                                 |
|:---------------------------------------------------------------:|
|![MAS AI](/MAS/Logo/image.png)                                   |

> ⭐ **Please star this project if you find it useful!**

## 🆕 New Documentation

We've added comprehensive guides to help you get the most out of MASAI:

- **[Comprehensive Model Configuration Guide](docs/MODEL_CONFIG_COMPREHENSIVE.md)** - **NEW!** Complete parameter reference for OpenAI and Google Gemini with all supported parameters, examples, constraints, and troubleshooting
- **[Model Parameters Guide](docs/MODEL_PARAMETERS.md)** - Complete reference for all supported models (Gemini, OpenAI, Anthropic) with ALL parameters, examples, and best practices
- **[Tools Guide](docs/TOOLS_GUIDE.md)** - How to define tools, use them, implement Redis caching, and integrate with agents
- **[Singular Agent Guide](docs/SINGULAR_AGENT_GUIDE.md)** - Complete guide for single agent architecture, execution, memory, and tools
- **[Multi-Agent System Guide](docs/MULTIAGENT_SYSTEM_GUIDE.md)** - Comprehensive guide for decentralized and hierarchical multi-agent coordination
- **[OMAN Guide](docs/OMAN_GUIDE.md)** - Orchestrated Multi-Agent Network for enterprise-level multi-domain systems

---

## 📋 Quick Navigation

### Getting Started
| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICK_START.md) | Get started in 5 minutes |
| [Installation](docs/INSTALLATION.md) | Setup instructions and requirements |
| [Configuration](docs/CONFIGURATION.md) | Configuration options and setup |

### Core Concepts
| Document | Description |
|----------|-------------|
| [Framework Overview](docs/FRAMEWORK_OVERVIEW.md) | Architecture and design principles |
| **[Comprehensive Model Configuration](docs/MODEL_CONFIG_COMPREHENSIVE.md)** | **NEW!** Complete OpenAI & Gemini parameters with all constraints and examples |
| [Model Parameters](docs/MODEL_PARAMETERS.md) | Complete model configuration guide |
| [Tools Guide](docs/TOOLS_GUIDE.md) | Tool definition, usage, and caching |
| [Memory System](docs/MEMORY_SYSTEM.md) | Persistent memory and long-context management |

### Agent Systems
| Document | Description |
|----------|-------------|
| [Agent Manager Detailed](docs/AGENTMANAGER_DETAILED.md) | AgentManager API and usage |
| [Singular Agent Guide](docs/SINGULAR_AGENT_GUIDE.md) | **NEW!** Single agent architecture and usage |
| [Multi-Agent System Guide](docs/MULTIAGENT_SYSTEM_GUIDE.md) | **NEW!** Decentralized and hierarchical MAS |
| [OMAN Guide](docs/OMAN_GUIDE.md) | **NEW!** Orchestrated Multi-Agent Network |

### Advanced Topics
| Document | Description |
|----------|-------------|
| [Advanced Usage](docs/ADVANCED.md) | Expert patterns and techniques |
| [Multi-Agent Orchestration](docs/MULTI_AGENT_ORCHESTRATION.md) | Complex multi-agent workflows |
| [LangChain Agnostic Guide](docs/LANGCHAIN_AGNOSTIC_GUIDE.md) | Using MASAI without LangChain |

### Reference
| Document | Description |
|----------|-------------|
| [API Reference](docs/API_REFERENCE.md) | Complete API documentation |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [Usage Guide](docs/USAGE_GUIDE.md) | Common usage patterns |

---

## Quick Start

### Installation

```bash
pip install masai-framework
```

### Basic Usage

```python
from masai.AgentManager import AgentManager, AgentDetails
import asyncio

# Create agent manager
manager = AgentManager(user_id="user_123")

# Create an agent
agent = manager.create_agent(
    agent_name="assistant",
    tools=[],  # Add LangChain tools here
    agent_details=AgentDetails(
        capabilities=["analysis", "reasoning"],
        description="Helpful assistant",
        style="concise"
    )
)

# Use the agent - Full execution
result = await agent.initiate_agent(
    query="What is 2+2?",
    passed_from="user"
)
print(result["answer"])

# Or use streaming for real-time updates
async for state in agent.initiate_agent_astream(
    query="What is 2+2?",
    passed_from="user"
):
    # Unpack tuple: (mode, {node_name: state_dict})
    mode, update_data = state
    state_value = next(iter(update_data.values()))

    # Access state information
    print(f"Node: {state_value.get('current_node')}")
    if state_value.get("answer"):
        print(f"Answer: {state_value['answer']}")
```

**See [docs/QUICK_START.md](docs/QUICK_START.md) for detailed examples.**

---

## Core Features

### 🧠 Multi-Agent Architecture
- **Router-Evaluator-Reflector Pattern**: Sophisticated agent decision-making
- **Agent Orchestration**: Coordinate multiple agents for complex tasks
- **Tool Integration**: Seamless integration with LangChain tools
- **Streaming Support**: Real-time response streaming

### 💾 Persistent Memory
- **Redis Backend**: Fast vector storage with RediSearch
- **Qdrant Backend**: Distributed vector database support
- **User Isolation**: Multi-user support with automatic filtering
- **Deduplication**: Automatic duplicate detection and merging

### 🔄 Long-Context Management
- **Context Summarization**: Automatic summarization of long conversations
- **Memory Overflow Handling**: Intelligent flushing to persistent storage
- **Semantic Search**: Find relevant memories using embeddings
- **Category Filtering**: Organize memories by categories

### 🎯 Flexible Configuration
- **Multiple LLM Providers**: OpenAI, Google Gemini, Anthropic Claude
- **Custom Embeddings**: Support for any embedding model
- **Scalable Parameters**: Configure all model parameters via config
- **Component Customization**: Override any component behavior

### 🤝 Multi Agent Orchestration
- **Sequential Workflow**: Fixed agent pipeline
- **Hierarchical Workflow**: Supervisor-based delegation
- **Decentralized Workflow**: Peer-to-peer collaboration
- **Orchestrated Multi-Agent Network (OMAN)**: Coordinate multiple MAS instances
- **Data & Context Management**: Shared memory, context propagation, isolation
- **See [docs/MULTI_AGENT_ORCHESTRATION.md](docs/MULTI_AGENT_ORCHESTRATION.md) for details**

---

## 📚 Documentation

### Core Documentation
- **[Quick Start](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Architecture](docs/ARCHITECTURE.md)** - System design and components
- **[Memory System](docs/MEMORY_SYSTEM.md)** - Persistent memory and long-context management
- **[Installation](docs/INSTALLATION.md)** - Complete setup guide
- **[Configuration](docs/CONFIGURATION.md)** - All configuration options

### Advanced Documentation
- **[Multi-Agent Orchestration](docs/MULTI_AGENT_ORCHESTRATION.md)** - Sequential, hierarchical, decentralized, and OMAN patterns
- **[Data & Context Management](docs/DATA_CONTEXT_MANAGEMENT.md)** - Data flow, context sharing, and isolation
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Common usage patterns
- **[Advanced Topics](docs/ADVANCED.md)** - Expert patterns and customization
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation

---

## Architecture

### System Overview

```
User Application
       ↓
AgentManager (Orchestrator)
       ↓
Agent (Router-Evaluator-Reflector)
       ├─ MASGenerativeModel (LLM + Memory)
       ├─ Tool Executor
       └─ State Manager
       ↓
Memory System
       ├─ LongTermMemory
       ├─ Redis/Qdrant Backend
       └─ Embedding Model
```

**See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.**

---

## Installation

### Requirements
- Python 3.8+
- Redis (for persistent memory) or Qdrant
- API keys for LLM providers (OpenAI, Google, etc.)

### Setup

```bash
# Clone repository
git clone https://github.com/shaunthecomputerscientist/mas-ai.git
cd mas-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

**See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed setup.**

---

## Usage Guide

### Creating Agents

```python
agent = manager.create_agent(
    agent_name="research_agent",
    agent_details=AgentDetails(
        capabilities=["research", "analysis"],
        description="Research specialist",
        style="detailed"
    ),
    tools=[]  # Add LangChain tools here
)
```

### Executing Agent

```python
# Full execution
result = await agent.initiate_agent(
    query="Explain quantum computing",
    passed_from="user"
)
print(result["answer"])
print(f"Reasoning: {result['reasoning']}")
print(f"Satisfied: {result['satisfied']}")
```

### Streaming Responses

```python
async for state in agent.initiate_agent_astream(
    query="Tell me a story",
    passed_from="user"
):
    # Unpack tuple: (mode, {node_name: state_dict})
    mode, update_data = state
    state_value = next(iter(update_data.values()))

    # Access state information
    if state_value.get("answer"):
        print(state_value["answer"])
```

**See [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) for comprehensive examples.**

---

## Memory System

### Persistent Memory Setup

MASAI supports two vector database backends for persistent memory: **Redis** and **Qdrant**.

#### Option 1: Redis Backend

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.Memory.LongTermMemory import RedisConfig
from langchain_openai import OpenAIEmbeddings

# Configure Redis backend
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,  # Must match embedding model output
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    dedup_mode="similarity",  # Options: "none", "similarity", "hash"
    dedup_similarity_threshold=0.95
)

# Create manager with memory config
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json",
    memory_config=redis_config
)
```

#### Option 2: Qdrant Backend

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.Memory.LongTermMemory import QdrantConfig
from langchain_openai import OpenAIEmbeddings

# Configure Qdrant backend (local)
qdrant_config = QdrantConfig(
    url="http://localhost:6333",  # Qdrant server URL
    collection_name="masai_memories",
    vector_size=1536,  # Must match embedding model output
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    distance="cosine",  # Options: "cosine", "dot", "euclid"
    dedup_mode="similarity",  # Options: "none", "similarity", "hash"
    dedup_similarity_threshold=0.9
)

# For Qdrant Cloud
qdrant_cloud_config = QdrantConfig(
    url="https://your-cluster.qdrant.io",
    api_key="your-qdrant-api-key",
    collection_name="masai_memories",
    vector_size=1536,
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Create manager with memory config
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json",
    memory_config=qdrant_config  # or qdrant_cloud_config
)
```

#### Create Agent with Persistent Memory

```python
# Create agent with persistent memory enabled
agent = manager.create_agent(
    agent_name="assistant",
    agent_details=AgentDetails(
        capabilities=["reasoning"],
        description="Assistant"
    ),
    persist_memory=True,  # Enable persistence (requires memory_config in AgentManager)
    long_context=True,
    long_context_order=5  # Flush to persistent storage when summaries exceed this
)
```

### Memory Operations

```python
from masai.schema import Document

# Save memories manually (in addition to automatic overflow)
await agent.llm_router.long_term_memory.save(
    user_id="user_123",
    documents=[
        Document(page_content="User prefers dark mode"),
        "Plain string also works",
        {"page_content": "Dict format supported", "metadata": {"category": "preferences"}}
    ]
)

# Search memories with semantic similarity
memories = await agent.llm_router.long_term_memory.search(
    user_id="user_123",
    query="What are user preferences?",
    k=5,
    categories=["preferences"]  # Optional category filter
)

# Access via AgentManager (recommended - canonical reference)
await manager.long_term_memory.save(user_id="user_123", documents=[...])
await manager.long_term_memory.search(user_id="user_123", query="...", k=5)
```

### Memory Flow

Memory flows through two paths:
1. **Automatic**: When `context_summaries` exceeds `long_context_order`, overflow is flushed to persistent storage
2. **Manual**: Direct calls to `long_term_memory.save()` for on-demand persistence

**See [docs/MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md) for detailed memory docs.**

---

## Dynamic Context via Callables

MASAI supports dynamic context injection through **context callables** - functions that can provide real-time data to your agent based on user queries.

### Basic Concept

Context callables are functions that:
- Receive the user query as input
- Return additional context information
- Are called during inference to enrich the LLM prompt
- Support both user-level and node-level customization

### Pattern 1: User-Level Context (Simple)

User-level callables execute for all user queries:

```python
# Define context providers
def get_user_preferences(query: str) -> str:
    """Fetch user preferences based on query"""
    return "User prefers: dark mode, Python, verbose explanations"

def get_recent_history(query: str) -> str:
    """Fetch recent interaction history"""
    return "Recent: Discussed quantum computing, asked about Python libraries"

# Create agent with context_callable (single)
agent = manager.create_agent(
    agent_name="assistant",
    agent_details=AgentDetails(
        capabilities=["reasoning", "analysis"],
        description="Helpful assistant with user context"
    ),
    context_callable=get_user_preferences  # Single callable
)

# Or with list of callables (results combined)
agent = manager.create_agent(
    agent_name="assistant",
    agent_details=AgentDetails(
        capabilities=["reasoning", "analysis"],
        description="Helpful assistant with multiple contexts"
    ),
    context_callable=[get_user_preferences, get_recent_history]  # Multiple
)

# Query execution
result = await agent.initiate_agent(
    query="What should I build?",
    passed_from="user"
)
# LLM receives:
# - Original query
# - Results from get_user_preferences()
# - Results from get_recent_history()
```

### Pattern 2: Node-Level Context (Advanced)

Different callables for different agent components:

```python
# Define specialized context providers
def get_research_context(query: str) -> str:
    """Context for router (decision-making)"""
    return "Research sources available: Papers, Articles, Books"

def get_evaluation_context(query: str) -> str:
    """Context for evaluator (answer quality)"""
    return "Quality criteria: Accuracy, Completeness, Clarity"

def get_reflection_context(query: str) -> str:
    """Context for reflector (reasoning)"""
    return "Reasoning approach: Evidence-based, Multi-perspective"

# Create agent with callable_config (node-specific)
agent = manager.create_agent(
    agent_name="research_agent",
    agent_details=AgentDetails(
        capabilities=["research", "analysis"],
        description="Research specialist"
    ),
    callable_config={
        'router': get_research_context,        # Single callable for router
        'evaluator': get_evaluation_context,   # Single callable for evaluator
        'reflector': get_reflection_context    # Single callable for reflector
    }
)

# Query execution
result = await agent.initiate_agent(
    query="Research AI safety",
    passed_from="user"
)
# Each node receives its specific context
```

### Pattern 3: Mixed Approach (User + Node Level)

Combine user-level and node-specific contexts:

```python
# User-level context (all nodes)
def get_global_context(query: str) -> dict:
    return "Global: All documentation available"

# Node-specific contexts
def router_context(query: str) -> str:
    return "Router: Focus on task decomposition"

def evaluator_context(query: str) -> str:
    return "Evaluator: Check against requirements"

def reflector_context(query: str) -> list:
    return ["Reflector: Consider edge cases", "Reflector: Verify logic"]

# Create agent with both
agent = manager.create_agent(
    agent_name="hybrid_agent",
    agent_details=AgentDetails(
        capabilities=["analysis", "validation"],
        description="Hybrid agent"
    ),
    context_callable=get_global_context,  # User-level (all nodes)
    callable_config={                     # Node-specific (overrides)
        'router': router_context,
        'evaluator': evaluator_context,
        'reflector': reflector_context
    }
)
```

### Pattern 4: Multiple Callables per Node

List of callables for each node (results combined with newlines):

```python
def preference_1(query: str) -> str:
    return "Preference 1: Be concise"

def preference_2(query: str) -> str:
    return "Preference 2: Use examples"

def preference_3(query: str) -> str:
    return "Preference 3: Explain trade-offs"

# Multiple callables per node
agent = manager.create_agent(
    agent_name="detailed_agent",
    agent_details=AgentDetails(
        capabilities=["explanation"],
        description="Detailed agent with multiple preferences"
    ),
    context_callable=[preference_1, preference_2, preference_3]  # List for user level
)

# Or with node-specific lists
agent = manager.create_agent(
    agent_name="detailed_agent",
    agent_details=AgentDetails(
        capabilities=["explanation"],
        description="Detailed agent with node-specific lists"
    ),
    callable_config={
        'router': [preference_1, preference_2],           # List for router
        'evaluator': preference_3,                        # Single for evaluator
        'reflector': [preference_2, preference_3]         # List for reflector
    }
)
```

### Pattern 5: Real-World Example

Database/API context for agents:

```python
import aiohttp

async def fetch_user_data(query: str) -> str:
    """Fetch user data from API"""
    # Simulate API call
    return "User: John, Tier: Premium, Usage: 80%"

async def fetch_system_status(query: str) -> str:
    """Fetch system status"""
    return "System: All services operational"

async def fetch_knowledge_base(query: str) -> str:
    """Search knowledge base"""
    return "Found: 5 relevant articles on the topic"

# Create agent with async callables
agent = manager.create_agent(
    agent_name="support_agent",
    agent_details=AgentDetails(
        capabilities=["support", "troubleshooting"],
        description="Support agent with live data"
    ),
    context_callable=[
        fetch_user_data,
        fetch_system_status,
        fetch_knowledge_base
    ]
)

# Query execution
result = await agent.initiate_agent(
    query="Why is my service slow?",
    passed_from="user"
)
# Agent receives live user data, system status, and KB results
```

### Context Callable Return Types

Callables can return different types - all are converted to strings:

```python
def returns_string(query: str) -> str:
    return "Context as string"

def returns_dict(query: str) -> dict:
    return {"key": "value", "info": "data"}

def returns_list(query: str) -> list:
    return ["Item 1", "Item 2", "Item 3"]

def returns_number(query: str) -> int:
    return 42

# All work - converted to string representation
agent = manager.create_agent(
    agent_name="flexible_agent",
    agent_details=AgentDetails(
        capabilities=["reasoning"],
        description="Agent with flexible return types"
    ),
    context_callable=[
        returns_string,
        returns_dict,
        returns_list,
        returns_number
    ]
)
```

### When Callables Are Invoked

Important: Context callables are only invoked:
- When `passed_from="user"` (user query)
- NOT for agent-to-agent delegation
- NOT for internal node-to-node processing

```python
# User query → callables invoked
result = await agent.initiate_agent(
    query="What is AI?",
    passed_from="user"  # ✅ Callables called
)

# Agent-to-agent → callables NOT invoked
result = await agent.initiate_agent(
    query="Analyze this",
    passed_from="agent1"  # ❌ Callables NOT called
)
```

### Best Practices

```python
# ✅ DO: Keep callables fast (< 100ms)
def fast_context(query: str) -> str:
    return cached_data.get(query)

# ❌ DON'T: Slow blocking operations
def slow_context(query: str) -> str:
    time.sleep(5)  # Bad for latency
    return fetch_from_slow_api()

# ✅ DO: Use async for I/O operations
async def async_context(query: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

# ✅ DO: Error handling
def safe_context(query: str) -> str:
    try:
        return fetch_data()
    except Exception as e:
        return f"Context unavailable: {str(e)}"

# ✅ DO: Query-specific context
def smart_context(query: str) -> str:
    if "price" in query.lower():
        return get_pricing_info()
    elif "api" in query.lower():
        return get_api_docs()
    else:
        return get_general_info()
```

---

## Configuration

### Agent Creation Parameters

When creating agents with `manager.create_agent()`, you can configure these parameters:

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | str | Required | Unique identifier for the agent |
| `agent_details` | AgentDetails | Required | Agent capabilities, description, and style |
| `tools` | list | [] | LangChain tools available to agent |

#### Memory Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `persist_memory` | bool | None | Enable persistent memory (requires memory_config in AgentManager) |
| `long_context` | bool | True | Enable long-context summarization for long conversations |
| `long_context_order` | int | 20 | Flush to storage when context summaries exceed this count |
| `memory_order` | int | 10 | Number of recent messages to keep in memory |

#### LLM Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.2 | Sampling temperature (0.0-1.0) |
| `plan` | bool | False | Enable planner component |
| `character_factor` | int | None | Character-level truncation factor |
| `config_dict` | dict | {} | Node-specific config (e.g., `{'evaluator_streaming': False}`) |

#### Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_callable` | Callable or List[Callable] | None | User-level context function(s) - called for all user queries |
| `callable_config` | Dict | None | Node-specific context mapping (router, evaluator, reflector) |

#### Execution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tool_output_words` | int | 3000 | Maximum words for tool output |
| `retain_messages_order` | int | 10 | Messages to retain in execution |

### Complete Agent Creation Example

```python
from masai.AgentManager import AgentManager, AgentDetails

manager = AgentManager(user_id="user_123", model_config_path="model_config.json")

# Define context providers
def get_user_context(query: str) -> str:
    return "User preferences and settings"

def get_router_context(query: str) -> str:
    return "Task routing guidelines"

def get_evaluator_context(query: str) -> str:
    return "Quality evaluation criteria"

def get_reflector_context(query: str) -> list:
    return ["Reasoning checkpoints", "Logic validation"]

# Create fully configured agent
agent = manager.create_agent(
    # Core
    agent_name="assistant",
    agent_details=AgentDetails(
        capabilities=["reasoning", "analysis", "research"],
        description="Intelligent assistant with context awareness",
        style="detailed and thoughtful"
    ),
    tools=[],  # Add LangChain tools here
    
    # Memory
    persist_memory=True,
    long_context=True,
    long_context_order=5,
    memory_order=10,
    
    # LLM
    temperature=0.7,
    plan=False,
    character_factor=20,
    config_dict={
        'evaluator_streaming': False,
        'reflector_streaming': False
    },
    
    # Context
    context_callable=get_user_context,  # User-level context
    callable_config={                   # Node-specific context
        'router': get_router_context,
        'evaluator': get_evaluator_context,
        'reflector': get_reflector_context
    },
    
    # Execution
    max_tool_output_words=5000,
    retain_messages_order=15
)

# Use the agent
result = await agent.initiate_agent(
    query="Analyze this problem",
    passed_from="user"
)
```

---

## Agent Creation Patterns

### Pattern 1: Minimal Agent (In-Memory Only)

```python
agent = manager.create_agent(
    agent_name="simple_agent",
    agent_details=AgentDetails(
        capabilities=["basic_qa"],
        description="Simple QA agent"
    )
)
```

### Pattern 2: Agent with Persistent Memory

```python
# Requires memory_config in AgentManager
agent = manager.create_agent(
    agent_name="persistent_agent",
    agent_details=AgentDetails(
        capabilities=["learning"],
        description="Learns from conversations"
    ),
    persist_memory=True,
    long_context=True,
    long_context_order=10
)
```

### Pattern 3: Agent with User Context

```python
def get_user_profile(query: str) -> str:
    return "User: Premium member, 100+ interactions, prefers technical explanations"

agent = manager.create_agent(
    agent_name="context_agent",
    agent_details=AgentDetails(
        capabilities=["personalized_qa"],
        description="Personalized assistant"
    ),
    context_callable=get_user_profile
)
```

### Pattern 4: Agent with Node-Specific Context

```python
def router_context(query: str) -> str:
    return "Available tools: calculator, search, database"

def evaluator_context(query: str) -> str:
    return "Quality: accuracy > completeness > speed"

agent = manager.create_agent(
    agent_name="specialized_agent",
    agent_details=AgentDetails(
        capabilities=["tool_use", "reasoning"],
        description="Tool-aware agent"
    ),
    callable_config={
        'router': router_context,
        'evaluator': evaluator_context
    }
)
```

### Pattern 5: Agent with Multiple Context Providers

```python
def get_database_context(query: str) -> str:
    # Query relevant database records
    return "DB: Found 3 matching records"

def get_api_context(query: str) -> str:
    # Query relevant APIs
    return "API: Data available from 2 services"

def get_cache_context(query: str) -> str:
    # Check cache
    return "Cache: 5 cached responses available"

agent = manager.create_agent(
    agent_name="data_agent",
    agent_details=AgentDetails(
        capabilities=["data_retrieval", "synthesis"],
        description="Data-aware agent"
    ),
    context_callable=[get_database_context, get_api_context, get_cache_context]
)
```

### Pattern 6: Agent with Full Configuration

```python
def user_context(query: str) -> str:
    return "User: Admin, Full access, Prefers detailed responses"

def router_tasks(query: str) -> str:
    return "Available: research, code, design, planning"

def evaluator_criteria(query: str) -> list:
    return [
        "Completeness: Answer all parts of query",
        "Accuracy: Verify with sources",
        "Clarity: Explain technical terms"
    ]

def reflector_checks(query: str) -> str:
    return "Check: Logic consistency, No contradictions, Evidence-based"

agent = manager.create_agent(
    # Identity
    agent_name="enterprise_agent",
    agent_details=AgentDetails(
        capabilities=["enterprise_ai", "advanced_reasoning", "complex_analysis"],
        description="Enterprise-grade AI assistant with full context awareness",
        style="professional, thorough, evidence-based"
    ),
    tools=[],  # Add enterprise tools
    
    # Memory Configuration
    persist_memory=True,
    long_context=True,
    long_context_order=15,
    memory_order=20,
    
    # LLM Configuration
    temperature=0.6,
    plan=True,
    character_factor=30,
    config_dict={
        'evaluator_streaming': False,
        'reflector_streaming': False,
        'router_streaming': True
    },
    
    # Context Injection
    context_callable=user_context,
    callable_config={
        'router': router_tasks,
        'evaluator': evaluator_criteria,
        'reflector': reflector_checks
    },
    
    # Execution Control
    max_tool_output_words=8000,
    retain_messages_order=25
)
```

---

## Agent Details Configuration

The `AgentDetails` class defines agent personality and capabilities:

```python
from masai.AgentManager import AgentDetails

details = AgentDetails(
    # List of capabilities (string descriptions)
    capabilities=[
        "complex reasoning",
        "code generation",
        "mathematical analysis",
        "research synthesis"
    ],
    
    # Longer description of the agent's role
    description="""
    An advanced AI assistant capable of handling complex tasks.
    You excel at breaking down problems, conducting research,
    and providing well-reasoned solutions with code examples.
    """,
    
    # Style/personality guidance for responses
    style="detailed, professional, with code examples and explanations"
)

agent = manager.create_agent(
    agent_name="advanced_assistant",
    agent_details=details
)
```

---

## Advanced Agent Creation

### Custom Tool Integration

```python
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions"""
    return str(eval(expression))

agent = manager.create_agent(
    agent_name="calculator",
    tools=[calculate]
)
```

### Custom Tool Integration with Context

```python
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant articles"""
    return f"Found 5 articles about {query}"

def kb_context(query: str) -> str:
    """Provide KB search context"""
    return search_knowledge_base(query)

agent = manager.create_agent(
    agent_name="kb_agent",
    agent_details=AgentDetails(
        capabilities=["knowledge_search", "synthesis"],
        description="Agent with KB integration"
    ),
    tools=[search_knowledge_base],
    context_callable=kb_context
)
```

### Streaming with Dynamic Context

```python
# Stream responses with dynamic context injection
async for state in agent.initiate_agent_astream(
    query="What is quantum computing?",
    passed_from="user"
):
    mode, update_data = state
    state_value = next(iter(update_data.values()))
    
    # Streaming includes context from callables
    if state_value.get("answer"):
        print(state_value["answer"], end="", flush=True)

print("\n")  # Final newline
```

### Memory with Context Callables

When using persistent memory with context callables:

```python
def get_learning_context(query: str) -> str:
    """Provide learning context from previous interactions"""
    return "Previously learned: Python, Data Science, ML Basics"

agent = manager.create_agent(
    agent_name="learning_agent",
    agent_details=AgentDetails(
        capabilities=["learning", "adaptation"],
        description="Agent that learns from interactions"
    ),
    persist_memory=True,      # Remember past interactions
    long_context=True,        # Summarize long conversations
    context_callable=get_learning_context  # Reference past learning
)

# Each query enriched with:
# 1. Past interaction summaries (from long_term_memory)
# 2. Learning context (from context_callable)
# 3. Recent messages (from memory_order)
```

### Multi-Agent Orchestration

```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig

# Decentralized MAS (peer-to-peer)
mas = MultiAgentSystem(agentManager=manager)
result = await mas.initiate_decentralized_mas(
    query="Complex task",
    set_entry_agent=agent1,
    memory_order=3
)

# Hierarchical MAS (supervisor-based)
supervisor_config = SupervisorConfig(
    model_name="gpt-4o",
    temperature=0.7,
    model_category="openai",
    memory_order=20,
    memory=True,
    extra_context={}
)

mas_hierarchical = MultiAgentSystem(
    agentManager=manager,
    supervisor_config=supervisor_config
)
result = await mas_hierarchical.initiate_hierarchical_mas(query="Complex task")
```

**See [docs/ADVANCED.md](docs/ADVANCED.md) for advanced patterns.**

---

## API Reference

### Core Classes

- **AgentManager**: Orchestrates agent creation and management
- **Agent**: Router-Evaluator-Reflector architecture
- **MASGenerativeModel**: LLM with memory management
- **LongTermMemory**: Persistent memory interface
- **RedisConfig/QdrantConfig**: Backend configuration

**See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete API.**

---

## Troubleshooting

### Redis Connection Refused
```bash
redis-server
# or
docker run -d -p 6379:6379 redis:latest
```

### Qdrant Connection Refused
```bash
# Local Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:latest
```

### OpenAI API Key Not Found
```bash
export OPENAI_API_KEY="your-key-here"
```

### Memory Not Being Retrieved
```python
# Verify context overflow (access through LLM component)
print(f"Summaries: {len(agent.llm_router.context_summaries)}")
print(f"Long context order: {agent.llm_router.long_context_order}")
print(f"Persist memory: {agent.llm_router.persist_memory}")
print(f"Long term memory: {agent.llm_router.long_term_memory}")
```

**See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.**

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issues](https://github.com/shaunthecomputerscientist/mas-ai/issues)
- 💬 [Discussions](https://github.com/shaunthecomputerscientist/mas-ai/discussions)

---

**Last Updated**: October 31, 2025

