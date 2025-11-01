# MASAI Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   AgentManager          │
        │  (Orchestrator)         │
        │  - Agent creation       │
        │  - Config management    │
        │  - Memory coordination  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────────────────┐
        │   Agent (LangGraph)                 │
        │  ┌──────────────────────────────┐   │
        │  │ Router Node                  │   │
        │  │ (Decision making)            │   │
        │  └──────────────────────────────┘   │
        │  ┌──────────────────────────────┐   │
        │  │ Evaluator Node               │   │
        │  │ (Tool output evaluation)     │   │
        │  └──────────────────────────────┘   │
        │  ┌──────────────────────────────┐   │
        │  │ Reflector Node               │   │
        │  │ (Self-reflection)            │   │
        │  └──────────────────────────────┘   │
        │  ┌──────────────────────────────┐   │
        │  │ Tool Executor                │   │
        │  │ (LangChain tools)            │   │
        │  └──────────────────────────────┘   │
        └────────────┬────────────────────────┘
                     │
        ┌────────────▼────────────────────────┐
        │   MASGenerativeModel                │
        │  - LLM wrapper                      │
        │  - Memory management                │
        │  - Context handling                 │
        └────────────┬────────────────────────┘
                     │
        ┌────────────▼────────────────────────┐
        │   Memory System                     │
        │  ┌──────────────────────────────┐   │
        │  │ LongTermMemory               │   │
        │  │ - Search interface           │   │
        │  │ - Save interface             │   │
        │  └──────────────────────────────┘   │
        │  ┌──────────────────────────────┐   │
        │  │ Backend Adapter              │   │
        │  │ - Redis (RediSearch)         │   │
        │  │ - Qdrant (Vector DB)         │   │
        │  └──────────────────────────────┘   │
        │  ┌──────────────────────────────┐   │
        │  │ Embedding Model              │   │
        │  │ - OpenAI                     │   │
        │  │ - HuggingFace                │   │
        │  │ - Custom                     │   │
        │  └──────────────────────────────┘   │
        └──────────────────────────────────────┘
```

## Component Details

### 1. AgentManager

**Responsibility**: Orchestrate agent creation and configuration

**Key Methods**:
- `create_agent()`: Create new agent with configuration
- `get_agent()`: Retrieve existing agent
- `list_agents()`: List all agents

**Features**:
- Centralized configuration management
- Shared memory across agents
- User isolation

### 2. Agent (LangGraph)

**Architecture**: Router-Evaluator-Reflector pattern

**Nodes**:
1. **Router**: Initial decision-making
   - Analyzes user query
   - Decides if tool is needed
   - Routes to appropriate node

2. **Evaluator**: Tool output evaluation
   - Evaluates tool results
   - Decides next action
   - Checks satisfaction

3. **Reflector**: Self-reflection
   - Analyzes reasoning
   - Identifies issues
   - Suggests improvements

4. **Tool Executor**: Execute LangChain tools/MASAI Tools
   - Runs selected tool
   - Handles errors
   - Returns results

**State Management**:
- Maintains conversation state
- Tracks tool usage
- Manages reflection counter

### 3. MASGenerativeModel

**Responsibility**: LLM wrapper with memory management

**Key Features**:
- Long-context management
- Memory summarization
- Context overflow handling
- Streaming support

**Methods**:
- `initiate_agent()`: Execute agent fully
- `initiate_agent_astream()`: Stream agent execution
- `set_context()`: Set agent context
- `generate_response_mas()`: Internal LLM call (used by nodes)
- `astream_response_mas()`: Internal streaming LLM call

### 4. Memory System

#### LongTermMemory

**Interface**:
```python
class LongTermMemory:
    async def save(user_id, documents) -> None
    async def search(user_id, query, k) -> List[Document]
    async def delete(user_id, doc_id) -> None
```

#### Backend Adapters

**Redis Adapter**:
- Uses RediSearch for vector search
- Fast in-memory storage
- Automatic indexing
- TTL support

**Qdrant Adapter**:
- Distributed vector database
- Persistent storage
- Advanced filtering
- Scalable

#### Embedding Models

Supported:
- OpenAI (text-embedding-3-small)
- HuggingFace (all-MiniLM-L6-v2)
- Custom callable functions

## Data Flow

### Query Processing

```
User Query
    ↓
agent.initiate_agent(query) or agent.initiate_agent_astream(query)
    ↓
Planner Node (optional)
    ├─ Create action plan
    └─ Route to Router
    ↓
Router Node
    ├─ Analyze query
    ├─ Retrieve memories (if context overflow)
    ├─ Decide if tool needed
    └─ Route to Tool Executor or Reflector
    ↓
Tool Executor (if needed)
    ├─ Execute selected tool
    ├─ Handle errors
    └─ Return results
    ↓
Evaluator Node
    ├─ Evaluate tool results
    ├─ Check satisfaction
    └─ Route to next node
    ↓
Reflector Node (if needed)
    ├─ Analyze reasoning
    ├─ Identify issues
    └─ Suggest improvements
    ↓
Return Response (Dict with answer, reasoning, etc.)
```

### Memory Flow

```
User Query
    ↓
MASGenerativeModel.generate_response_mas()
    ↓
_load_long_context_docs()
    ├─ Load context_summaries
    └─ Search long-term memory (if overflow)
    ↓
Generate Response
    ↓
_update_long_context_background()
    ├─ Summarize chat history
    ├─ Check if overflow
    └─ Flush to long-term memory (if needed)
    ↓
Store in Redis/Qdrant
```

## Configuration Hierarchy

```
model_config.json (highest priority)
    ↓
AgentManager config_dict
    ↓
Agent-specific config
    ↓
Component defaults (lowest priority)
```

## Scalability

### Horizontal Scaling
- Multiple agents per user
- Multiple users per system
- Distributed memory backends

### Vertical Scaling
- Larger context windows
- More tools per agent
- Complex agent workflows

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Agent creation | O(1) | Constant time |
| Query processing | O(n) | Linear in context size |
| Memory search | O(log n) | Vector search |
| Memory save | O(1) | Constant time |
| Context summarization | O(n) | Linear in messages |

## Security Considerations

1. **User Isolation**: All queries filtered by user_id
2. **API Key Management**: Environment variables only
3. **Memory Encryption**: Optional at backend level
4. **Tool Sandboxing**: Tools run in isolated context

## Extension Points

1. **Custom Tools**: Add LangChain tools
2. **Custom Embeddings**: Implement embedding interface
3. **Custom Backends**: Implement adapter interface
4. **Custom LLMs**: Use any LangChain LLM

See [ADVANCED.md](ADVANCED.md) for extension examples.