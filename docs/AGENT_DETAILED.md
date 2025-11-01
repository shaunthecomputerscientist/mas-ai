# Agent - Comprehensive Guide

## Overview

**Agent** is a LangGraph StateGraph wrapper that orchestrates the Router → Evaluator → Reflector → Planner workflow. It is **NOT an LLM wrapper** - it coordinates multiple LLM components.

---

## Architecture

### Agent is NOT an LLM

```python
# ❌ WRONG - Agent is not an LLM
agent.generate_response_mas(...)  # This doesn't exist

# ✅ CORRECT - Agent orchestrates components
result = await agent.initiate_agent(query="...", passed_from="user")
```

### Component Access

```python
# Access LLM components through Agent
agent.llm_router        # Router LLM (MASGenerativeModel)
agent.llm_evaluator     # Evaluator LLM (MASGenerativeModel)
agent.llm_reflector     # Reflector LLM (MASGenerativeModel)
agent.llm_planner       # Planner LLM (MASGenerativeModel)

# Access memory through components
agent.llm_router.chat_history
agent.llm_router.context_summaries
agent.llm_router.long_term_memory
```

---

## State Machine Flow

### Basic Flow

```
User Query
    ↓
[Router] - Route to tools/agents
    ↓
[Evaluator] - Evaluate router's decision
    ├─ Satisfied? → Return answer
    └─ Not satisfied? → Continue
    ↓
[Reflector] - Deep reasoning
    ├─ Satisfied? → Return answer
    └─ Not satisfied? → Continue
    ↓
[Planner] (optional) - Plan multi-step workflow
    ↓
Return final answer
```

### State Transitions

```python
# State structure
state = {
    "question": "User query",
    "answer": "Current answer",
    "messages": [...],
    "reasoning": "Why this answer",
    "satisfied": False,
    "current_node": "router",
    "tool_output": "Tool result",
    "delegate_to_agent": None
}
```

---

## initiate_agent() Method

### Signature

```python
async def initiate_agent(
    query: str,
    passed_from: Optional[str] = None,
    previous_node: Optional[str] = None
) -> Dict[str, Any]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | str | User question/prompt |
| `passed_from` | str | Source identifier (e.g., "user", "agent_name") |
| `previous_node` | str | Previous node in workflow (for multi-agent) |

### Return Value

```python
{
    "answer": "The final response",
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "reasoning": "Why this answer was chosen",
    "satisfied": True,
    "current_node": "reflector",
    "tool_output": "Output from executed tool",
    "delegate_to_agent": None  # or agent name if delegated
}
```

### Example

```python
result = await agent.initiate_agent(
    query="What is machine learning?",
    passed_from="user"
)

print(result["answer"])
print(f"Reasoning: {result['reasoning']}")
print(f"Satisfied: {result['satisfied']}")
```

---

## initiate_agent_astream() Method

### Signature

```python
async def initiate_agent_astream(
    query: str,
    passed_from: Optional[str] = None,
    previous_node: Optional[str] = None
) -> AsyncIterator[Tuple[str, Dict[str, Any]]]
```

### Streaming Pattern

```python
async for state in agent.initiate_agent_astream(
    query="Tell me about AI",
    passed_from="user"
):
    # state is a tuple: (node_name, state_dict)
    node_name, state_dict = state
    
    # Extract state value
    state_value = [v for k, v in state_dict.items()][0]
    
    # Process state
    print(f"Node: {state_value['current_node']}")
    if state_value.get("answer"):
        print(f"Answer: {state_value['answer']}")
```

### Real-World Example

```python
async for state in agent.initiate_agent_astream(
    query="Research quantum computing",
    passed_from="user"
):
    node_name, state_dict = state
    state_value = [v for k, v in state_dict.items()][0]
    
    current_node = state_value.get("current_node")
    
    if current_node == "router":
        print("🔀 Routing query...")
    elif current_node == "evaluator":
        print("✓ Evaluating response...")
    elif current_node == "reflector":
        print("🧠 Deep reasoning...")
    elif current_node == "planner":
        print("📋 Planning workflow...")
    
    if state_value.get("tool_output"):
        print(f"Tool output: {state_value['tool_output'][:100]}...")
    
    if state_value.get("satisfied"):
        print(f"✅ Final answer: {state_value['answer']}")
```

---

## set_context() Method

### Signature

```python
def set_context(
    context: Union[str, Dict[str, Any]],
    mode: str = "replace"
) -> None
```

### Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `context` | str/dict | - | Context to set |
| `mode` | str | "replace", "append" | How to set context |

### Example

```python
# Replace context
agent.set_context(
    context="You are a helpful research assistant",
    mode="replace"
)

# Append to context
agent.set_context(
    context={"domain": "medical", "expertise": "cardiology"},
    mode="append"
)
```

---

## Memory Access

### Short-Term Memory

```python
# Access chat history
chat_history = agent.llm_router.chat_history
print(f"Recent messages: {len(chat_history)}")

for msg in chat_history:
    print(f"{msg['role']}: {msg['content'][:50]}...")
```

### Context Summaries

```python
# Access summarized context
summaries = agent.llm_router.context_summaries
print(f"Summaries: {len(summaries)}")

for summary in summaries:
    print(f"Summary: {summary[:100]}...")
```

### Long-Term Memory

```python
# Search persistent memory
results = await agent.llm_router.long_term_memory.search(
    query="machine learning",
    user_id="user_123",
    top_k=5
)

for doc in results:
    print(f"Found: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

---

## Tool Execution

### Tool Output Handling

```python
result = await agent.initiate_agent(
    query="Search for quantum computing papers"
)

if result.get("tool_output"):
    print(f"Tool executed: {result['tool_output']}")
else:
    print("No tool was executed")
```

### Tool Delegation

```python
result = await agent.initiate_agent(
    query="Complex task"
)

if result.get("delegate_to_agent"):
    delegated_agent_name = result["delegate_to_agent"]
    print(f"Delegated to: {delegated_agent_name}")
```

---

## Error Handling

### Try-Catch Pattern

```python
try:
    result = await agent.initiate_agent(
        query="What is AI?",
        passed_from="user"
    )
    print(result["answer"])
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Error: {e}")
```

### Validation

```python
# Check if satisfied
result = await agent.initiate_agent(query="...")

if not result["satisfied"]:
    print("Agent not satisfied with answer")
    print(f"Reasoning: {result['reasoning']}")
```

---

## Real-World Example

### Research Assistant

```python
from masai.AgentManager import AgentManager, AgentDetails

manager = AgentManager(
    model_config_path="model_config.json",
    user_id="researcher_1"
)

agent = manager.create_agent(
    agent_name="researcher",
    tools=[search_tool, web_tool],
    agent_details=AgentDetails(
        capabilities=["research", "analysis"],
        description="Research assistant"
    ),
    plan=True,
    persist_memory=True
)

# Full execution
result = await agent.initiate_agent(
    query="Research latest AI breakthroughs",
    passed_from="user"
)

print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Satisfied: {result['satisfied']}")

# Streaming execution
print("\n--- Streaming ---")
async for state in agent.initiate_agent_astream(
    query="Analyze the research findings",
    passed_from="user"
):
    node_name, state_dict = state
    state_value = [v for k, v in state_dict.items()][0]
    
    if state_value.get("current_node") == "reflector":
        print(f"Reflecting: {state_value.get('answer', '')[:100]}...")
```

---

## Troubleshooting

### Issue: "Agent not satisfied"

```python
result = await agent.initiate_agent(query="...")
if not result["satisfied"]:
    print(f"Not satisfied: {result['reasoning']}")
```

**Solution**: Adjust model parameters or provide more context.

### Issue: Tool not executed

```python
if not result.get("tool_output"):
    print("No tool was executed")
```

**Solution**: Check tool availability and query relevance.

### Issue: Memory access error

```python
# ❌ WRONG
agent.long_term_memory.search(...)

# ✅ CORRECT
agent.llm_router.long_term_memory.search(...)
```

---

## See Also

- [FRAMEWORK_OVERVIEW.md](FRAMEWORK_OVERVIEW.md) - Architecture overview
- [AGENTMANAGER_DETAILED.md](AGENTMANAGER_DETAILED.md) - AgentManager guide
- [MEMORY_ARCHITECTURE_DEEP_DIVE.md](MEMORY_ARCHITECTURE_DEEP_DIVE.md) - Memory system
- [QUICK_START.md](QUICK_START.md) - Quick start guide

