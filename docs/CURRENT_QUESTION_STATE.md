# `current_question` State Variable

## Overview

The `current_question` state variable tracks the current question being processed by the agent across all nodes and workflow executions. This provides a reliable, consistent way to reference the original user query without relying on the messages array.

## Motivation

### Problem with Previous Approach

**Before:**
```python
original_question = messages[0]['content'] if messages else "No original question found."
```

**Issues:**
1. **Fragile**: Assumes messages array always has at least one element
2. **Inconsistent**: Messages array can be modified during workflow
3. **Unclear**: Not obvious that `messages[0]` represents the current question
4. **Error-Prone**: Can fail if messages array is empty or restructured

### Solution: Dedicated State Variable

**After:**
```python
original_question = state.get('current_question', 'No original question found.')
```

**Benefits:**
1. ✅ **Explicit**: Clear purpose and meaning
2. ✅ **Reliable**: Always available in state
3. ✅ **Consistent**: Updated on each agent invocation
4. ✅ **Maintainable**: Easy to track and update

---

## Implementation

### 1. State Definition

**File:** `src/masai/Agents/base_agent.py`

```python
class State(TypedDict):
    messages: List[Dict[str, str]]
    current_tool: str
    tool_input: Any
    tool_output: Any
    answer: str
    satisfied: bool
    reasoning: str
    delegate_to_agent: Optional[str]
    current_node: str
    previous_node: Optional[str]
    plan: Optional[dict]
    passed_from: Optional[str]
    reflection_counter: int
    tool_loop_counter: int
    tool_decided_by: Optional[str]
    current_question: str  # ✅ NEW: Track the current question
```

---

### 2. Initialization in `initiate_agent`

**File:** `src/masai/Agents/singular_agent.py`

#### Fresh Start (No Retained State)

```python
initial_state = State(
    messages=[{"role": "user", "content": new_query}],
    current_tool="", tool_input=None, tool_output="", answer="",
    satisfied=False, reasoning="", delegate_to_agent=None,
    current_node='planner' if self.plan else 'router',
    previous_node=previous_node, plan={}, passed_from=passed_from,
    reflection_counter=0, tool_loop_counter=0, tool_decided_by=None,
    current_question=new_query  # ✅ Set current question
)
```

#### With Retained State

```python
# Start with retained messages and add new query
initial_messages = self.retained_state.get("messages", []).copy()
initial_messages.append({"role": "user", "content": new_query})

initial_state = State(
    messages=initial_messages,
    current_tool="", tool_input=None, tool_output="", answer="",
    satisfied=False, reasoning="", delegate_to_agent=None,
    current_node='planner' if self.plan else 'router',
    previous_node=previous_node, plan={}, passed_from=passed_from,
    reflection_counter=0, tool_loop_counter=0, tool_decided_by=None,
    current_question=new_query  # ✅ Set current question (new query)
)
```

---

### 3. Initialization in `initiate_agent_astream`

**File:** `src/masai/Agents/singular_agent.py`

Same pattern as `initiate_agent`:

```python
# Fresh start
initial_state = State(
    messages=[{"role": "user", "content": new_query}],
    # ... other fields ...
    current_question=new_query  # ✅ Set current question
)

# With retained state
initial_state = State(
    messages=initial_messages,
    # ... other fields ...
    current_question=new_query  # ✅ Set current question
)
```

---

### 4. Usage in `_format_node_prompt`

**File:** `src/masai/Agents/singular_agent.py`

**Before:**
```python
async def _format_node_prompt(self, state: State, node: str) -> str:
    messages = state.get('messages', [])
    original_question = messages[0]['content'] if messages else "No original question found."
    # ... rest of method
```

**After:**
```python
async def _format_node_prompt(self, state: State, node: str) -> str:
    # Use current_question from state instead of messages[0]['content']
    original_question = state.get('current_question', 'No original question found.')
    # ... rest of method
```

**Used in all node prompts:**
- ✅ Router: `ROUTER_NODE_PROMPT.format(original_question=original_question, ...)`
- ✅ Evaluator: `EVALUATOR_NODE_PROMPT.format(original_question=original_question, ...)`
- ✅ Reflector: `REFLECTOR_NODE_PROMPT.format(original_question=original_question, ...)`
- ✅ Planner: `PLANNER_NODE_PROMPT.format(original_question=original_question, ...)`

---

## Workflow

### Single Agent Invocation

```
User Query: "What is the weather in Paris?"
    ↓
initiate_agent(query="What is the weather in Paris?")
    ↓
initial_state.current_question = "What is the weather in Paris?"
    ↓
Router Node: Uses state.current_question → "What is the weather in Paris?"
    ↓
Execute Tool
    ↓
Evaluator Node: Uses state.current_question → "What is the weather in Paris?"
    ↓
Final Answer
```

### Multi-Turn Conversation (Retained State)

```
Turn 1:
User Query: "What is the weather in Paris?"
    ↓
initiate_agent(query="What is the weather in Paris?")
    ↓
state.current_question = "What is the weather in Paris?"
    ↓
[Workflow executes]
    ↓
Final Answer: "It's 15°C and sunny in Paris"

Turn 2:
User Query: "What about London?"
    ↓
initiate_agent(query="What about London?")
    ↓
state.current_question = "What about London?"  # ✅ Updated!
    ↓
state.messages = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "It's 15°C and sunny in Paris"},
    {"role": "user", "content": "What about London?"}  # ✅ New message
]
    ↓
Router Node: Uses state.current_question → "What about London?"
    ↓
[Workflow executes with context of previous conversation]
```

---

## Benefits

### 1. **Clarity**
```python
# Before: Unclear what messages[0] represents
original_question = messages[0]['content']

# After: Explicit and clear
original_question = state.get('current_question')
```

### 2. **Reliability**
```python
# Before: Can fail if messages is empty
original_question = messages[0]['content'] if messages else "No original question found."

# After: Always available with fallback
original_question = state.get('current_question', 'No original question found.')
```

### 3. **Consistency**
- Current question is always the query passed to `initiate_agent()`
- Not affected by message array modifications
- Updated on each new agent invocation

### 4. **Multi-Turn Support**
- Each turn has its own `current_question`
- Previous questions remain in `messages` array for context
- Clear separation between current query and conversation history

---

## Use Cases

### Use Case 1: Simple Query
```python
response = await agent.initiate_agent("What is 2+2?")
# state.current_question = "What is 2+2?"
# All nodes reference this question
```

### Use Case 2: Follow-Up Query
```python
# First query
response1 = await agent.initiate_agent("What is the capital of France?")
# state.current_question = "What is the capital of France?"

# Follow-up query (with retained state)
response2 = await agent.initiate_agent("What is its population?")
# state.current_question = "What is its population?"
# state.messages contains both queries for context
```

### Use Case 3: Complex Multi-Step Task
```python
response = await agent.initiate_agent(
    "Analyze sales data for Q1 2024 and create a report"
)
# state.current_question = "Analyze sales data for Q1 2024 and create a report"
# Router, Evaluator, Reflector all reference this exact question
# Even if workflow takes multiple steps and reflections
```

---

## Comparison: Before vs After

| Aspect | Before (`messages[0]`) | After (`current_question`) |
|--------|------------------------|----------------------------|
| **Clarity** | Unclear purpose | Explicit purpose |
| **Reliability** | Can fail if empty | Always available |
| **Consistency** | Can be modified | Immutable per invocation |
| **Multi-Turn** | Confusing | Clear separation |
| **Maintenance** | Hard to track | Easy to track |
| **Error Handling** | Requires checks | Built-in fallback |

---

## Testing

### Test Case 1: Fresh Start
```python
agent = Agent(...)
response = await agent.initiate_agent("Test query")

assert response['current_question'] == "Test query"
```

### Test Case 2: Retained State
```python
agent = Agent(...)
response1 = await agent.initiate_agent("First query")
response2 = await agent.initiate_agent("Second query")

assert response2['current_question'] == "Second query"
assert len(response2['messages']) == 4  # 2 user + 2 assistant
```

### Test Case 3: Empty Query Handling
```python
agent = Agent(...)
response = await agent.initiate_agent("")

assert response['current_question'] == ""  # Empty but defined
```

---

## Migration Guide

### For Existing Code

If you have custom code that accesses `messages[0]['content']`:

**Before:**
```python
def my_custom_function(state: State):
    original_query = state['messages'][0]['content']
    # ... use original_query
```

**After:**
```python
def my_custom_function(state: State):
    original_query = state.get('current_question', '')
    # ... use original_query
```

### For Custom Nodes

If you've created custom nodes:

**Before:**
```python
async def my_custom_node(state: State) -> State:
    messages = state.get('messages', [])
    query = messages[0]['content'] if messages else ""
    # ... process query
```

**After:**
```python
async def my_custom_node(state: State) -> State:
    query = state.get('current_question', '')
    # ... process query
```

---

## Summary

| Feature | Description |
|---------|-------------|
| **Variable Name** | `current_question` |
| **Type** | `str` |
| **Location** | `State` TypedDict |
| **Initialized** | `initiate_agent()` and `initiate_agent_astream()` |
| **Updated** | On each agent invocation |
| **Used By** | Router, Evaluator, Reflector, Planner prompts |
| **Benefits** | Clarity, reliability, consistency, multi-turn support |
| **Version** | 0.2.5+ |

---

## Version

- **Added in:** v0.2.5
- **Status:** Production Ready ✅
- **Breaking Changes:** None (backward compatible)

