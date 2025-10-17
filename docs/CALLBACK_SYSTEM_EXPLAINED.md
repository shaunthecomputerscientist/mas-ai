# How the Callback System Works

## Overview

This document explains how the streaming callback system works, why there's no conflict between multiple requests, and how events are routed to the correct handler.

---

## Your Questions

### Q1: "There is only one streaming class right? Then how will there be different emits to the user?"

**Answer:** Each request gets its own callback through **Context Variables** (`ContextVar`). Even though there's only one class, each request is isolated.

### Q2: "How is the `_event_callback` and `stream_agent_execution` part working?"

**Answer:** The callback is set per-request using context variables, which are thread-safe and async-safe.

---

## The Magic: Context Variables

### What is ContextVar?

`ContextVar` is Python's way of providing **per-request/per-task isolation**. It's like thread-local storage, but for async tasks.

```python
from contextvars import ContextVar

# Global variable, but each async task gets its own value
_streaming_callback: ContextVar[Optional[callable]] = ContextVar('streaming_callback', default=None)
```

**Key Properties:**
- ✅ **Per-task isolation** - Each async task has its own value
- ✅ **Thread-safe** - Safe for concurrent requests
- ✅ **Async-safe** - Works with asyncio
- ✅ **Automatic cleanup** - Value is cleared when task completes

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multiple Concurrent Requests                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Request 1 (User A)                    Request 2 (User B)               │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ handler1 = Enhanced...  │          │ handler2 = Enhanced...  │      │
│  │ handler1.event_queue    │          │ handler2.event_queue    │      │
│  └────────────┬────────────┘          └────────────┬────────────┘      │
│               │                                    │                    │
│               ▼                                    ▼                    │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ set_streaming_callback( │          │ set_streaming_callback( │      │
│  │   handler1._event_...   │          │   handler2._event_...   │      │
│  │ )                       │          │ )                       │      │
│  └────────────┬────────────┘          └────────────┬────────────┘      │
│               │                                    │                    │
│               ▼                                    ▼                    │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ ContextVar for Request 1│          │ ContextVar for Request 2│      │
│  │ callback = handler1...  │          │ callback = handler2...  │      │
│  └────────────┬────────────┘          └────────────┬────────────┘      │
│               │                                    │                    │
│               ▼                                    ▼                    │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ Agent executes          │          │ Agent executes          │      │
│  │ Tool calls emit_event() │          │ Tool calls emit_event() │      │
│  └────────────┬────────────┘          └────────────┬────────────┘      │
│               │                                    │                    │
│               ▼                                    ▼                    │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ get_streaming_callback()│          │ get_streaming_callback()│      │
│  │ Returns: handler1...    │          │ Returns: handler2...    │      │
│  └────────────┬────────────┘          └────────────┬────────────┘      │
│               │                                    │                    │
│               ▼                                    ▼                    │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ handler1._event_callback│          │ handler2._event_callback│      │
│  │ Puts event in queue1    │          │ Puts event in queue2    │      │
│  └────────────┬────────────┘          └────────────┬────────────┘      │
│               │                                    │                    │
│               ▼                                    ▼                    │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ handler1.event_queue    │          │ handler2.event_queue    │      │
│  │ Event for User A        │          │ Event for User B        │      │
│  └────────────┬────────────┘          └────────────┬────────────┘      │
│               │                                    │                    │
│               ▼                                    ▼                    │
│  ┌─────────────────────────┐          ┌─────────────────────────┐      │
│  │ Stream to User A        │          │ Stream to User B        │      │
│  └─────────────────────────┘          └─────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Request Arrives

```python
# Request 1 from User A
@app.post("/agent/stream")
async def agent_stream(req: AgentQueryRequest):
    handler1 = EnhancedStreamHandler(model="mas-ai-general")
    return StreamingResponse(
        handler1.stream_agent_execution(agent, query),
        media_type='text/event-stream'
    )

# Request 2 from User B (concurrent)
@app.post("/agent/stream")
async def agent_stream(req: AgentQueryRequest):
    handler2 = EnhancedStreamHandler(model="mas-ai-general")
    return StreamingResponse(
        handler2.stream_agent_execution(agent, query),
        media_type='text/event-stream'
    )
```

**Result:**
- Two separate `EnhancedStreamHandler` instances
- Two separate event queues
- Two separate async tasks

---

### Step 2: Set Callback (Per-Request)

```python
# In handler1.stream_agent_execution() - Request 1
async def stream_agent_execution(self, agent, query, ...):
    self.event_queue = asyncio.Queue()  # Queue for Request 1
    
    # Set callback for THIS request
    set_streaming_callback(self._event_callback)  # ← Sets callback in ContextVar
    
    # ... execute agent ...
```

**What happens:**

```python
# In streaming_events.py
def set_streaming_callback(callback: callable):
    """Set the streaming callback for emitting events."""
    _streaming_callback.set(callback)  # ← Sets value in ContextVar
```

**ContextVar behavior:**

```
Request 1 (async task 1):
  _streaming_callback.get() → handler1._event_callback

Request 2 (async task 2):
  _streaming_callback.get() → handler2._event_callback

Each request has its own value!
```

---

### Step 3: Agent Executes and Emits Events

```python
# Agent executes (could be Request 1 or Request 2)
async def execute_tool(self, state):
    # Emit tool call event
    await emit_tool_call(tool_name, tool_input)
    
    # Execute tool
    result = await tool.ainvoke(input_data)
    
    # Emit tool output event
    await emit_tool_output(tool_name, result)
```

**What happens:**

```python
# In streaming_events.py
async def emit_tool_call(tool_name: str, tool_input: Any, node: str = None):
    await emit_event(
        "tool_call",
        {"tool_name": tool_name, "tool_input": tool_input},
        {"node": node}
    )

async def emit_event(event_type, data, metadata):
    # Get callback for THIS request
    callback = get_streaming_callback()  # ← Gets value from ContextVar
    
    if callback:
        event = StreamingEvent(event_type, data, metadata)
        await callback(event)  # ← Calls the correct handler's callback
```

**ContextVar ensures correct routing:**

```
Request 1 emits event:
  get_streaming_callback() → handler1._event_callback
  await handler1._event_callback(event)
  → Event goes to handler1.event_queue

Request 2 emits event:
  get_streaming_callback() → handler2._event_callback
  await handler2._event_callback(event)
  → Event goes to handler2.event_queue

No mixing! Each event goes to the correct queue!
```

---

### Step 4: Callback Puts Event in Queue

```python
# In EnhancedStreamHandler
async def _event_callback(self, event: StreamingEvent):
    """Callback for receiving streaming events."""
    if self.event_queue and self._should_emit_event(event):
        await self.event_queue.put(event)  # ← Put in THIS handler's queue
```

**For Request 1:**
```
handler1._event_callback(event)
  → await handler1.event_queue.put(event)
  → Event in handler1's queue
```

**For Request 2:**
```
handler2._event_callback(event)
  → await handler2.event_queue.put(event)
  → Event in handler2's queue
```

---

### Step 5: Main Loop Processes Events

```python
# In handler1.stream_agent_execution() - Request 1
while not agent_task.done() or not self.event_queue.empty():
    event = await asyncio.wait_for(
        self.event_queue.get(),  # ← Get from handler1's queue
        timeout=0.1
    )
    
    sse_data = event.to_openai_sse(self.model)  # ← Use handler1's model
    yield sse_data  # ← Stream to User A
```

**For Request 2:**
```python
# In handler2.stream_agent_execution() - Request 2
while not agent_task.done() or not self.event_queue.empty():
    event = await asyncio.wait_for(
        self.event_queue.get(),  # ← Get from handler2's queue
        timeout=0.1
    )
    
    sse_data = event.to_openai_sse(self.model)  # ← Use handler2's model
    yield sse_data  # ← Stream to User B
```

---

## Why `event.to_openai_sse(self.model)` Works

### The Question

> "Here openai_sse takes model name inside but the class decides the self.event_type but there is only one streaming class right? then how will there be different emits to the user?"

### The Answer

**Each handler instance has its own `self.model`:**

```python
# Request 1
handler1 = EnhancedStreamHandler(model="mas-ai-general")
handler1.model = "mas-ai-general"

# Request 2
handler2 = EnhancedStreamHandler(model="custom-model")
handler2.model = "custom-model"
```

**When converting to SSE:**

```python
# Request 1
event.to_openai_sse(handler1.model)  # Uses "mas-ai-general"
# Output: {"model": "mas-ai-general", ...}

# Request 2
event.to_openai_sse(handler2.model)  # Uses "custom-model"
# Output: {"model": "custom-model", ...}
```

**The event type is in the event itself:**

```python
class StreamingEvent:
    def __init__(self, event_type: EventType, data: Dict, metadata: Dict):
        self.event_type = event_type  # ← Event type stored here
        self.data = data
        self.metadata = metadata
    
    def to_openai_sse(self, model: str) -> str:
        # Use self.event_type to determine formatting
        if self.event_type == "tool_call":
            delta = {"tool_calls": [...]}
        elif self.event_type == "custom":
            delta = {"content": self.data.get("content")}
        # ...
        
        # Use provided model name
        return f"data: {json.dumps({'model': model, ...})}\n\n"
```

---

## Isolation Guarantees

### ContextVar Provides:

1. **Per-Request Isolation**
   ```
   Request 1: callback = handler1._event_callback
   Request 2: callback = handler2._event_callback
   No mixing!
   ```

2. **Thread Safety**
   ```
   Multiple threads can call set_streaming_callback() simultaneously
   Each thread gets its own value
   ```

3. **Async Safety**
   ```
   Multiple async tasks can call set_streaming_callback() simultaneously
   Each task gets its own value
   ```

4. **Automatic Cleanup**
   ```
   When async task completes, ContextVar value is automatically cleared
   No memory leaks
   ```

---

## Example: Two Concurrent Requests

```python
# Time: 0.0s - Request 1 arrives
handler1 = EnhancedStreamHandler(model="model-a")
set_streaming_callback(handler1._event_callback)
# ContextVar for Request 1: callback = handler1._event_callback

# Time: 0.1s - Request 2 arrives (concurrent)
handler2 = EnhancedStreamHandler(model="model-b")
set_streaming_callback(handler2._event_callback)
# ContextVar for Request 2: callback = handler2._event_callback

# Time: 0.5s - Request 1 emits event
await emit_custom_event("Progress from Request 1")
  ↓
callback = get_streaming_callback()  # Returns handler1._event_callback
  ↓
await handler1._event_callback(event)
  ↓
await handler1.event_queue.put(event)  # Event in handler1's queue

# Time: 0.6s - Request 2 emits event
await emit_custom_event("Progress from Request 2")
  ↓
callback = get_streaming_callback()  # Returns handler2._event_callback
  ↓
await handler2._event_callback(event)
  ↓
await handler2.event_queue.put(event)  # Event in handler2's queue

# Time: 0.7s - Request 1 processes event
event = await handler1.event_queue.get()  # Gets Request 1's event
sse = event.to_openai_sse(handler1.model)  # Uses "model-a"
yield sse  # Streams to User A

# Time: 0.8s - Request 2 processes event
event = await handler2.event_queue.get()  # Gets Request 2's event
sse = event.to_openai_sse(handler2.model)  # Uses "model-b"
yield sse  # Streams to User B
```

---

## Summary

### Key Points

1. **ContextVar provides per-request isolation**
   - Each request has its own callback
   - No mixing between requests

2. **Each handler has its own queue**
   - Events go to the correct queue
   - No cross-contamination

3. **Each handler has its own model name**
   - `event.to_openai_sse(self.model)` uses the handler's model
   - Different requests can use different models

4. **The system is thread-safe and async-safe**
   - Multiple concurrent requests work correctly
   - No race conditions

### Visual Summary

```
Request 1                          Request 2
   ↓                                  ↓
handler1                           handler2
   ↓                                  ↓
handler1.event_queue              handler2.event_queue
   ↓                                  ↓
set_callback(handler1.cb)         set_callback(handler2.cb)
   ↓                                  ↓
ContextVar[Request1] = handler1.cb ContextVar[Request2] = handler2.cb
   ↓                                  ↓
emit_event()                       emit_event()
   ↓                                  ↓
get_callback() → handler1.cb       get_callback() → handler2.cb
   ↓                                  ↓
handler1.event_queue.put()         handler2.event_queue.put()
   ↓                                  ↓
handler1.event_queue.get()         handler2.event_queue.get()
   ↓                                  ↓
to_openai_sse(handler1.model)      to_openai_sse(handler2.model)
   ↓                                  ↓
Stream to User A                   Stream to User B
```

**No mixing! Each request is completely isolated!** ✅

