# Enhanced Streaming in MASAI Framework

## Overview

MASAI now supports **enhanced streaming** that goes beyond basic state updates to include intermediate events like tool calls, tool outputs, and custom events from tools. This provides real-time visibility into agent execution while maintaining backward compatibility with existing streaming implementations.

---

## Features

### ✅ What Can Be Streamed

1. **State Updates** (existing)
   - Reasoning updates
   - Final answers
   - Node transitions

2. **Tool Events** (new)
   - Tool call events (when a tool is invoked)
   - Tool output events (when a tool returns results)

3. **Custom Events** (new)
   - Progress updates from tools
   - Custom data from tool execution
   - Real-time status messages

4. **Decentralized MAS** (new)
   - Multi-agent delegation chains
   - Agent-to-agent handoffs
   - Shared memory updates

---

## Architecture

### Event Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MASAI Agent Execution                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Agent Node (Router/Planner)                            │
│     └─> State Update ──────────────────┐                   │
│                                         │                   │
│  2. Execute Tool Node                   │                   │
│     ├─> emit_tool_call() ──────────────┤                   │
│     ├─> Tool Execution                  │                   │
│     │   └─> emit_custom_event() ────────┤                   │
│     └─> emit_tool_output() ─────────────┤                   │
│                                         │                   │
│  3. Evaluator Node                      │                   │
│     └─> State Update ──────────────────┤                   │
│                                         │                   │
│  4. Final Answer                        │                   │
│     └─> State Update ──────────────────┤                   │
│                                         ▼                   │
│                                  Event Queue                │
│                                         │                   │
└─────────────────────────────────────────┼───────────────────┘
                                          │
                                          ▼
                              ┌───────────────────┐
                              │ Stream Handler    │
                              │ (EnhancedStream)  │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ SSE Format        │
                              │ (OpenAI-like)     │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ Client (Browser)  │
                              └───────────────────┘
```

---

## Usage

### 1. Basic Enhanced Streaming

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from masai.Tools.utilities.enhanced_streaming import EnhancedStreamHandler

app = FastAPI()

@app.post("/agent/stream")
async def stream_agent(query: str):
    # Create handler
    handler = EnhancedStreamHandler(
        model="mas-ai-general",
        enable_tool_events=True,      # Stream tool calls/outputs
        enable_custom_events=True,    # Stream custom events from tools
        enable_node_transitions=False # Usually too verbose
    )
    
    # Stream execution
    return StreamingResponse(
        handler.stream_agent_execution(agent, query),
        media_type='text/event-stream',
        headers=handler.get_headers()
    )
```

### 2. Simple Streaming (Backward Compatible)

```python
from masai.Tools.utilities.enhanced_streaming import SimpleStreamHandler

@app.post("/agent/stream/simple")
async def stream_agent_simple(query: str):
    # Simple handler (only state updates)
    handler = SimpleStreamHandler(model="mas-ai-general")
    
    return StreamingResponse(
        handler.stream_agent_execution(agent, query),
        media_type='text/event-stream'
    )
```

### 3. Custom Events from Tools

```python
from langchain.tools import tool
from masai.Tools.utilities.streaming_events import emit_custom_event

@tool("search_database")
async def search_database(query: str) -> str:
    """Search database with progress updates."""
    
    # Emit progress: Starting
    await emit_custom_event(
        content="🔍 Starting search...",
        custom_data={"progress": 0.0, "stage": "init"}
    )
    
    # Perform search
    results = await perform_search(query)
    
    # Emit progress: Complete
    await emit_custom_event(
        content=f"✅ Found {len(results)} results",
        custom_data={"progress": 1.0, "result_count": len(results)}
    )
    
    return results
```

### 4. Decentralized MAS Streaming

```python
from masai.MultiAgents.MultiAgent import MultiAgentSystem

@app.post("/mas/stream")
async def stream_mas(query: str):
    mas = MultiAgentSystem(agentManager=manager)
    handler = EnhancedStreamHandler(enable_tool_events=True)
    
    async def stream_execution():
        handler.event_queue = asyncio.Queue()
        set_streaming_callback(handler._event_callback)
        
        try:
            async for state in mas.initiate_decentralized_mas_astream(
                query=query,
                set_entry_agent=entry_agent
            ):
                # Process and yield states
                yield process_state(state)
                
                # Yield queued events
                while not handler.event_queue.empty():
                    event = handler.event_queue.get_nowait()
                    yield event.to_openai_sse(handler.model)
        finally:
            clear_streaming_callback()
    
    return StreamingResponse(stream_execution(), media_type='text/event-stream')
```

---

## Event Types

### 1. State Events

**Type:** `"state"`

**Data:**
```json
{
  "event_type": "state",
  "data": {
    "reasoning": "Analyzing the query...",
    "answer": "Here is the result",
    "satisfied": true
  }
}
```

### 2. Tool Call Events

**Type:** `"tool_call"`

**Data:**
```json
{
  "event_type": "tool_call",
  "data": {
    "tool_name": "search_database",
    "tool_input": {"query": "AI trends", "limit": 10}
  },
  "metadata": {"node": "execute_tool"}
}
```

**SSE Format:**
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "search_database",
          "arguments": "{\"query\": \"AI trends\", \"limit\": 10}"
        }
      }]
    }
  }]
}
```

### 3. Tool Output Events

**Type:** `"tool_output"`

**Data:**
```json
{
  "event_type": "tool_output",
  "data": {
    "tool_name": "search_database",
    "output": "Found 5 results: ..."
  },
  "metadata": {"node": "execute_tool"}
}
```

**SSE Format:**
```json
{
  "choices": [{
    "delta": {
      "content": "\n[Tool: search_database]\nFound 5 results: ...\n"
    }
  }]
}
```

### 4. Custom Events

**Type:** `"custom"`

**Data:**
```json
{
  "event_type": "custom",
  "data": {
    "content": "🔍 Searching 100 records...",
    "custom_data": {
      "progress": 0.5,
      "stage": "searching",
      "records_processed": 50
    }
  }
}
```

**SSE Format:**
```json
{
  "choices": [{
    "delta": {
      "content": "🔍 Searching 100 records...",
      "custom_data": {
        "progress": 0.5,
        "stage": "searching",
        "records_processed": 50
      }
    }
  }]
}
```

---

## Configuration Options

### EnhancedStreamHandler

```python
handler = EnhancedStreamHandler(
    model="mas-ai-general",           # Model name for SSE responses
    enable_tool_events=True,          # Stream tool calls/outputs
    enable_node_transitions=False,    # Stream node transitions
    enable_custom_events=True,        # Stream custom events from tools
    event_filter={"tool_call", "custom"}  # Only stream specific types
)
```

### Event Filtering

```python
# Only stream tool events
handler = EnhancedStreamHandler(
    event_filter={"tool_call", "tool_output"}
)

# Only stream custom events
handler = EnhancedStreamHandler(
    event_filter={"custom"}
)

# Stream everything (default)
handler = EnhancedStreamHandler(
    event_filter=None
)
```

---

## Client-Side Integration

### JavaScript/TypeScript

```javascript
async function streamAgent(query) {
    const response = await fetch('/agent/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') break;
                
                const event = JSON.parse(data);
                handleEvent(event);
            }
        }
    }
}

function handleEvent(event) {
    const eventType = event.event_type;
    const delta = event.choices[0].delta;
    
    switch (eventType) {
        case 'tool_call':
            console.log('🔧 Tool:', delta.tool_calls[0].function.name);
            break;
        case 'tool_output':
            console.log('📤 Output:', delta.content);
            break;
        case 'custom':
            console.log('✨ Progress:', delta.custom_data.progress);
            break;
        default:
            console.log('💬 Content:', delta.content);
    }
}
```

---

## Best Practices

### 1. Use Custom Events for Progress

```python
@tool("long_running_task")
async def long_running_task(data: str) -> str:
    """Long running task with progress updates."""
    
    total_steps = 10
    for i in range(total_steps):
        await emit_custom_event(
            content=f"Processing step {i+1}/{total_steps}",
            custom_data={"progress": (i+1) / total_steps}
        )
        await process_step(i)
    
    return "Complete"
```

### 2. Disable Verbose Events in Production

```python
# Development: Enable all events
handler = EnhancedStreamHandler(
    enable_tool_events=True,
    enable_node_transitions=True,
    enable_custom_events=True
)

# Production: Only essential events
handler = EnhancedStreamHandler(
    enable_tool_events=False,
    enable_node_transitions=False,
    enable_custom_events=True
)
```

### 3. Handle Errors Gracefully

```python
async def stream_with_error_handling():
    try:
        async for event in handler.stream_agent_execution(agent, query):
            yield event
    except Exception as e:
        yield handler._format_sse_event({
            "delta": {"content": f"Error: {str(e)}"},
            "finish_reason": "stop"
        })
        yield "data: [DONE]\n\n"
```

---

## Backward Compatibility

✅ **Existing streaming code continues to work without changes**

Your current implementation:
```python
async for state in agent.initiate_agent_astream(query):
    # Process state
    pass
```

Still works exactly the same way. Enhanced streaming is **opt-in** through:
1. Using `EnhancedStreamHandler`
2. Calling `emit_custom_event()` in tools
3. Setting streaming callback with `set_streaming_callback()`

---

## Performance Considerations

1. **Event Queue**: Uses `asyncio.Queue` for efficient event handling
2. **No Blocking**: All events are emitted asynchronously
3. **Minimal Overhead**: Events only processed if callback is set
4. **Configurable**: Disable events you don't need

---

## See Also

- [examples/enhanced_streaming_example.py](../examples/enhanced_streaming_example.py) - Complete working examples
- [src/masai/Tools/utilities/streaming_events.py](../src/masai/Tools/utilities/streaming_events.py) - Event utilities
- [src/masai/Tools/utilities/enhanced_streaming.py](../src/masai/Tools/utilities/enhanced_streaming.py) - Stream handlers

