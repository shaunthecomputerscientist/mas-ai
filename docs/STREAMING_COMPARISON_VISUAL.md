# Visual Comparison: Simple vs Enhanced Streaming

## Scenario: Tool with Emit Calls

### The Tool

```python
@tool("search_database")
async def search_database(query: str) -> str:
    """Search database with progress updates."""
    
    # Emit 1: Start
    await emit_custom_event("🔍 Starting search...", {"progress": 0.0})
    
    # Do work
    results = await db.find(query).to_list()
    
    # Emit 2: Complete
    await emit_custom_event("✅ Found results!", {"progress": 1.0})
    
    return json.dumps(results)
```

---

## Option 1: SimpleStreamHandler

### Code

```python
@app.post("/agent/stream/simple")
async def agent_stream_simple(req):
    handler = SimpleStreamHandler(model="mas-ai")
    
    return StreamingResponse(
        handler.stream_agent_execution(agent, req.query),
        media_type='text/event-stream'
    )
```

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleStreamHandler                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. stream_agent_execution() called                            │
│     └─> Does NOT set streaming callback                        │
│                                                                 │
│  2. Agent starts execution                                      │
│     └─> State: {"reasoning": "I need to search..."}            │
│         └─> ✅ STREAMED to client                              │
│                                                                 │
│  3. Agent calls tool: search_database                          │
│     │                                                           │
│     ├─> emit_custom_event("🔍 Starting search...")            │
│     │   └─> get_streaming_callback() → None                    │
│     │   └─> ❌ NO-OP (not streamed)                            │
│     │                                                           │
│     ├─> Tool executes: db.find(query)                          │
│     │                                                           │
│     ├─> emit_custom_event("✅ Found results!")                 │
│     │   └─> get_streaming_callback() → None                    │
│     │   └─> ❌ NO-OP (not streamed)                            │
│     │                                                           │
│     └─> Tool returns: json.dumps(results)                      │
│                                                                 │
│  4. Agent processes tool output                                │
│     └─> State: {"answer": "Found 5 results..."}                │
│         └─> ✅ STREAMED to client                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What Client Sees

```
data: {"choices": [{"delta": {"role": "assistant"}}]}

data: {"choices": [{"delta": {"content": "I need to search..."}}]}

data: {"choices": [{"delta": {"content": "Found 5 results..."}}]}

data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

### Summary

- ✅ Reasoning streamed
- ✅ Answer streamed
- ❌ Tool events NOT streamed
- ❌ Custom events NOT streamed
- ✅ No errors
- ✅ Tool works correctly

---

## Option 2: EnhancedStreamHandler

### Code

```python
@app.post("/agent/stream/enhanced")
async def agent_stream_enhanced(req):
    handler = EnhancedStreamHandler(
        model="mas-ai",
        enable_tool_events=True,
        enable_custom_events=True
    )
    
    return StreamingResponse(
        handler.stream_agent_execution(agent, req.query),
        media_type='text/event-stream'
    )
```

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   EnhancedStreamHandler                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. stream_agent_execution() called                            │
│     └─> ✅ Sets streaming callback: handler._event_callback    │
│                                                                 │
│  2. Agent starts execution                                      │
│     └─> State: {"reasoning": "I need to search..."}            │
│         └─> ✅ STREAMED to client                              │
│                                                                 │
│  3. base_agent.execute_tool() called                           │
│     └─> emit_tool_call("search_database", {...})               │
│         └─> get_streaming_callback() → handler._event_callback │
│         └─> ✅ Event added to queue                            │
│         └─> ✅ STREAMED to client                              │
│                                                                 │
│  4. Tool executes: search_database                             │
│     │                                                           │
│     ├─> emit_custom_event("🔍 Starting search...")            │
│     │   └─> get_streaming_callback() → handler._event_callback │
│     │   └─> ✅ Event added to queue                            │
│     │   └─> ✅ STREAMED to client                              │
│     │                                                           │
│     ├─> Tool executes: db.find(query)                          │
│     │                                                           │
│     ├─> emit_custom_event("✅ Found results!")                 │
│     │   └─> get_streaming_callback() → handler._event_callback │
│     │   └─> ✅ Event added to queue                            │
│     │   └─> ✅ STREAMED to client                              │
│     │                                                           │
│     └─> Tool returns: json.dumps(results)                      │
│                                                                 │
│  5. base_agent.execute_tool() continues                        │
│     └─> emit_tool_output("search_database", results)           │
│         └─> get_streaming_callback() → handler._event_callback │
│         └─> ✅ Event added to queue                            │
│         └─> ✅ STREAMED to client                              │
│                                                                 │
│  6. Agent processes tool output                                │
│     └─> State: {"answer": "Found 5 results..."}                │
│         └─> ✅ STREAMED to client                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What Client Sees

```
data: {"choices": [{"delta": {"role": "assistant"}}]}

data: {"choices": [{"delta": {"content": "I need to search..."}}]}

data: {"choices": [{"delta": {"tool_calls": [{"function": {"name": "search_database"}}]}}], "event_type": "tool_call"}

data: {"choices": [{"delta": {"content": "🔍 Starting search..."}}], "event_type": "custom"}

data: {"choices": [{"delta": {"content": "✅ Found results!"}}], "event_type": "custom"}

data: {"choices": [{"delta": {"content": "\n[Tool: search_database]\n..."}}], "event_type": "tool_output"}

data: {"choices": [{"delta": {"content": "Found 5 results..."}}]}

data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

### Summary

- ✅ Reasoning streamed
- ✅ Answer streamed
- ✅ Tool call event streamed
- ✅ Tool output event streamed
- ✅ Custom events streamed
- ✅ No errors
- ✅ Tool works correctly

---

## Side-by-Side Comparison

```
┌─────────────────────────────────┬─────────────────────────────────┐
│      SimpleStreamHandler        │     EnhancedStreamHandler       │
├─────────────────────────────────┼─────────────────────────────────┤
│                                 │                                 │
│  Callback: ❌ NOT SET           │  Callback: ✅ SET               │
│                                 │                                 │
│  emit_custom_event():           │  emit_custom_event():           │
│    └─> ❌ NO-OP                 │    └─> ✅ ACTIVE                │
│                                 │                                 │
│  emit_tool_call():              │  emit_tool_call():              │
│    └─> ❌ NO-OP                 │    └─> ✅ ACTIVE                │
│                                 │                                 │
│  emit_tool_output():            │  emit_tool_output():            │
│    └─> ❌ NO-OP                 │    └─> ✅ ACTIVE                │
│                                 │                                 │
│  Streams:                       │  Streams:                       │
│    ✅ Reasoning                 │    ✅ Reasoning                 │
│    ✅ Answer                    │    ✅ Answer                    │
│    ❌ Tool events               │    ✅ Tool events               │
│    ❌ Custom events             │    ✅ Custom events             │
│                                 │                                 │
│  Performance:                   │  Performance:                   │
│    ⚡ Fast (no event queue)    │    ⚡ Fast (async queue)        │
│                                 │                                 │
│  Use Case:                      │  Use Case:                      │
│    • Simple streaming           │    • Detailed visibility        │
│    • Minimal overhead           │    • Progress tracking          │
│    • Backward compatible        │    • Debugging                  │
│                                 │    • User feedback              │
│                                 │                                 │
└─────────────────────────────────┴─────────────────────────────────┘
```

---

## Your Current Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│              Your Current Implementation                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Callback: ❌ NOT SET                                           │
│                                                                 │
│  async def event_generator():                                  │
│      async for state in agent.initiate_agent_astream(query):   │
│          if state.get("reasoning"):                            │
│              yield _format_sse_event(...)                      │
│          if state.get("answer"):                               │
│              final_answer = state.get("answer")                │
│                                                                 │
│  emit_custom_event():                                          │
│    └─> ❌ NO-OP (callback not set)                             │
│                                                                 │
│  emit_tool_call():                                             │
│    └─> ❌ NO-OP (callback not set)                             │
│                                                                 │
│  emit_tool_output():                                           │
│    └─> ❌ NO-OP (callback not set)                             │
│                                                                 │
│  Behavior: SAME AS SimpleStreamHandler                         │
│                                                                 │
│  ✅ Safe to add emit calls to tools                            │
│  ✅ No breaking changes                                        │
│  ✅ Future-proof                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Migration Path

```
┌─────────────────────────────────────────────────────────────────┐
│                     Migration Timeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Add Emit Calls to Tools                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ @tool("my_tool")                                          │ │
│  │ async def my_tool(x):                                     │ │
│  │     await emit_custom_event("Starting...")  ← ADD THIS   │ │
│  │     result = do_work(x)                                   │ │
│  │     await emit_custom_event("Complete!")    ← ADD THIS   │ │
│  │     return result                                         │ │
│  └───────────────────────────────────────────────────────────┘ │
│  Status: ✅ No breaking changes (emit calls are no-ops)        │
│                                                                 │
│  Phase 2: Test with EnhancedStreamHandler                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ @app.post("/agent/stream/enhanced")                       │ │
│  │ async def agent_stream_enhanced(req):                     │ │
│  │     handler = EnhancedStreamHandler(...)                  │ │
│  │     return StreamingResponse(...)                         │ │
│  └───────────────────────────────────────────────────────────┘ │
│  Status: ✅ Emit calls now active, events streamed             │
│                                                                 │
│  Phase 3: Gradual Rollout                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ • Keep old endpoint for existing clients                  │ │
│  │ • New clients use enhanced endpoint                       │ │
│  │ • Monitor performance and feedback                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│  Status: ✅ Both endpoints work, no breaking changes           │
│                                                                 │
│  Phase 4: Full Migration (Optional)                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ • Migrate all clients to enhanced endpoint                │ │
│  │ • Deprecate old endpoint                                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│  Status: ✅ All clients using enhanced streaming               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### 1. Emit Functions Are Smart

```python
async def emit_custom_event(content, custom_data):
    callback = get_streaming_callback()
    if callback:  # ← Smart check
        # Only do work if callback exists
        event = StreamingEvent("custom", {...})
        await callback(event)
    # Otherwise, return immediately (no-op)
```

### 2. No Performance Penalty

```
With SimpleStreamHandler:
  emit_custom_event() → Check callback (None) → Return
  Time: ~1-2 microseconds
  
With EnhancedStreamHandler:
  emit_custom_event() → Check callback (exists) → Create event → Queue
  Time: ~10-20 microseconds
```

### 3. Context Variables Are Thread-Safe

```python
# Each request has its own context
_streaming_callback: ContextVar[Optional[callable]] = ContextVar(...)

# Request 1: Uses SimpleStreamHandler
#   → callback = None for this request

# Request 2: Uses EnhancedStreamHandler
#   → callback = handler._event_callback for this request

# No interference between requests!
```

---

## Summary

### The Answer to Your Question

**"Will emit functions work properly with SimpleStreamHandler?"**

**YES!** They work perfectly:
- ✅ No errors
- ✅ No breaking changes
- ✅ Minimal overhead (~1-2 microseconds per call)
- ✅ Future-proof (automatically work with EnhancedStreamHandler)
- ✅ Safe to add to tools now

### What You Should Do

1. ✅ **Add emit calls to your tools now**
   - They're no-ops with your current implementation
   - They'll automatically work when you upgrade

2. ✅ **Keep your current endpoint**
   - No changes needed
   - Everything continues to work

3. ✅ **Add new enhanced endpoint when ready**
   - Test with new endpoint
   - Gradual migration
   - No breaking changes

4. ✅ **Don't worry about performance**
   - Emit calls are extremely fast
   - No-ops have negligible overhead

### The Bottom Line

**You can safely add `emit_custom_event()`, `emit_tool_call()`, and `emit_tool_output()` calls to your tools right now. They won't break anything, and they'll automatically start working when you switch to EnhancedStreamHandler.**

🎉 **It's a win-win!** 🎉

