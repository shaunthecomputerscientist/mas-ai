"""
Streaming Events Utility for MASAI Framework

This module provides utilities for emitting intermediate events during agent execution,
including tool calls, tool outputs, and custom events from tools.

Features:
- Emit tool call events
- Emit tool output events
- Emit custom events from tools
- Compatible with existing state streaming
- OpenAI-compatible SSE format
"""

import json
import time
from typing import Dict, Any, Optional, Literal
from contextvars import ContextVar

# Context variable to store streaming callback
_streaming_callback: ContextVar[Optional[callable]] = ContextVar('streaming_callback', default=None)

# Event types
EventType = Literal[
    "state",           # Regular state update
    "tool_call",       # Tool is being called
    "tool_output",     # Tool has returned output
    "node_transition", # Agent moved to a new node
    "custom",          # Custom event from tool
    "error",           # Error occurred
    "reasoning",       # Reasoning update
    "answer"           # Final answer
]


class StreamingEvent:
    """Represents a streaming event that can be emitted during agent execution."""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a streaming event.
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Optional metadata (timestamp, node, etc.)
        """
        self.event_type = event_type
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def to_openai_sse(self, model: str = "mas-ai-general") -> str:
        """
        Convert event to OpenAI-compatible SSE format.
        
        Args:
            model: Model name to include in response
            
        Returns:
            SSE-formatted string
        """
        # Map event types to OpenAI-like structure
        if self.event_type == "tool_call":
            choice_data = {
                "delta": {
                    "tool_calls": [{
                        "id": f"call_{int(self.timestamp * 1000)}",
                        "type": "function",
                        "function": {
                            "name": self.data.get("tool_name"),
                            "arguments": json.dumps(self.data.get("tool_input", {}))
                        }
                    }]
                },
                "finish_reason": None
            }
        elif self.event_type == "tool_output":
            choice_data = {
                "delta": {
                    "content": f"\n[Tool: {self.data.get('tool_name')}]\n{self.data.get('output', '')}\n"
                },
                "finish_reason": None
            }
        elif self.event_type == "reasoning":
            choice_data = {
                "delta": {
                    "content": self.data.get("reasoning", "")
                },
                "finish_reason": None
            }
        elif self.event_type == "answer":
            choice_data = {
                "delta": {
                    "content": self.data.get("answer", "")
                },
                "finish_reason": None
            }
        elif self.event_type == "node_transition":
            choice_data = {
                "delta": {
                    "content": f"\n[Node: {self.data.get('from_node')} → {self.data.get('to_node')}]\n"
                },
                "finish_reason": None
            }
        elif self.event_type == "custom":
            # Custom events from tools
            choice_data = {
                "delta": {
                    "content": self.data.get("content", ""),
                    "custom_data": self.data.get("custom_data", {})
                },
                "finish_reason": None
            }
        elif self.event_type == "error":
            choice_data = {
                "delta": {
                    "content": f"\n[Error: {self.data.get('error', 'Unknown error')}]\n"
                },
                "finish_reason": "stop"
            }
        else:
            # Default: treat as content
            choice_data = {
                "delta": {
                    "content": json.dumps(self.data)
                },
                "finish_reason": None
            }
        
        # Wrap in OpenAI format
        sse_data = {
            "id": f"chatcmpl-{int(self.timestamp * 1000)}",
            "object": "chat.completion.chunk",
            "created": int(self.timestamp),
            "model": model,
            "choices": [{"index": 0, **choice_data}],
            "event_type": self.event_type,  # Add event type for filtering
            "metadata": self.metadata
        }
        
        return f"data: {json.dumps(sse_data)}\n\n"


def set_streaming_callback(callback: callable):
    """
    Set the streaming callback for emitting events.
    
    Args:
        callback: Async function that accepts StreamingEvent
    """
    _streaming_callback.set(callback)


def clear_streaming_callback():
    """Clear the streaming callback."""
    _streaming_callback.set(None)


def get_streaming_callback() -> Optional[callable]:
    """Get the current streaming callback."""
    return _streaming_callback.get()


async def emit_event(
    event_type: EventType,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Emit a streaming event if callback is set.
    
    Args:
        event_type: Type of event
        data: Event data
        metadata: Optional metadata
    """
    callback = get_streaming_callback()
    if callback:
        event = StreamingEvent(event_type, data, metadata)
        await callback(event)


async def emit_tool_call(tool_name: str, tool_input: Any, node: str = None):
    """
    Emit a tool call event.
    
    Args:
        tool_name: Name of the tool being called
        tool_input: Input to the tool
        node: Current node name
    """
    await emit_event(
        "tool_call",
        {
            "tool_name": tool_name,
            "tool_input": tool_input
        },
        {"node": node}
    )


async def emit_tool_output(tool_name: str, output: Any, node: str = None):
    """
    Emit a tool output event.
    
    Args:
        tool_name: Name of the tool
        output: Tool output
        node: Current node name
    """
    await emit_event(
        "tool_output",
        {
            "tool_name": tool_name,
            "output": str(output)
        },
        {"node": node}
    )


async def emit_node_transition(from_node: str, to_node: str):
    """
    Emit a node transition event.
    
    Args:
        from_node: Previous node
        to_node: New node
    """
    await emit_event(
        "node_transition",
        {
            "from_node": from_node,
            "to_node": to_node
        }
    )


async def emit_custom_event(content: str, custom_data: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
    """
    Emit a custom event from a tool.
    
    This can be called from within tools to emit custom streaming data.
    
    Args:
        content: Text content to stream
        custom_data: Additional custom data
        metadata: Optional metadata
        
    Example:
        ```python
        from masai.Tools.utilities.streaming_events import emit_custom_event
        
        @tool("search_web")
        async def search_web(query: str) -> str:
            # Emit progress update
            await emit_custom_event(
                content=f"Searching for: {query}",
                custom_data={"progress": 0.3}
            )
            
            results = await search_api(query)
            
            # Emit another update
            await emit_custom_event(
                content=f"Found {len(results)} results",
                custom_data={"progress": 1.0, "result_count": len(results)}
            )
            
            return results
        ```
    """
    await emit_event(
        "custom",
        {
            "content": content,
            "custom_data": custom_data or {}
        },
        metadata
    )

