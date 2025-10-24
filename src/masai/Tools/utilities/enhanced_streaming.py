"""
Enhanced Streaming Utilities for MASAI Framework

This module provides enhanced streaming capabilities for FastAPI endpoints,
including support for intermediate events (tool calls, tool outputs, etc.)
while maintaining backward compatibility with existing state streaming.

Usage:
    from masai.Tools.utilities.enhanced_streaming import EnhancedStreamHandler
    
    async def agent_query_stream(req: AgentQueryRequest):
        handler = EnhancedStreamHandler(
            model="mas-ai-general",
            enable_tool_events=True,
            enable_node_transitions=False
        )
        
        return StreamingResponse(
            handler.stream_agent_execution(agent, query),
            media_type='text/event-stream',
            headers=handler.get_headers()
        )
"""

import json
import time
import asyncio
from typing import Dict, Any, AsyncGenerator, Optional, Set
from queue import Queue
import threading

from .streaming_events import (
    StreamingEvent,
    set_streaming_callback,
    clear_streaming_callback
)


class EnhancedStreamHandler:
    """
    Enhanced streaming handler that supports both state updates and intermediate events.
    
    Features:
    - Stream state updates (reasoning, answer)
    - Stream tool calls and outputs
    - Stream node transitions
    - Stream custom events from tools
    - Configurable event filtering
    - OpenAI-compatible SSE format
    """
    
    def __init__(
        self,
        model: str = "mas-ai-general",
        enable_tool_events: bool = True,
        enable_node_transitions: bool = False,
        enable_custom_events: bool = True,
        event_filter: Optional[Set[str]] = None
    ):
        """
        Initialize the enhanced stream handler.
        
        Args:
            model: Model name for OpenAI-compatible responses
            enable_tool_events: Whether to stream tool call/output events
            enable_node_transitions: Whether to stream node transition events
            enable_custom_events: Whether to stream custom events from tools
            event_filter: Set of event types to include (None = all enabled types)
        """
        self.model = model
        self.enable_tool_events = enable_tool_events
        self.enable_node_transitions = enable_node_transitions
        self.enable_custom_events = enable_custom_events
        self.event_filter = event_filter
        
        # Queue for collecting events from callback
        self.event_queue: asyncio.Queue = None
        
    def get_headers(self) -> Dict[str, str]:
        """Get recommended headers for streaming response."""
        return {
            'Cache-Control': 'no-cache, no-transform',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    
    def _format_sse_event(self, choice_data: Dict[str, Any]) -> str:
        """Format data as OpenAI-compatible SSE event."""
        data = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{"index": 0, **choice_data}],
        }
        return f"data: {json.dumps(data)}\n\n"
    
    def _should_emit_event(self, event: StreamingEvent) -> bool:
        """Check if event should be emitted based on configuration."""
        # Check event filter
        if self.event_filter and event.event_type not in self.event_filter:
            return False
        
        # Check feature flags
        if event.event_type in ("tool_call", "tool_output") and not self.enable_tool_events:
            return False
        if event.event_type == "node_transition" and not self.enable_node_transitions:
            return False
        if event.event_type == "custom" and not self.enable_custom_events:
            return False
        
        return True
    
    async def _event_callback(self, event: StreamingEvent):
        """Callback for receiving streaming events."""
        if self.event_queue and self._should_emit_event(event):
            await self.event_queue.put(event)
    
    async def stream_agent_execution(
        self,
        agent,
        query: str,
        passed_from: str = "user",
        agent_type: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent execution with enhanced events.
        
        Args:
            agent: Agent instance
            query: User query
            passed_from: Source of query
            agent_type: Agent type for role field
            
        Yields:
            SSE-formatted strings
        """
        # Initialize event queue
        self.event_queue = asyncio.Queue()
        
        # Set streaming callback
        set_streaming_callback(self._event_callback)

        try:
            # Initial assistant role signal (OpenAI-style)
            yield self._format_sse_event({
                "delta": {"role": agent_type or "assistant"},
                "finish_reason": None
            })
            
            # Create task for agent execution
            agent_task = asyncio.create_task(
                self._run_agent_stream(agent, query, passed_from)
            )
            
            # Process events as they arrive
            while not agent_task.done() or not self.event_queue.empty():
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=0.1
                    )
                    
                    # Convert event to SSE and yield
                    sse_data = event.to_openai_sse(self.model)
                    yield sse_data
                    
                except asyncio.TimeoutError:
                    # No event available, continue
                    continue
            
            # Wait for agent task to complete
            await agent_task

            # NOTE: All streaming logic is centralized in _run_agent_stream():
            # - Tool events: Emitted by tools via emit_event() → callback → queue
            # - Reasoning events: Extracted from state and emitted in _run_agent_stream()
            # - Answer events: Extracted from final state and emitted in _run_agent_stream()
            # This ensures ALL events flow through the same queue → SSE path.

            # Stream completion footer
            yield self._format_sse_event({"delta": {}, "finish_reason": "stop"})
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield self._format_sse_event({
                "delta": {"content": f"\n\nError: {str(e)}"},
                "finish_reason": "stop",
            })
            yield "data: [DONE]\n\n"
            
        finally:
            # Clear streaming callback
            clear_streaming_callback()
            self.event_queue = None
    
    async def _run_agent_stream(self, agent, query: str, passed_from: str) -> Dict[str, Any]:
        """
        Run agent with streaming and emit reasoning/answer events.

        This method centralizes all streaming logic:
        - Extracts reasoning and emits as events (immediately)
        - Stores answer and emits at the end (with satisfied status)
        - Tool events are already emitted by tools via emit_event()
        """
        from .streaming_events import StreamingEvent

        final_state = {}
        last_reasoning = None
        final_answer = None

        async for state in agent.initiate_agent_astream(query, passed_from=passed_from):
            # Unwrap possible (node_name, state_dict) tuples
            actual = state
            if isinstance(state, tuple) and len(state) > 1:
                maybe = state[1]
                if isinstance(maybe, dict) and maybe:
                    # take first value when dict-of-states
                    actual = next(iter(maybe.values()), maybe)
                else:
                    actual = maybe

            if actual and isinstance(actual, dict):
                final_state = actual

                # Emit reasoning events immediately (thinking process)
                current_reasoning = actual.get("reasoning")
                if current_reasoning and current_reasoning != last_reasoning:
                    last_reasoning = current_reasoning
                    reasoning_event = StreamingEvent(
                        event_type="reasoning",
                        data={"reasoning": current_reasoning},
                        metadata={"agent": agent.agent_name, "node": actual.get("current_node")}
                    )
                    await self.event_queue.put(reasoning_event)

                # Store answer but don't emit yet (wait for end)
                current_answer = actual.get("answer")
                if current_answer:
                    final_answer = current_answer

        # After workflow completes, emit final answer with satisfied status
        if final_answer and final_state:
            is_satisfied = final_state.get("satisfied", False)
            answer_event = StreamingEvent(
                event_type="answer",
                data={
                    "answer": final_answer,
                    "satisfied": is_satisfied
                },
                metadata={"agent": agent.agent_name, "node": final_state.get("current_node")}
            )
            await self.event_queue.put(answer_event)

        return final_state


class SimpleStreamHandler:
    """
    Simple streaming handler for backward compatibility.

    Now emits events with event_type field for frontend compatibility.
    Note: This handler does NOT support tool events (tool_call, tool_output, custom).
    For full event support, use EnhancedStreamHandler instead.
    """

    def __init__(self, model: str = "mas-ai-general"):
        self.model = model

    def _format_sse_event(self, choice_data: Dict[str, Any], event_type: Optional[str] = None) -> str:
        """
        Format SSE event with optional event_type field.

        Args:
            choice_data: OpenAI-style choice data
            event_type: Optional event type (reasoning, answer, etc.)
        """
        data = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{"index": 0, **choice_data}],
        }

        # Add event_type if provided (for frontend compatibility)
        if event_type:
            data["event_type"] = event_type

        return f"data: {json.dumps(data)}\n\n"
    
    async def stream_agent_execution(
        self,
        agent,
        query: str,
        passed_from: str = "user",
        agent_type: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent execution (simple version).
        
        Args:
            agent: Agent instance
            query: User query
            passed_from: Source of query
            agent_type: Agent type for role field
            
        Yields:
            SSE-formatted strings
        """
        final_answer = None
        final_state = None
        seen_reasoning = set()  # Track seen reasoning to avoid duplicates

        try:
            # Initial assistant role signal (OpenAI-style)
            yield self._format_sse_event({
                "delta": {"role": agent_type or "assistant"},
                "finish_reason": None
            })

            async for state in agent.initiate_agent_astream(query, passed_from=passed_from):
                # Unwrap possible (node_name, state_dict) tuples
                actual = state
                if isinstance(state, tuple) and len(state) > 1:
                    maybe = state[1]
                    if isinstance(maybe, dict) and maybe:
                        # take first value when dict-of-states
                        actual = next(iter(maybe.values()), maybe)
                    else:
                        actual = maybe

                if not actual:
                    continue

                if isinstance(actual, dict):
                    # Store final state for later
                    final_state = actual

                    # Stream reasoning as it arrives with event_type (avoid duplicates)
                    current_reasoning = actual.get("reasoning")
                    if current_reasoning and current_reasoning not in seen_reasoning:
                        seen_reasoning.add(current_reasoning)
                        yield self._format_sse_event(
                            choice_data={
                                "delta": {"content": current_reasoning},
                                "finish_reason": None,
                            },
                            event_type="reasoning"  # ✅ Add event_type for frontend
                        )

                    # Keep latest answer; defer emitting until the end
                    if actual.get("answer"):
                        final_answer = actual.get("answer")

            # Emit final answer at the end with satisfied status
            # This allows frontend to distinguish between successful and failed answers
            if final_answer and final_state:
                is_satisfied = final_state.get("satisfied", False)
                yield self._format_sse_event(
                    choice_data={
                        "delta": {
                            "content": final_answer,
                            "satisfied": is_satisfied  # Include status for frontend
                        },
                        "finish_reason": None,
                    },
                    event_type="answer"  # ✅ Add event_type for frontend
                )
            
            # Stream completion footer
            yield self._format_sse_event({"delta": {}, "finish_reason": "stop"})
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield self._format_sse_event({
                "delta": {"content": f"Error: {str(e)}"},
                "finish_reason": "stop",
            })
            yield "data: [DONE]\n\n"

