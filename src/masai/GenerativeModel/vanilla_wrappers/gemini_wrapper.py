"""
Google Gemini Chat Model Wrapper using vanilla Google Generative AI SDK.

Drop-in replacement for langchain_google_genai.chat_models.ChatGoogleGenerativeAI
"""

import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union
from pydantic import BaseModel
import json

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Google Generative AI SDK not installed. Install with: pip install google-generativeai>=0.8.5"
    )

from .base_chat_model import BaseChatModel, AIMessage


class ChatGoogleGenerativeAI(BaseChatModel):
    """
    Google Gemini chat model wrapper using vanilla Google Generative AI SDK.

    Compatible with LangChain's ChatGoogleGenerativeAI interface:
    - invoke(), ainvoke(), stream(), astream()
    - with_structured_output()

    Supports:
    - Gemini 2.5 Pro, Gemini 2.5 Flash
    - Gemini 2.0 Flash, Gemini 1.5 Pro/Flash
    - Structured output via response_schema
    - Thinking models with thinkingBudget

    ═══════════════════════════════════════════════════════════════════════════
    📋 AVAILABLE PARAMETERS FOR GEMINI MODELS
    ═══════════════════════════════════════════════════════════════════════════

    Use these parameter names in model_config.json or config_dict:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ CORE PARAMETERS (Required)                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • model: str - Model name (e.g., "gemini-2.5-pro")                      │
    │ • temperature: float - Sampling temperature (0.0-2.0, default: 0.7)     │
    │ • api_key: str - Google API key (or set GOOGLE_API_KEY env var)         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ GENERATION PARAMETERS                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • max_output_tokens: int - Maximum response length (default: model max) │
    │ • top_p: float - Nucleus sampling (0.0-1.0, default: 0.95)              │
    │ • top_k: int - Top-k sampling (1-100, default: 40)                      │
    │ • stop_sequences: List[str] - Stop generation at these strings          │
    │ • candidate_count: int - Number of response variations (1-8)            │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ SAFETY PARAMETERS                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • safety_settings: List[Dict] - Safety category/threshold pairs         │
    │   Example:                                                               │
    │   [                                                                      │
    │     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},│
    │     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}│
    │   ]                                                                      │
    │   Categories: HARASSMENT, HATE_SPEECH, SEXUALLY_EXPLICIT,               │
    │               DANGEROUS_CONTENT                                          │
    │   Thresholds: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE,      │
    │               BLOCK_LOW_AND_ABOVE                                        │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ THINKING/REASONING PARAMETERS (Gemini 2.5+ models)                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • thinking_budget: int - Token budget for thinking                      │
    │   • -1: Dynamic (model decides)                                          │
    │   • 0: Thinking disabled                                                 │
    │   • 1-10000: Fixed token budget                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ LOGPROBS PARAMETERS                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • response_logprobs: bool - Return log probabilities (default: False)   │
    │ • logprobs: int - Number of top candidate tokens (1-20)                 │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ PENALTY PARAMETERS                                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • presence_penalty: float - Penalize existing tokens (-2.0 to 2.0)      │
    │ • frequency_penalty: float - Penalize by frequency (-2.0 to 2.0)        │
    │ • seed: int - For deterministic output                                   │
    └─────────────────────────────────────────────────────────────────────────┘

    ═══════════════════════════════════════════════════════════════════════════
    📝 USAGE EXAMPLES
    ═══════════════════════════════════════════════════════════════════════════

    Example 1: In model_config.json
    --------------------------------
    {
        "router": {
            "model_name": "gemini-2.5-pro",
            "category": "gemini",
            "temperature": 0.2,
            "max_output_tokens": 2048,
            "top_k": 20,
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
            ],
            "thinking_budget": -1
        }
    }

    Example 2: In config_dict (runtime override)
    --------------------------------------------
    config_dict = {
        "router_temperature": 0.3,
        "router_max_output_tokens": 4096,
        "router_top_k": 40,
        "router_safety_settings": [...]
    }

    ═══════════════════════════════════════════════════════════════════════════
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        api_key: Optional[str] = None,

        # Standard parameters (all models)
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,

        # Gemini-specific parameters
        safety_settings: Optional[List[Dict[str, str]]] = None,
        thinking_budget: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        candidate_count: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
        logprobs: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,

        # Catch-all for future parameters
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize Google Gemini chat model.

        See class docstring for comprehensive parameter documentation.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            verbose=verbose,
            **kwargs
        )

        # Store all parameters
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.safety_settings = safety_settings or self._get_default_safety_settings()
        self.thinking_budget = thinking_budget
        self.stop_sequences = stop_sequences
        self.candidate_count = candidate_count
        self.response_logprobs = response_logprobs
        self.logprobs = logprobs
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.model_kwargs = model_kwargs or {}

        # Configure Google Generative AI
        genai.configure(api_key=self.api_key)

        # Check if this is a thinking model
        self._is_thinking_model = self._check_thinking_model()

        if self.verbose:
            print(f"✅ Initialized {self.__class__.__name__}(model='{self.model}', thinking={self._is_thinking_model})")
    
    def _check_thinking_model(self) -> bool:
        """Check if the model is a thinking/reasoning model."""
        return (
            self.model.startswith('gemini-2.5') or
            'thinking' in self.model.lower()
        )

    def _get_default_safety_settings(self) -> List[Dict[str, str]]:
        """
        Get default safety settings for Gemini API.

        Default: BLOCK_ONLY_HIGH for all categories (balanced approach)

        Returns:
            List of safety settings
        """
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
    
    def _prepare_generation_config(self) -> Dict[str, Any]:
        """
        Prepare generation configuration for Gemini API.

        Returns:
            Dict of generation config parameters
        """
        config = {
            "temperature": self.temperature,
        }

        # Add standard parameters
        if self.max_output_tokens is not None:
            config["max_output_tokens"] = self.max_output_tokens

        if self.top_p is not None:
            config["top_p"] = self.top_p

        if self.top_k is not None:
            config["top_k"] = self.top_k

        # Add thinking budget for thinking models
        if self._is_thinking_model:
            if self.thinking_budget is not None:
                config["thinking_budget"] = self.thinking_budget
            elif "thinkingBudget" in self.model_kwargs:
                # Backward compatibility
                config["thinking_budget"] = self.model_kwargs["thinkingBudget"]

        # Add stop sequences
        if self.stop_sequences is not None:
            config["stop_sequences"] = self.stop_sequences

        # Add candidate count
        if self.candidate_count is not None:
            config["candidate_count"] = self.candidate_count

        # Add response logprobs
        if self.response_logprobs is not None:
            config["response_logprobs"] = self.response_logprobs

        # Add logprobs
        if self.logprobs is not None:
            config["logprobs"] = self.logprobs

        # Add presence penalty
        if self.presence_penalty is not None:
            config["presence_penalty"] = self.presence_penalty

        # Add frequency penalty
        if self.frequency_penalty is not None:
            config["frequency_penalty"] = self.frequency_penalty

        # Add seed
        if self.seed is not None:
            config["seed"] = self.seed

        # Add structured output configuration
        if self._is_structured_output_enabled():
            # Get raw Pydantic schema (NOT OpenAI-processed schema)
            # OpenAI's _get_json_schema() forces all fields to be required, which we don't want for Gemini
            schema = self._structured_output_schema.model_json_schema()
            cleaned_schema = self._convert_to_gemini_schema(schema)

            # Debug: Print schema being sent (can be disabled in production)
            if self.verbose:
                import json
                print("🔍 DEBUG: Gemini schema being sent:")
                print(json.dumps(cleaned_schema, indent=2))

            config["response_mime_type"] = "application/json"
            config["response_schema"] = cleaned_schema

        # Add any extra model_kwargs (but not thinkingBudget again)
        for key, value in self.model_kwargs.items():
            if key not in ["thinkingBudget"] and key not in config:
                config[key] = value

        return config
    
    def _convert_to_gemini_schema(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON schema to Gemini schema format.

        Gemini uses a slightly different schema format than standard JSON Schema.
        Removes unsupported fields like 'default', 'additionalProperties', etc.
        Handles anyOf/allOf/oneOf by flattening to the first type or merging properties.

        Args:
            json_schema: Standard JSON schema from Pydantic

        Returns:
            Gemini-compatible schema
        """
        import copy

        # Deep copy to avoid modifying original
        cleaned_schema = copy.deepcopy(json_schema)

        # Clean up schema for Gemini compatibility
        def clean_schema(obj):
            """Remove unsupported fields from schema recursively."""
            if isinstance(obj, dict):
                # Handle anyOf/allOf/oneOf (Gemini doesn't support these)
                if 'anyOf' in obj:
                    # Flatten anyOf - prefer string type for flexibility
                    any_of_schemas = obj.pop('anyOf')

                    # Strategy: Find string type first (most flexible for Gemini)
                    # Otherwise use first non-null type
                    string_schema = None
                    first_non_null = None

                    for schema in any_of_schemas:
                        if isinstance(schema, dict):
                            if schema.get('type') == 'string':
                                string_schema = schema
                                break
                            elif schema.get('type') != 'null' and not first_non_null:
                                first_non_null = schema

                    # Prefer string, fallback to first non-null
                    chosen_schema = string_schema or first_non_null
                    if chosen_schema:
                        obj.update(chosen_schema)

                if 'allOf' in obj:
                    # Merge all schemas in allOf
                    all_of_schemas = obj.pop('allOf')
                    for schema in all_of_schemas:
                        if isinstance(schema, dict):
                            # Merge properties
                            if 'properties' in schema:
                                if 'properties' not in obj:
                                    obj['properties'] = {}
                                obj['properties'].update(schema['properties'])
                            # Merge other fields
                            for key, value in schema.items():
                                if key not in ['properties']:
                                    obj[key] = value

                if 'oneOf' in obj:
                    # Flatten oneOf - prefer string type
                    one_of_schemas = obj.pop('oneOf')
                    if one_of_schemas:
                        # Try to find string type first
                        string_schema = next((s for s in one_of_schemas if isinstance(s, dict) and s.get('type') == 'string'), None)
                        obj.update(string_schema or one_of_schemas[0])

                # Gemini requires non-empty properties for OBJECT type
                # If type is object but properties is empty or missing, change to string
                if obj.get('type') == 'object':
                    if 'properties' not in obj or not obj['properties']:
                        # Change to string type (more flexible for Gemini)
                        obj['type'] = 'string'
                        obj.pop('properties', None)
                        obj.pop('required', None)

                # Remove unsupported fields
                unsupported_fields = ['default', 'additionalProperties', 'title', '$defs']
                for field in unsupported_fields:
                    obj.pop(field, None)

                # Recursively clean nested objects
                for value in list(obj.values()):  # Use list() to avoid dict size change during iteration
                    clean_schema(value)
            elif isinstance(obj, list):
                for item in obj:
                    clean_schema(item)

        clean_schema(cleaned_schema)
        return cleaned_schema
    
    def _normalize_messages_to_gemini(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Convert message list to Gemini prompt format.

        Gemini's generate_content expects a single prompt string or
        a list of Content objects. For MASAI's use case (single formatted prompt),
        we use a string for simplicity.

        Note: Gemini doesn't natively support system messages. Following LangChain's
        approach, we prepend system content to the first user message.

        MASAI-specific behavior:
        - First message from user has role="user" (from initiate_agent)
        - Subsequent messages use agent names or tool names as roles
        - This preserves multi-agent conversation context

        Args:
            messages: Normalized message list [{"role": "user/assistant/system/<agent_name>/<tool_name>", "content": "..."}]

        Returns:
            Single prompt string suitable for Gemini
        """
        # Handle system message by prepending to first user message (LangChain approach)
        system_content = None
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if not content:  # Skip empty messages
                continue

            if role == "system":
                # Store system content to prepend to first user message
                system_content = content
            elif role == "user":
                # Prepend system content if present (only once)
                if system_content:
                    prompt_parts.append(f"{system_content}\n\n{content}")
                    system_content = None  # Clear after use
                else:
                    prompt_parts.append(content)
            elif role == "assistant":
                # For multi-turn conversations, label assistant messages
                prompt_parts.append(f"Assistant: {content}")
            else:
                # MASAI uses agent names and tool names as roles
                # Format: "<AgentName>: <content>" or "<ToolName>: <content>"
                # This preserves multi-agent conversation context
                prompt_parts.append(f"{role}: {content}")

        # If only system message exists (edge case), return it as user message
        if not prompt_parts and system_content:
            return system_content

        return "\n\n".join(prompt_parts)
    
    def invoke(self, messages: Union[str, List[Dict[str, str]]]) -> AIMessage:
        """
        Synchronously generate a response.

        Args:
            messages: String prompt or list of message dicts

        Returns:
            Pydantic model if structured output enabled, otherwise AIMessage
        """
        normalized_messages = self._normalize_messages(messages)
        prompt = self._normalize_messages_to_gemini(normalized_messages)
        config = self._prepare_generation_config()

        if self.verbose:
            print(f"🟢 Gemini invoke: {self.model}")

        # Create model instance
        model_instance = genai.GenerativeModel(
            model_name=self.model,
            generation_config=config
        )

        # Make API call
        response = model_instance.generate_content(prompt)

        # Check if response has valid content
        try:
            content = response.text
        except ValueError as e:
            # Gemini blocked response or returned empty content
            # Check finish_reason to determine cause
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                error_msg = f"Gemini finish_reason={finish_reason}: {str(e)}"

                # Map finish_reason to human-readable message
                reason_map = {
                    1: "STOP (no content generated - likely safety filter or empty prompt)",
                    2: "MAX_TOKENS (response too long)",
                    3: "SAFETY (content filtered by safety settings)",
                    4: "RECITATION (content blocked due to recitation)",
                    5: "OTHER (unknown error)"
                }
                readable_reason = reason_map.get(finish_reason, f"Unknown ({finish_reason})")
                error_msg = f"Gemini error: {readable_reason}. Original error: {str(e)}"
            else:
                error_msg = f"Gemini returned no content: {str(e)}"

            # Log the error
            if self.verbose:
                print(f"❌ {error_msg}")

            # Raise exception with detailed message
            raise ValueError(error_msg)

        # Parse structured output if enabled
        if self._is_structured_output_enabled():
            # Gemini returns parsed object directly
            if hasattr(response, 'parsed') and response.parsed:
                # Return Pydantic model directly (MASAI expects .model_dump() method)
                return response.parsed
            else:
                # Fallback to manual parsing
                parsed = self._parse_structured_output(content)
                # Return Pydantic model directly (MASAI expects .model_dump() method)
                return parsed

        return AIMessage(content=content)
    
    async def ainvoke(self, messages: Union[str, List[Dict[str, str]]]) -> AIMessage:
        """
        Asynchronously generate a response.
        
        Args:
            messages: String prompt or list of message dicts
        
        Returns:
            AIMessage with response (or Pydantic model if structured output)
        """
        # Note: Google SDK doesn't have native async support yet
        # We'll use asyncio.to_thread to run sync code in async context
        import asyncio
        return await asyncio.to_thread(self.invoke, messages)
    
    def stream(self, messages: Union[str, List[Dict[str, str]]]):
        """
        Synchronously stream response chunks.

        Args:
            messages: String prompt or list of message dicts

        Yields:
            AIMessage chunks with incremental content
        """
        normalized_messages = self._normalize_messages(messages)
        prompt = self._normalize_messages_to_gemini(normalized_messages)
        config = self._prepare_generation_config()

        if self.verbose:
            print(f"🟢 Gemini stream: {self.model}")

        # Create model instance
        model_instance = genai.GenerativeModel(
            model_name=self.model,
            generation_config=config
        )

        # Stream API call
        response_stream = model_instance.generate_content(prompt, stream=True)

        accumulated_content = ""
        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                content = chunk.text
                accumulated_content += content

                # For structured output, yield parsed chunks
                if self._is_structured_output_enabled():
                    # Try to parse accumulated content
                    try:
                        parsed = self._parse_structured_output(accumulated_content)
                        yield AIMessage(content=accumulated_content, parsed=parsed)
                    except:
                        # Not yet complete JSON, yield raw content
                        yield AIMessage(content=content)
                else:
                    yield AIMessage(content=content)
    
    async def astream(self, messages: Union[str, List[Dict[str, str]]]) -> AsyncGenerator[AIMessage, None]:
        """
        Asynchronously stream response chunks.

        Args:
            messages: String prompt or list of message dicts

        Yields:
            AIMessage chunks with incremental content
        """
        normalized_messages = self._normalize_messages(messages)
        prompt = self._normalize_messages_to_gemini(normalized_messages)
        config = self._prepare_generation_config()

        if self.verbose:
            print(f"🟢 Gemini astream: {self.model}")

        # Note: Google SDK doesn't have native async streaming yet
        # We'll use asyncio.to_thread to run sync streaming in async context
        import asyncio

        # Create a queue for chunks
        from queue import Queue
        chunk_queue = Queue()
        done_sentinel = object()

        def stream_worker():
            last_parsed = None
            try:
                # Create model instance
                model_instance = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config=config
                )

                # Stream API call
                response_stream = model_instance.generate_content(prompt, stream=True)

                accumulated_content = ""
                for chunk in response_stream:
                    # Check if chunk has text content
                    if hasattr(chunk, 'text'):
                        try:
                            content = chunk.text
                            accumulated_content += content

                            # For structured output, yield parsed chunks
                            if self._is_structured_output_enabled():
                                try:
                                    parsed = self._parse_structured_output(accumulated_content)
                                    last_parsed = parsed
                                    chunk_queue.put(AIMessage(content=accumulated_content, parsed=parsed))
                                except:
                                    chunk_queue.put(AIMessage(content=content))
                            else:
                                chunk_queue.put(AIMessage(content=content))
                        except ValueError as e:
                            # Gemini blocked response (safety filter or validation error)
                            # Check finish_reason to determine cause
                            error_msg = f"Gemini blocked response: {str(e)}"
                            if hasattr(chunk, 'candidates') and chunk.candidates:
                                finish_reason = chunk.candidates[0].finish_reason

                                # Map finish_reason to human-readable message
                                reason_map = {
                                    1: "STOP (no content generated - likely safety filter or empty prompt)",
                                    2: "MAX_TOKENS (response too long)",
                                    3: "SAFETY (content filtered by safety settings)",
                                    4: "RECITATION (content blocked due to recitation)",
                                    5: "OTHER (unknown error)"
                                }
                                readable_reason = reason_map.get(finish_reason, f"Unknown ({finish_reason})")
                                error_msg = f"Gemini error: {readable_reason}. Original error: {str(e)}"

                            # Raise exception to be caught by outer try-except
                            raise ValueError(error_msg)

                # For structured output, yield the final Pydantic model instance
                # This is what MASAI expects (has model_dump() method)
                if self._is_structured_output_enabled() and last_parsed:
                    chunk_queue.put(last_parsed)
                elif self._is_structured_output_enabled() and not last_parsed:
                    # No valid parsed response - raise error
                    raise ValueError("No valid structured output received from Gemini")
            finally:
                chunk_queue.put(done_sentinel)

        # Start streaming in background thread
        import threading
        thread = threading.Thread(target=stream_worker)
        thread.start()

        # Yield chunks from queue
        while True:
            chunk = await asyncio.to_thread(chunk_queue.get)
            if chunk is done_sentinel:
                break
            yield chunk

