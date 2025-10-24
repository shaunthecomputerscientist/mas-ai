"""
Base Chat Model - Abstract base class for vanilla SDK wrappers.

This module provides a LangChain-compatible interface for chat models
using vanilla provider SDKs (OpenAI, Google Gemini, etc.) instead of LangChain wrappers.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union
from pydantic import BaseModel
import json


class AIMessage:
    """Simple message class compatible with LangChain's AIMessage."""
    
    def __init__(self, content: str, **kwargs):
        self.content = content
        self.additional_kwargs = kwargs
    
    def __repr__(self):
        return f"AIMessage(content='{self.content[:50]}...')"


class BaseChatModel(ABC):
    """
    Abstract base class for chat model wrappers.
    
    Provides a LangChain-compatible interface:
    - invoke(messages) -> AIMessage
    - ainvoke(messages) -> AIMessage
    - stream(messages) -> Generator[AIMessage]
    - astream(messages) -> AsyncGenerator[AIMessage]
    - with_structured_output(schema, method) -> BaseChatModel
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the base chat model.
        
        Args:
            model: Model name/identifier
            temperature: Sampling temperature (0.0 to 1.0)
            api_key: API key for the provider
            verbose: Enable verbose logging
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.verbose = verbose
        self.extra_kwargs = kwargs
        
        # Structured output configuration
        self._structured_output_schema: Optional[Type[BaseModel]] = None
        self._structured_output_method: Optional[str] = None
    
    @abstractmethod
    def invoke(self, messages: Union[str, List[Dict[str, str]]]) -> AIMessage:
        """
        Synchronously generate a response.
        
        Args:
            messages: Either a string prompt or list of message dicts
                     [{"role": "user", "content": "..."}]
        
        Returns:
            AIMessage with the response content
        """
        pass
    
    @abstractmethod
    async def ainvoke(self, messages: Union[str, List[Dict[str, str]]]) -> AIMessage:
        """
        Asynchronously generate a response.
        
        Args:
            messages: Either a string prompt or list of message dicts
        
        Returns:
            AIMessage with the response content
        """
        pass
    
    @abstractmethod
    def stream(self, messages: Union[str, List[Dict[str, str]]]):
        """
        Synchronously stream response chunks.
        
        Args:
            messages: Either a string prompt or list of message dicts
        
        Yields:
            AIMessage chunks with incremental content
        """
        pass
    
    @abstractmethod
    async def astream(self, messages: Union[str, List[Dict[str, str]]]) -> AsyncGenerator[AIMessage, None]:
        """
        Asynchronously stream response chunks.
        
        Args:
            messages: Either a string prompt or list of message dicts
        
        Yields:
            AIMessage chunks with incremental content
        """
        pass
    
    def with_structured_output(
        self,
        schema: Type[BaseModel],
        method: str = "json_mode"
    ) -> "BaseChatModel":
        """
        Create a new instance configured for structured output.
        
        Args:
            schema: Pydantic model defining the output structure
            method: Method to use ("json_mode" or "function_calling")
        
        Returns:
            New instance of the model configured for structured output
        """
        # Create a copy of the current instance
        new_instance = self.__class__(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            verbose=self.verbose,
            **self.extra_kwargs
        )
        
        # Configure structured output
        new_instance._structured_output_schema = schema
        new_instance._structured_output_method = method
        
        return new_instance
    
    def _normalize_messages(
        self,
        messages: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Normalize input messages to standard format.

        MASAI-specific behavior:
        - First message from user has role="user" (from initiate_agent)
        - Subsequent messages use agent names or tool names as roles
        - These custom roles are preserved for Gemini (converted to labeled text)
        - For OpenAI, custom roles are mapped to "assistant" (handled in subclass)

        Args:
            messages: String or list of message dicts

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            # Validate message format
            normalized = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Handle both dict and LangChain message objects
                    if "role" in msg and "content" in msg:
                        # Keep role as-is (including agent names, tool names)
                        normalized.append(msg)
                    elif "type" in msg and "content" in msg:
                        # LangChain format: {"type": "human", "content": "..."}
                        role_map = {"human": "user", "ai": "assistant", "system": "system"}
                        role = role_map.get(msg["type"], "user")
                        normalized.append({"role": role, "content": msg["content"]})
                    else:
                        raise ValueError(f"Invalid message format: {msg}")
                else:
                    raise ValueError(f"Message must be dict, got {type(msg)}")
            return normalized
        else:
            raise ValueError(f"Messages must be string or list, got {type(messages)}")
    
    def _parse_structured_output(self, json_str: str) -> BaseModel:
        """
        Parse JSON string into Pydantic model.
        
        Args:
            json_str: JSON string from model response
        
        Returns:
            Pydantic model instance
        """
        if not self._structured_output_schema:
            raise ValueError("No structured output schema configured")
        
        try:
            # Parse JSON
            data = json.loads(json_str)
            
            # Create Pydantic model instance
            return self._structured_output_schema(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nContent: {json_str}")
        except Exception as e:
            raise ValueError(f"Failed to create model instance: {e}\nData: {json_str}")
    
    def _get_json_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema from Pydantic model.

        Returns:
            JSON schema dict with additionalProperties: false for OpenAI compatibility
        """
        if not self._structured_output_schema:
            raise ValueError("No structured output schema configured")

        schema = self._structured_output_schema.model_json_schema()

        # OpenAI strict mode requirements:
        # 1. All objects must have additionalProperties: false
        # 2. All properties must be in the required array
        # 3. No default values allowed in required fields

        def process_schema_for_openai(obj):
            if isinstance(obj, dict):
                # Handle objects with type="object"
                if obj.get("type") == "object":
                    # Set additionalProperties to false (even if it was true)
                    obj["additionalProperties"] = False

                    # For OpenAI strict mode: all properties must be required
                    if "properties" in obj:
                        # Get all property names
                        all_props = list(obj["properties"].keys())
                        # Set required to include all properties
                        obj["required"] = all_props

                        # Remove default values from properties (OpenAI strict mode doesn't allow them)
                        for prop_name, prop_schema in obj["properties"].items():
                            if "default" in prop_schema:
                                del prop_schema["default"]

                # Recursively process all values
                for value in obj.values():
                    process_schema_for_openai(value)
            elif isinstance(obj, list):
                # Recursively process all items in arrays (including anyOf, oneOf, allOf)
                for item in obj:
                    process_schema_for_openai(item)

        process_schema_for_openai(schema)
        return schema
    
    def _is_structured_output_enabled(self) -> bool:
        """Check if structured output is configured."""
        return self._structured_output_schema is not None
    
    def __repr__(self):
        structured = f", structured={self._structured_output_schema.__name__}" if self._is_structured_output_enabled() else ""
        return f"{self.__class__.__name__}(model='{self.model}', temperature={self.temperature}{structured})"

