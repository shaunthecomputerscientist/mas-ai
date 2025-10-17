"""
Custom ChatPromptTemplate classes to replace LangChain's prompt templates.
These are drop-in replacements that maintain API compatibility.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel


class PromptTemplate:
    """
    A simple prompt template that formats strings with variables.
    
    This is a drop-in replacement for langchain_core.prompts.PromptTemplate.
    """
    
    def __init__(self, template: str, input_variables: Optional[List[str]] = None):
        """
        Initialize a PromptTemplate.
        
        Args:
            template (str): The template string with {variable} placeholders.
            input_variables (List[str], optional): List of variable names in the template.
        """
        self.template = template
        self.input_variables = input_variables or []
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variable names and their values.
        
        Returns:
            str: Formatted template string.
        """
        return self.template.format(**kwargs)
    
    def format_prompt(self, **kwargs) -> str:
        """
        Format the template (alias for format method).
        
        Args:
            **kwargs: Variable names and their values.
        
        Returns:
            str: Formatted template string.
        """
        return self.format(**kwargs)


class BaseMessagePromptTemplate:
    """Base class for message prompt templates."""
    
    def __init__(self, prompt: Union[PromptTemplate, str]):
        """
        Initialize a message prompt template.
        
        Args:
            prompt (Union[PromptTemplate, str]): The prompt template or string.
        """
        if isinstance(prompt, str):
            self.prompt = PromptTemplate(template=prompt)
        else:
            self.prompt = prompt
    
    def format(self, **kwargs) -> str:
        """
        Format the prompt with provided variables.
        
        Args:
            **kwargs: Variable names and their values.
        
        Returns:
            str: Formatted prompt string.
        """
        return self.prompt.format(**kwargs)


class SystemMessagePromptTemplate(BaseMessagePromptTemplate):
    """
    A prompt template for system messages.
    
    This is a drop-in replacement for langchain_core.prompts.SystemMessagePromptTemplate.
    """
    
    def __init__(self, prompt: Union[PromptTemplate, str]):
        """
        Initialize a system message prompt template.
        
        Args:
            prompt (Union[PromptTemplate, str]): The prompt template or string.
        """
        super().__init__(prompt)
        self.role = "system"


class HumanMessagePromptTemplate(BaseMessagePromptTemplate):
    """
    A prompt template for human messages.
    
    This is a drop-in replacement for langchain_core.prompts.HumanMessagePromptTemplate.
    """
    
    def __init__(self, prompt: Union[PromptTemplate, str]):
        """
        Initialize a human message prompt template.
        
        Args:
            prompt (Union[PromptTemplate, str]): The prompt template or string.
        """
        super().__init__(prompt)
        self.role = "user"


class AIMessagePromptTemplate(BaseMessagePromptTemplate):
    """
    A prompt template for AI messages.
    
    This is a drop-in replacement for langchain_core.prompts.AIMessagePromptTemplate.
    """
    
    def __init__(self, prompt: Union[PromptTemplate, str]):
        """
        Initialize an AI message prompt template.
        
        Args:
            prompt (Union[PromptTemplate, str]): The prompt template or string.
        """
        super().__init__(prompt)
        self.role = "assistant"


class ChatPromptTemplate:
    """
    A chat prompt template that combines multiple message templates.
    
    This is a drop-in replacement for langchain_core.prompts.ChatPromptTemplate.
    Maintains full API compatibility with LangChain's implementation.
    """
    
    def __init__(self, messages: List[BaseMessagePromptTemplate]):
        """
        Initialize a ChatPromptTemplate.
        
        Args:
            messages (List[BaseMessagePromptTemplate]): List of message templates.
        """
        self.messages = messages
    
    @classmethod
    def from_messages(cls, messages: List[Union[BaseMessagePromptTemplate, tuple]]) -> "ChatPromptTemplate":
        """
        Create a ChatPromptTemplate from a list of messages.
        
        Args:
            messages (List[Union[BaseMessagePromptTemplate, tuple]]): 
                List of message templates or tuples of (role, content).
        
        Returns:
            ChatPromptTemplate: A new ChatPromptTemplate instance.
        
        Example:
            >>> template = ChatPromptTemplate.from_messages([
            ...     SystemMessagePromptTemplate(prompt=PromptTemplate(template="You are a helpful assistant.")),
            ...     HumanMessagePromptTemplate(prompt=PromptTemplate(template="Hello {name}!"))
            ... ])
        """
        processed_messages = []
        for msg in messages:
            if isinstance(msg, BaseMessagePromptTemplate):
                processed_messages.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role == "system":
                    processed_messages.append(SystemMessagePromptTemplate(prompt=content))
                elif role in ["user", "human"]:
                    processed_messages.append(HumanMessagePromptTemplate(prompt=content))
                elif role in ["assistant", "ai"]:
                    processed_messages.append(AIMessagePromptTemplate(prompt=content))
                else:
                    raise ValueError(f"Unknown role: {role}")
            else:
                raise ValueError(f"Invalid message format: {msg}")
        
        return cls(messages=processed_messages)
    
    def format(self, **kwargs) -> str:
        """
        Format all messages with provided variables and return as a string.

        This matches LangChain's behavior where .format() returns a human-readable string
        with role labels (System:, Human:, AI:).

        Args:
            **kwargs: Variable names and their values.

        Returns:
            str: Formatted messages as a string with role labels.

        Example:
            >>> template.format(name="Alice")
            'System: You are a helpful assistant.\\n\\nHuman: Hello Alice!'
        """
        formatted_parts = []
        for msg in self.messages:
            # Map role to display label (matching LangChain's format)
            role_label_map = {
                "system": "System",
                "user": "Human",
                "assistant": "AI"
            }
            role_label = role_label_map.get(msg.role, msg.role.capitalize())
            content = msg.format(**kwargs)
            formatted_parts.append(f"{role_label}: {content}")

        # Join with single newline to match LangChain's behavior
        # (LangChain uses "\n" between messages, not "\n\n")
        return "\n".join(formatted_parts)
    
    def format_messages(self, **kwargs) -> List[Dict[str, str]]:
        """
        Format all messages and return as a list of message dictionaries.

        This is different from .format() which returns a string.
        This method returns the structured message format needed for LLM APIs.

        Args:
            **kwargs: Variable names and their values.

        Returns:
            List[Dict[str, str]]: List of formatted messages with 'role' and 'content'.

        Example:
            >>> template.format_messages(name="Alice")
            [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello Alice!'}
            ]
        """
        formatted_messages = []
        for msg in self.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.format(**kwargs)
            })
        return formatted_messages
    
    def format_prompt(self, **kwargs) -> str:
        """
        Format all messages (alias for format method that returns string).

        Args:
            **kwargs: Variable names and their values.

        Returns:
            str: Formatted messages as a string.
        """
        return self.format(**kwargs)
    
    def __str__(self) -> str:
        """String representation of the template."""
        return f"ChatPromptTemplate(messages={len(self.messages)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the template."""
        return f"ChatPromptTemplate(messages={self.messages})"


# For backward compatibility with LangChain imports
__all__ = [
    "PromptTemplate",
    "ChatPromptTemplate",
    "SystemMessagePromptTemplate",
    "HumanMessagePromptTemplate",
    "AIMessagePromptTemplate",
    "BaseMessagePromptTemplate"
]

