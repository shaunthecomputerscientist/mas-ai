"""
Deduplication utilities for MASAI framework.
This module provides utilities to detect and remove duplicate tool outputs
from chat history and component context to prevent redundant information
from accumulating in LLM prompts.
"""
import re
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from difflib import SequenceMatcher
class ToolOutputDeduplicator:
    """
    Handles deduplication of tool outputs in chat history and component context.
    """
    def __init__(self, similarity_threshold: float = 0.85, max_tool_output_length: int = 1000):
        """
        Initialize the deduplicator.
        Args:
            similarity_threshold: Threshold for considering two tool outputs as similar (0.0-1.0)
            max_tool_output_length: Maximum length of tool output to consider for deduplication
        """
        self.similarity_threshold = similarity_threshold
        self.max_tool_output_length = max_tool_output_length
        self._tool_output_hashes: Set[str] = set()
        self._tool_output_cache: Dict[str, str] = {}
    def extract_tool_output_from_content(self, content: str) -> Optional[str]:
        """
        Extract tool output from message content using various patterns.
        Args:
            content: Message content that may contain tool output
        Returns:
            Extracted tool output or None if not found
        """
        if not content or not isinstance(content, str):
            return None
        # Pattern 1: Look for tool output in formatted prompts
        tool_output_patterns = [
            r'<TOOL OUTPUT>:(.*?)</TOOL OUTPUT>',
            r'Tool Output:\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'tool_output["\']?\s*[:=]\s*["\']?(.*?)["\']?(?:\n|$)',
            r'Output:\s*(.*?)(?:\n\n|\n[A-Z]|$)',
        ]
        for pattern in tool_output_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the first substantial match
                for match in matches:
                    cleaned_match = match.strip()
                    if len(cleaned_match) > 10:  # Ignore very short matches
                        return cleaned_match[:self.max_tool_output_length]
        # Pattern 2: If content looks like a tool output (starts with common tool output indicators)
        tool_indicators = [
            'success', 'error', 'result', 'data', 'response', 'output',
            '{', '[', 'true', 'false', 'null'
        ]
        content_lower = content.lower().strip()
        if any(content_lower.startswith(indicator) for indicator in tool_indicators):
            if len(content) > 20 and len(content) < self.max_tool_output_length:
                return content.strip()
        return None
    def get_content_hash(self, content: str) -> str:
        """
        Generate a hash for content to enable fast duplicate detection.
        Args:
            content: Content to hash
        Returns:
            SHA256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    def are_tool_outputs_similar(self, output1: str, output2: str) -> bool:
        """
        Check if two tool outputs are similar enough to be considered duplicates.
        Args:
            output1: First tool output
            output2: Second tool output
        Returns:
            True if outputs are similar enough to be considered duplicates
        """
        if not output1 or not output2:
            return False
        # Exact match
        if output1.strip() == output2.strip():
            return True
        # Hash-based quick check
        hash1 = self.get_content_hash(output1)
        hash2 = self.get_content_hash(output2)
        if hash1 == hash2:
            return True
        # Similarity-based check for longer outputs
        if len(output1) > 50 and len(output2) > 50:
            similarity = SequenceMatcher(None, output1, output2).ratio()
            return similarity >= self.similarity_threshold
        return False
    def deduplicate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate tool outputs from a list of messages.
        Args:
            messages: List of message dictionaries
        Returns:
            Deduplicated list of messages
        """
        if not messages:
            return messages
        deduplicated = []
        seen_tool_outputs = []
        for message in messages:
            content = message.get('content', '')
            role = message.get('role', '')
            if not content:
                deduplicated.append(message)
                continue
            # Special handling for tool messages - content is the raw tool output
            if role == 'tool':
                tool_output = content.strip()
            else:
                # Extract potential tool output from formatted content
                tool_output = self.extract_tool_output_from_content(content)
            if tool_output:
                # Check if this tool output is similar to any we've seen
                is_duplicate = False
                for seen_output in seen_tool_outputs:
                    if self.are_tool_outputs_similar(tool_output, seen_output):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    seen_tool_outputs.append(tool_output)
                    deduplicated.append(message)
                else:
                    # It's a duplicate, skip this message
                    continue
            else:
                # Not a tool output, keep the message
                deduplicated.append(message)
        return deduplicated
    def deduplicate_component_context(
        self,
        component_context: List[Dict[str, Any]],
        current_tool_output: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate component context, optionally removing messages that contain
        the current tool output.
        Args:
            component_context: List of context messages
            current_tool_output: Current tool output to check against
        Returns:
            Deduplicated component context
        """
        if not component_context:
            return component_context
        # First, general deduplication
        deduplicated = self.deduplicate_messages(component_context)
        # If we have a current tool output, remove any messages containing it
        if current_tool_output:
            filtered = []
            for message in deduplicated:
                content = message.get('content', '')
                extracted_output = self.extract_tool_output_from_content(content)
                if extracted_output:
                    if not self.are_tool_outputs_similar(extracted_output, current_tool_output):
                        filtered.append(message)
                else:
                    filtered.append(message)
            return filtered
        return deduplicated
    def clean_chat_history(self, chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean chat history by removing duplicate tool outputs while preserving
        conversation flow.
        Args:
            chat_history: List of chat history messages
        Returns:
            Cleaned chat history
        """
        return self.deduplicate_messages(chat_history)
# Global instance for use across the framework
_global_deduplicator = ToolOutputDeduplicator()
def get_deduplicator() -> ToolOutputDeduplicator:
    """Get the global deduplicator instance."""
    return _global_deduplicator
def deduplicate_tool_outputs(
    messages: List[Dict[str, Any]],
    current_tool_output: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to deduplicate tool outputs from messages.
    Args:
        messages: List of messages to deduplicate
        current_tool_output: Current tool output to check against
    Returns:
        Deduplicated messages
    """
    deduplicator = get_deduplicator()
    if current_tool_output:
        return deduplicator.deduplicate_component_context(messages, current_tool_output)
    else:
        return deduplicator.deduplicate_messages(messages)






