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
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        content_stripped = content.strip()

        # Pattern 1: Look for tool output in formatted prompts with tags
        tool_output_patterns = [
            r'<PREVIOUS TOOL OUTPUT START>(.*?)<PREVIOUS TOOL OUTPUT END>',
            r'<TOOL OUTPUT>:(.*?)</TOOL OUTPUT>',
            r'Tool Output:\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'tool_output["\']?\s*[:=]\s*["\']?(.*?)["\']?(?:\n|$)',
            r'Output:\s*(.*?)(?:\n\n|\n[A-Z]|$)',
        ]
        for pattern in tool_output_patterns:
            matches = re.findall(pattern, content_stripped, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the first substantial match
                for match in matches:
                    cleaned_match = match.strip()
                    if len(cleaned_match) > 10:  # Ignore very short matches
                        return cleaned_match[:self.max_tool_output_length]

        # Pattern 2: If content looks like a tool output (starts with common tool output indicators)
        tool_indicators = [
            'success', 'error', 'result', 'data', 'response', 'output',
            '{', '[', 'true', 'false', 'null', '##', '#', '|'  # Added markdown indicators
        ]
        content_lower = content_stripped.lower()
        if any(content_lower.startswith(indicator) for indicator in tool_indicators):
            if len(content_stripped) > 20 and len(content_stripped) < self.max_tool_output_length:
                return content_stripped

        # Pattern 3: If content is substantial and looks like structured output (markdown tables, JSON, etc.)
        # This catches tool outputs that don't start with specific indicators
        if len(content_stripped) > 50:
            # Check for markdown table indicators
            if '|' in content_stripped and '\n' in content_stripped:
                return content_stripped[:self.max_tool_output_length]
            # Check for JSON/dict-like content
            if (content_stripped.count('{') > 0 or content_stripped.count('[') > 0) and \
               (content_stripped.count('}') > 0 or content_stripped.count(']') > 0):
                return content_stripped[:self.max_tool_output_length]
            # Check for markdown headers and content
            if content_stripped.count('\n') > 2 and ('##' in content_stripped or '#' in content_stripped):
                return content_stripped[:self.max_tool_output_length]

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

    def find_substring_overlap(self, reference: str, content: str, min_chunk_size: int = 50) -> Optional[Tuple[int, int, str]]:
        """
        Find if reference tool output appears as a substring in content.
        Returns the position and the overlapping substring if found.

        Args:
            reference: Reference tool output to search for
            content: Content to search in
            min_chunk_size: Minimum size of substring to consider as overlap

        Returns:
            Tuple of (start_pos, end_pos, overlapping_text) or None if no overlap found
        """
        if not reference or not content or len(reference) < min_chunk_size:
            return None

        # Split reference into chunks and look for them in content
        ref_lines = reference.split('\n')

        # Try to find substantial chunks of reference in content
        for i in range(len(ref_lines)):
            for j in range(i + 1, len(ref_lines) + 1):
                chunk = '\n'.join(ref_lines[i:j])
                if len(chunk) >= min_chunk_size:
                    if chunk in content:
                        start_pos = content.find(chunk)
                        end_pos = start_pos + len(chunk)
                        return (start_pos, end_pos, chunk)

        # Also try word-based matching for non-newline content
        if '\n' not in reference:
            words = reference.split()
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    chunk = ' '.join(words[i:j])
                    if len(chunk) >= min_chunk_size and chunk in content:
                        start_pos = content.find(chunk)
                        end_pos = start_pos + len(chunk)
                        return (start_pos, end_pos, chunk)

        return None

    def remove_substring_overlap(self, content: str, reference: str, min_chunk_size: int = 50) -> str:
        """
        Remove any substring from content that appears in reference tool output.

        Args:
            content: Content to clean
            reference: Reference tool output
            min_chunk_size: Minimum size of substring to consider for removal

        Returns:
            Content with overlapping substrings removed or truncated
        """
        if not content or not reference:
            return content

        overlap = self.find_substring_overlap(reference, content, min_chunk_size)
        if overlap:
            start_pos, end_pos, overlapping_text = overlap
            # Remove the overlapping substring
            cleaned = content[:start_pos] + content[end_pos:]
            # Clean up extra whitespace
            cleaned = re.sub(r'\n\s*\n', '\n', cleaned).strip()
            return cleaned

        return content
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
        the current tool output or substrings of it.
        Args:
            component_context: List of context messages
            current_tool_output: Current tool output to check against
        Returns:
            Deduplicated component context with overlapping substrings removed
        """
        if not component_context:
            return component_context
        # First, general deduplication
        deduplicated = self.deduplicate_messages(component_context)
        # If we have a current tool output, remove any messages containing it or its substrings
        if current_tool_output:
            filtered = []
            for message in deduplicated:
                content = message.get('content', '')
                if not content:
                    filtered.append(message)
                    continue

                # Check for exact match or high similarity
                extracted_output = self.extract_tool_output_from_content(content)
                if extracted_output and self.are_tool_outputs_similar(extracted_output, current_tool_output):
                    # Skip this message entirely - it's a duplicate
                    continue

                # Check for substring overlap and remove it
                cleaned_content = self.remove_substring_overlap(content, current_tool_output, min_chunk_size=50)

                # Only keep the message if it still has substantial content after cleaning
                if cleaned_content and len(cleaned_content.strip()) > 20:
                    message_copy = message.copy()
                    message_copy['content'] = cleaned_content
                    filtered.append(message_copy)
                elif not cleaned_content or len(cleaned_content.strip()) <= 20:
                    # Message became too small after removing overlap, skip it
                    continue
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


# ============================================================================
# NEW CENTRALIZED DEDUPLICATION FUNCTIONS FOR COMPONENT CONTEXT & CHAT HISTORY
# ============================================================================

def extract_tool_output_from_prompt(prompt: str) -> Optional[str]:
    """
    Extract tool output from prompt's <PREVIOUS TOOL OUTPUT START>...<END> tags.

    Handles various edge cases:
    - Different whitespace patterns (newlines, spaces, tabs)
    - Content with special characters
    - Multiple occurrences (returns first)
    - Empty or malformed tags

    Args:
        prompt: The prompt that may contain tool output tags

    Returns:
        Extracted tool output content, or None if not found
    """
    if not prompt or not isinstance(prompt, str):
        return None

    # Try multiple patterns to handle different whitespace/formatting
    patterns = [
        # Pattern 1: Strict - with newlines
        r'<PREVIOUS TOOL OUTPUT START>\n(.*?)\n<PREVIOUS TOOL OUTPUT END>',
        # Pattern 2: Flexible - any whitespace
        r'<PREVIOUS TOOL OUTPUT START>\s*(.*?)\s*<PREVIOUS TOOL OUTPUT END>',
        # Pattern 3: No whitespace requirement
        r'<PREVIOUS TOOL OUTPUT START>(.*?)<PREVIOUS TOOL OUTPUT END>',
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # Only return if we got substantial content
            if extracted and len(extracted) > 0:
                return extracted

    return None


def calculate_similarity(text1: str, text2: str, threshold: float = 0.75) -> float:
    """
    Calculate similarity between two texts using SequenceMatcher.
    Handles minor character variations, whitespace differences, special characters, etc.

    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (0.0-1.0) - not used but kept for API compatibility

    Returns:
        Similarity ratio (0.0-1.0)
    """
    if not text1 or not text2:
        return 0.0

    # Exact match check first (fastest)
    if text1.strip() == text2.strip():
        return 1.0

    # Normalize texts: remove extra whitespace, convert to lowercase
    # This handles: multiple spaces, tabs, newlines, case differences
    norm_text1 = ' '.join(text1.lower().split())
    norm_text2 = ' '.join(text2.lower().split())

    # If normalized texts are identical, return 1.0
    if norm_text1 == norm_text2:
        return 1.0

    # Calculate similarity using SequenceMatcher
    # This handles: character variations, minor differences
    return SequenceMatcher(None, norm_text1, norm_text2).ratio()


def is_content_similar(content1: str, content2: str, similarity_threshold: float = 0.75) -> bool:
    """
    Check if two content strings are similar (handles minor variations).

    Args:
        content1: First content
        content2: Second content
        similarity_threshold: Threshold for considering content as similar

    Returns:
        True if content is similar enough to be considered duplicate
    """
    if not content1 or not content2:
        return False

    # Exact match (fastest)
    if content1.strip() == content2.strip():
        return True

    # Similarity-based match (handles variations)
    similarity = calculate_similarity(content1, content2, similarity_threshold)
    return similarity >= similarity_threshold


def truncate_overlapping_content(content: str, overlapping_content: str, max_words: int = 30) -> str:
    """
    Truncate content if it contains overlapping_content as substring.

    Args:
        content: Content to potentially truncate
        overlapping_content: Content to check for overlap
        max_words: Maximum words to keep if truncated

    Returns:
        Truncated content or original content
    """
    if not content or not overlapping_content:
        return content

    # Check if overlapping_content is a substring of content
    if overlapping_content.strip() in content:
        # Truncate to max_words
        words = content.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + '...'

    return content


def deduplicate_and_truncate_chat_history(
    chat_history: List[Dict[str, Any]],
    component_context: Optional[List[Dict[str, Any]]] = None,
    current_prompt: Optional[str] = None,
    similarity_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    CENTRALIZED DEDUPLICATION FUNCTION

    Extends chat history with component context, then deduplicates and truncates
    all content to prevent accumulation of duplicate tool outputs.

    Process:
    1. Extract tool output from current prompt
    2. Extend chat_history with component_context (if present)
    3. For each message in chat_history:
       - Extract tool output from message content
       - Check for similarity with current tool output
       - Truncate if similar to reduce token usage
    4. Remove exact duplicate messages

    Args:
        chat_history: Current chat history
        component_context: Component context messages to add
        current_prompt: Current prompt (may contain tool output)
        similarity_threshold: Threshold for similarity matching (0.0-1.0)

    Returns:
        Deduplicated and truncated chat history
    """
    if not chat_history:
        chat_history = []

    # Step 1: Extract tool output from current prompt
    current_tool_output = extract_tool_output_from_prompt(current_prompt) if current_prompt else None

    # Step 2: Extend chat_history with component_context
    extended_history = chat_history.copy()
    if component_context:
        extended_history.extend(component_context)

    # Step 3: Truncate all messages in parallel using ThreadPoolExecutor
    def truncate_message(msg):
        """
        Truncate tool output in a single message if similar to current tool output.

        Handles edge cases:
        - Different whitespace patterns
        - Special characters in content
        - Multiple tool output sections
        - Empty or malformed tags
        - JSON/Plotly outputs with few words but many characters
        """
        if not isinstance(msg, dict):
            return msg

        content = msg.get('content', '')
        if not content:
            return msg

        # Extract tool output from this message
        tool_output_in_msg = extract_tool_output_from_prompt(content) if '<PREVIOUS TOOL OUTPUT START>' in content else None

        # Truncate if similar to current tool output
        if current_tool_output and tool_output_in_msg:
            if is_content_similar(tool_output_in_msg, current_tool_output, similarity_threshold):
                # Use CHARACTER-LEVEL truncation instead of word-level
                # This handles JSON/Plotly outputs that have few words but many characters
                # Keep first 500 characters + truncation marker
                char_limit = 250
                preview = tool_output_in_msg[:char_limit]
                truncated_content = f'{preview}\n[TRUNCATED - Similar to current output]'

                content = re.sub(
                    r'<PREVIOUS TOOL OUTPUT START>\s*.*?\s*<PREVIOUS TOOL OUTPUT END>',
                    f'<PREVIOUS TOOL OUTPUT START>\n{truncated_content}\n<PREVIOUS TOOL OUTPUT END>',
                    content,
                    flags=re.DOTALL
                )

        # Return message with updated content
        processed_msg = msg.copy()
        processed_msg['content'] = content
        return processed_msg

    # Apply truncation to all messages in parallel using ThreadPoolExecutor
    truncated_messages = [None] * len(extended_history)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(truncate_message, msg): i for i, msg in enumerate(extended_history)}
        for future in as_completed(futures):
            idx = futures[future]
            truncated_messages[idx] = future.result()

    # Step 4: Remove exact duplicates while preserving role and content
    processed_history = []
    seen_contents = set()

    for msg in truncated_messages:
        if not isinstance(msg, dict):
            processed_history.append(msg)
            continue

        content = msg.get('content', '')
        content_normalized = content.strip()

        # Skip exact duplicate content
        if content_normalized in seen_contents:
            continue

        seen_contents.add(content_normalized)
        processed_history.append(msg)  # Preserve both role and content

    return processed_history


def truncate_similar_substrings_in_history(
    chat_history: List[Dict[str, Any]],
    current_prompt: Optional[str] = None,
    similarity_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    SUBSTRING TRUNCATION FUNCTION

    Truncates any substring in chat_history content that is similar to
    the current prompt's tool output. This prevents large tool outputs
    from accumulating across multiple calls.

    Process:
    1. Extract tool output from current prompt
    2. For each message in chat_history:
       - Find all substrings that are similar to current tool output
       - Replace similar substrings with [TRUNCATED] marker
       - Reduce token size significantly
    3. Return history with truncated content

    Args:
        chat_history: List of chat history messages
        current_prompt: Current prompt that may contain tool output
        similarity_threshold: Threshold for similarity matching (0.0-1.0)

    Returns:
        Chat history with truncated similar substrings
    """
    if not chat_history or not current_prompt:
        return chat_history

    # Extract tool output from current prompt
    current_tool_output = extract_tool_output_from_prompt(current_prompt)
    if not current_tool_output or len(current_tool_output.strip()) < 50:
        # If no significant tool output, return as-is
        return chat_history

    # Process all messages in parallel
    def truncate_substrings_in_message(msg):
        """Truncate similar substrings in a single message."""
        if not isinstance(msg, dict):
            return msg

        content = msg.get('content', '')
        if not content or not isinstance(content, str):
            return msg

        # Truncate similar substrings in this message
        truncated_content = _truncate_similar_substrings(
            content,
            current_tool_output,
            similarity_threshold
        )

        # Update message with truncated content, preserving role
        processed_msg = msg.copy()
        processed_msg['content'] = truncated_content
        return processed_msg

    # Apply truncation to all messages in parallel using ThreadPoolExecutor
    processed_history = [None] * len(chat_history)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(truncate_substrings_in_message, msg): i for i, msg in enumerate(chat_history)}
        for future in as_completed(futures):
            idx = futures[future]
            processed_history[idx] = future.result()

    return processed_history


def _truncate_similar_substrings(
    content: str,
    reference_text: str,
    similarity_threshold: float = 0.75,
    min_substring_length: int = 100
) -> str:
    """
    Find and truncate substrings in content that are similar to reference_text.

    Args:
        content: Text to search for similar substrings
        reference_text: Reference text to match against
        similarity_threshold: Threshold for similarity (0.0-1.0)
        min_substring_length: Minimum length of substring to consider

    Returns:
        Content with similar substrings replaced with [TRUNCATED] marker
    """
    if not content or not reference_text:
        return content

    # Normalize reference text
    ref_normalized = ' '.join(reference_text.lower().split())
    ref_length = len(ref_normalized)

    if ref_length < min_substring_length:
        return content

    # Find similar substrings using sliding window
    content_normalized = ' '.join(content.lower().split())
    truncated_content = content

    # Try to find matching substrings of various lengths
    for window_size in range(ref_length, min_substring_length - 1, -100):
        if window_size < min_substring_length:
            break

        for i in range(len(content_normalized) - window_size + 1):
            substring = content_normalized[i:i + window_size]

            # Calculate similarity
            similarity = SequenceMatcher(None, ref_normalized, substring).ratio()

            if similarity >= similarity_threshold:
                # Find the original substring in content (case-insensitive)
                # and replace it
                pattern = re.escape(substring)
                truncated_content = re.sub(
                    pattern,
                    '[TRUNCATED - Similar to current output]',
                    truncated_content,
                    flags=re.IGNORECASE
                )
                # Only truncate once per message to avoid over-truncation
                break

        if '[TRUNCATED' in truncated_content:
            break

    return truncated_content




