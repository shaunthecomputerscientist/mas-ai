import json
import re
import logging

# Setup logger for JSON parser
logger = logging.getLogger(__name__)

def repair_json_string(json_str: str) -> str:
    """
    Attempt to repair common JSON string issues:
    1. Invalid escape sequences (\' which is not valid JSON)
    2. Missing closing brackets/braces

    IMPORTANT: This function does NOT escape control characters in field values!
    - Field values (like code) should be preserved as-is
    - Only JSON structure issues are fixed
    - Control characters in values are the responsibility of the LLM to escape

    Uses a stack-based approach to determine the correct order of closing characters.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    # Remove leading/trailing whitespace
    json_str = json_str.strip()

    repairs_made = []

    # REMOVED: Control character escaping
    # Reason: This was too aggressive and modified field values (like code)
    # The LLM should generate proper JSON with escaped control characters
    # If the LLM generates actual newlines, json.loads() will fail and we'll
    # get a clear error message instead of silently corrupting the data

    # Fix 1: Handle invalid escape sequences carefully
    # \' is not a valid JSON escape sequence (only \" \\ \/ \b \f \n \r \t \uXXXX are valid)
    #
    # The LLM sometimes generates: {"code": "db[\'tasks\']"}
    # This is INVALID JSON because \' is not a valid escape sequence
    # It should be: {"code": "db['tasks']"} (no escape needed for single quotes in JSON)
    #
    # Strategy: Remove the backslash before single quotes ONLY when inside JSON string values
    def fix_invalid_escapes(json_str):
        """Remove invalid escape sequences like \' from JSON strings"""
        result = []
        in_string = False
        escape_next = False
        i = 0

        while i < len(json_str):
            char = json_str[i]

            if escape_next:
                # Previous char was backslash
                if char == "'":
                    # \' is invalid in JSON - just keep the quote, drop the backslash
                    result.append("'")
                else:
                    # Valid escape sequence - keep both backslash and char
                    result.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\' and in_string:
                # Check if next char is a single quote
                if i + 1 < len(json_str) and json_str[i + 1] == "'":
                    # Don't add the backslash yet, mark for next iteration
                    escape_next = True
                    i += 1
                    continue
                else:
                    # Valid escape sequence
                    result.append(char)
            elif char == '"':
                # Toggle string state
                in_string = not in_string
                result.append(char)
            else:
                result.append(char)

            i += 1

        return ''.join(result)

    # Try to fix invalid escape sequences
    try:
        if "\\'" in json_str:
            json_str = fix_invalid_escapes(json_str)
            repairs_made.append("fixed invalid escape sequences")
    except Exception as e:
        logger.warning(f"Failed to fix invalid escape sequences: {e}")

    # Fix 2: Use a stack to track opening brackets/braces and determine what's missing
    stack = []
    for char in json_str:
        if char in ('{', '['):
            stack.append(char)
        elif char == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif char == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    # If stack is not empty, we have unclosed brackets/braces
    if stack:
        # Add closing characters in reverse order
        closing_chars = []
        for open_char in reversed(stack):
            if open_char == '{':
                closing_chars.append('}')
            elif open_char == '[':
                closing_chars.append(']')

        json_str = json_str + ''.join(closing_chars)
        repairs_made.append(f"added {len(closing_chars)} closing character(s): {''.join(closing_chars)}")

    if repairs_made:
        logger.warning(f"Repaired JSON string: {'; '.join(repairs_made)}")

    return json_str


def clean_input(text):
    """
    Cleans the input text to make it more JSON-compliant.
    - Replaces single quotes with double quotes.
    - Handles boolean values (True/False) and None.
    """
    # Replace single quotes with double quotes
    text = text.replace("'", '"')

    # Replace Python-style booleans and None with JSON-compatible values
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    return text

def extract_delimited_content(text, start_delimiter, end_delimiter):
    """
    Extract content between delimiters.

    Args:
        text: Input text containing delimited content
        start_delimiter: Opening delimiter (e.g., '<PYTHON>')
        end_delimiter: Closing delimiter (e.g., '</PYTHON>')

    Returns:
        Extracted content (without delimiters), or None if not found

    Example:
        text = 'code: <PYTHON>print("hello")</PYTHON>'
        extract_delimited_content(text, '<PYTHON>', '</PYTHON>')
        # Returns: 'print("hello")'
    """
    # Escape special regex characters in delimiters
    start_pattern = re.escape(start_delimiter)
    end_pattern = re.escape(end_delimiter)

    # Create pattern to match content between delimiters
    # re.DOTALL makes . match newlines too
    pattern = rf'{start_pattern}(.*?){end_pattern}'

    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        # Strip leading/trailing whitespace (but preserve internal formatting)
        return content.strip()

    return None


def parse_json_with_delimiters(text):
    """
    Parse JSON that may contain delimited content (e.g., <PYTHON>code</PYTHON>).

    This function:
    1. Extracts delimited content (e.g., code between <PYTHON></PYTHON>)
    2. Replaces it with a placeholder in the JSON
    3. Parses the JSON
    4. Restores the delimited content

    Supported delimiters:
    - <PYTHON>...</PYTHON> for Python code
    - <SQL>...</SQL> for SQL queries
    - <CODE>...</CODE> for generic code

    Handles both formats:
    - Without quotes: {"code": <PYTHON>...</PYTHON>}
    - With quotes: {"code": "<PYTHON>...</PYTHON>"}

    Args:
        text: Input text containing JSON with possible delimited content

    Returns:
        Parsed dictionary with delimited content restored

    Example:
        Input: '{"code": <PYTHON>print("hello")</PYTHON>, "query": "test"}'
        Output: {"code": 'print("hello")', "query": "test"}
    """
    # Define supported delimiters
    delimiters = [
        ('<PYTHON>', '</PYTHON>'),
        ('<SQL>', '</SQL>'),
        ('<CODE>', '</CODE>'),
    ]

    # Store extracted content
    extracted_content = {}
    modified_text = text

    # Extract all delimited content and replace with placeholders
    for start_delim, end_delim in delimiters:
        delimiter_type = start_delim.strip('<>')
        counter = 0

        while True:
            content = extract_delimited_content(modified_text, start_delim, end_delim)
            if content is None:
                break

            # Create unique placeholder
            placeholder = f'__DELIMITED_{delimiter_type}_{counter}__'
            extracted_content[placeholder] = content

            # Replace delimited content with placeholder
            # Handle both quoted and unquoted delimiters:
            # 1. "<PYTHON>...</PYTHON>" (with quotes)
            # 2. <PYTHON>...</PYTHON> (without quotes)
            pattern = re.escape(start_delim) + r'.*?' + re.escape(end_delim)

            # Check if delimiter is inside quotes
            # Pattern: "delimiter...content...delimiter"
            quoted_pattern = r'"' + pattern + r'"'
            if re.search(quoted_pattern, modified_text, re.DOTALL):
                # Replace quoted delimiter with placeholder (keep quotes)
                modified_text = re.sub(quoted_pattern, f'"{placeholder}"', modified_text, count=1, flags=re.DOTALL)
            else:
                # Replace unquoted delimiter with placeholder (add quotes)
                modified_text = re.sub(pattern, f'"{placeholder}"', modified_text, count=1, flags=re.DOTALL)

            counter += 1

    # Now parse the modified JSON (with placeholders)
    try:
        parsed = json.loads(modified_text)
    except json.JSONDecodeError as e:
        # Try to repair and parse
        logger.warning(f"Initial JSON parsing failed: {e}. Attempting repair...")
        try:
            repaired = repair_json_string(modified_text)
            parsed = json.loads(repaired)
            logger.info(f"Successfully parsed repaired JSON string")
        except json.JSONDecodeError as repair_error:
            raise ValueError(f"Failed to parse JSON. Original error: {e}. Repair attempt also failed: {repair_error}")

    # Restore delimited content
    def restore_content(obj):
        """Recursively restore delimited content in parsed object"""
        if isinstance(obj, dict):
            return {k: restore_content(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [restore_content(item) for item in obj]
        elif isinstance(obj, str):
            # Check if this is a placeholder
            if obj in extracted_content:
                content = extracted_content[obj]

                # CRITICAL FIX: The content was extracted BEFORE JSON parsing,
                # but json.loads() decodes escape sequences in the placeholder.
                #
                # Example flow:
                # 1. LLM generates: {"code": "<PYTHON>f'''text\\n\\n{var}'''</PYTHON>"}
                # 2. We extract: f'''text\\n\\n{var}''' (with \\n)
                # 3. We replace with placeholder: {"code": "__DELIMITED_PYTHON_0__"}
                # 4. json.loads() parses this - placeholder is fine
                # 5. We restore content: f'''text\\n\\n{var}''' (still with \\n)
                #
                # BUT if LLM generates actual newlines:
                # 1. LLM generates: {"code": "<PYTHON>f'''text\n\n{var}'''</PYTHON>"} (actual newlines)
                # 2. repair_json_string() escapes them: {"code": "<PYTHON>f'''text\\n\\n{var}'''</PYTHON>"}
                # 3. We extract: f'''text\\n\\n{var}''' (with \\n - looks good!)
                # 4. We replace with placeholder: {"code": "__DELIMITED_PYTHON_0__"}
                # 5. json.loads() parses - placeholder is fine
                # 6. We restore content: f'''text\\n\\n{var}''' (with \\n - still good!)
                #
                # So actually, the content should be correct as-is!
                # The issue must be elsewhere...

                return content
            return obj
        else:
            return obj

    return restore_content(parsed)


def parse_json(text):
    """
    Attempts to parse the cleaned text as JSON with automatic repair for malformed JSON.

    This function now supports:
    1. Standard JSON parsing
    2. JSON with delimited content (e.g., <PYTHON>code</PYTHON>)
    3. Automatic repair for common JSON issues
    4. Handling double-escaped sequences (\\\\n → \\n → actual newline)

    Returns a dictionary if successful, otherwise raises an error.
    """
    # First, check if text contains delimiters
    has_delimiters = any(delim in text for delim in ['<PYTHON>', '<SQL>', '<CODE>'])

    if has_delimiters:
        # Use delimiter-aware parsing
        try:
            return parse_json_with_delimiters(text)
        except Exception as e:
            logger.warning(f"Delimiter-based parsing failed: {e}. Falling back to standard parsing...")

    # Standard JSON parsing (with repair)
    try:
        # First attempt: direct parsing
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Second attempt: try to repair and parse
        logger.warning(f"Initial JSON parsing failed: {e}. Attempting repair...")
        try:
            repaired = repair_json_string(text)
            result = json.loads(repaired)
            logger.info(f"Successfully parsed repaired JSON string")
            return result
        except json.JSONDecodeError as repair_error:
            raise ValueError(f"Failed to parse JSON. Original error: {e}. Repair attempt also failed: {repair_error}")

def unescape_json_string(s: str) -> str:
    """
    Unescape a JSON string value (mimics what json.loads() does).

    This is needed when we manually extract string values from malformed JSON.
    Standard JSON escape sequences:
    - \\" → "
    - \\\\ → \\
    - \\/ → /
    - \\b → backspace
    - \\f → form feed
    - \\n → newline
    - \\r → carriage return
    - \\t → tab
    - \\uXXXX → unicode character

    Args:
        s: String with JSON escape sequences

    Returns:
        Unescaped string
    """
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == '"':
                result.append('"')
                i += 2
            elif next_char == '\\':
                result.append('\\')
                i += 2
            elif next_char == '/':
                result.append('/')
                i += 2
            elif next_char == 'b':
                result.append('\b')
                i += 2
            elif next_char == 'f':
                result.append('\f')
                i += 2
            elif next_char == 'n':
                result.append('\n')
                i += 2
            elif next_char == 'r':
                result.append('\r')
                i += 2
            elif next_char == 't':
                result.append('\t')
                i += 2
            elif next_char == 'u' and i + 5 < len(s):
                # Unicode escape: \uXXXX
                try:
                    code_point = int(s[i+2:i+6], 16)
                    result.append(chr(code_point))
                    i += 6
                except (ValueError, OverflowError):
                    # Invalid unicode escape - keep as-is
                    result.append(s[i])
                    i += 1
            else:
                # Unknown escape sequence - keep the backslash
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1

    return ''.join(result)


def parse_task_string(task_str: str) -> list[str]:
    """
    Parses a comma-separated string of quoted tasks into individual tasks.
    Handles commas within task descriptions and varying quotation marks.

    Args:
        task_str: Input string containing tasks (e.g. '"task 1", "task 2, with comma"')

    Returns:
        List of cleaned task strings
    """
    # Remove surrounding brackets if present
    cleaned = task_str.strip('[]')

    # Split on commas followed by optional whitespace and a quote
    task_split = re.split(r',(?=\s*["\'])', cleaned)

    # Clean whitespace and quotes from each task
    return [task.strip(' "\'') for task in task_split]


def handle_tool_input(parsed_dict):
    """
    Ensures that 'tool_input' in the parsed dictionary is always a valid dictionary.
    If 'tool_input' is not a dictionary, attempts to parse it into one with automatic repair.
    """
    if "tool_input" in parsed_dict:
        if isinstance(parsed_dict["tool_input"], str):
            try:
                # First attempt: direct parsing
                parsed_dict["tool_input"] = json.loads(parsed_dict["tool_input"])
            except json.JSONDecodeError as e:
                # Second attempt: try to repair and parse
                logger.warning(f"tool_input JSON parsing failed: {e}. Attempting repair...")
                try:
                    repaired = repair_json_string(parsed_dict["tool_input"])
                    parsed_dict["tool_input"] = json.loads(repaired)
                    logger.info(f"Successfully parsed repaired tool_input JSON string")
                except json.JSONDecodeError as repair_error:
                    raise ValueError(
                        f"Invalid tool_input format: {parsed_dict['tool_input']}. "
                        f"Original error: {e}. Repair attempt also failed: {repair_error}"
                    )
        elif not isinstance(parsed_dict["tool_input"], dict):
            raise ValueError(f"tool_input must be a dictionary or valid JSON string, got: {type(parsed_dict['tool_input'])}")

    return parsed_dict

def parser(stream):
    # Step 1: Clean the input
    cleaned_text = clean_input(stream)

    # Step 2: Parse the cleaned text into a dictionary
    parsed_dict = parse_json(cleaned_text)

    # Step 3: Handle tool_input field specifically
    parsed_dict = handle_tool_input(parsed_dict)

    return parsed_dict

def parse_tool_input(tool_input:dict, fields):
    """
    Parses tool_input according to the provided fields, handling various formats and edge cases.

    Parameters:
    tool_input (str or dict): The input to parse. Can be a dictionary, JSON string, or malformed JSON.
    fields (list): List of field names to extract.

    Returns:
    dict: A dictionary containing the parsed data for the specified fields.

    Flow:
    1. If tool_input is a dict → Direct field extraction (fast path)
    2. If tool_input is a string → Try json.loads(), then repair, then manual extraction
    3. Extract fields using regex-based parsing for malformed JSON
    """

    # Fields that may contain code or structured content with escape sequences
    # These are preserved during manual extraction but json.loads() will still unescape them
    CODE_FIELDS = {'code', 'script', 'program', 'source', 'query_code', 'python_code'}

    def clean_text(text):
        """Cleans and normalizes the input text."""
        return text.strip().replace("\\'", "'").replace('\\"', '"')

    def parse_json_like(text):
        """
        Attempts to parse a JSON-like string. Handles various quote styles and ensures nested quotes are replaced.
        """
        def replace_quotes(s):
            state = {'in_string': False, 'quote_char': None}
            result = []
            i = 0
            while i < len(s):
                if s[i] in ['"', "'"]:
                    if not state['in_string']:
                        state['in_string'] = True
                        state['quote_char'] = s[i]
                        result.append('"')
                    elif state['quote_char'] == s[i]:
                        state['in_string'] = False
                        result.append('"')
                    else:
                        result.append(s[i])
                elif s[i] == '\\' and i + 1 < len(s):
                    result.append(s[i:i+2])
                    i += 1
                else:
                    result.append(s[i])
                i += 1
            return ''.join(result)

        try:
            # Replace quotes and safely parse the modified string
            processed_text = replace_quotes(text)
            return json.loads(processed_text)
        except json.JSONDecodeError:
            return None


    def extract_field_value(data, field):
        """
        Extracts the value for a single field from the input data.
        Handles all JSON types correctly: strings, numbers, booleans, null, objects, arrays.

        CRITICAL PRINCIPLES:
        1. Preserve ENTIRE value without truncation
        2. Preserve original type (string stays string, bool stays bool, etc.)
        3. Don't convert types unless explicitly needed
        4. Handle nested structures correctly

        Strategy:
        - If data is already a dict → Direct extraction (fast path, type-safe)
        - If data is a string → Try JSON parsing first (handles all types correctly)
        - Only use manual extraction as last resort for truly malformed input
        """
        # FAST PATH: If data is already a dict, extract field directly
        # This preserves all types correctly (strings, bools, numbers, nested structures)
        if isinstance(data, dict):
            if field in data:
                value = data[field]
                # Return value as-is, preserving its type
                # Don't try to parse strings that look like JSON - they might be intentional
                return value
            return None

        # If `data` is not a string at this point, return an error
        if not isinstance(data, str):
            return None

        # MANUAL EXTRACTION FALLBACK: Only used when JSON parsing fails
        # This handles malformed JSON with mixed quotes, unescaped characters, etc.
        # Example: {"code": "x = db['clients'].find({'key': 'value'})", "other": "value"}

        # Find the field position
        field_pattern = rf'["\']?{re.escape(field)}["\']?\s*:\s*'
        field_match = re.search(field_pattern, data)
        if not field_match:
            return None

        start_pos = field_match.end()
        if start_pos >= len(data):
            return None

        first_char = data[start_pos]

        # Handle different value types based on first character

        # TYPE 1: Boolean values (true/false)
        if data[start_pos:start_pos+4] == 'true':
            return True
        if data[start_pos:start_pos+5] == 'false':
            return False

        # TYPE 2: Null value
        if data[start_pos:start_pos+4] == 'null':
            return None

        # TYPE 3: Numbers (int or float)
        if first_char.isdigit() or first_char == '-':
            # Extract number
            num_match = re.match(r'-?\d+\.?\d*', data[start_pos:])
            if num_match:
                num_str = num_match.group(0)
                if '.' in num_str:
                    return float(num_str)
                else:
                    return int(num_str)

        # TYPE 4: Quoted strings (most common, most complex)
        if first_char in ['"', "'"]:
            # ROBUST STRING EXTRACTION with multiple strategies:
            # 1. Track bracket/brace/paren depth for nested structures
            # 2. Handle escaped characters properly
            # 3. Use lookahead to verify closing quotes (handles unescaped quotes from LLM)

            quote_char = first_char
            value_start = start_pos + 1
            i = value_start
            value_chars = []

            # Track state
            escape_next = False
            bracket_depth = 0  # Track [], {}, () nesting
            paren_depth = 0
            brace_depth = 0

            def is_valid_closing_quote(pos):
                """
                Check if a quote at position 'pos' is a valid closing quote.
                A valid closing quote should be followed by:
                - Whitespace + comma (next field)
                - Whitespace + } (end of object)
                - Whitespace + ] (end of array)
                - End of string

                This handles cases where LLM generates unescaped quotes inside values.
                Example: "code": "parts.append("test")"
                         The quote before "test" is NOT valid (followed by 't')
                         The final quote IS valid (followed by comma or })
                """
                # Look ahead to see what follows this quote
                j = pos + 1
                # Skip whitespace
                while j < len(data) and data[j] in ' \t\n\r':
                    j += 1

                if j >= len(data):
                    # End of string - valid closing quote
                    return True

                next_char = data[j]
                # Valid if followed by comma, closing brace, or closing bracket
                if next_char in ',}]':
                    return True

                # Invalid - probably an unescaped quote inside the value
                return False

            while i < len(data):
                current_char = data[i]

                if escape_next:
                    # Previous character was backslash - this char is escaped
                    value_chars.append(current_char)
                    escape_next = False
                    i += 1
                    continue

                if current_char == '\\':
                    # Start of escape sequence
                    value_chars.append(current_char)
                    escape_next = True
                    i += 1
                    continue

                # Track bracket/brace/paren depth (only when not in escape sequence)
                if current_char == '[':
                    bracket_depth += 1
                    value_chars.append(current_char)
                    i += 1
                    continue
                elif current_char == ']':
                    bracket_depth -= 1
                    value_chars.append(current_char)
                    i += 1
                    continue
                elif current_char == '{':
                    brace_depth += 1
                    value_chars.append(current_char)
                    i += 1
                    continue
                elif current_char == '}':
                    brace_depth -= 1
                    value_chars.append(current_char)
                    i += 1
                    continue
                elif current_char == '(':
                    paren_depth += 1
                    value_chars.append(current_char)
                    i += 1
                    continue
                elif current_char == ')':
                    paren_depth -= 1
                    value_chars.append(current_char)
                    i += 1
                    continue

                # Check for closing quote
                if current_char == quote_char:
                    # Strategy: Check both depth AND lookahead
                    at_depth_zero = (bracket_depth == 0 and brace_depth == 0 and paren_depth == 0)

                    if at_depth_zero:
                        # At depth 0 - could be closing quote OR unescaped quote
                        # Use lookahead to verify
                        if is_valid_closing_quote(i):
                            # This is the real closing quote!
                            value = ''.join(value_chars)

                            # CRITICAL: Unescape JSON escape sequences ONLY for non-code fields
                            # Unescape JSON escape sequences for non-code fields
                            # Code fields preserve escape sequences in their original form
                            if field not in CODE_FIELDS:
                                value = unescape_json_string(value)
                                logger.debug(f"Extracted field '{field}' value (length: {len(value)}, verified closing quote, unescaped)")
                            else:
                                logger.debug(f"Extracted field '{field}' value (length: {len(value)}, verified closing quote, preserved as-is)")

                            return value
                        else:
                            # Unescaped quote inside value - include it
                            logger.debug(f"Found unescaped quote at position {i} (not followed by comma/brace/bracket)")
                            value_chars.append(current_char)
                            i += 1
                            continue
                    else:
                        # Inside nested structure - this quote is part of the value
                        value_chars.append(current_char)
                        i += 1
                        continue

                # Regular character
                value_chars.append(current_char)
                i += 1

            # Reached end without finding closing quote
            value = ''.join(value_chars)

            # Unescape JSON escape sequences for non-code fields
            # Code fields preserve escape sequences in their original form
            if field not in CODE_FIELDS:
                value = unescape_json_string(value)
                logger.warning(f"No closing quote found for field '{field}', returning value (length: {len(value)}, unescaped)")
            else:
                logger.warning(f"No closing quote found for field '{field}', returning value (length: {len(value)}, preserved as-is)")

            return value

        # TYPE 5: Objects {} - extract with bracket counting
        if first_char == '{':
            depth = 1
            i = start_pos + 1
            value_chars = ['{']
            escape_next = False
            in_string = False
            string_char = None

            while i < len(data) and depth > 0:
                char = data[i]
                value_chars.append(char)

                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char in ['"', "'"]:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                elif not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1

                i += 1

            value_str = ''.join(value_chars)
            # Try to parse as JSON
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                # Try repair
                try:
                    repaired = repair_json_string(value_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    # Return as string if can't parse
                    return value_str

        # TYPE 6: Arrays [] - extract with bracket counting
        if first_char == '[':
            depth = 1
            i = start_pos + 1
            value_chars = ['[']
            escape_next = False
            in_string = False
            string_char = None

            while i < len(data) and depth > 0:
                char = data[i]
                value_chars.append(char)

                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char in ['"', "'"]:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                elif not in_string:
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1

                i += 1

            value_str = ''.join(value_chars)
            # Try to parse as JSON
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                # Try repair
                try:
                    repaired = repair_json_string(value_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    # Return as string if can't parse
                    return value_str

        # If we get here, couldn't extract value
        logger.warning(f"Could not extract value for field '{field}'")
        return None

    def extract_fields(data, fields):
        """
        Extracts values for the specified fields from the input data.
        Handles cases where `tool_input` is a serialized JSON string or a dictionary.
        """
        result={field: extract_field_value(data, field) for field in fields}
        # print(result)
        for field in fields:
            if field not in result.keys():
                # print(field)
                result[field] = None
        return result

    # OPTIMIZATION: If tool_input is a string, try to parse it as JSON first
    # This converts it to a dict for faster field extraction
    # NOTE: json.loads() automatically unescapes ALL string fields (standard JSON behavior)
    # Tools that need escape sequences preserved should handle this in their own processing
    if isinstance(tool_input, str):
        try:
            # Try direct JSON parsing
            tool_input = json.loads(tool_input)
            logger.debug("Successfully parsed tool_input string as JSON")
        except json.JSONDecodeError as e:
            # Try with repair
            try:
                repaired = repair_json_string(tool_input)
                tool_input = json.loads(repaired)
                logger.debug("Successfully parsed repaired tool_input string as JSON")
            except json.JSONDecodeError:
                # Try one more time with aggressive cleaning
                try:
                    # Remove any leading/trailing whitespace and quotes
                    cleaned = tool_input.strip().strip('"').strip("'")
                    # Try parsing the cleaned version
                    tool_input = json.loads(cleaned)
                    logger.debug("Successfully parsed cleaned tool_input string as JSON")
                except json.JSONDecodeError:
                    # All JSON parsing failed - use direct field extraction
                    # This is normal for malformed JSON (mixed quotes, unescaped chars, etc.)
                    logger.debug(f"Using direct field extraction for malformed JSON (length: {len(tool_input)})")
                    # Let it fall through to extract_fields which handles malformed JSON

    try:
        parsed_tool_input = extract_fields(tool_input, fields)

        # Log the parsed values for debugging
        for field in fields:
            if field in parsed_tool_input and parsed_tool_input[field] is not None:
                value = parsed_tool_input[field]
                if isinstance(value, str):
                    logger.debug(f"Parsed field '{field}': length={len(value)}, preview={value[:100]}...")
                else:
                    logger.debug(f"Parsed field '{field}': type={type(value).__name__}")

        return parsed_tool_input
    except Exception as e:
        logger.error(f"Error parsing tool input: {e}", exc_info=True)
        return None