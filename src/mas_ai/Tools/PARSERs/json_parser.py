import json
import re

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

def parse_json(text):
    """
    Attempts to parse the cleaned text as JSON.
    Returns a dictionary if successful, otherwise raises an error.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

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
    If 'tool_input' is not a dictionary, attempts to parse it into one.
    """
    if "tool_input" in parsed_dict:
        if isinstance(parsed_dict["tool_input"], str):
            try:
                # Attempt to parse tool_input as JSON
                parsed_dict["tool_input"] = json.loads(parsed_dict["tool_input"])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid tool_input format: {parsed_dict['tool_input']}")
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
    Parses `tool_input` according to the provided fields, handling various formats and edge cases.

    Parameters:
    text (str or dict): The input to parse. It can be a dictionary, a JSON-like string, or a raw string.
    fields (list): List of field names to extract.

    Returns:
    dict: A dictionary containing the parsed data for the specified fields.
    """

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
        Handles quoted values, JSON-like structures, unquoted key-value pairs, and boolean values.
        """
        # Ensure `data` is in string format
        if isinstance(data, dict):
            try:
                data = json.dumps(data)  # Serialize dictionary to a JSON string
            except Exception:
                pass

        # If `data` is not a string at this point, return an error
        if not isinstance(data, str):
            return None

        # Define patterns to match different value formats
        patterns = [
            rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"',    # Double-quoted value
            rf"'{field}'\s*:\s*'((?:[^'\\]|\\.)*)'",    # Single-quoted value
            rf'"{field}"\s*:\s*(\{{[^}}]*\}})',         # JSON-like dictionary
            rf'"{field}"\s*:\s*(\[[^\]]*\])',           # JSON-like list
            rf'"{field}"\s*:\s*(true|false)',           # Boolean values (JSON-style)
            rf'"{field}"\s*:\s*(\d+)',                  # Match only numbers
            rf'"{field}"\s*:\s*([^\s,]+)'               # Unquoted value (general case)
        ]

        # Apply patterns to extract the value
        for pattern in patterns:
            match = re.search(pattern, data, re.DOTALL)
            if match:
                value = next((g for g in match.groups() if g is not None), None)
                # print(match,value)
                if value:
                    # Convert JSON boolean values to Python booleans
                    if 'true' in value.lower():
                        return True
                    if 'false' in value.lower():
                        return False
                    if (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    return value.strip("'\"")
                
        
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
        
    try:
        parsed_tool_input = extract_fields(tool_input,fields)
        return parsed_tool_input
    except Exception as e:
        print(e)
        return None