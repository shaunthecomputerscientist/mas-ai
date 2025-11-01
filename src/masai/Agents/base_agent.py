import asyncio # Import asyncio
import os
from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional, Callable, AsyncGenerator # Added Callable, AsyncGenerator
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Assuming these imports are correct relative to your project structure
from ..GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel
from ..Tools.logging_setup.logger import setup_logger
from ..Tools.PARSERs.json_parser import parse_tool_input, parse_task_string
from ..langgraph.graph import StateGraph
from ..langgraph.graph.state import CompiledStateGraph
from ..Tools.utilities.streaming_events import emit_tool_call, emit_tool_output
import inspect

load_dotenv()

# Use the State definition from the Agent snippet as it includes reflection_counter etc.
class State(TypedDict):
    messages: List[Dict[str, str]]
    current_tool: str
    tool_input: Any # Allow Dict or str from gettoolinput/parsing
    tool_output: Any # Tools can return various types, handle appropriately
    answer: str
    satisfied: bool
    reasoning: str
    delegate_to_agent: Optional[str] # Mark as Optional
    current_node: str
    previous_node: Optional[str] # Mark as Optional
    plan: Optional[dict] # Mark as Optional
    passed_from: Optional[str] # Mark as Optional
    reflection_counter: int # Defaults set in Agent's initial_state
    tool_loop_counter: int # Defaults set in Agent's initial_state
    tool_decided_by: Optional[str] # Track which node decided to use the current tool
    current_question: str # Track the current question being processed

class BaseAgent:
    _logger = None

    def __init__(self, agent_name: str, logging: bool = True, shared_memory_order: int = 5, retain_messages_order: int = 30, max_tool_output_words: int = 3000,**kwargs):
        """Base class for agents with common functionality.

        Args:
            agent_name: Name of the agent
            logging: Enable logging
            shared_memory_order: Number of messages to keep in shared memory
            retain_messages_order: Number of messages to retain across executions
            max_tool_output_words: Maximum number of words to include from tool output in LLM prompts (default: 3000)
            character_factor: Number of characters per word (default: 10)
            **kwargs: Additional keyword arguments (e.g., character_factor)
        """
        self.agent_name = agent_name.lower()
        self.logging = logging
        self.shared_memory_order = shared_memory_order
        self.retain_messages_order = retain_messages_order or 10 # Keep default if 20 not desired
        self.max_tool_output_words = max_tool_output_words  # Configurable tool output limit
        self.app = None  # StateGraph workflow, to be set by subclasses
        self.graph = None
        # These will be set properly by the inheriting Agent class
        self.tool_mapping: Dict[str, Any] = {}
        self.pydanticmodel: Optional[Type[BaseModel]] = None
        self.agent_context: Optional[Dict[str, Any]] = None
        self.character_factor = kwargs.get('character_factor', 10)

        # NEW: Retained state for continuity across workflow executions
        self.retained_state: Optional[Dict[str, Any]] = None

        if self.logging:
            # Use a shared logger instance across all agents if desired
            # Or initialize per instance: self.logger = setup_logger(self.agent_name)
            if BaseAgent._logger is None:
                BaseAgent._logger = setup_logger() # Setup once
            self.logger = BaseAgent._logger
        else:
            self.logger = None

    # --- Synchronous Helper Methods ---

    def _truncate_tool_output(self, tool_output: str, max_words: Optional[int] = None) -> str:
        """
        Truncate tool output to a maximum number of words.

        Args:
            tool_output: The tool output string to truncate
            max_words: Maximum number of words (uses self.max_tool_output_words if None)

        Returns:
            Truncated tool output with truncation notice if truncated
        """
        if not tool_output or tool_output == 'N/A':
            return tool_output

        max_words = max_words or self.max_tool_output_words
        max_characters = max_words * self.character_factor
        # Convert to string if not already
        tool_output_str = str(tool_output)
       

        # Check if truncation is needed
        if  len(tool_output_str) <= max_characters:
            return tool_output_str
        else:
            tool_output_str_start = str(tool_output)[:max_characters//2]
            tool_output_str_end = str(tool_output)[-max_characters//2:] 
            # Truncate and add notice
            truncation_notice = f"\n\n[... OUTPUT TRUNCATED:Showing first {max_characters//2} characters and last {max_characters//2} characters out of {len(tool_output_str)} total characters. Full output stored in state ...]"
            truncated = tool_output_str_start + truncation_notice + tool_output_str_end

        if self.logger:
            self.logger.info(f"Tool output truncated from {len(tool_output_str)} words to {max_words} words for LLM prompt")

        return truncated

    async def gettoolinput(self, tool_input: Union[Dict, str], tool_name: str) -> Union[Dict, str]:
        """Parse and sanitize tool input based on the tool's schema."""
        # Ensure tool_mapping and tool exist before accessing args_schema
        if tool_name not in self.tool_mapping:
             if self.logger: self.logger.error(f"Tool '{tool_name}' not found in tool_mapping during gettoolinput.")
             return {"error": f"Tool '{tool_name}' not found."} # Return error dict

        tool = self.tool_mapping[tool_name]
        if not hasattr(tool, 'args_schema') or not tool.args_schema:
             if self.logger: self.logger.warning(f"Tool '{tool_name}' has no args_schema. Passing input as is.")
             return tool_input if isinstance(tool_input, dict) else {"input": tool_input} # Basic wrapping if string

        # Proceed with parsing using args_schema if available
        try:
            # Ensure expected_args gets populated correctly
            schema_properties = tool.args_schema.model_json_schema().get('properties', {})
            expected_args = list(schema_properties.keys())
            if not expected_args:
                 if self.logger: self.logger.warning(f"Tool '{tool_name}' args_schema has no properties defined.")
                 # Handle case with no defined properties - maybe expect a single 'input'?
                 # This depends on how parse_tool_input behaves with empty expected_args
                 return parse_tool_input(tool_input, []) # Or handle differently


            return parse_tool_input(tool_input, expected_args)
        except Exception as e:
             if self.logger: self.logger.error(f"Error getting/parsing schema for tool '{tool_name}': {e}", exc_info=True)
             return {"error": f"Failed to parse input for tool '{tool_name}': {e}"}


    def display(self):
        """Display the agent's workflow graph if it exists."""
        if not self.graph:
            if self.app:
                self.graph = self.app.get_graph() # Try to get graph if app exists
            else:
                raise ValueError("Workflow graph not initialized; ensure workflow is compiled (`self.app = self.agentworkflow()`)")

        try:
            png_data = self.graph.draw_mermaid_png()
            mermaid_dir = os.path.join('MAS', 'Database', 'mermaid') # Consider making path configurable
            os.makedirs(mermaid_dir, exist_ok=True)
            png_file_path = os.path.join(mermaid_dir, f"{self.agent_name}_diagram.png")
            with open(png_file_path, "wb") as f:
                f.write(png_data)
            if self.logger: self.logger.info(f"Agent graph saved to {png_file_path}")
        except Exception as e:
             if self.logger: self.logger.error(f"Failed to draw or save agent graph: {e}", exc_info=True)
             # Optionally re-raise or handle

    async def _update_state(self, current_state: State, parsed_response: Dict, node: str) -> State:
        """Update the agent state based on a parsed response (synchronous)."""
        # Use .get for safer dictionary access
        tool_name = parsed_response.get('tool') # Assumes LLM returns 'tool' key
        tool_input_raw = parsed_response.get('tool_input') # Assumes LLM returns 'tool_input'

        # Update previous/current node first
        current_state['previous_node'] = node # The node we're entering
        current_state['current_node'] = None  # Reset current

        # Filter out Pydantic model name (happens with function_calling method)
        # When using method="function_calling", OpenAI may return the model class name as tool
        if tool_name and tool_name.upper() in ["ANSWERFORMAT", "ANSWER_FORMAT"]:
            if self.logger: self.logger.debug(f"Filtered out Pydantic model name '{tool_name}' from tool field")
            tool_name = None

        if tool_name and tool_name not in ["None", None,'','null','none']:
            current_state["current_tool"] = tool_name
            # Track which node decided to use this tool (for component context)
            current_state["tool_decided_by"] = node
            # Parse tool input safely
            parsed_tool_input = await self.gettoolinput(tool_input_raw, tool_name)

            # Check for tool loops
            if isinstance(parsed_tool_input, dict) and "error" in parsed_tool_input:
                 # Handle parsing error - maybe log and don't update input/loop counter?
                 if self.logger: self.logger.error(f"Failed to parse tool input from LLM for tool {tool_name}: {parsed_tool_input['error']}")
                 # Decide how to proceed - maybe clear tool/input? For now, log and potentially let evaluator handle.
                 current_state['tool_input'] = parsed_tool_input # Store the error dict
                 current_state['tool_loop_counter'] = 0 # Reset counter on error?
            elif parsed_tool_input == current_state.get('tool_input') and parsed_tool_input not in [None, "None", {}]: # Compare parsed input
                current_state['tool_loop_counter'] = current_state.get('tool_loop_counter', 0) + 1
                if self.logger: self.logger.warning(f"Potential tool loop detected. Count: {current_state['tool_loop_counter']}")
            else:
                current_state['tool_loop_counter'] = 0 # Reset if input changed

            current_state['tool_input'] = parsed_tool_input # Store parsed input (or error dict)
            if self.logger:
                self.logger.warning("-------------------------------------Tool Input Parsed---------------------------------")
                self.logger.warning(f"Tool: {current_state['current_tool']}, Input: {current_state['tool_input']}")
        else:
             # Clear tool info if no tool selected by LLM
             current_state["current_tool"] = None
             current_state["tool_input"] = None
             current_state["tool_decided_by"] = None  # Clear decision maker
             current_state['tool_loop_counter'] = 0 # Reset counter if no tool


        # Handle plan update specifically for planner node
        if node == 'planner' and 'answer' in parsed_response:
             try:
                 # Initialize plan dict if it's None or not present
                 if 'plan' not in current_state or current_state['plan'] is None:
                     current_state['plan'] = {}
                 plans: list = parse_task_string(parsed_response['answer']) # Assuming this returns a list of plan steps
                 # Update plan dictionary - overwrite or append? Overwriting based on index.
                 current_state['plan'] = {i: plan for i, plan in enumerate(plans)} # Store plan steps
                 if self.logger: self.logger.info(f"Plan updated: {current_state['plan']}")
             except Exception as e:
                 if self.logger: self.logger.error(f"Failed to parse plan string: {parsed_response['answer']}. Error: {e}")
                 # Decide how to handle plan parsing failure

        # Update common state fields from parsed LLM response
        current_state.update({
            "answer": parsed_response.get('answer', current_state.get('answer')), # Keep old answer if new one not provided
            "satisfied": parsed_response.get('satisfied', False), # Default to False if not provided
            "reasoning": parsed_response.get('reasoning', ''),
            # current_tool updated above
            "delegate_to_agent": parsed_response.get('delegate_to_agent')
        })

        # Append assistant message (LLM Answer)
        # Avoid appending if answer is None or just plan info
        assistant_content = current_state.get('answer')
        if assistant_content:
             current_state["messages"].append({"role": self.agent_name, "content": assistant_content})

        # Retain message history limit
        if len(current_state["messages"]) > self.retain_messages_order:
            # Keep first (user query) and N last messages
            current_state['messages'] = [current_state['messages'][0]] + current_state["messages"][-self.retain_messages_order:]

        if self.logger: self.logger.debug(f"State updated after node: {node}. Satisfied: {current_state['satisfied']}. Tool: {current_state['current_tool']}")
        return current_state


    async def _sanitize_query(self, query: str) -> str:
        """Sanitize input query by removing/replacing problematic characters."""
        # Be cautious with over-sanitization, might break legitimate inputs (e.g., JSON)
        # Consider a more targeted approach if needed. This basic version is kept from original.
        replacements = {
            "'": '', "\\": "", """: '"', """: '"', '"': "" # Original replacements
        }
        sanitized = str(query)
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        return sanitized

    # --- Methods to be Implemented/Overridden by Subclasses ---

    def set_context(self, context: Optional[Dict] = None, mode: str = "set"):
        """Set or update context for all components (optional, can be overridden)."""
        # Base implementation does nothing, subclasses (like Agent) should override
        pass

    async def node_handler(self, state: State, llm: BaseGenerativeModel, prompt: str, component_context: Optional[List] = None, node: Optional[str] = None) -> State:
        """Handle node-specific LLM responses and update state (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement async node_handler")

    async def initiate_agent(self, query: str, passed_from: Optional[str] = None, stream: bool = False) -> Union[AsyncGenerator[Dict, None], Dict]:
        """Initiate the agent workflow asynchronously (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement async initiate_agent")

    def agentworkflow(self) -> CompiledStateGraph:
        """Compile and return the agent's workflow graph (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement agentworkflow")

    def check_condition(self, state: State) -> Literal["continue", "end", "reflection"]:
        """Base condition checker (to be overridden or specialized by subclasses)."""
        # Renamed to check_condition for consistency if preferred, Agent uses checkroutingcondition
        raise NotImplementedError("Subclasses must implement check_condition (or checkroutingcondition)")

    # --- Async Tool Execution Method ---

    async def execute_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool based on the current state asynchronously,
        differentiating between async and sync based on the underlying tool func
        and trusting ainvoke if func inspection fails but ainvoke exists.

        Emits streaming events for tool calls and outputs if streaming callback is set.
        """
        # Import streaming utilities
        tool_name = state.get("current_tool")
        tool_input = state.get("tool_input")

        state['current_node'] = 'execute_tool'

        if not tool_name or tool_name == "None":
            # ... (no changes needed in this block) ...
            if self.logger: self.logger.warning("execute_tool called but no tool name provided.")
            state['tool_output'] = "No tool to execute."
            if "messages" not in state: state["messages"] = []
            state["messages"].append({"role": "tool", "name": "system", "content": "No tool to execute."})

            # Manual state update for early return
            state['previous_node'] = 'execute_tool'
            state['current_node'] = None
            return state


        if tool_name not in self.tool_mapping:
            # ... (no changes needed in this block) ...
            if self.logger: self.logger.error(f"Tool '{tool_name}' not found in tool_mapping.")
            state['tool_output'] = f"Error: Tool '{tool_name}' not found."
            state['satisfied'] = False
            state['current_tool'] = None  # Clear invalid tool to prevent infinite loop
            if "messages" not in state: state["messages"] = []
            state["messages"].append({"role": "tool", "name": tool_name, "content": state['tool_output']})

            # Manual state update for early return
            state['previous_node'] = 'execute_tool'
            state['current_node'] = None
            return state


        tool = self.tool_mapping[tool_name]
        if self.logger:
            self.logger.warning(f"Attempting to execute tool: {tool_name} with input: {tool_input}")

        # Emit tool call event
        await emit_tool_call(tool_name, tool_input, node=state.get('current_node'))

        result = None
        try:
            # Prepare input data (same logic as before)
            if tool_input is None:
                 input_data = {}
            elif isinstance(tool_input, dict):
                input_data = tool_input
            else:
                # Consider if the tool expects input directly or wrapped, e.g. tool.args_schema
                input_data = {"input": tool_input} # Default wrapping, adjust if needed

            if isinstance(input_data, dict) and "error" in input_data:
                 raise ValueError(f"Invalid tool input provided: {input_data['error']}")

            # --- Determine Execution Strategy ---
            has_ainvoke = hasattr(tool, "ainvoke") and callable(getattr(tool, "ainvoke"))
            has_invoke = hasattr(tool, "invoke") and callable(getattr(tool, "invoke"))
            has_func = hasattr(tool, "func") and callable(getattr(tool, "func"))
            # Check the underlying function's type *if* possible
            is_underlying_async = has_func and inspect.iscoroutinefunction(getattr(tool, "func"))

            if self.logger:
                 self.logger.warning(f"Tool '{tool_name}' | Has ainvoke: {has_ainvoke} | Has invoke: {has_invoke} | Has func: {has_func} | Underlying func async: {is_underlying_async}")

            # --- Execution Logic ---
            should_use_ainvoke = False
            if has_ainvoke:
                if is_underlying_async:
                    # Case 1: Explicitly async underlying function found.
                    should_use_ainvoke = True
                    if self.logger: self.logger.warning(f"Choosing ainvoke for '{tool_name}' (has_ainvoke=True, is_underlying_async=True).")
                elif not has_func:
                    # Case 2: Cannot inspect .func, but ainvoke exists.
                    # TRUST ainvoke as the intended async path (handles StructuredTool etc.)
                    should_use_ainvoke = True
                    if self.logger: self.logger.warning(f"Choosing ainvoke for '{tool_name}' (has_ainvoke=True, has_func=False). Trusting ainvoke.")
                # else: has_ainvoke=True, has_func=True, is_underlying_async=False
                # -> We will fall through to 'invoke' below, as ainvoke might fail for sync func.

            if should_use_ainvoke:
                # Execute using ainvoke
                if self.logger: self.logger.warning(f"Executing '{tool_name}' using tool.ainvoke.")
                result = await tool.ainvoke(input_data)

            elif has_invoke:
                # Execute using invoke (only if we didn't decide to use ainvoke)
                if self.logger:
                     log_msg = f"Executing '{tool_name}' using tool.invoke wrapped in asyncio.to_thread"
                     # Add context for logging
                     if has_func and not is_underlying_async: log_msg += " (underlying func is sync)."
                     elif not has_func and not has_ainvoke: log_msg += " (underlying func info unavailable, no ainvoke)."
                     elif has_ainvoke and has_func and not is_underlying_async: log_msg += " (ainvoke exists but underlying func is sync)."
                     # The case that caused the previous error (has_ainvoke=T, has_func=F) is now handled by should_use_ainvoke=True
                     else: log_msg += "."
                     self.logger.warning(log_msg)

                # Run sync invoke in a thread
                result = await asyncio.to_thread(tool.invoke, input_data)

            # --- Fallback for Raw Callables ---
            elif callable(tool):
                 # ... (no changes needed in this block) ...
                 is_direct_async = inspect.iscoroutinefunction(tool)
                 if self.logger: self.logger.warning(f"Tool '{tool_name}' lacks invoke/ainvoke. Attempting direct call (is_async={is_direct_async}).")
                 args_to_pass, kwargs_to_pass = [], {}
                 if isinstance(input_data, dict): kwargs_to_pass = input_data
                 elif input_data is not None: args_to_pass = [input_data]

                 if is_direct_async:
                     result = await tool(*args_to_pass, **kwargs_to_pass)
                 else:
                     result = await asyncio.to_thread(lambda: tool(*args_to_pass, **kwargs_to_pass))
            else:
                 raise AttributeError(f"Tool '{tool_name}' is not executable: lacks callable 'ainvoke'/'invoke' methods and is not directly callable.")

        except ValidationError as e:
            # ... (no changes needed in error handling blocks) ...
            result = f"Validation Error for tool '{tool_name}': {str(e)}. Input: {tool_input}. Ensure format matches requirements."
            if self.logger: self.logger.warning(f"Validation error executing tool '{tool_name}': {e}. Input: {tool_input}")
            state['satisfied'] = False
            state['tool_output'] = result
            state['current_tool'] = None  # Clear invalid tool to prevent infinite loop
            if "messages" not in state: state["messages"] = []
            state["messages"].append({"role": "tool", "name": tool_name, "content": state['tool_output']})

            # Manual state update for early return
            state['previous_node'] = 'execute_tool'
            state['current_node'] = None
            return state
        except Exception as e:
            # Catch the specific error if needed for debugging, otherwise general Exception
            result = f"Error executing tool '{tool_name}': {str(e)}. Input: {tool_input}"
            if self.logger:
                self.logger.error(f"Unexpected error executing tool '{tool_name}': {e}", exc_info=True)
            state['satisfied'] = False
            state['tool_output'] = result
            state['current_tool'] = None  # Clear invalid tool to prevent infinite loop
            if "messages" not in state: state["messages"] = []
            state["messages"].append({"role": "tool", "name": tool_name, "content": state['tool_output']})

            # Manual state update for early return
            state['previous_node'] = 'execute_tool'
            state['current_node'] = None
            return state

        # Post-processing: Handle result and return_direct logic
        # Wrap in try-except to catch any errors during result processing
        try:
            if "messages" not in state: state["messages"] = []
            if self.logger:
                self.logger.warning("-------------------------------------Tool Output---------------------------------")
                log_output = str(result)
                # if len(log_output) > 500: log_output = log_output[:500] + "... (truncated)"
                self.logger.warning(f"Tool: {tool_name}, Output: {log_output}")
                self.logger.warning("-----------------------------------------------------------------------------------")

            # Check for return_direct in three places with proper priority:
            # Priority: Return Value > Argument > Decorator
            # 1. Return value from tool execution (HIGHEST priority)
            # 2. Dynamic return_direct from tool input argument (MEDIUM priority)
            # 3. Tool decorator's return_direct attribute (LOWEST priority)

            tool_decorator_return_direct = hasattr(tool, 'return_direct') and tool.return_direct
            input_return_direct = None  # None means not provided
            result_return_direct = None  # None means not provided
            actual_result = result

            # Check if tool_input contains return_direct argument
            if isinstance(tool_input, dict) and 'return_direct' in tool_input:
                input_return_direct = tool_input.get('return_direct')
                if self.logger:
                    self.logger.info(f"Tool '{tool_name}' has return_direct={input_return_direct} from tool_input argument")

            # Check if result is a dict with return_direct key (tool decided internally during execution)
            if isinstance(result, dict) and 'return_direct' in result:
                result_return_direct = result.get('return_direct')
                # Extract actual result (remove return_direct from output)
                actual_result = {k: v for k, v in result.items() if k != 'return_direct'}
                if self.logger:
                    self.logger.info(f"Tool '{tool_name}' returned return_direct={result_return_direct} in result (runtime decision)")

            # Implement proper priority: Return Value > Argument > Decorator
            should_return_direct = False
            source = None

            if result_return_direct is not None:
                # HIGHEST PRIORITY: Return value from function execution
                should_return_direct = bool(result_return_direct)
                source = "return value (runtime decision)"
            elif input_return_direct is not None:
                # MEDIUM PRIORITY: Argument passed to function
                should_return_direct = bool(input_return_direct)
                source = "input argument"
            else:
                # LOWEST PRIORITY: Decorator
                should_return_direct = tool_decorator_return_direct
                source = "decorator"

            # Store the actual result (without return_direct metadata)
            state['tool_output'] = str(actual_result)
            state['passed_from'] = tool_name
            state["messages"].append({"role": tool_name, "content": state['tool_output']})

            # Emit tool output event
            await emit_tool_output(tool_name, actual_result, node=state.get('current_node'))

            if should_return_direct:
                 if self.logger:
                     self.logger.warning(f"Tool '{tool_name}' requested return_direct ({source}). Setting final answer.")
                 state.update({
                     'current_tool': None, 'tool_input': None, 'answer': state['tool_output'],
                     'reasoning': f"Result directly provided by tool '{tool_name}'.",
                     'satisfied': True, 'delegate_to_agent': None,
                 })
            elif 'satisfied' not in state:
                 state['satisfied'] = False

        except Exception as e:
            # Catch any errors during result processing (e.g., str() conversion, dict operations, message append)
            error_msg = f"Error processing tool result for '{tool_name}': {str(e)}. Raw result type: {type(result).__name__}"
            if self.logger:
                self.logger.error(f"Error during tool result processing for '{tool_name}': {e}", exc_info=True)

            state['satisfied'] = False
            state['tool_output'] = error_msg
            state['current_tool'] = None
            if "messages" not in state: state["messages"] = []
            # Use try-except for message append in case state["messages"] is corrupted
            try:
                state["messages"].append({"role": "tool", "name": tool_name, "content": error_msg})
            except Exception:
                # If even this fails, create new messages list
                state["messages"] = [{"role": "tool", "name": tool_name, "content": error_msg}]

        # Manual state update to be consistent with other nodes (don't use _update_state as it expects LLM response)
        state['previous_node'] = 'execute_tool'
        state['current_node'] = None
        return state