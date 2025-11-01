import asyncio # Ensure asyncio is imported
from ..langgraph.graph import END, StateGraph, START
from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional, Generator, AsyncGenerator, Callable # Ensure all types needed are imported
from ..langgraph.graph.state import CompiledStateGraph
# Import the modified BaseAgent and other dependencies
from .base_agent import BaseAgent, State # Assuming BaseAgent is in base_agent.py
from pydantic import BaseModel, Field, ValidationError

from ..GenerativeModel.generativeModels import MASGenerativeModel,GenerativeModel # Keep if needed
from ..GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel
from ..prompts.prompt_templates import TOOL_LOOP_WARNING_PROMPT, EVALUATOR_NODE_PROMPT, REFLECTOR_NODE_PROMPT, PLANNER_NODE_PROMPT, ROUTER_NODE_PROMPT
from ..Config import config

class Agent(BaseAgent): # Inherit from the modified BaseAgent
    def __init__(self, agent_name: str, llm_router: BaseGenerativeModel, llm_evaluator: BaseGenerativeModel,
                 llm_reflector: BaseGenerativeModel, llm_planner: Optional[BaseGenerativeModel] = None,
                 tool_mapping: Optional[Dict[str, Any]] = None, # Use Any for Langchain tools, or specific Tool type
                 AnswerFormat: Optional[Type[BaseModel]] = None,
                 logging: bool = True, agent_context: Optional[Dict[str, Any]] = None, shared_memory_order: int = 5,
                 retain_messages_order: int = 30, max_tool_output_words: int = 3000, **kwargs):

        """Initialize an async agent with router-evaluator-reflector architecture and optional planner.
        Inherits async execute_tool from BaseAgent.

        Args:
            agent_name: Name of the agent
            llm_router: LLM for router node
            llm_evaluator: LLM for evaluator node
            llm_reflector: LLM for reflector node
            llm_planner: Optional LLM for planner node
            tool_mapping: Dictionary of available tools
            AnswerFormat: Pydantic model for structured output (with _cached_schema attribute)
            logging: Enable logging
            agent_context: Context dictionary for the agent
            shared_memory_order: Number of messages to keep in shared memory
            retain_messages_order: Number of messages to retain across executions
            max_tool_output_words: Maximum number of words from tool output to include in LLM prompts (default: 3000)
            **kwargs: Additional keyword arguments (e.g., max_tool_loop, character_factor)
        """
        super().__init__(agent_name, logging, shared_memory_order, retain_messages_order, max_tool_output_words, **kwargs)
        self.llm_router = llm_router
        self.llm_evaluator = llm_evaluator
        self.llm_reflector = llm_reflector
        self.llm_planner = llm_planner
        self.plan = bool(llm_planner)
        self.tool_mapping = tool_mapping or {} # Set tool_mapping for BaseAgent's use
        self.pydanticmodel = AnswerFormat # Set pydanticmodel for BaseAgent's use
        self.agent_context = agent_context # Set agent_context for BaseAgent's use
        self.MAX_TOOL_LOOP = kwargs.get('max_tool_loop', config.max_tool_loops)

        # Compile the workflow using async nodes
        self.app: CompiledStateGraph = self.agentworkflow()

    # Override set_context from BaseAgent
    def set_context(self, context: Optional[Dict] = None, mode: str = "set"):
        """Set or update context for all LLM components."""
        components = [self.llm_router, self.llm_evaluator, self.llm_reflector, self.llm_planner]
        if context:
            for component in components:
                if component:
                    if not hasattr(component, 'info') or component.info is None:
                         component.info = {} # Initialize if not present

                    if mode == 'set':
                        component.info = context.copy()
                    elif mode == 'update':
                        # Ensure info exists before updating
                        if component.info is None: component.info = {}
                        component.info.update(context)

    # Override node_handler from BaseAgent -> Make it async
    async def node_handler(self, state: State, llm: BaseGenerativeModel, prompt: str, component_context: Optional[List] = None, node: Optional[str] = None) -> State:
        """Handle node-specific LLM responses and update state asynchronously."""
        node_name = node or 'default_async_node'
        if self.logger: self.logger.debug(f"Entering async node_handler for node: {node_name}\n")

        # Prepare arguments for the threaded call
        llm_call_args = {
            "prompt": prompt,
            "output_structure": self.pydanticmodel,  # Contains _cached_schema attribute
            "agent_context": self.agent_context,
            "agent_name": self.agent_name,
            "component_context": component_context or [],
            "passed_from": state.get('passed_from'),
            "query": state.get('current_question'),
            "node": node
        }

        parsed_response = None
        try:
            # Run synchronous LLM call in a separate thread
            if llm.streaming:
                parsed_response = await llm.astream_response_mas(**llm_call_args)
            else:
                parsed_response = await llm.generate_response_mas(**llm_call_args)

            # Check if response is a string (error) instead of dict
            if isinstance(parsed_response, str):
                # LLM returned an error string, convert to proper error response
                raise Exception(parsed_response)

            if self.logger:
                log_answer = parsed_response.get('answer', 'N/A')
                self.logger.info(f"Node {node_name} LLM response answer (truncated): {str(log_answer)[:config.truncated_response_length]}...")
                # Optionally log other parts like reasoning, tool selection

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during LLM call in node {node_name}: {e}", exc_info=True)
            # Create a default error response matching Pydantic model fields if possible
            parsed_response = {
                "answer": f"Error in {node_name}: Failed to get valid response from LLM. Error: {e}",
                "satisfied": False,
                "reasoning": f"LLM call failed with error: {e}",
                "tool": None, # Explicitly set tool to None on error
                "tool_input": None,
                "delegate_to_agent": None
            }
            # Ensure all keys expected by _update_state / Pydantic model are present
            if self.pydanticmodel:
                 # Get required fields or all fields
                 # model_fields = self.pydanticmodel.model_fields.keys()
                 schema_props = self.pydanticmodel.model_json_schema().get('properties', {}).keys()
                 for key in schema_props:
                     if key not in parsed_response:
                         parsed_response[key] = None # Assign default


        # Reset passed_from after it's been used (or attempted)
        # if state.get('passed_from') is not None: state['passed_from'] = None # Handled in _update_state? Check BaseAgent _update_state logic. Seems node_handler should clear it.
        state['passed_from'] = None # Clear it here after LLM call attempt

        # Call BaseAgent's synchronous state update method
        updated_state = await self._update_state(state, parsed_response, node_name)
        return updated_state # type: ignore


    # --- Synchronous Condition/Helper Methods (Keep from original Agent) ---

    async def checkroutingcondition(self, state: State) -> Literal["continue", "end", "reflection"]:
        """
        Determine the next step in the workflow (synchronous). Overrides BaseAgent.check_condition.

        NOTE: This is called AFTER _update_state, so state is already updated.
        We save the state here for continuity across workflow executions.
        """
        if self.logger:
            self.logger.info('----------------------------Deciding Node--------------------------------')
            self.logger.info(f"Checking routing: satisfied={state.get('satisfied')}, current_tool='{state.get('current_tool')}', reflection_counter={state.get('reflection_counter')}, tool_loop_counter={state.get('tool_loop_counter')}")

        satisfied = state.get("satisfied", False)
        current_tool = state.get("current_tool", None)
        reflection_counter = state.get('reflection_counter', 0)
        MAX_REFLECTIONS = config.MAX_REFLECTION_COUNT # Define max reflections

        # NEW: Save state snapshot for continuity (before routing decision)
        # This captures the current conversation state including messages, answer, etc.
        self._save_state_snapshot(state)

        # End condition: Satisfied and no tool/delegation needed
        # Check for tool.return_direct case handled in execute_tool setting satisfied=True & tool=None
        if satisfied and current_tool in [None, "None", ""]:
            if self.logger: self.logger.info("Routing decision: end (satisfied, no tool)")
            return "end"

        # Continue condition: Tool identified (and not satisfied yet, usually)
        # If a tool is selected, we generally continue to execute it.
        if current_tool not in [None, "None", ""]:
            if self.logger: self.logger.info(f"Routing decision: continue (tool '{current_tool}' identified)")
            return "continue"

        # Reflection condition: Not satisfied, no tool selected, and within reflection limits
        if not satisfied and current_tool in [None, "None", ""] and reflection_counter < MAX_REFLECTIONS:
            if self.logger: self.logger.info(f"Routing decision: reflection (not satisfied, no tool, reflection #{reflection_counter+1})")
            return "reflection"

        # Fallback End condition: Reached max reflections or other terminal state without satisfaction/tool
        if self.logger: self.logger.warning(f"Routing decision: end (fallback/max reflections - satisfied={satisfied}, tool='{current_tool}', reflections={reflection_counter})")
        # Ensure the final answer reflects the situation (e.g., failure to satisfy)
        # This might need adjustment in the Reflector node logic.
        return "end"

    def _save_state_snapshot(self, state: State) -> None:
        """
        Save a snapshot of the current state for continuity across workflow executions.

        This is called in checkroutingcondition (after _update_state) to capture:
        - Recent conversation messages
        - Last answer
        - Reasoning
        - Any other relevant context
        """
        try:
            # Get answer and ensure it's a string (not None)
            last_answer = state.get("answer") or ""
            last_reasoning = state.get("reasoning") or ""

            # Create a lightweight snapshot (don't store everything)
            self.retained_state = {
                # Keep recent messages for context (limited by retain_messages_order)
                "messages": state.get("messages", [])[-self.retain_messages_order:],
                "last_answer": last_answer,
                "last_reasoning": last_reasoning,
                "satisfied": state.get("satisfied", False),
                "tool_output": state.get("tool_output", None),
                "current_tool": state.get("current_tool", None),
                "tool_input": state.get("tool_input", None),
                "tool_decided_by": state.get("tool_decided_by", None),
                "previous_node": state.get("previous_node",None),
                # Don't retain counters (fresh start for each query)
            }

            if self.logger:
                self.logger.debug(f"State snapshot saved: {len(self.retained_state.get('messages', []))} messages, "
                                f"last_answer length: {len(last_answer)}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving state snapshot: {e}", exc_info=True)
            # Don't fail the workflow if snapshot fails
            self.retained_state = None


    async def _tool_loop_warning_prompt(self, state: State) -> str:
        """Generate warning if stuck in a tool loop (synchronous)."""
        if state.get('tool_loop_counter', 0) > self.MAX_TOOL_LOOP: # Check counter updated in _update_state
            return TOOL_LOOP_WARNING_PROMPT
        else:
            return ""

    async def _format_node_prompt(self, state: State, node: str) -> str:
        """Format the prompt for a specific node (synchronous)."""
        # Get tool output from state (full version stored in state)
        tool_output_raw = state.get('tool_output', 'N/A')

        # Truncate tool output for LLM prompt to prevent excessive token usage
        # Full output remains in state for reference
        tool_output = self._truncate_tool_output(tool_output_raw)

        # Use current_question from state instead of messages[0]['content']
        original_question = state.get('current_question', 'No original question found.')
        tool_input = state.get('tool_input', 'N/A')

        # Safely access plan, format if exists
        plan_dict = state.get('plan')
        plan_str = ""
        if plan_dict and isinstance(plan_dict, dict):
             plan_items = "\n".join([f"  - Step {k+1}: {v}" for k, v in sorted(plan_dict.items())])
             plan_str = f"\n<CURRENT PLAN>:\n{plan_items} <\CURRENT PLAN>"
        # plan_str = f"\n{'<PLAN>: '+str(state['plan']) if state.get('plan') else ''}" # Old format

        warning = await self._tool_loop_warning_prompt(state=state)
        reflection_count_display = state.get('reflection_counter', 0) + 1 # For display in prompt

        if node == 'evaluator':
            return EVALUATOR_NODE_PROMPT.format(
                    warning=warning,
                    original_question=original_question,
                    tool_output=tool_output,
                    plan_str=plan_str,
                    tool_input=tool_input
                )

        elif node == 'reflector':
            return REFLECTOR_NODE_PROMPT.format(
                    warning=warning,
                    original_question=original_question,
                    tool_output=tool_output,
                    plan_str=plan_str,
                    reflection_count_display=reflection_count_display,
                    tool_input=tool_input
                )

        elif node == 'planner':
             # Planner prompt needs to ask for the plan AND the first action
             return PLANNER_NODE_PROMPT.format(
                    warning=warning,
                    original_question=original_question
                )

        elif node == 'router': # Added case for router prompt formatting
             return ROUTER_NODE_PROMPT.format(
                    warning=warning,
                    original_question=original_question,
                    plan_str=plan_str,
                    tool_output=tool_output,
                )


        else:
             if self.logger: self.logger.warning(f"Prompt formatting not defined for node: {node}")
             return original_question # Fallback prompt

    # --- Async Node Methods (Overrides BaseAgent placeholders if needed, otherwise new) ---

    async def router(self, state: State) -> State:
        """Route the query to a tool or delegate asynchronously."""
        state['current_node'] = 'router'
        if self.logger: self.logger.info("--- Entering Router Node (Async) ---")
        prev_node = state.get('previous_node')

        # Build component context with priority order
        component_context = []

        # Priority 1: If coming directly from evaluator or reflector
        if prev_node == 'evaluator' and self.llm_evaluator:
             component_context = self.llm_evaluator.chat_history[-self.shared_memory_order:]
        elif prev_node == 'reflector' and self.llm_reflector:
             component_context = self.llm_reflector.chat_history[-self.shared_memory_order:]
        elif prev_node == "execute_tool":
            tool_decided_by = state.get('tool_decided_by')
            if tool_decided_by == 'router' and self.llm_router:
                pass
            elif tool_decided_by == 'evaluator' and self.llm_evaluator:
                component_context = self.llm_evaluator.chat_history[-self.shared_memory_order:]
            elif tool_decided_by == 'reflector' and self.llm_reflector:
                component_context = self.llm_reflector.chat_history[-self.shared_memory_order:]
        else:
            # Priority 2: Fallback to state messages
            if prev_node != 'router': # Log only if not coming from router (common on first loop)
                component_context = state.get('messages', [])[-self.shared_memory_order:]
                # remove the last messae since it's same as query. We are providing query separately to prompt.
                component_context = component_context[:-1]


        prompt = await self._format_node_prompt(state=state, node='router')
        return await self.node_handler(state, self.llm_router, prompt, component_context=component_context, node='router')

    async def evaluator(self, state: State) -> State:
        """Evaluate tool output asynchronously."""
        state['current_node'] = 'evaluator'
        if self.logger: self.logger.info("--- Entering Evaluator Node (Async) ---")
        prev_node = state.get('previous_node')
        component_context = []
        # Evaluator often needs context from the node *before* the tool execution
        # Or from the tool execution itself (passed via state['tool_output'])
        # Let's primarily use context from router/planner/reflector that *led* to the tool call
        if prev_node == 'execute_tool':
             # Use context from the node that actually decided to use the tool
             # Tool output is already included in the prompt via {tool_output}
             tool_decided_by = state.get('tool_decided_by')
             if tool_decided_by == 'router' and self.llm_router:
                 component_context = self.llm_router.chat_history[-self.shared_memory_order:]
             elif tool_decided_by == 'reflector' and self.llm_reflector:
                 component_context = self.llm_reflector.chat_history[-self.shared_memory_order:]

             # Fallback to router/planner if tool_decided_by is not set (backward compatibility)
             elif not tool_decided_by:
                 if self.llm_router: component_context = self.llm_router.chat_history[-self.shared_memory_order:]
                 elif self.llm_planner: component_context = self.llm_planner.chat_history[-self.shared_memory_order:]

        elif prev_node == 'reflector' and self.llm_reflector: component_context = self.llm_reflector.chat_history[-self.shared_memory_order:]
        elif prev_node == 'router' and self.llm_router: component_context = self.llm_router.chat_history[-self.shared_memory_order:]
        elif prev_node == 'planner' and self.llm_planner: component_context = self.llm_planner.chat_history[-self.shared_memory_order:]
        else:
            if prev_node!= 'evaluator': # Log only if not coming from router (common on first loop)
                component_context = state.get('messages', [])[-self.shared_memory_order:]


        prompt = await self._format_node_prompt(state=state, node='evaluator')
        return await self.node_handler(state, self.llm_evaluator, prompt, component_context=component_context, node='evaluator')

    async def reflection(self, state: State) -> State:
        """Reflect on progress and generate a final answer asynchronously."""
        state['current_node'] = 'reflector'
        if self.logger: self.logger.info(f"--- Entering Reflection Node (Async) #{state.get('reflection_counter', 0) + 1} ---")

        # Increment reflection counter - handled by _update_state now? No, _update_state doesn't increment reflection counter. Do it here.
        state['reflection_counter'] = state.get('reflection_counter', 0) + 1

        prev_node = state.get('previous_node')
        component_context = []
        # Gather context from relevant pr4evious steps
        if prev_node == 'router' and self.llm_router: component_context = self.llm_router.chat_history[-self.shared_memory_order:]
        elif prev_node == 'evaluator' and self.llm_evaluator: component_context = self.llm_evaluator.chat_history[-self.shared_memory_order:]
        elif prev_node == 'planner' and self.llm_planner: component_context = self.llm_planner.chat_history[-self.shared_memory_order:]
        elif prev_node == 'execute_tool':
             # Use context from the node that actually decided to use the tool
             # Tool output is already included in the prompt via {tool_output}
             tool_decided_by = state.get('tool_decided_by')
             if tool_decided_by == 'router' and self.llm_router:
                 component_context = self.llm_router.chat_history[-self.shared_memory_order:]
             elif tool_decided_by == 'evaluator' and self.llm_evaluator:
                 component_context = self.llm_evaluator.chat_history[-self.shared_memory_order:]
        else:
            if prev_node!="reflector": # Log only if not coming from reflector (common on first loop)
                component_context = state.get('messages', [])[-self.shared_memory_order:]


        prompt = await self._format_node_prompt(state=state, node='reflector')
        current_state = await self.node_handler(state, self.llm_reflector, prompt, component_context=component_context, node='reflector')

        if self.logger: self.logger.info("--- Exiting Reflection Node (Async) ---")
        return current_state

    async def planner(self, state: State) -> State:
        """Plan tasks for complex queries asynchronously."""
        state['current_node'] = 'planner'
        if self.logger: self.logger.info("--- Entering Planner Node (Async) ---")
        prev_node = state.get('previous_node')
        component_context = [] # Planner usually starts fresh or gets context from reflection if replanning
        if prev_node == 'reflector' and self.llm_reflector:
             component_context = self.llm_reflector.chat_history[-self.shared_memory_order:]

        prompt = await self._format_node_prompt(state=state, node='planner')
        if not self.llm_planner:
             if self.logger: self.logger.error("Planner node reached, but no llm_planner configured.")
             # Create error response using _update_state structure
             error_response = {
                 "answer": "Error: Planning component is not available.",
                 "satisfied": False,
                 "reasoning": "Internal configuration error: llm_planner missing.",
                 "tool": None, "tool_input": None, "delegate_to_agent": None
                 }
             return self._update_state(state, error_response, 'planner_error') # type: ignore

        return await self.node_handler(state, self.llm_planner, prompt, component_context=component_context, node='planner')

    # --- Workflow Compilation (Overrides BaseAgent placeholder) ---

    def agentworkflow(self) -> CompiledStateGraph:
        """Compile the agent's workflow graph with async nodes."""
        workflow = StateGraph(State) # type: ignore

        # Add async nodes from this class
        workflow.add_node("evaluator", self.evaluator)
        workflow.add_node("reflection", self.reflection)

        # Add async execute_tool inherited from BaseAgent
        workflow.add_node("execute_tool", self.execute_tool) # This is now async def in BaseAgent

        if self.plan:
            workflow.add_node("planner", self.planner)
            workflow.set_entry_point("planner")
            # Planner output (via node_handler -> _update_state) should set tool/satisfied
            # Use checkroutingcondition to decide where to go after planner
            workflow.add_conditional_edges("planner", self.checkroutingcondition, {
                "continue": "execute_tool", # Planner decided tool is first step
                "reflection": "reflection", # Planner decided reflection is needed first
                "end": END # Planner answered directly (unlikely)
            })
        else:
            workflow.add_node("router", self.router)
            workflow.set_entry_point("router")
            # Router decides first step
            workflow.add_conditional_edges("router", self.checkroutingcondition, {
                "continue": "execute_tool",
                "reflection": "reflection",
                "end": END
            })

        # --- Edges from intermediate nodes ---

        # After executing a tool, decide next step (evaluator or end if return_direct)
        workflow.add_conditional_edges("execute_tool", self.checkroutingcondition, {
            "continue": "evaluator",  # Normal case: evaluate tool output
            "reflection": "reflection",  # Tool execution suggests reflection needed
            "end": END  # Tool has return_direct=True, go directly to end
        })

        # After evaluating, decide next step based on satisfaction and tool needs
        workflow.add_conditional_edges("evaluator", self.checkroutingcondition, {
             "continue": "execute_tool", # Evaluation identified another tool needed
             "reflection": "reflection", # Evaluation suggests reflection is needed
             "end": END # Evaluation confirmed goal is met
        })

        # After reflecting, decide next step
        workflow.add_conditional_edges("reflection", self.checkroutingcondition, {
             "continue": "execute_tool", # Reflection identified a tool needed
             "reflection": "reflection", # Needs more reflection (loop guard in checkroutingcondition)
             "end": END # Reflection concluded answer is ready or cannot proceed
        })

        # Compile the graph
        return workflow.compile()



    async def initiate_agent(self, query: str, passed_from: Optional[str] = None, previous_node: Optional[str]= None) -> Dict:
        """
        Initiate the agent asynchronously and return the final result.

        Args:
            query: The user query to process
            passed_from: Optional identifier of the component that passed the query

        Returns:
            Dict: The final state containing the agent's response
        """
        if not self.app:
            raise ValueError("Agent workflow not compiled. Cannot initiate.")
        if self.logger: self.logger.debug(f"Agent: {self.agent_name}")

        new_query = await self._sanitize_query(query)

        # NEW: Check if we have retained state from previous execution
        if self.retained_state:
            if self.logger:
                self.logger.info(f"Restoring state from previous execution: "
                               f"{len(self.retained_state.get('messages', []))} messages in history")

            # Start with retained messages and add new query
            initial_messages = self.retained_state.get("messages", []).copy()
            initial_messages.append({"role": "user", "content": new_query})

            # Create initial state with restored context
            initial_state = State(
                messages=initial_messages,
                current_tool=self.retained_state.get("current_tool", ""), tool_input=self.retained_state.get("tool_input", None), tool_output=self.retained_state.get("tool_output", ""), answer=self.retained_state.get("last_answer", ""),
                satisfied=False, reasoning="", delegate_to_agent=None,
                current_node='planner' if self.plan else 'router',
                previous_node=self.retained_state.get("previous_node", None), plan={}, passed_from=passed_from,
                reflection_counter=0, tool_loop_counter=0, tool_decided_by=None,
                current_question=new_query  # Track current question
            )
        else:
            if self.logger:
                self.logger.info("No retained state found, starting fresh")

            # Create initial state (fresh start)
            initial_state = State(
                messages=[{"role": "user", "content": new_query}],
                current_tool="", tool_input=None, tool_output="", answer="",
                satisfied=False, reasoning="", delegate_to_agent=None,
                current_node='planner' if self.plan else 'router',
                previous_node=previous_node, plan={}, passed_from=passed_from,
                reflection_counter=0, tool_loop_counter=0, tool_decided_by=None,
                current_question=new_query  # Track current question
            )

        configuration = {"recursion_limit": config.MAX_RECURSION_LIMIT}

        try:
            # Execute fully and return the final state
            final_state = await self.app.ainvoke(initial_state, config=configuration)
            # Return the final state dictionary directly
            return dict(final_state)  # Ensure it's a plain dict

        except Exception as e:
            if self.logger: self.logger.error(f"Error during agent invocation: {e}", exc_info=True)
            error_state = initial_state.copy()  # Start with initial state
            error_state.update({
                "answer": f"Agent execution failed with error: {e}",
                "satisfied": False,
                "reasoning": f"Error during ainvoke: {e}",
                "current_node": "error",
                "tool_output": f"Error: {e}"  # Add error to tool_output maybe
            })
            return dict(error_state)  # Return as dict


    async def initiate_agent_astream(self, query: str, passed_from: Optional[str] = None, previous_node: Optional[str]= None) -> AsyncGenerator[Dict, None]:
        """
        Initiate the agent asynchronously with streaming updates.

        Args:
            query: The user query to process
            passed_from: Optional identifier of the component that passed the query

        Yields:
            Dict: State updates as the agent processes the query
        """
        if not self.app:
            raise ValueError("Agent workflow not compiled. Cannot initiate.")
        if self.logger: self.logger.debug(f"Agent: {self.agent_name}")

        new_query = await self._sanitize_query(query)

        # NEW: Check if we have retained state from previous execution
        if self.retained_state:
            if self.logger:
                self.logger.info(f"Restoring state from previous execution (streaming): "
                               f"{len(self.retained_state.get('messages', []))} messages in history")

            # Start with retained messages and add new query
            initial_messages = self.retained_state.get("messages", []).copy()
            initial_messages.append({"role": "user", "content": new_query})

            # Create initial state with restored context
            initial_state = State(
                messages=initial_messages,
                current_tool=self.retained_state.get("current_tool", ""), tool_input=self.retained_state.get("tool_input", None), tool_output=self.retained_state.get("tool_output", ""), answer=self.retained_state.get("last_answer", ""),
                satisfied=False, reasoning="", delegate_to_agent=None,
                current_node='planner' if self.plan else 'router',
                previous_node=self.retained_state.get("previous_node", None), plan={}, passed_from=passed_from,
                reflection_counter=0, tool_loop_counter=0, tool_decided_by=None,
                current_question=new_query  # Track current question
            )
        else:
            if self.logger:
                self.logger.info("No retained state found, starting fresh (streaming)")

            # Create initial state (fresh start)
            initial_state = State(
                messages=[{"role": "user", "content": new_query}],
                current_tool="", tool_input=None, tool_output="", answer="",
                satisfied=False, reasoning="", delegate_to_agent=None,
                current_node='planner' if self.plan else 'router',
                previous_node=previous_node, plan={}, passed_from=passed_from,
                reflection_counter=0, tool_loop_counter=0, tool_decided_by=None,
                current_question=new_query  # Track current question
            )

        configuration = {"recursion_limit": config.MAX_RECURSION_LIMIT}

        try:
            # Stream state updates
            async for state in self.app.astream(initial_state, config=configuration, stream_mode=[config.stream_mode]):
                yield state

        except Exception as e:
            if self.logger: self.logger.error(f"Error during streaming: {e}", exc_info=True)
            yield {
                "answer": f"Streaming error: {e}",
                "satisfied": False,
                "current_node": "error"
            }
