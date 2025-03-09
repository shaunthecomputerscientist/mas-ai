from langgraph.graph import END, StateGraph, START
from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()
from ..GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel
from ..Tools.logging_setup.logger import setup_logger
from ..Tools.PARSERs.json_parser import parse_tool_input, parse_task_string
from langchain.schema import Document

class State(TypedDict):
    messages: List[Dict[str, str]]
    current_tool: str
    tool_input: str
    tool_output: str
    answer: str
    satisfied: bool
    reasoning: str
    delegate_to_agent: str
    current_node: str
    previous_node: str
    plan: List[str]
    passed_from: str
    tool_loop_counter:int = 0

class BaseAgent:
    _logger = None

    def __init__(self, agent_name: str, logging: bool = True, shared_memory_order: int = 5, retain_messages_order: int = 20):
        """Base class for agents with common functionality."""
        self.agent_name = agent_name.lower()
        self.logging = logging
        self.shared_memory_order = shared_memory_order
        self.retain_messages_order = retain_messages_order
        self.app = None  # StateGraph workflow, to be set by subclasses
        self.graph = None
        self.tool_mapping: Dict[str, Any] = {}
        self.pydanticmodel: Optional[Type[BaseModel]] = None
        self.agent_context: Optional[Dict[str, Any]] = None
        self.node = 'evaluator'

        if self.logging:
            if BaseAgent._logger is None:
                BaseAgent._logger = setup_logger()
            self.logger = BaseAgent._logger
        else:
            self.logger = None

    def set_context(self, context: Optional[Dict] = None, mode: str = "set"):
        """Set or update context for all components (optional, can be overridden)."""
        pass

    def gettoolinput(self, tool_input: dict, tool_name: str) -> Union[Dict, str]:
        """Parse and sanitize tool input."""
        return parse_tool_input(tool_input, list(self.tool_mapping[tool_name].args_schema.schema()['properties'].keys()))

    def display(self):
        """Display the agent's workflow graph if it exists."""
        if self.graph:
            png_data = self.graph.draw_mermaid_png()
            mermaid_dir = os.path.join('MAS', 'Database', 'mermaid')
            os.makedirs(mermaid_dir, exist_ok=True)
            png_file_path = os.path.join(mermaid_dir, f"{self.agent_name}_diagram.png")
            with open(png_file_path, "wb") as f:
                f.write(png_data)
        else:
            raise ValueError("Graph not initialized; ensure workflow is compiled")

    def _update_state(self, current_state: State, parsed_response: Dict, node: str) -> State:
        """Update the agent state based on a parsed response."""
        if parsed_response['tool'] not in ["None", None]:
            current_state["current_tool"] = parsed_response['tool']
            current_tool_input = self.gettoolinput(parsed_response['tool_input'], current_state['current_tool'])
            if current_tool_input==current_state['tool_input'] and current_tool_input not in ['None',None]: # checks with previous state
                current_state['tool_loop_counter']+=1
            else:
                current_state['tool_loop_counter']=1
            
            current_state['tool_input'] =  current_tool_input
            if self.logger:
                self.logger.warning("-------------------------------------Tool Input---------------------------------\n\n")
                self.logger.warning(current_state['tool_input'])

        if node == 'planner':
            current_state['plan'] = parse_task_string(parsed_response['answer'])
        
        self.node=node

        current_state.update({
            "previous_node": node,
            "current_node": node,
            "answer": parsed_response['answer'],
            "satisfied": parsed_response['satisfied'],
            "reasoning": parsed_response['reasoning'],
            "current_tool": parsed_response['tool'],
            "delegate_to_agent": parsed_response['delegate_to_agent']
        })

        current_state["messages"].append({"role": "assistant", "content": current_state['answer']})
        if len(current_state["messages"]) > self.retain_messages_order:
            current_state['messages'] = [current_state['messages'][0]] + current_state["messages"][-self.retain_messages_order:]
        return current_state

    def node_handler(self, state: State, llm: BaseGenerativeModel, prompt: str, component_context: Optional[List] = None, node: Optional[str] = None) -> State:
        """Handle node-specific LLM responses and update state (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement node_handler for specific LLMs")

    def _sanitize_query(self, query: str) -> str:
        """Sanitize input query by removing/replacing problematic characters."""
        replacements = {
            "'": '', "\\": "", """: '"', """: '"', '"': ""
        }
        sanitized = str(query)
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        return sanitized

    def initiate_agent(self, query: str, passed_from: Optional[str] = None) -> Dict:
        """Initiate the agent workflow with a query (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement initiate_agent with their workflow")

    def agentworkflow(self) -> StateGraph:
        """Compile and return the agent's workflow graph (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement agentworkflow to define their state machine")

    def check_condition(self, state: State) -> Literal["continue", "end", "reflection"]:
        """Base condition checker (to be overridden or specialized by subclasses)."""
        raise NotImplementedError("Subclasses must implement check_condition for their workflow")

    def execute_tool(self, state: State) -> State:
        """Execute a tool based on the current state (shared logic for both workflows)."""
        tool_name = state["current_tool"]
        tool_input = state["tool_input"]
        tool = self.tool_mapping[tool_name]

        result = tool.invoke(input=tool_input)
        if self.logger:
            self.logger.warning("-------------------------------------Tool Output---------------------------------\n\n")
            self.logger.warning(result)

        if tool.return_direct:
            state["tool_input"] = 'None'
            state["messages"].append({"role": f"Tool: {tool_name}", "content": str(result)})
            state.update({
                'current_tool': 'None',
                'answer': str(result),
                'reasoning': "",
                'satisfied': True,
                'delegate_to_agent': None
            })
            self.logger.info("RETURNING DIRECT")
            return state

        state['tool_output'] = str(result)
        state['passed_from'] = tool_name
        state["messages"].append({"role": f"Tool: {tool_name}", "tool_output": state['tool_output']})
        return state