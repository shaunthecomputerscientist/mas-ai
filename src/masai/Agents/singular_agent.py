from langgraph.graph import END, StateGraph, START
from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional
from pydantic import BaseModel, Field
import ast, os
from dotenv import load_dotenv
load_dotenv()
from ..GenerativeModel.generativeModels import MASGenerativeModel,GenerativeModel
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
    delegate_to_agent:str
    current_node: str
    previous_node: str
    plan: List[str]
    passed_from: str
class Agent:
    """Agent Made Out of Routing-Evaluator-Reflector Architecture"""
    _logger=None
    def __init__(self,agent_name,llm_router, llm_evaluator, llm_reflector, llm_planner=None, tool_mapping=None, AnswerFormat:BaseModel=None,logging=True, agent_context=None, shared_memory_order:int=5):
        """Initialize an agent with router-evaluator-reflector architecture and optional planner.
        
        The agent uses a state machine workflow to process queries through specialized LLMs:
        - Router: Determines which tool to use or agent to delegate to
        - Evaluator: Evaluates tool outputs and determines next steps
        - Reflector: Reflects on overall progress and generates final answers
        - Planner (optional): Creates execution plans for complex tasks

        Args:
            agent_name (str): Name identifier for the agent instance
            llm_router (BaseGenerativeModel): Language model for routing decisions - determines which tool to use or agent to delegate to
            llm_evaluator (BaseGenerativeModel): Language model for evaluation - processes tool outputs and determines next steps
            llm_reflector (BaseGenerativeModel): Language model for reflection - analyzes overall progress and generates final answers
            llm_planner (BaseGenerativeModel, optional): Language model for planning complex tasks. Defaults to None.
            tool_mapping (Dict[str, Callable], optional): Mapping of tool names to their function implementations. Defaults to None.
            AnswerFormat (BaseModel, optional): Pydantic model defining the structure of agent responses. Defaults to None.
            logging (bool, optional): Enable/disable logging functionality. Defaults to True.
            agent_context (Dict[str, Any], optional): Additional context information for the agent in multi agent system, providing context about other agents it should interact with. Defaults to None.
            shared_memory_order (int, optional): Number of previous interactions to maintain in shared memory among individual components of an agent. Defaults to 5.
            retain_messages_order (int, optional): Number of previous interactions to maintain in agent's system memory.This includes short term memory of all components within the agent. Defaults to 20.
        """
        self.agent_name = agent_name
        self.llm_evaluator :BaseGenerativeModel =llm_evaluator
        self.llm_router : BaseGenerativeModel=llm_router
        self.llm_reflector : BaseGenerativeModel=llm_reflector
        if llm_planner:
            self.llm_planner = llm_planner
            self.plan=True
        else:
            self.llm_planner = None
            self.plan=False
        self.app = self.agentworkflow()
        self.graph = self.app.get_graph()
        self.tool_mapping : dict= tool_mapping
        self.pydanticmodel : BaseModel = AnswerFormat
        self.logging = logging
        self.shared_memory_order=shared_memory_order

        if self.logging:
            if Agent._logger is None:
                Agent._logger = setup_logger()
            self.logger = Agent._logger
        else:
            self.logger = None


        self.agent_context = agent_context
        self.node:str='evaluator'
        self.retain_messages_order=20


    def set_context(self,context:Optional[Dict]=None,mode="set"):
        """Set Context Either to override existing context passed to Agent Manager or Define new Context for all components.
        Two modes: update or set.
        """
        components = [self.llm_evaluator,self.llm_reflector, self.llm_router,self.llm_planner]
        try:
            if context:
                for component in components:
                    if component is not None:
                        if mode=='set':
                            component.info=context
                        elif mode=='update':
                            component.info.update(context)
        except Exception as e:
            raise e
        
                        
                    
        
    def gettoolinput(self, tool_input : dict, tool_name: str)->Union[Dict,str]:
        tool_input = parse_tool_input(tool_input, list((self.tool_mapping[tool_name]).args_schema.schema()['properties'].keys()))
        return tool_input

    def display(self):
        """Display the graph of the agent"""
        png_data = self.graph.draw_mermaid_png()
        # Save the PNG image to a file
        mermaid_dir = os.path.join('MAS','Database','mermaid')
        os.makedirs(mermaid_dir, exist_ok=True)
        png_file_path = os.path.join(mermaid_dir, "diagram.png")
        with open(png_file_path, "wb") as f:
            f.write(png_data)

    def _update_state(self,current_state:State, parsed_response,node):
        if not (parsed_response['tool']=="none" or parsed_response['tool']==None):
            current_state["current_tool"] = parsed_response['tool']
            current_state['tool_input'] = self.gettoolinput(parsed_response['tool_input'],current_state['current_tool'])
            if self.logger:
                self.logger.warning("-------------------------------------Tool Input---------------------------------\n\n")
                self.logger.warning(current_state['tool_input'])

        
        if node=='planner':
            current_state['plan']=parse_task_string(parsed_response['answer'])
            for ele in current_state['plan']:
                print(ele)
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
        if len(current_state["messages"])>self.retain_messages_order:
            current_state['messages']=[current_state['messages'][0]].extend(current_state["messages"][-self.retain_messages_order:])
        return current_state

    def node_handler(self,state: State, llm : MASGenerativeModel, prompt:str, component_context=None, node=None):
        
            parsed_response = llm.generate_response_mas(prompt,
                            output_structure=self.pydanticmodel, 
                            agent_context=self.agent_context if self.agent_context else None, 
                            agent_name=self.agent_name,
                            component_context=component_context if component_context else [],
                            passed_from=state['passed_from'])
            if self.logger:
                self.logger.info(f"{parsed_response['answer']}")
            if state['passed_from'] is not None:
                state['passed_from']=None
            current_state = state
            current_state = self._update_state(current_state, parsed_response,node)
            return current_state

    def checkroutingcondition(self,state):
        if self.logger:
            self.logger.info('----------------------------Deciding Node--------------------------------')
        if state["satisfied"] and not(state["current_tool"]==None or state["current_tool"]== "None"):
            return "continue" #continue when satisfied is true but tool is provided
        
        elif state["satisfied"] and (state["current_tool"]==None or state["current_tool"] == "None"):

            return "end" #end when satisfied and tool is none
        
        elif not state["satisfied"] and (state["current_tool"]==None or state["current_tool"]== "None"):
            return "reflection" #reflect when not satisfied and tool is not provided
        elif state["current_tool"]==None or state["current_tool"]== "None":
            return "end" #end when no tool is chosen
        
        return "continue"


    def router(self,state: State) -> State:
        messages = state["messages"]
        state['current_node']='router'
        if self.node=='evaluator':
            component_context = self.llm_evaluator.chat_history[-self.shared_memory_order:]
        elif self.node=='reflector':
            component_context = self.llm_reflector.chat_history[-self.shared_memory_order:]
        elif self.node=='router':
            component_context=[]
        prompt = messages[0]['content'] if messages else ""
        current_state = self.node_handler(state, self.llm_router, prompt,component_context=component_context,node='router')
        return current_state

    def execute_tool(self,state: State) -> State:
        tool_name = state["current_tool"]
        tool_input = state["tool_input"]
        tool=self.tool_mapping[tool_name]

        result = tool.invoke(input=tool_input)
       
        if self.logger:
            self.logger.warning("-------------------------------------Tool Output---------------------------------\n\n")
            self.logger.warning(result)


        if tool.return_direct:
            state["tool_input"] = 'None'
            state["messages"].append({"role": f"Tool: {tool_name}", "content": str(result)})
            state['current_tool'] = 'None'
            state['answer']= str(result)
            state['reasoning'] = ""
            state['satisfied'] = True
            state['delegate_to_agent']=None
            self.logger.info("RETURNING DIRECT")
            return state

        state['tool_output'] = str(result)
        state['passed_from']=tool_name
        state["messages"].append({"role": f"Tool: {tool_name}", "tool_output": state['tool_output']})
        return state

    # Define the evaluation function
    def evaluator(self,state: State) -> Dict[str, Any]:
        messages = state["messages"]
        state['current_node'] = 'evaluator'
        tool_output = state["tool_output"]
        if state['previous_node']=='reflector':
            component_context=self.llm_reflector.chat_history[-self.shared_memory_order:]
        elif state['previous_node']=='router':
            component_context=self.llm_router.chat_history[-self.shared_memory_order:]
        elif state['previous_node']=='planner':
            component_context=self.llm_planner.chat_history[-self.shared_memory_order:]
        else:
            component_context=[]
            
        if state['previous_node']=='planner':
            prompt = f"\n\n<ORIGINAL QUESTION>: {messages[0]['content']}\n\n <PREVIOUS TOOL>:{state['current_tool']}\n\n<TOOL OUTPUT>: {tool_output}\n\n<PLAN>: {state['plan']}\n\n"
        else:
            prompt = f"\n\n<ORIGINAL QUESTION>: {messages[0]['content']}\n\n <PREVIOUS TOOL>:{state['current_tool']}\n\n<TOOL OUTPUT>: {tool_output}\n\n"
        current_state = self.node_handler(state,self.llm_evaluator,prompt,component_context=component_context,node='evaluator')

        return current_state

    def reflection(self, state: State):
        messages = state["messages"]
        if self.logger:
            self.logger.info("\n\n--------------------Reasoning and Reflecting-----------------------------\n\n")
        state['current_node'] = 'reflector'
        if state['previous_node']=='router':
            component_context=self.llm_router.chat_history[-self.shared_memory_order:]
        elif state['previous_node']=='evaluator':
            component_context=self.llm_evaluator.chat_history[-self.shared_memory_order:]
        elif state['previous_node']=='planner':
            component_context=self.llm_planner.chat_history[-self.shared_memory_order:]
        else:
            component_context=[]
        if state['previous_node']=='planner':
            prompt = f"""<CURRENT STAGE>: REFLECTION STAGE\n\n <GOAL>: Reflect/Reason on gathered component_context, think and arrive at solution. \n\n<LAST USED TOOL>{state['current_tool']}\n\n<TOOL OUTPUT>{state['tool_output']} \n\n<QUESTION> : {messages[0]['content']}\n\n<PLAN>: {state['plan']}"""
        else:
            prompt = f"""<CURRENT STAGE>: REFLECTION STAGE\n\n <GOAL>: Reflect/Reason on gathered component_context, think and arrive at solution. \n\n<LAST USED TOOL>{state['current_tool']}\n\n<TOOL OUTPUT>{state['tool_output']} \n\n<QUESTION> : {messages[0]['content']}"""
        current_state = self.node_handler(state, self.llm_reflector,prompt,component_context=component_context,node='reflector')
        if self.logger:
            self.logger.info("--------------------Reflection End-----------------------------")
        return current_state
    
    def planner(self,state: State):
        messages = state["messages"]
        state['current_node'] = 'planner'
        if self.node=='evaluator':
            component_context=self.llm_evaluator.chat_history[-self.shared_memory_order:]
        elif self.node=='reflection':
            component_context=self.llm_reflector.chat_history[-self.shared_memory_order:]
        else:
            component_context=[]
        prompt = f"""<CURRENT STAGE>: PLANNING STAGE\n\n <GOAL>: Plan tasks logically to accomplish the goal. \n\n<QUESTION> : {messages[0]['content']}"""
        current_state = self.node_handler(state, self.llm_planner,prompt,component_context=component_context,node='planner')
        return current_state

    def agentworkflow(self):
        workflow = StateGraph(State)
        nodes = ["execute_tool", "evaluator", "reflection"]
        if self.plan:
            nodes.append("planner")
        else:
            nodes.append("router")

        for node in nodes:
            workflow.add_node(node, getattr(self, node))

        # Dynamic edges
        if self.plan:
            workflow.add_edge(START, "planner")
            workflow.add_edge("planner", "evaluator")
        else:
            workflow.add_edge(START, "router")
            workflow.add_conditional_edges("router", self.checkroutingcondition, {
                "end": END, "reflection": "reflection", "continue": "execute_tool"
            })

        for node in ["execute_tool", "evaluator", "reflection"]:
            workflow.add_conditional_edges(node, self.checkroutingcondition, {
                "end": END, "continue": "evaluator" if node == "execute_tool" else "execute_tool", "reflection": "reflection"
            })

        workflow.set_entry_point("planner" if self.plan else "router")
        return workflow.compile()
    def _sanitize_query(self, query: str) -> str:
        """Sanitize input query by removing/replacing problematic characters."""
        replacements = {
            "'": '',      # Remove single quotes
            "\\": "",     # Remove backslashes
            """: '"',     # Replace smart quotes
            """: '"',     # Replace smart quotes
            '"': ""       # Remove double quotes
        }
        
        sanitized = str(query)
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        return sanitized

    def initiate_agent(self, query: str, passed_from:str=None):
        """Takes in query. Optional passed_from to let llm know if it's an agent, tool, or user who initated this agent."""
        new_query = self._sanitize_query(query)
        if self.logger:
            self.logger.debug(self.agent_name)
        initial_state = State(
            messages=[{"role": "user", "content": new_query}],
            current_tool="",
            tool_input="",
            tool_output="",
            answer="",
            satisfied=False,
            reasoning="",
            delegate_to_agent=None,
            current_node='router',
            previous_node=None,
            passed_from=passed_from
        )
        response = self.app.invoke(initial_state, {"recursion_limit": 100})
        return response





