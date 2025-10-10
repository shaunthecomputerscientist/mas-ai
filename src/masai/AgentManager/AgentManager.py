import os, json
from typing import List, Tuple, Type, Union, Literal, Dict, Optional, Callable
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from ..GenerativeModel.generativeModels import MASGenerativeModel
from ..Agents.singular_agent import Agent
from ..pydanticModels.AnswerModel import answermodel
from dataclasses import dataclass
from ..prompts.prompt_templates import get_agent_prompts
@dataclass
class AgentDetails:
    capabilities: List[str]  # e.g., ["reasoning", "coding", "science"]
    description: str = ""  # Optional additional description
    style: str = "gives very elaborate answers"  # Communication style
    
class AgentManager:
    def __init__(self, logging=True, context:dict=None, model_config_path=None, chat_log:str=None, streaming:bool= False, streaming_callback : Optional[Callable]=None):
        """Initialize the AgentManager with an empty registry of agents.

            The AgentManager class serves as a central registry for creating, managing, and
            coordinating multiple agents in a multi-agent system.

            Args:
                logging (bool, optional): Enable or disable logging of agent activities.
                    Defaults to True.
                context (dict, optional): Additional contextual information to be shared
                    with all agents. Defaults to None.
                model_config_path (str): Path to the model configuration file.
                    This is a required argument.
                chat_log (str, optional): Path to file where chat log needs to be saved.
                    Defaults to None.
                streaming (bool, optional): Enable or disable streaming of responses from
                    LLMs. Defaults to False.
                streaming_callback (Optional[Callable], optional): An async callable
                    function that will be called with chunks of streamed content if
                    `streaming` is True. Must be provided if `streaming` is True.
                    Defaults to None.

        Attributes:
            agents (dict[str, Agent]): Dictionary storing agent instances,
                where keys are agent names and values are Agent objects.
            agent_prompts (dict): Dictionary storing system prompts for each agent.
            logging (bool): Flag to control logging behavior.
            context (dict): Shared context available to all agents.
            model_config_path (str): Path to the model configuration file.
            chat_log (str or None): Path to the chat log file, or None if logging
                to file is disabled.
            streaming (bool): Flag indicating if streaming is enabled.
            streaming_callback (Optional[Callable]): The callback function for streaming,
                or None if streaming is disabled.

        Raises:
            ValueError: If `model_config_path` is not provided or if `streaming` is
                        True but `streaming_callback` is not provided.
        """
        self.agents = {}
        self.agent_prompts = {}
        self.logging = logging
        self.context = context
        self.chat_log=chat_log
        self.streaming = streaming
        self.streaming_callback = streaming_callback
        if self.streaming and not self.streaming_callback:
            raise ValueError("Streaming callback needs to be provided for streaming. Should be async callable that takes in chunks.")
        
        # model_config_path should be provided by user
        if not model_config_path:
            raise ValueError("model_config_path must be provided")
        self.model_config_path = model_config_path

    def load_prompts(self) -> Tuple[str, str, str]:
        """Load prompts from module."""
        return get_agent_prompts()

    def promptformatter(self, router_prompt: str, evaluator_prompt: str, reflector_prompt: str, planner_prompt: str, system_prompt: str) -> Tuple[ChatPromptTemplate, ChatPromptTemplate, ChatPromptTemplate, ChatPromptTemplate]:
        """Format prompts into ChatPromptTemplates."""
        input_variables = ['question', 'history', 'schema','current_time','useful_info','coworking_agents_info','long_context']
        template = """
        <INFO>:{useful_info}</INFO>
        \n\n<TIME>:{current_time}</TIME>
        \n\n<AVAILABLE COWORKING AGENTS>:{coworking_agents_info}</AVAILABLE COWORKING AGENTS>
        \n\n<RESPONSE FORMAT>:{schema}</RESPONSE FORMAT>
        \n\n<CHAT HISTORY>:{history}</CHAT HISTORY>
        \n\n<EXTENDED CONTEXT>:{long_context}</EXTENDED CONTEXT>
        \n<QUESTION>:{question}</QUESTION>
        """
        
        human_message_template = HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=input_variables, template=template)
        )
        
        system_message_template_1 = SystemMessagePromptTemplate(
            prompt=PromptTemplate(template =system_prompt + "\nFOLLOW THESE INSTRUCTIONS:" + router_prompt )
        )
        system_message_template_2 = SystemMessagePromptTemplate(
            prompt=PromptTemplate(template=system_prompt +"\nFOLLOW THESE INSTRUCTIONS:" + evaluator_prompt )
        )
        system_message_template_3 = SystemMessagePromptTemplate(
            prompt=PromptTemplate(template=system_prompt + "\nFOLLOW THESE INSTRUCTIONS:" + reflector_prompt )
        )
        
        system_message_template_4 = SystemMessagePromptTemplate(
            prompt=PromptTemplate(template=system_prompt + "\nFOLLOW THESE INSTRUCTIONS:" + planner_prompt if planner_prompt else "")
        )
        
        router_chat_prompt = ChatPromptTemplate.from_messages([system_message_template_1, human_message_template])
        evaluator_chat_prompt = ChatPromptTemplate.from_messages([system_message_template_2, human_message_template])
        reflector_chat_prompt = ChatPromptTemplate.from_messages([system_message_template_3, human_message_template])
        if planner_prompt:
            planner_chat_prompt = ChatPromptTemplate.from_messages([system_message_template_4, human_message_template])
        else:
            planner_chat_prompt = None
        return router_chat_prompt, evaluator_chat_prompt, reflector_chat_prompt, planner_chat_prompt

    def _load_model_config(self, agent_name: str) -> dict:
        """Load model configuration from a JSON file."""
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(f"Model config file not found at {self.model_config_path}.")
        
        with open(self.model_config_path, "r") as f:
            data=json.load(f)
            
        if agent_name in data:
            return data[agent_name]
        elif 'all' in data:
            return data['all']

    def create_agent(self, agent_name: str, tools: List[object], agent_details: AgentDetails, 
                 memory_order: int = 10, long_context: bool = True,long_context_order: int = 20, shared_memory_order: int = 10, 
                 plan: bool = False,temperature=0.2,context_callable:Optional[Callable]=None,retain_messages_order: int = 10,**kwargs):
        """Create and register a new agent in the AgentManager.

        Args:
            agent_name (str): Unique identifier for the agent (converted to lowercase).
            tools (List[object]): Tools the agent can use, each with a 'name' attribute.
            agent_details (AgentDetails): Configuration with capabilities, description, and style.
            memory_order (int, optional): Number of past interactions to keep. Defaults to 20.
            long_context (bool, optional): Use long context if True. Defaults to True.
            long_context_order (int, optional): Number of past interactions summary to keep in long context. Defaults to 10.
            shared_memory_order (int, optional): Shared memory size for components. Defaults to 10.
            plan (bool, optional): Include planner if True. Defaults to False.
            context_callable (Optional[Callable]): Callable that uses user input to give more context to the llm during inference.
            retain_messages_order (int, optional): Number of past interactions to keep in memory for an agent's internal state across multiple queries. Defaults to 10.
            
            **kwargs: Additional keyword arguments.  Can include:
                - `config_dict` (dict, optional): A dictionary specifying memory order overrides for individual LLMs.
                  The dictionary should have the following structure:
                  ```
                  {
                      "router_memory_order": int,  # Memory order for the router LLM
                      "router_long_context_context": int, # Long context order for the router LLM
                      "router_temperature": int, # temperature for router
                      "evaluator_memory_order": int, # Memory order for the evaluator LLM
                      "evaluator_long_context_order": int, # Long context order for the evaluator LLM
                      "evaluator_temperature": int, # temperature for evaluator
                      "reflector_memory_order": int, # Memory order for the reflector LLM
                      "reflector_long_context_order": int, # Long context order for the reflector LLM
                      "reflector_temperature": int, # temperature for reflector
                      "planner_memory_order": int, # Memory order for the planner LLM (if plan is True)
                      "planner_long_context_order": int # Long context order for the planner LLM (if plan is True)
                      "planner_temperature": int, # temperature for planner
                  }
                  ```
                
                If a specific component's memory order is not provided in the dictionary, the default `memory_order` and `long_context_order` values will be used.
                
                - `in_memory_store (InMemoryDocStore) : from masai.Memory.InMemoryStore import InMemoryDocStore and set it while using LTIMS variable.
                - `top_k (int, optional) : returns top k elements from memory store matching the query`

        Raises:
            ValueError: If agent_name already exists.
            FileNotFoundError: If prompts file is missing.
        """
        agent_name = agent_name.lower()
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' already exists.")

        # Load and format prompts
        prompts = self.load_prompts()
        system_prompt = self._create_system_prompt(agent_name, agent_details)
        chat_prompts = self.promptformatter(*prompts, system_prompt=system_prompt)

        # Configure tools and answer format
        tool_mapping = {tool.name: tool for tool in tools}
        AnswerFormat = answermodel(tool_names=list(tool_mapping.keys()) + ['None'], tools=tools)

        # Initialize LLM models
        model_config = self._load_model_config(agent_name)
        llm_args = {"temperature": temperature, "memory_order": memory_order, 
                    "extra_context": self.context, 
                    "long_context": long_context,
                    "long_context_order":long_context_order,
                    "chat_log":self.chat_log,
                    "streaming": self.streaming,
                    "streaming_callback": self.streaming_callback,
                    "context_callable": context_callable
                    }
        if kwargs.get('in_memory_store'):
            llm_args["memory_store"] = kwargs['in_memory_store']
            llm_args['k'] = kwargs.get('top_k')
        
        
        def override_config(component, llm_args, memory_order, long_context_order,temperature, **kwargs):
            temp_args=llm_args.copy()
            if "config_dict" in kwargs:
                config_dict = kwargs["config_dict"]
                temp_args["temperature"] = config_dict.get(f"{component}_temperature",temperature)
                temp_args["memory_order"] = config_dict.get(f"{component}_memory_order", memory_order)
                temp_args["long_context_order"] = config_dict.get(f"{component}_long_context_order", long_context_order)
            return temp_args
        
        llm_router_args = override_config("router", llm_args, memory_order, long_context_order,temperature)
        llm_router = MASGenerativeModel(model_config["router"]["model_name"], category=model_config["router"]["category"], prompt_template=chat_prompts[0], **llm_router_args)

        llm_evaluator_args = override_config("evaluator", llm_args, memory_order, long_context_order,temperature)
        llm_evaluator = MASGenerativeModel(model_config["evaluator"]["model_name"], category=model_config["evaluator"]["category"], prompt_template=chat_prompts[1], **llm_evaluator_args)

        llm_reflector_args = override_config("reflector", llm_args, memory_order, long_context_order,temperature)
        llm_reflector = MASGenerativeModel(model_config["reflector"]["model_name"], category=model_config["reflector"]["category"], prompt_template=chat_prompts[2], **llm_reflector_args)

        if plan:
            llm_planner_args = override_config("planner", llm_args, memory_order, long_context_order, temperature)
            llm_planner = MASGenerativeModel(model_config["planner"]["model_name"], category=model_config["planner"]["category"], prompt_template=chat_prompts[3], **llm_planner_args)
        else:
            llm_planner = None

        agent = Agent(agent_name, llm_router, llm_evaluator, llm_reflector, llm_planner, tool_mapping, AnswerFormat, self.logging, shared_memory_order=shared_memory_order,retain_messages_order=retain_messages_order)
        self.agents[agent_name.lower()] = agent
        self.agent_prompts[agent_name.lower()] = system_prompt
    def _compile_agents(self,type='decentralized',agent_context:dict=None):
        """Share agent system prompts among all registered agents.

        This method ensures each agent is aware of other agents' capabilities and characteristics
        by sharing their system prompts. For each agent, it creates a dictionary of all other 
        agents' prompts (excluding itself) and stores it in the agent's context.

        Example:
            If there are agents A, B, and C:
            - Agent A will receive prompts from B and C
            - Agent B will receive prompts from A and C
            - Agent C will receive prompts from A and B

        Note:
            This method should be called after all agents have been created and before
            starting any agent interactions to ensure proper inter-agent awareness.
        """
        if type=='decentralized':
            for agents in self.agents.values():
                prompts ={}
                for agent_name in self.agent_prompts:
                    if agent_name!=agents.agent_name:
                        prompts[agent_name]=self.agent_prompts[agent_name]

                # print(prompts)
                agents.agent_context = prompts
        elif type=='hierarchical':
            for agents in self.agents.values():
                agents.agent_context = agent_context
        return

    def _create_system_prompt(self, agent_name: str, details: AgentDetails) -> str:
        """Convert AgentDetails into a system prompt."""
        capabilities_str = ",".join(details.capabilities)
        
        prompt_parts = [
            f"\n NAME: {agent_name}.\n YOUR CHARACTERISTICS AND CAPABILITIES {capabilities_str}",
            f"RESPONSE STYLE: {details.style}."
        ]
        
        if details.description:
            prompt_parts.append(details.description)
                
        return "\n".join(prompt_parts)
    

    def get_agent(self, agent_name: str)->Agent:
        """Retrieve an agent by name."""
        if agent_name.lower() not in self.agents:
            raise ValueError(f"No agent found with name '{agent_name}'.")
        
        return self.agents[agent_name.lower()]

    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self.agents.keys())

    def cleanup(self) -> None:
        """
        Cleanup all resources held by AgentManager.

        This method is called when AgentManager is removed from cache (after 30 min TTL).
        It ensures all agents, LLM instances, and their memory are properly freed.
        """
        try:
            # Clear all agents and their resources
            for agent_name, agent in self.agents.items():
                # Clear agent's retained state
                if hasattr(agent, 'retained_state'):
                    agent.retained_state = None

                # Clear LLM chat histories
                if hasattr(agent, 'llm_router') and agent.llm_router:
                    if hasattr(agent.llm_router, 'chat_history'):
                        agent.llm_router.chat_history.clear()

                if hasattr(agent, 'llm_evaluator') and agent.llm_evaluator:
                    if hasattr(agent.llm_evaluator, 'chat_history'):
                        agent.llm_evaluator.chat_history.clear()

                if hasattr(agent, 'llm_reflector') and agent.llm_reflector:
                    if hasattr(agent.llm_reflector, 'chat_history'):
                        agent.llm_reflector.chat_history.clear()

                if hasattr(agent, 'llm_planner') and agent.llm_planner:
                    if hasattr(agent.llm_planner, 'chat_history'):
                        agent.llm_planner.chat_history.clear()

                # Clear tool mapping
                if hasattr(agent, 'tool_mapping'):
                    agent.tool_mapping.clear()

            # Clear agent registry
            self.agents.clear()
            self.agent_prompts.clear()

            # Clear context
            if self.context:
                self.context.clear()

        except Exception as e:
            # Don't fail cleanup, just log if possible
            print(f"Error during AgentManager cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup when AgentManager is garbage collected."""
        self.cleanup()
