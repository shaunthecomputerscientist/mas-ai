from typing import List, Dict, Optional, Tuple
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from ..MultiAgents.MultiAgent import MultiAgentSystem
from ..GenerativeModel.generativeModels import BaseGenerativeModel
from ..GenerativeModel.generativeModels import MASGenerativeModel  # Assuming this is the path to your LLM class
from ..prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from ..schema import Document
from ..pydanticModels.omanModel import structure_oman_supervisor



"""Currently in progress : Orchestrate the networks and agents."""
class OrchestratedMultiAgentNetwork:
    
    def __init__(self, mas_instances: List[MultiAgentSystem], network_memory_order: int = 3, oman_llm_config: Dict = None, extra_context: Dict = None):
        """Initialize the OMAN with a list of MultiAgentSystem instances.
        oman_llm_config: Dict = None,
        network_memory_order: int = 3,
        mas_instances: List[MultiAgentSystem] = None,
        extra_context: Dict = None
        
        oman_llm_config format:
        
        oman_llm_config = {
                "model_name": "gemini-2.0-flash-001",
                "category": "gemini",
                "temperature": 0.2,
                "memory_order": 3
            }
        """
        self.networks: Dict[str, MultiAgentSystem] = {}
        self.shared_memory = {"tasks": [], "outcomes": [], "capabilities": {}}
        self.network_memory_order = network_memory_order
        self.task_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.extra_context = extra_context
        
            
            
        for idx, mas in enumerate(mas_instances):
            network_name = f"Network_{idx + 1}"
            self.networks[network_name] = mas
        
        if not oman_llm_config:
            oman_llm_config = {
                "model_name": "gemini-2.0-flash-001",
                "category": "gemini",
                "temperature": 0.2,
                "memory_order": network_memory_order
            }
        
        # Assign names and populate networks

        self.oman_llm = BaseGenerativeModel(
            model_name=oman_llm_config["model_name"],
            category=oman_llm_config["category"],
            prompt_template=self._create_oman_prompt_template(),
            temperature=oman_llm_config.get("temperature", 0.2),
            memory_order=oman_llm_config.get("memory_order", 3),
            info=self.extra_context
        )
        
        

    def _create_oman_prompt_template(self) -> ChatPromptTemplate:
        """Create a prompt template for the OMAN LLM to act as a router."""
        system_prompt = """
        You are a supervisor/router for a network of Multi-Agent Systems (MAS). 
        Your task is to analyze the user's query and determine which network is best suited to handle it based on their capabilities and agent prompts.
        Each Network is made up of multiple agents that solves a problem.
        """
        template = """
        Below is the information about each network and their agent prompts:

        {network_info}

        Analyze the query and select the most appropriate network based on its specialization.
        """
        input_variables = ["network_info"]
        
        system_message_template = SystemMessagePromptTemplate(
            prompt=PromptTemplate(template=system_prompt)
        )
        human_message_template = HumanMessagePromptTemplate(
            prompt=PromptTemplate(template=system_prompt, input_variables=input_variables)
        )
        return ChatPromptTemplate.from_messages([system_message_template, human_message_template])

    def _format_network_info(self) -> str:
        """Format network information for the OMAN LLM prompt."""
        network_info = ""
        for network_name, mas in self.networks.items():
            network_info += f"\n[{network_name}]\n"
            network_info += "Agent Prompts:\n"
            for agent_name, prompt in mas.agentManager.agent_prompts.items():
                network_info += f"- {agent_name}: {prompt}\n\n"
        return network_info

    def delegate_task(self, query: str) -> str:
        """Delegate a query to the appropriate network and return the answer directly."""
        # Use OMAN LLM to route the query
        network_info = self._format_network_info()
        routed_network_name = self.oman_llm.generate_response(
            prompt=f"Query: {query}",  # This will be handled by the prompt template
            output_structure=structure_oman_supervisor(self.networks.keys()),
            custom_inputs={'network_info':network_info}
        )

        if routed_network_name['delegate_to_network'] not in self.networks:
            raise ValueError(f"Routed network '{routed_network_name}' not found in OMAN.")

        # Execute the task in the selected network
        selected_network = self.networks[routed_network_name]
        result = selected_network.initiate_decentralized_mas(
            query=query,
            set_entry_agent=selected_network.agentManager.get_agent(list(selected_network.agentManager.agents.keys())[0])  # Use first agent as entry
        )
        self._update_shared_memory(query, result, routed_network_name)

        return result["answer"]

    def _update_shared_memory(self, query: str, result: Dict, network_name: str):
        """Update shared memory with task outcomes."""
        self.shared_memory["tasks"].append({"query": query, "network": network_name, "result": result["answer"]})
        self.shared_memory["outcomes"].append({"network": network_name, "success": result.get("success", True)})
        if len(self.shared_memory["tasks"]) > self.network_memory_order:
            self.shared_memory["tasks"] = self.shared_memory["tasks"][-self.network_memory_order:]
            self.shared_memory["outcomes"] = self.shared_memory["outcomes"][-self.network_memory_order:]
    def list_networks(self) -> List[str]:
        """List all network names."""
        return list(self.networks.keys())