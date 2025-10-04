from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional, Callable
from pydantic import BaseModel, Field
import json
import os
from ..GenerativeModel.generativeModels import MASGenerativeModel
from ..Agents.singular_agent import Agent
from ..AgentManager.AgentManager import AgentManager
from ..pydanticModels.supervisorModels import structure_supervisor
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from ..Tools.logging_setup.logger import setup_logger
import time
from ..Tools.utilities.tokenGenerationTool import token_stream
from importlib import resources
from ..prompts.prompt_templates import get_supervisor_prompt
import threading
import asyncio
from typing import Awaitable
from .TaskManager import TaskManager
@dataclass
class SupervisorConfig:
    model_name: str
    temperature: float
    model_category: str
    memory_order: int
    memory: bool
    extra_context: dict
    supervisor_system_prompt: Optional[str]
    


class MultiAgentState(TypedDict):
    last_agent_answers: List[str]
    last_agent: Optional[str]
    last_agent_input:Optional[str]
    agent_reasoning: Optional[str]
    pending_tasks: Dict[str, Callable]

class MultiAgentSystem:
    def __init__(self,agentManager:AgentManager, supervisor_config: Optional[SupervisorConfig] = None, heirarchical_mas_result_callback=None, agent_return_direct:bool=False):
        """Initialize a Multi Agent System to grop many agents together in varios ways.

        Args:
            agentManager (AgentManager): AgentManager instance for agents.
            supervisor_config (Optional[SupervisorConfig], optional): If heirarchical mas is used, supervisor config is important to initialize a supervisor. Defaults to None.
            heirarchical_mas_result_callback (_type_, optional): Callback for the answers generated while executing heirarchical mas. Defaults to None.
            agent_return_direct (bool, optional): If False, supervisor evaluates agent response before returning. Defaults to False.
        """
        self.agentManager = agentManager
        self.num_agents = len(self.agentManager.agents)
        
        # Get prompt file path that works in both dev and prod
        self.answermodel = structure_supervisor(self.agentManager.agents)
        self.state = MultiAgentState(last_agent_answers=[], last_agent=None, last_agent_input=None, pending_tasks={})
        if supervisor_config:
            self.supervisor = MASGenerativeModel(
                model_name=supervisor_config.model_name,
                temperature=supervisor_config.temperature,
                category=supervisor_config.model_category,
                prompt_template=self._load_supervisor_promptTemplate(),
                memory_order=supervisor_config.memory_order,
                extra_context=supervisor_config.extra_context
            )
            self.mode = "hierarchical"
            self.supervisorModel=structure_supervisor(self.agentManager.agents)
            self.agentManager._compile_agents()
            self.task_manager = TaskManager(self.agentManager.agents, 
                                            self.agentManager.agent_prompts,
                                            self.supervisor, 
                                            self.supervisorModel, 
                                            self.initiate_decentralized_mas,
                                            result_callback=heirarchical_mas_result_callback,
                                            logging_enabled=self.agentManager.logging, 
                                            return_direct=agent_return_direct)
            self.task_ids = []

        else:
            self.supervisor=None
            self.supervisorModel=None
            self.mode="decentralized"
            self.agentManager._compile_agents(type=self.mode)
            
        
        
        self.logger = setup_logger()
        
    
    async def _update_state(self,message, agent, input, reasoning):
        if len(self.state['last_agent_answers'])>5:
            self.state['last_agent_answers']=self.state['last_agent_answers'][-4:]
        else:
            self.state['last_agent_answers'].append(message)
        self.state['last_agent']=agent
        self.state['last_agent_input']= input
        self.state['agent_reasoning']=reasoning
    
    def _load_supervisorPrompt(self):
        """Load supervisor prompt from module."""
        return get_supervisor_prompt()
    
    def _load_supervisor_promptTemplate(self, supervisor_system_prompt: str=None):
        if supervisor_system_prompt:
            system_prompt = supervisor_system_prompt
        else:
            system_prompt = self._load_supervisorPrompt()
        input_variables = ['question', 'history', 'schema','current_time','useful_info','coworking_agents_info']
        template = """
        INFO:{useful_info} 
        \n\nTIME:{current_time}, 
        \nQUESTION: {question},
        \n\nRESPONSE FORMAT : {schema},
        \n\nAGENTS YOU HAVE:{coworking_agents_info}
        \n\nCHAT HISTORY: {history}
        """
        
        human_message_template = HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=input_variables, template=template)
        )
        
        system_message_template = SystemMessagePromptTemplate(
            prompt=PromptTemplate(template = system_prompt )
        )
        supervisor_chatPrompt_template = ChatPromptTemplate.from_messages([system_message_template, human_message_template])
        return supervisor_chatPrompt_template
    
    
    
    async def initiate_sequential_mas(self, query: str, agent_sequence: List[str], memory_order: int = 3):
        """
        Execute agents in a sequential order, passing output from one agent to the next.
        
        Args:
            query (str): Initial query to process
            agent_sequence (List[str]): List of agent names in the order they should be executed
            memory_order (int): Number of previous messages to keep in memory
        
        Returns:
            str: Final output from the last agent in the sequence
        """
        current_query = query
        
        for i, agent_name in enumerate(agent_sequence):
            try:
                current_agent = self.agentManager.agents[agent_name.lower()]
                
                # Prepare prompt with context from previous agent if not first agent
                if i > 0:
                    agent_prompt = (
                        f"TASK IS PASSED TO YOU FROM {self.state['last_agent']}\n\n"
                        f"<PREVIOUS AGENT REASONING>: {self.state['agent_reasoning']}\n\n"
                        f"<PREVIOUS AGENT OUTPUT>: {self.state['last_agent_answers'][-1] if self.state['last_agent_answers'] else 'No previous output'}\n\n"
                        f"<ORIGINAL QUESTION>: {query}\n\n"
                        f"<YOUR TASK>: Process the previous agent's output and continue the task."
                    )
                    passed_from=self.state['last_agent']
                else:
                    agent_prompt = current_query
                    passed_from="user"
                
                # Execute current agent
                agent_output = await current_agent.initiate_agent(query=agent_prompt,passed_from=passed_from)
                
                # Update state with current agent's output
                await self._update_state(
                    message='|'.join(str(output.get('content', output.get('tool_output', ''))) 
                                    for output in agent_output['messages'][-memory_order:]),
                    agent=current_agent.agent_name,
                    input=agent_output['answer'],
                    reasoning=agent_output['reasoning']
                )
                
                # Update query for next agent
                current_query = agent_output.get('answer',"")
                
                self.logger.info(f"Agent {agent_name} completed processing")
                
            except KeyError as e:
                raise ValueError(f"Agent '{agent_name}' not found in agent manager: {e}")
            except Exception as e:
                self.logger.error(f"Error during sequential processing at agent {agent_name}: {e}")
                raise
        
        return current_query

    async def initiate_decentralized_mas(self, query: str, set_entry_agent: Agent,memory_order:int=3, passed_from:str="user"):
        """
        This function is used to initiate the decentralized multi-agent system.
        It is decentralized because control can be passed to any agent at any time.
        The system stores the state of the last agent that processed the query.
        If another query is passed to the system within the same process, the last agent will become the entry point.
        The system will continue to process the query until relevant agents have processed the query and have returned the answer.

        Args:
            query (str): Query to be processed
            set_entry_agent (Agent): Agent that is responsible for delegating the first task or entry point agent
            memory_order (int, optional): Number of previous messages from each agent to keep in memory to pass as context to next agent. Defaults to 3.

        Raises:
            ValueError: if agent is not found

        Returns:
            dict: Agent output
        """
        if self.state['last_agent'] in self.agentManager.agents.keys():
            agent_output = await self.agentManager.agents[self.state['last_agent']].initiate_agent(query=query, passed_from=passed_from)
            await self._update_state(
                    message='|'.join(str(output.get('content', output.get('tool_output', ''))) 
                                    for output in agent_output['messages'][-memory_order:]),
                    agent=self.state['last_agent'],
                    input=agent_output['answer'],
                    reasoning=agent_output['reasoning']
                )
        else:
            agent_output = await set_entry_agent.initiate_agent(query=query,passed_from=passed_from)
            await self._update_state(
                        message='|'.join(str(output.get('content', output.get('tool_output', ''))) 
                                        for output in agent_output['messages'][-memory_order:]),
                        agent=set_entry_agent.agent_name,
                        input=agent_output['answer'],
                        reasoning=agent_output['reasoning']
                    )
        while agent_output['delegate_to_agent']:
            try:
                next_agent: Agent = self.agentManager.agents[agent_output['delegate_to_agent'].lower()]
            except Exception as e:
                raise ValueError(e)
            agent_prompt = (
                        f"TASK IS DELEGATED TO YOU BY {self.state['last_agent']}\n\n"
                        f"<REASONING OF {self.state['last_agent']} AGENT>: {self.state['agent_reasoning']}\n\n"
                        f"<LAST AGENT ANSWERS>: {self.state['last_agent_answers']}\n\n" #shared memory accross multiple agents
                        f"<ORIGINAL QUESTION>: {query}"
                    )
            agent_output=await next_agent.initiate_agent(query=agent_prompt, passed_from=self.state['last_agent'])

            await self._update_state(
                    message='|'.join(str(output.get('content', output.get('tool_output', ''))) 
                                    for output in agent_output['messages'][-memory_order:]),
                    agent=next_agent.agent_name,
                    input=agent_output['answer'],
                    reasoning=agent_output['reasoning'])
            
        return agent_output
    
    async def initiate_hierarchical_mas(self, query: str) -> Dict[str, Any]:
            """Initiate a task using the async TaskManager."""
            if not self.task_manager:
                raise ValueError("TaskManager not configured for hierarchical MAS")

            # ---- Add 'await' here ----
            result: Dict[str, Any] = await self.task_manager.initiate_task(query)
            # ---- End Change ----

            # Now 'result' should be the dictionary returned by initiate_task
            try:
                if "task_id" in result: # Check if task_id exists (it might be a direct failure)
                    self.task_ids.append(result["task_id"])
                else:
                    # Handle cases where initiate_task might return failure without a standard task_id
                    self.task_manager.logger.warning(f"No 'task_id' found in result from initiate_task: {result}")
            except Exception as e:
                # Log potential issues if result isn't as expected, even after await
                if hasattr(self, 'task_manager') and self.task_manager.logging_enabled:
                    self.task_manager.logger.error(f"Error processing result after awaiting initiate_task: {e}. Result was: {result}", exc_info=True)
                # Decide how to handle this - re-raise, return error state?
                # For now, let's return the potentially problematic result
                pass # Allow returning the original result even if appending task_id failed

            # Optional: Start global monitor if needed (though TaskManager handles callbacks internally now)
            # ...

            return result # Return the actual dictionary result

    async def _monitor_and_cleanup_tasks(self):
        """Global async monitor that periodically checks for all completed tasks,
        invokes callbacks, and cleans them up.
        """
        while True:
            print('inside monitor task')
            await asyncio.sleep(self.task_manager.check_interval)
            # Wrap the synchronous get_completed_tasks in asyncio.to_thread
            completed_tasks = await self.task_manager.get_completed_tasks()
            print("Async monitor found completed tasks:", completed_tasks)
            for task in completed_tasks:
                if self.task_manager.result_callback:
                    if asyncio.iscoroutinefunction(self.task_manager.result_callback):
                        await self.task_manager.result_callback(task)
                    else:
                        self.task_manager.result_callback(task)
                await self.task_manager._cleanup_task(task["task_id"])
