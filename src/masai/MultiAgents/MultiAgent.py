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
    def __init__(self,agentManager:AgentManager, supervisor_config: Optional[SupervisorConfig] = None, result_callback=None):
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
            # self.task_queue = Queue()
            # self.executor = ThreadPoolExecutor(max_workers=5)  # Adjust based on needs
            # self.callbacks: Dict[str, Queue] = {}  # Store callback queues for each task
            # self.agentManager._compile_agents(type=self.mode,agent_context={'SUPERVISOR AGENT':self._load_supervisorPrompt()})
            self.agentManager._compile_agents()
            self.task_manager = TaskManager(self.agentManager.agents, self.supervisor, self.supervisorModel, self.initiate_decentralized_mas,result_callback=result_callback, logging=self.agentManager.logging)
            self.task_ids = []

        else:
            self.supervisor=None
            self.supervisorModel=None
            self.mode="decentralized"
            self.agentManager._compile_agents(type=self.mode)
            
        
        
        self.logger = setup_logger()
        
    
    def _update_state(self,message, agent, input, reasoning):
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
    
    
    
    def initiate_sequential_mas(self, query: str, agent_sequence: List[str], memory_order: int = 3):
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
                agent_output = current_agent.initiate_agent(query=agent_prompt,passed_from=passed_from)
                
                # Update state with current agent's output
                self._update_state(
                    message='|'.join(str(output.get('content', output.get('tool_output', ''))) 
                                    for output in agent_output['messages'][-memory_order:]),
                    agent=current_agent.agent_name,
                    input=agent_output['answer'],
                    reasoning=agent_output['reasoning']
                )
                
                # Update query for next agent
                current_query = agent_output['answer']
                
                self.logger.info(f"Agent {agent_name} completed processing")
                
            except KeyError as e:
                raise ValueError(f"Agent '{agent_name}' not found in agent manager: {e}")
            except Exception as e:
                self.logger.error(f"Error during sequential processing at agent {agent_name}: {e}")
                raise
        
        return current_query

    def initiate_decentralized_mas(self, query: str, set_entry_agent: Agent,memory_order:int=3):
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
            agent_output = self.agentManager.agents[self.state['last_agent']].initiate_agent(query=query, passed_from="user")
            self._update_state(
                    message='|'.join(str(output.get('content', output.get('tool_output', ''))) 
                                    for output in agent_output['messages'][-memory_order:]),
                    agent=self.state['last_agent'],
                    input=agent_output['answer'],
                    reasoning=agent_output['reasoning']
                )
        else:
            agent_output = set_entry_agent.initiate_agent(query=query,passed_from="user")
            self._update_state(
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
            agent_output=next_agent.initiate_agent(query=agent_prompt, passed_from=self.state['last_agent'])

            self._update_state(
                    message='|'.join(str(output.get('content', output.get('tool_output', ''))) 
                                    for output in agent_output['messages'][-memory_order:]),
                    agent=next_agent.agent_name,
                    input=agent_output['answer'],
                    reasoning=agent_output['reasoning'])
            
        return agent_output
    
    def initiate_hierarchical_mas(self, query: str, callback=None) -> Dict[str, Any]:
            """Initiate a task and optionally monitor for completion non-blockingly.
            This function is used to initiate a task and monitor for completion non-blockingly.
            It is hierarchical because it uses a supervisor to delegate the task to the appropriate agent.
            Parameters:
                query (str): The query to be processed
                callback (function, optional): A callback function to be called when the task is completed. Takes in task as argument.
            Returns:
                dict: The result of the task. contains key 'answer'.
            """
            if not self.task_manager:
                raise ValueError("TaskManager not configured")
            
            result = self.task_manager.initiate_task(query)
            self.task_ids.append(result["task_id"])

            # If a callback is provided and task is queued, monitor asynchronously
            if callback and result["status"] == "queued":
                threading.Thread(
                    target=self._monitor_task,
                    args=(result["task_id"], callback),
                    daemon=True
                ).start()

            return result

    async def _monitor_task(self, task_id: str, callback: Callable[[Dict], None] | Callable[[Dict], Awaitable[None]]):
        while task_id in self.task_manager.pending_tasks:
            await asyncio.sleep(1)
            completed = await self.task_manager.get_completed_tasks()
            for task in completed:
                if task["task_id"] == task_id:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task)
                    else:
                        callback(task)
                    return


class TaskManager:
    def __init__(self, agents: Dict[str, Any], supervisor: Any, supervisorModel, decentralized_mas_function,
                 result_callback=None, logging=False):
        self.agents = agents
        self.supervisor = supervisor
        self.supervisorModel = supervisorModel
        self.decentralized_mas_function = decentralized_mas_function
        self.task_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.pending_tasks = {}
        self.completed_tasks = Queue()
        self.logger = setup_logger()
        self.check_interval = 2  # Check interval in seconds
        self.result_callback = result_callback  # Callback for completed tasks
        self.logging = logging

        # Start periodic check in a background thread
        self.periodic_check_thread = threading.Thread(target=self._periodic_check, daemon=True)
        self.periodic_check_thread.start()
    def initiate_task(self, query: str) -> Dict[str, Any]:
        """Initiate a task and return immediately with either a result or task ID."""
        supervisor_prompt = (
            f"METADATA"
            f"Current pending tasks: {self.pending_tasks}\n"
            f"Processing: {self.task_queue}\n"
            f"---------------------------------------------"
            f"QUESTION: {query}\n"
        )
        decision = self.supervisor.generate_response_mas(supervisor_prompt, self.supervisorModel, self.agents)
        if str(decision.get("delegate_to_agent")).lower() == "none":
            # Supervisor provided a direct answer; return it immediately
            return {"status": "completed", "answer": decision['answer'], "task_id": f"direct_{time.time()}"}
        else:
            # Delegate to an agent; process in background and return task ID
            task_id = f"task_{len(self.pending_tasks) + 1}"
            self.pending_tasks[task_id] = {
                "agent": decision["delegate_to_agent"],
                "status": "queued",
                "query": query,
                "agent_input": decision["agent_input"],
                "supervisor_reasoning": decision["reasoning"],
                "created_at": time.time()
            }
            self.task_queue.put((
                task_id,
                decision["delegate_to_agent"],
                query,
                decision["agent_input"],
                decision["reasoning"]
            ))
            self._process_queue()
            return {"status": "queued", "task_id": task_id, "answer": decision['answer']}
    def _process_queue(self):
        """Process tasks from the queue using the thread pool."""
        while not self.task_queue.empty():
            task_data = self.task_queue.get()
            task_id = task_data[0]
            future = self.executor.submit(self._execute_task, *task_data)
            future.add_done_callback(lambda f, tid=task_id: self._handle_task_completion(f, tid))

    def _execute_task(self, task_id: str, agent_name: str, query: str, agent_input: str, supervisor_reasoning: str) -> Dict[str, Any]:
        """Execute a task by invoking the appropriate agent."""
        try:
            agent = self.agents[agent_name.lower()]
            agent_output = self.decentralized_mas_function(query=agent_input, set_entry_agent=agent)
            return self._notify_supervisor(
                task_id=task_id,
                agent_name=agent_name,
                agent_output=agent_output,
                original_query=query,
                supervisor_reasoning=supervisor_reasoning
            )
        except Exception as e:
            if self.logging:
                self.logger.error(f"Task {task_id} failed: {str(e)}")
            return {"status": "failed", "error": str(e), "task_id": task_id}

    def _handle_task_completion(self, future, task_id: str):
        """Handle task completion, updating status and storing results."""
        try:
            result = future.result()
            if result["status"] == "requires_revision":
                if self.logging:
                    self.logger.info(f"Re-queueing task {task_id} for revision")
                self.task_queue.put((
                    task_id,
                    result["delegate_to_agent"],
                    result["original_query"],
                    result["new_input"],
                    result["reasoning"]
                ))
                self.pending_tasks[task_id]["status"] = "requeued"
                self._process_queue()
            else:
                self.pending_tasks[task_id].update({
                    "status": "completed",
                    "completed_at": time.time(),
                    "result": result.get("answer")
                })
                self.completed_tasks.put({"task_id": task_id, "answer": result.get("answer")})
        except Exception as e:
            if self.logging:
                self.logger.error(f"Task completion handling failed: {str(e)}")
            self.pending_tasks[task_id]["status"] = "failed"
            self.completed_tasks.put({"task_id": task_id, "result": {"status": "failed", "error": str(e)}})
        finally:
            if self.pending_tasks[task_id]["status"] in ["completed", "failed"]:
                del self.pending_tasks[task_id]

    def _notify_supervisor(self, task_id: str, agent_name: str, agent_output: Dict, original_query: str,
                          supervisor_reasoning: str) -> Dict[str, Any]:
        """Notify the supervisor to review the task result."""
        supervisor_prompt = (
            f"Current pending tasks: {self.pending_tasks}\n"
            f"Task ID: {task_id}\n"
            f"Agent: {agent_name}\n"
            f"Original Query: {original_query}\n"
            f"Agent Steps: {agent_output['messages'][-3:] if agent_output.get('messages') else 'No steps'}\n"
            f"Agent Answer: {agent_output['answer']}\n"
        )
        
        decision = self.supervisor.generate_response_mas(
            prompt=supervisor_prompt,
            output_structure=self.supervisorModel,
            agent_context=self.agents
        )
        print(decision)
        if str(decision.get("delegate_to_agent")).lower() != "none":
            return {
                "status": "requires_revision",
                "delegate_to_agent": decision["delegate_to_agent"],
                "new_input": decision["agent_input"],
                "reasoning": decision["reasoning"],
                "original_query": original_query
            }
        return {
            "status": "completed",
            "answer": decision["answer"],
            "task_id": task_id
        }

    def get_completed_tasks(self) -> list:
        """Retrieve all completed task results."""
        results = []
        while not self.completed_tasks.empty():
            results.append(self.completed_tasks.get())
        return results
    def _cleanup_task(self, task_id: str):
        """Clean up completed task data."""
        try:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            if self.logging:
                self.logger.info(f"Cleaned up task {task_id}")
        except Exception as e:
            if self.logging:
                self.logger.error(f"Error cleaning up task {task_id}: {e}")
            
    def _periodic_check(self):
        """Periodically check for completed tasks, invoke callback, and clean up."""
        while True:
            time.sleep(self.check_interval)
            completed_tasks = self.get_completed_tasks()
            if completed_tasks:
                for task in completed_tasks:
                    # Invoke callback if provided
                    if self.result_callback:
                        self.result_callback(task)
                    # Optionally log or print for debugging
                    if self.logging:
                        self.logger.info(f"Task {task['task_id']} completed with result: {task['answer']}")
                    self._cleanup_task(task['task_id'])