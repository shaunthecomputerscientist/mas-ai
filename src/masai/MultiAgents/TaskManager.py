from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional, Callable
from pydantic import BaseModel, Field
import json
import os
from ..GenerativeModel.generativeModels import MASGenerativeModel
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from ..Tools.logging_setup.logger import setup_logger
import time
import asyncio
from typing import Awaitable
import threading

class TaskManager:
    def __init__(
        self,
        agents: Dict[str, Any],
        supervisor: Any,
        supervisorModel,
        decentralized_mas_function,
        result_callback: Callable = None,
        logging: bool = False,
        return_direct: bool = False
    ):
        self.agents = agents
        self.supervisor = supervisor  # instance of MASGenerativeModel
        self.supervisorModel = supervisorModel
        self.decentralized_mas_function = decentralized_mas_function
        self.task_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.retain_task_history_order=5
        self.pending_tasks = {}
        self.completed_tasks = Queue()
        self.logger = setup_logger()
        self.check_interval = 2  # seconds between checks
        self.result_callback = result_callback  # Callback for completed tasks
        self.logging = logging
        self.return_direct = return_direct
        self.completed_task_history = []


        # Control flag for our monitoring thread
        self._stop_event = threading.Event()

        # Store the running event loop (if any) for scheduling async callbacks from thread
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

        # Start the single background thread for periodic checking
        # Uncomment the next two lines if you want thread-based monitoring:
        self.periodic_thread = threading.Thread(target=self._periodic_check, daemon=True)
        self.periodic_thread.start()

    def initiate_task(self, query: str) -> Dict[str, Any]:
        """Initiate a task and return immediately with either a result or task ID."""
        supervisor_prompt = (
            f"METADATA\n"
            f"Current pending tasks: {self.pending_tasks}\n"
            f"""Completed: {self.completed_task_history[-self.retain_task_history_order:] 
            if len(self.completed_task_history)>self.retain_task_history_order 
            else self.completed_task_history}\n"""
            f"\n---------------------------------------------\n"
            f"QUESTION: {query}\n"
        )
        print(supervisor_prompt)
        decision = self.supervisor.generate_response_mas(
            supervisor_prompt, self.supervisorModel, self.agents, passed_from='Supervisor'
        )
        if str(decision.get("delegate_to_agent")).lower() == "none":
            # Supervisor provided a direct answer; return it immediately
            return {"status": "completed", "answer": decision['answer'], "task_id": f"direct_{time.time()}"}
        else:
            # Delegate to an agent; process in background and return task ID
            task_id = f"task_{len(self.pending_tasks) + 1}"
            key = task_id if task_id not in self.pending_tasks else f"task_{len(self.pending_tasks) + 5}"
            self.pending_tasks[key] = {
                "agent": decision["delegate_to_agent"],
                "status": "queued",
                "query": query,
                "agent_input": decision["agent_input"],
                "supervisor_reasoning": decision["reasoning"],
                "created_at": time.time()
            }
            self.task_queue.put((
                key,
                decision["delegate_to_agent"],
                query,
                decision["agent_input"],
                decision["reasoning"]
            ))
            self._process_queue()
            return {"status": "queued", "task_id": key, "answer": decision['answer']}

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
            agent_output = self.decentralized_mas_function(query=agent_input, set_entry_agent=agent, passed_from="Supervisor")
            if not self.return_direct:
                return self._notify_supervisor(
                    task_id=task_id,
                    agent_name=agent_name,
                    agent_output=agent_output,
                    original_query=query,
                    supervisor_reasoning=supervisor_reasoning
                )
            else:
                return {
                    "status": "completed",
                    "answer": agent_output["answer"],
                    "task_id": task_id
                }
        except Exception as e:
            if self.logging:
                self.logger.error(f"Task {task_id} failed: {str(e)}")
            return {"status": "failed", "error": str(e), "task_id": task_id}

    def _handle_task_completion(self, future, task_id: str):
        """Handle task completion, updating status and storing results."""
        try:
            result = future.result()
            print('checking result')
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
                print('adding to completed tasks')
                self.completed_tasks.put({"task_id": task_id, "answer": result.get("answer")})
                self.completed_task_history.append({"task_id": task_id, "answer": result.get("answer")})
                print(self.completed_tasks.queue)
        except Exception as e:
            if self.logging:
                self.logger.error(f"Task completion handling failed: {str(e)}")
            self.pending_tasks[task_id]["status"] = "failed"
            self.completed_tasks.put({"task_id": task_id, "result": {"status": "failed", "error": str(e)}})
        finally:
            if task_id in self.pending_tasks and self.pending_tasks[task_id]["status"] in ["completed", "failed"]:
                del self.pending_tasks[task_id]

    def _notify_supervisor(self, task_id: str, agent_name: str, agent_output: Dict, original_query: str,
                           supervisor_reasoning: str) -> Dict[str, Any]:
        """Notify the supervisor to review the task result."""
        supervisor_prompt = (
            f"Current pending tasks: {self.pending_tasks}\n"
            f"This Task ID: {task_id}\n"
            f"Agent: {agent_name}\n"
            f"Original Query: {original_query}\n"
            f"Agent Steps: {agent_output['messages'][-3:] if agent_output.get('messages') else 'No steps'}\n"
            f"Agent Answer: {agent_output['answer']}\n"
        )
        decision = self.supervisor.generate_response_mas(
            prompt=supervisor_prompt,
            output_structure=self.supervisorModel,
            agent_context=self.agents,
            passed_from="Supervisor",
        )
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
        """Retrieve all completed task results synchronously."""
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
        """Periodically check for completed tasks, invoke callback, and clean them up.
           This runs in a background thread.
        """
        while not self._stop_event.is_set():
            time.sleep(self.check_interval)
            completed_tasks = self.get_completed_tasks()
            # if completed_tasks and self.logging:
            #     self.logger.info(f"Completed tasks: {completed_tasks}")
            for task in completed_tasks:
                if self.result_callback:
                    if asyncio.iscoroutinefunction(self.result_callback):
                        # Schedule the async callback in the stored event loop
                        if self.loop:
                            asyncio.run_coroutine_threadsafe(self.result_callback(task), self.loop)
                        else:
                            # Fallback: if no loop available, simply call synchronously (not recommended)
                            self.result_callback(task)
                    else:
                        self.result_callback(task)
                if self.logging:
                    self.logger.info(f"cleaned up task with id {task['task_id']} from pending tasks")
                self._cleanup_task(task['task_id'])

    def stop(self):
        """Stop the monitoring thread when needed."""
        self._stop_event.set()

    # ---------------- Async Monitoring Functions ----------------

    # async def _monitor_task(self, task_id: str, callback: Callable[[Dict], None] | Callable[[Dict], Awaitable[None]]):
    #     """Async monitor for a specific task.
    #        Wrap get_completed_tasks() with asyncio.to_thread to avoid blocking.
    #     """
    #     print(f"Starting _monitor_task for {task_id}. Pending tasks: {list(self.pending_tasks.keys())}")
    #     while task_id in self.pending_tasks:
    #         print("inside monitor task loop")
    #         await asyncio.sleep(0.5)
    #         # Run get_completed_tasks() in a thread to avoid blocking the async loop
    #         completed = await asyncio.to_thread(self.get_completed_tasks)
    #         for task in completed:
    #             if task["task_id"] == task_id:
    #                 if asyncio.iscoroutinefunction(callback):
    #                     await callback(task)
    #                 else:
    #                     callback(task)
    #                 return

    # async def _monitor_and_cleanup_tasks(self):
    #     """Global async monitor that periodically checks for all completed tasks,
    #     invokes callbacks, and cleans them up.
    #     """
    #     while True:
    #         await asyncio.sleep(self.task_manager.check_interval)
    #         completed_tasks = await asyncio.to_thread(self.task_manager.get_completed_tasks)
    #         print("Async monitor found completed tasks:", completed_tasks)
    #         for task in completed_tasks:
    #             if self.task_manager.result_callback:
    #                 if asyncio.iscoroutinefunction(self.task_manager.result_callback):
    #                     await self.task_manager.result_callback(task)
    #                 else:
    #                     self.task_manager.result_callback(task)
    #             self.task_manager._cleanup_task(task["task_id"])

class AsyncTaskManager:
    def __init__(
        self,
        agents: Dict[str, Any],
        supervisor: Any,
        supervisorModel,
        decentralized_mas_function,
        result_callback: Callable = None,
        logging: bool = False,
        return_direct: bool = False
    ):
        self.agents = agents
        self.supervisor = supervisor              # instance of MASGenerativeModel
        self.supervisorModel = supervisorModel
        self.decentralized_mas_function = decentralized_mas_function
        self.task_queue = asyncio.Queue()         # Async queue for tasks
        self.pending_tasks = {}                   # Dictionary to track pending tasks
        self.completed_tasks = asyncio.Queue()    # Async queue for completed tasks
        self.logger = setup_logger()
        self.check_interval = 2                   # seconds between checks
        self.result_callback = result_callback    # Callback for completed tasks
        self.logging = logging
        self.return_direct = return_direct

        # Start the global async monitor (this will run forever unless cancelled)
        self._global_monitor_task = asyncio.create_task(self._monitor_and_cleanup_tasks())

    async def initiate_task(self, query: str) -> Dict[str, Any]:
        """Initiate a task and return immediately with either a result or task ID."""
        supervisor_prompt = (
            f"METADATA\n"
            f"Current pending tasks: {self.pending_tasks}\n"
            f"Processing: {self.task_queue}\n"
            f"\n---------------------------------------------\n"
            f"QUESTION: {query}\n"
        )
        decision = self.supervisor.generate_response_mas(
            supervisor_prompt, self.supervisorModel, self.agents, passed_from='Supervisor'
        )
        if str(decision.get("delegate_to_agent")).lower() == "none":
            # Supervisor provided a direct answer; return it immediately.
            return {"status": "completed", "answer": decision['answer'], "task_id": f"direct_{time.time()}"}
        else:
            task_id = f"task_{len(self.pending_tasks) + 1}"
            key = task_id if task_id not in self.pending_tasks else f"task_{len(self.pending_tasks) + 5}"
            self.pending_tasks[key] = {
                "agent": decision["delegate_to_agent"],
                "status": "queued",
                "query": query,
                "agent_input": decision["agent_input"],
                "supervisor_reasoning": decision["reasoning"],
                "created_at": time.time()
            }
            await self.task_queue.put((
                key,
                decision["delegate_to_agent"],
                query,
                decision["agent_input"],
                decision["reasoning"]
            ))
            # Start processing the task queue asynchronously.
            asyncio.create_task(self._process_queue())
            return {"status": "queued", "task_id": key, "answer": decision['answer']}

    async def _process_queue(self):
        """Process tasks from the async queue."""
        while not self.task_queue.empty():
            task_data = await self.task_queue.get()
            task_id = task_data[0]
            result = await self._execute_task(*task_data)
            await self._handle_task_completion(result, task_id)

    async def _execute_task(self, task_id: str, agent_name: str, query: str, agent_input: str, supervisor_reasoning: str) -> Dict[str, Any]:
        """Execute a task by invoking the appropriate agent asynchronously."""
        try:
            agent = self.agents[agent_name.lower()]
            # Assume decentralized_mas_function is awaitable (if not, wrap with asyncio.to_thread)
            agent_output = await asyncio.to_thread(
                self.decentralized_mas_function,
                query=agent_input,
                set_entry_agent=agent,
                passed_from="Supervisor"
            )
            if not self.return_direct:
                return await self._notify_supervisor(task_id, agent_name, agent_output, query, supervisor_reasoning)
            else:
                return {
                    "status": "completed",
                    "answer": agent_output["answer"],
                    "task_id": task_id
                }
        except Exception as e:
            if self.logging:
                self.logger.error(f"Task {task_id} failed: {str(e)}")
            return {"status": "failed", "error": str(e), "task_id": task_id}

    async def _handle_task_completion(self, result: Dict[str, Any], task_id: str):
        """Handle task completion asynchronously."""
        try:
            if result["status"] == "requires_revision":
                if self.logging:
                    self.logger.info(f"Re-queueing task {task_id} for revision")
                await self.task_queue.put((
                    task_id,
                    result["delegate_to_agent"],
                    result["original_query"],
                    result["new_input"],
                    result["reasoning"]
                ))
                self.pending_tasks[task_id]["status"] = "requeued"
                # Optionally, process the queue immediately:
                asyncio.create_task(self._process_queue())
            else:
                self.pending_tasks[task_id].update({
                    "status": "completed",
                    "completed_at": time.time(),
                    "result": result.get("answer")
                })
                await self.completed_tasks.put({"task_id": task_id, "answer": result.get("answer")})
        except Exception as e:
            if self.logging:
                self.logger.error(f"Task completion handling failed: {str(e)}")
            self.pending_tasks[task_id]["status"] = "failed"
            await self.completed_tasks.put({"task_id": task_id, "result": {"status": "failed", "error": str(e)}})
        finally:
            if task_id in self.pending_tasks and self.pending_tasks[task_id]["status"] in ["completed", "failed"]:
                del self.pending_tasks[task_id]

    async def _notify_supervisor(self, task_id: str, agent_name: str, agent_output: Dict, original_query: str, supervisor_reasoning: str) -> Dict[str, Any]:
        """Notify the supervisor to review the task result asynchronously."""
        supervisor_prompt = (
            f"Current pending tasks: {self.pending_tasks}\n"
            f"This Task ID: {task_id}\n"
            f"Agent: {agent_name}\n"
            f"Original Query: {original_query}\n"
            f"Agent Steps: {agent_output['messages'][-3:] if agent_output.get('messages') else 'No steps'}\n"
            f"Agent Answer: {agent_output['answer']}\n"
        )
        decision = self.supervisor.generate_response_mas(
            prompt=supervisor_prompt,
            output_structure=self.supervisorModel,
            agent_context=self.agents,
            passed_from="Supervisor",
        )
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

    async def get_completed_tasks(self) -> list:
        """Retrieve all completed task results asynchronously."""
        results = []
        while not self.completed_tasks.empty():
            results.append(await self.completed_tasks.get())
        return results

    async def _cleanup_task(self, task_id: str):
        """Clean up completed task data."""
        try:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            if self.logging:
                self.logger.info(f"Cleaned up task {task_id}")
        except Exception as e:
            if self.logging:
                self.logger.error(f"Error cleaning up task {task_id}: {e}")

    async def _monitor_and_cleanup_tasks(self):
        """Global async monitor that periodically checks for all completed tasks,
           invokes callbacks, and cleans them up.
        """
        while True:
            await asyncio.sleep(self.check_interval)
            completed_tasks = await self.get_completed_tasks()
            print("Async monitor found completed tasks:", completed_tasks)
            for task in completed_tasks:
                if self.result_callback:
                    if asyncio.iscoroutinefunction(self.result_callback):
                        await self.result_callback(task)
                    else:
                        self.result_callback(task)
                await self._cleanup_task(task["task_id"])
