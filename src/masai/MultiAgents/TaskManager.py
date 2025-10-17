from ..prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import json
import os
from ..GenerativeModel.generativeModels import MASGenerativeModel
from ..Agents.singular_agent import Agent
from ..AgentManager.AgentManager import AgentManager
from ..pydanticModels.supervisorModels import structure_supervisor
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty  # Using standard queue for thread communication
from ..Tools.logging_setup.logger import setup_logger
import time
import threading
import asyncio

class TaskManager:
    def __init__(
        self,
        agents: Dict[str, Any],
        agent_context: Dict[str, Any],
        supervisor: Any,
        supervisor_model: Any,
        decentralized_mas_function: Callable[..., Awaitable[Dict[str, Any]]],
        result_callback: Optional[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]] = None,
        logging_enabled: bool = True,
        return_direct: bool = False,
        max_thread_workers: int = 5,
        check_interval: float = 1.0
    ):
        self.agents = {name.lower(): agent for name, agent in agents.items()}
        self.agent_context = agent_context
        self.supervisor = supervisor
        self.supervisor_model = supervisor_model
        self.decentralized_mas_function = decentralized_mas_function
        self.result_callback = result_callback
        self.logging_enabled = logging_enabled
        self.return_direct = return_direct
        self.logger = setup_logger()
        self.completed_tasks = []  # List to store completed tasks
        self.check_interval = check_interval
        
        self._last_n_completed_tasks = 10

        self._executor = ThreadPoolExecutor(max_workers=max_thread_workers)
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
        self._result_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._futures: Dict[str, Future] = {}
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("TaskManager must be initialized in an async context.")
        self._listener_task = asyncio.create_task(self._listen_for_results())
        self._completed_task_checker = asyncio.create_task(self._check_completed_tasks())
        if self.logging_enabled:
            self.logger.info(f"TaskManager initialized with {max_thread_workers} workers.")

    async def initiate_task(self, query: str) -> Dict[str, Any]:
        self.logger.info(f"Initiating task for query: {query}")
        task_id = f"task_{int(time.time() * 1000)}_{len(self.pending_tasks)}"
        # Build context from completed tasks
        completed_context = "\n".join([str(task) for task in self.completed_tasks[-self._last_n_completed_tasks:]]) if len(self.completed_tasks) > 0 else None        
        prompt = (
            f"Completed Tasks Context:\n{completed_context}\n\n"
            f"Pending Tasks: {self.pending_tasks}\n\n"
            f"QUESTION: {query}\n"
            f"Instructions: Set 'delegate_to_agent' to 'None' and provide 'answer' for direct response, "
            f"or specify 'delegate_to_agent' ({list(self.agents.keys())}), 'agent_input', and 'reasoning'."
        )
        if self.logging_enabled:
            self.logger.info(prompt)
        try:
            decision = await self.supervisor.generate_response_mas(
                prompt=prompt,
                output_structure=self.supervisor_model,
                agent_context=self.agent_context,
                passed_from="Supervisor"
            )
            if not isinstance(decision, dict):
                self.logger.error(f"Supervisor returned non-dict for task {task_id}: {decision}")
                return {"status": "failed", "task_id": task_id, "error": "Invalid supervisor response", "original_query": query}
        except Exception as e:
            self.logger.error(f"Supervisor failed for query '{query}': {e}", exc_info=True)
            return {"status": "failed", "task_id": task_id, "error": str(e), "original_query": query}

        delegate_to = str(decision.get("delegate_to_agent", "none")).lower()
        if delegate_to == "none":
            answer = decision.get("answer")
            if not answer:
                self.logger.error("Supervisor provided no answer for direct response.")
                return {"status": "failed", "task_id": task_id, "error": "No answer provided", "original_query": query}
            self.logger.info(f"Direct answer for '{query}': {answer}")
            return {"status": "completed", "task_id": task_id, "answer": answer, "original_query": query}

        agent_name = delegate_to
        agent_input = decision.get("agent_input")
        reasoning = decision.get("reasoning", "N/A")

        if not agent_input or agent_name not in self.agents:
            error = "Missing agent_input" if not agent_input else f"Unknown agent '{agent_name}'"
            self.logger.error(error)
            return {"status": "failed", "task_id": task_id, "error": error, "original_query": query}

        task_info = {
            "task_id": task_id,
            "agent_name": agent_name,
            "query": query,
            "agent_input": agent_input,
            "reasoning": reasoning,
            "status": "queued",
            "created_at": time.time()
        }
        self.pending_tasks[task_id] = task_info

        future = self._executor.submit(
            self._execute_task,
            task_id, agent_name, query, agent_input, reasoning
        )
        self._futures[task_id] = future
        def safe_callback(f):
            if self._main_loop.is_closed():
                self.logger.warning(f"Cannot schedule callback for task {task_id}: Event loop is closed.")
            else:
                asyncio.run_coroutine_threadsafe(self._process_future(f, task_id), self._main_loop)
        future.add_done_callback(safe_callback)

        self.logger.info(f"Task {task_id} queued for '{agent_name}'.")
        return {"status": "queued", "task_id": task_id, "original_query": query}

    def _execute_task(self, task_id: str, agent_name: str, query: str, agent_input: str, reasoning: str) -> Dict[str, Any]:
        self.logger.info(f"Executing task {task_id} for agent {agent_name}")
        try:
            agent = self.agents.get(agent_name.lower())
            if not agent:
                raise ValueError(f"Agent {agent_name} not found")

            # Run decentralized_mas_function in the main event loop
            loop = self._main_loop
            coro = self.decentralized_mas_function(
                query=agent_input,
                set_entry_agent=agent,
                passed_from="TaskManager"
            )
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            agent_output = future.result()

            if not isinstance(agent_output, dict):
                self.logger.error(f"Invalid agent output (non-dict) for task {task_id}: {agent_output}")
                return {"task_id": task_id, "status": "failed", "error": "Invalid agent output type", "original_query": query}
            if "answer" not in agent_output:
                self.logger.error(f"Agent output missing 'answer' for task {task_id}: {agent_output}")
                return {"task_id": task_id, "status": "failed", "error": "Missing answer in agent output", "original_query": query}

            self.logger.info(f"Task {task_id} received agent output: {agent_output['answer']}")
            if self.return_direct:
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "answer": agent_output["answer"],
                    "original_query": query,
                    "agent_name": agent_name
                }

            # Call _review_by_supervisor asynchronously
            review = asyncio.run_coroutine_threadsafe(
                self._review_by_supervisor(task_id, agent_name, agent_output, query, reasoning),
                loop
            ).result()
            return review

        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            return {"task_id": task_id, "status": "failed", "error": str(e), "original_query": query}

    async def _review_by_supervisor(self, task_id: str, agent_name: str, agent_output: Dict, query: str, reasoning: str) -> Dict[str, Any]:
        self.logger.info(f"Reviewing task {task_id} by supervisor")
        prompt = (
            f"REVIEW: Task {task_id}\n"
            f"Agent: {agent_name}\n"
            f"Query: {query}\n"
            f"Reasoning: {reasoning}\n"
            f"Agent Answer: {agent_output['answer']}\n"
            f"Instructions: Set 'delegate_to_agent' to 'None' and provide 'answer' if approved, "
            f"or specify 'delegate_to_agent', 'agent_input', and 'reasoning' for revision."
        )

        try:
            decision = await self.supervisor.generate_response_mas(
                prompt=prompt,
                output_structure=self.supervisor_model,
                agent_context=self.agent_context,
                passed_from="TaskManager_Review"
            )
            if not isinstance(decision, dict):
                self.logger.error(f"Supervisor review returned non-dict for task {task_id}: {decision}")
                return {"task_id": task_id, "status": "failed", "error": "Invalid review response", "original_query": query}

            delegate_to = str(decision.get("delegate_to_agent", "none")).lower()
            if delegate_to == "none":
                answer = decision.get("answer")
                if not answer:
                    return {"task_id": task_id, "status": "failed", "error": "No answer in review", "original_query": query}
                self.logger.info(f"Task {task_id} approved with answer: {answer}")
                return {"task_id": task_id, "status": "completed", "answer": answer, "original_query": query}

            new_agent = delegate_to
            new_input = decision.get("agent_input")
            new_reasoning = decision.get("reasoning", "N/A")
            if not new_input or new_agent not in self.agents:
                error = "Missing agent_input" if not new_input else f"Unknown agent '{new_agent}'"
                self.logger.error(f"Review error for task {task_id}: {error}")
                return {"task_id": task_id, "status": "failed", "error": error, "original_query": query}

            self.logger.info(f"Task {task_id} requires revision by '{new_agent}'")
            return {
                "task_id": task_id,
                "status": "requires_revision",
                "delegate_to_agent": new_agent,
                "agent_input": new_input,
                "reasoning": new_reasoning,
                "original_query": query
            }
        except Exception as e:
            self.logger.error(f"Supervisor review failed for task {task_id}: {e}", exc_info=True)
            return {"task_id": task_id, "status": "failed", "error": f"Review failed: {e}", "original_query": query}

    async def _process_future(self, future: Future, task_id: str):
        try:
            result = future.result()
            self.logger.info(f"Future completed for task {task_id}, queuing result: {result}")
            await self._result_queue.put(result)
        except Exception as e:
            self.logger.error(f"Future processing failed for task {task_id}: {e}", exc_info=True)
            await self._result_queue.put({
                "task_id": task_id,
                "status": "failed",
                "error": f"Thread execution failed: {e}",
                "original_query": self.pending_tasks.get(task_id, {}).get("query", "Unknown")
            })

    async def _check_completed_tasks(self):
        while not self._stop_event.is_set():
            try:
                completed_tasks = {tid: f for tid, f in self._futures.items() if f.done()}
                for task_id, future in completed_tasks.items():
                    if not self._main_loop.is_closed():
                        self.logger.info(f"Detected completed task {task_id} via periodic check")
                        await self._process_future(future, task_id)
                        del self._futures[task_id]
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in completed task checker: {e}", exc_info=True)

    async def _listen_for_results(self):
        while not self._stop_event.is_set() or not self._result_queue.empty():
            try:
                result = await asyncio.wait_for(self._result_queue.get(), timeout=self.check_interval)
                self.logger.info(f"Result dequeued: {result}")
                task_id = result.get("task_id")
                status = result.get("status")
                if task_id is None or status is None:
                    self.logger.error(f"Invalid result missing 'task_id' or 'status': {result}")
                    continue
                if status in ["completed", "failed"]:
                    if task_id in self.pending_tasks:
                        task_info = self.pending_tasks.pop(task_id)
                        task_info.update({"status": status, "result": result.get("answer"), "error": result.get("error")})
                        self.completed_tasks.append(task_info)
                        self.logger.info(f"Invoking callback for task {task_id} with answer: {result.get('answer')}")
                        await self._invoke_callback(result)
                        if task_id in self._futures:
                            del self._futures[task_id]
                        self.logger.info(f"Task {task_id} {status} and moved to completed tasks.")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in result listener: {e}", exc_info=True)

    async def _invoke_callback(self, result: Dict[str, Any]):
        if not self.result_callback:
            self.logger.info(f"No callback defined for result: {result}")
            return
        try:
            self.logger.info(f"Invoking callback for task {result['task_id']}")
            if asyncio.iscoroutinefunction(self.result_callback):
                await self.result_callback(result)
            else:
                await asyncio.to_thread(self.result_callback, result)
            self.logger.info(f"Callback invoked successfully for task {result['task_id']}")
        except Exception as e:
            self.logger.error(f"Callback failed for task {result['task_id']}: {e}", exc_info=True)

    async def stop(self, wait: bool = True):
        self.logger.info("Stopping TaskManager...")
        self._stop_event.set()
        if self._listener_task:
            try:
                await asyncio.wait_for(self._listener_task, timeout=self.check_interval * 2)
            except asyncio.TimeoutError:
                self.logger.warning("Listener task did not complete within timeout; forcing shutdown.")
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except asyncio.CancelledError:
                    pass
        if self._completed_task_checker:
            self._completed_task_checker.cancel()
            try:
                await self._completed_task_checker
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self._executor.shutdown, wait=wait)
        self.logger.info("TaskManager stopped.")

    def get_pending_tasks(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.pending_tasks)