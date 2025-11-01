from pydantic import BaseModel
from typing import Dict, List, Literal, Type, Tuple, Callable
from typing import Optional
from ..schema import Document
from datetime import datetime
from ..Memory.LongTermMemory import LongTermMemory
from ..GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel
from ..Tools.logging_setup.logger import setup_logger
from ..prompts.prompt_templates import SUMMARY_PROMPT
from ..Tools.utilities.deduplication_utils import deduplicate_and_truncate_chat_history
import asyncio, time



class GenerativeModel(BaseGenerativeModel):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        category: str,
        memory: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            category=category,
            memory=memory,
            **kwargs
        )
        self.category = category

    def generate_response(self, prompt: str, output_structure: Optional[Type[BaseModel]] = None):
        """
        Generates a response, optionally with a structured output.

        Args:
            prompt: The input prompt
            output_structure: Optional Pydantic model for structured output

        Returns:
            str or dict: If output_structure is None, returns response string.
                        If output_structure is provided, returns dict with model_dump().
                        On error, returns error string.

        Raises:
            Logs errors but returns error string instead of raising exception.
        """
        self.chat_history.append({'role': 'user', 'content': prompt})

        if output_structure is None:
            # Unstructured output
            if self.memory:
                messages = self.chat_history[-self.memory_order:] if len(self.chat_history) > self.memory_order else self.chat_history
                response = self.model.invoke(messages).content
            else:
                response = self.model.invoke(prompt).content

            self.chat_history.append({'role': 'assistant', 'content': response})
            return response

        # Structured output
        structured_llm = self.model.with_structured_output(output_structure)

        input_data = {
            "useful_info": self._format_info_for_llm(self.info) if self.info else "None",
            "current_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            "history": self.chat_history[-self.memory_order:] if len(self.chat_history) > self.memory_order else self.chat_history,
            "question": prompt,
            "schema": output_structure.model_json_schema(),
        }

        try:
            response = (structured_llm.invoke(
                self.prompt.format(**input_data)
            )).model_dump()
            self.chat_history.append({'role': 'assistant', 'content': response})
            return response

        except Exception as e:
            print(f'LLM Response Error Occurred: {e}')
            return str(e)

    async def generate_response_async(self, prompt: str, output_structure: Optional[Type[BaseModel]] = None):
        """
        Async version of generate_response for non-blocking background tasks.
        Uses ainvoke() instead of invoke() to avoid blocking the event loop.

        Args:
            prompt: The input prompt
            output_structure: Optional Pydantic model for structured output

        Returns:
            str or dict: If output_structure is None, returns response string.
                        If output_structure is provided, returns dict with model_dump().
                        On error, returns error string.

        Raises:
            Logs errors but returns error string instead of raising exception.
        """
        self.chat_history.append({'role': 'user', 'content': prompt})

        if output_structure is None:
            # Unstructured output
            if self.memory:
                messages = self.chat_history[-self.memory_order:] if len(self.chat_history) > self.memory_order else self.chat_history
                response = (await self.model.ainvoke(messages)).content
            else:
                response = (await self.model.ainvoke(prompt)).content

            self.chat_history.append({'role': 'assistant', 'content': response})
            return response

        # Structured output
        structured_llm = self.model.with_structured_output(output_structure)

        input_data = {
            "useful_info": self._format_info_for_llm(self.info) if self.info else "None",
            "current_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            "history": self.chat_history[-self.memory_order:] if len(self.chat_history) > self.memory_order else self.chat_history,
            "question": prompt,
            "schema": output_structure.model_json_schema(),
        }

        try:
            response = (await structured_llm.ainvoke(
                self.prompt.format(**input_data)
            )).model_dump()
            self.chat_history.append({'role': 'assistant', 'content': response})
            return response

        except Exception as e:
            print(f'LLM Response Error Occurred: {e}')
            return str(e)

class MASGenerativeModel(BaseGenerativeModel):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        category: str,
        prompt_template = None,
        memory_order: int = 5,
        extra_context: Optional[dict] = None,
        long_context: bool = False,
        long_context_order: int = 10,
        chat_log: str = None,
        streaming: bool = False,
        streaming_callback: Optional[Callable] = None,
        context_callable: Optional[Callable]=None,
        **kwargs
    ):
        """
            Initializes an instance of the class with the specified parameters.

            This method sets up the essential attributes for the object, including the model to use, its behavior, memory settings, and optional context features.

            Args:
                model_name (str): The name or identifier of the model to be utilized.
                temperature (float): A value that adjusts the randomness of the model's output. Lower values (e.g., close to 0) result in more predictable output, while higher values enhance creativity.
                category (str): The category of the model.
                prompt_template (Optional[str], optional): A template for prompt. Defaults to None.
                memory_order (int, optional): The number of previous interactions or data points to retain in memory. Defaults to 5.
                extra_context (Optional[dict], optional): Additional context for the model, such as metadata or user-specific details. Defaults to None.
                long_context (bool, optional): Flag to enable long context handling, allowing the model to process extended contextual information through summary. Defaults to True.
                long_context_order (int, optional): The size or order of the long context when enabled. Controls how much historical data is considered. Defaults to 10.
                chat_log (Optional[str], optional): File path to a chat log for loading past chat data as context. Defaults to None.
                context_callable (Optional[Callable]): Callable that uses user input to give more context to the llm.
                streaming (bool, optional): Enable or disable streaming of responses from
                    the model. Defaults to False.
                streaming_callback (Optional[Callable], optional): Async callback for streaming chunks. Defaults to None.
                **kwargs: Additional keyword arguments to be passed to the model.
            Kwargs:
                user_id (str, optional): User identifier for memory isolation
                long_term_memory (LongTermMemory, optional): Shared LongTermMemory instance from AgentManager.
                    If provided and persist_memory=True, uses this instead of creating new instance.
                memory_config (QdrantConfig|RedisConfig, optional): Configuration for persistent long-term memory backend.
                    Only used if long_term_memory is not provided.
                memory_config (QdrantConfig|RedisConfig, optional): [DEPRECATED] Use memory_config instead. Configuration for persistent memory.
                    Only used if long_term_memory is not provided and memory_config is not provided.
                qdrant_config (QdrantConfig, optional): [DEPRECATED] Use memory_config instead. Legacy configuration for persistent memory.
                    Only used if long_term_memory is not provided and memory_config is not provided.
                persist_memory (bool, optional): Enable persistent memory. If False, long_term_memory
                    is not used even if provided. Defaults to False.
                categories_resolver (Callable, optional): Function to extract categories from documents.
                    Only used if creating new LongTermMemory instance.
                k (int, optional): Returns top k elements from long-term memory matching the query.
                Defaults to 5.
                content_capping_limit (int, optional): Maximum number of characters to include from long-term memory. Defaults to 3000.
                callable_config (dict, optional): Configuration for the context_callable. Defaults to None.
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            category=category,
            memory_order=memory_order,
            prompt_template=prompt_template,
            info=extra_context if extra_context else {'USEFUL DATA': 'None'},
            **kwargs
        )

        self.content_capping_limit = kwargs.get('content_capping_limit', 3000) if kwargs else 3000


        self.category = category
        self.logger=setup_logger()
        self.streaming = streaming
        self.streaming_callback = streaming_callback
        self.context_callable=context_callable
        self.callable_config = kwargs.get('callable_config')


        self.long_context = long_context
        if self.long_context:
            self.llm_long_context = GenerativeModel(model_name=self.model_name,category=self.category,temperature=0.5,memory=False)
            self.context_summaries: List= []
            self._context_lock = asyncio.Lock()
        else:
            self.llm_long_context, self.context_summaries = None, None
            self._context_lock = None


        self.long_context_order = long_context_order
        self.chat_log = chat_log

        # Persistent memory configuration
        self.user_id = kwargs.get('user_id')
        self.persist_memory = bool(kwargs.get('persist_memory')) if kwargs.get('persist_memory') is not None else False

        # Use shared LongTermMemory from AgentManager if provided
        # Otherwise, create a new instance if memory_config (or legacy memory_config/qdrant_config) is provided
        shared_long_term_memory = kwargs.get('long_term_memory')
        memory_config = kwargs.get('memory_config')
        self.categories_resolver = kwargs.get('categories_resolver')

        # Initialize LongTermMemory
        self.long_term_memory = None
        if shared_long_term_memory and self.persist_memory:
            # Use shared LongTermMemory from AgentManager
            self.long_term_memory = shared_long_term_memory
            if self.logger:
                self.logger.debug(f"Using shared LongTermMemory for user_id: {self.user_id}")
        elif memory_config and self.persist_memory:
            # Create new LongTermMemory if memory_config provided but no shared instance
            try:
                self.long_term_memory = LongTermMemory(
                    backend_config=memory_config,
                    categories_resolver=self.categories_resolver,
                )
                if self.logger:
                    self.logger.debug(f"Created new LongTermMemory for user_id: {self.user_id}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to initialize LongTermMemory: {e}")
                self.long_term_memory = None

    async def _update_long_context(self, messages: List[Dict[str, str]]) -> List:
        """
        Updates the long-term context by summarizing messages.

        Args:
            messages: List of conversation messages

        Returns:
            Tuple of (updated context summaries, truncated messages)
        """

        try:
            summary_start = time.time()
            summary: str= self.llm_long_context.generate_response(SUMMARY_PROMPT.format(messages=messages[:-1]))
            context_summary_modified=False
            if summary is not None:
                # Extract categories if present at the end in a parseable format
                categories = []
                summary_text = summary
                try:
                    import re, json
                    m = re.search(r"CATEGORIES:\s*(\[.*?\])", summary, flags=re.IGNORECASE | re.DOTALL)
                    if m:
                        list_str = m.group(1)
                        try:
                            categories = json.loads(list_str)
                        except Exception:
                            categories = [c.strip(" '\"") for c in list_str.strip('[]').split(',') if c.strip()]
                        summary_text = summary[: m.start()].rstrip()
                except Exception:
                    pass

                self.context_summaries.append(Document(page_content=summary_text, metadata={"categories": categories} if categories else {}))

            if len(self.context_summaries) > self.long_context_order:
                # Flush overflowed summaries to long-term memory (if configured)
                if self.long_term_memory and self.persist_memory:
                    pass_summary = self.context_summaries[:-1]
                    await self._flush_to_long_term_memory(pass_summary)
                    self.context_summaries=self.context_summaries[-1:]
                    context_summary_modified=True
                if not context_summary_modified:
                    self.context_summaries = self.context_summaries[-self.long_context_order:]

                if self.context_summaries is None:
                   self.context_summaries = []
            self.chat_history = self.chat_history[-1:]


            if self.logger:
                self.logger.debug(f"Summarized For LSTM {time.time()-summary_start}")

            return self.context_summaries

        except Exception as e:

            self.logger.error(f"Error in long context summarization: {e}")
            raise e

    async def _update_long_context_background(self, messages: List[Dict[str, str]]):
        """
        Background task to update long context without blocking.
        Runs in parallel with LLM call.

        Args:
            messages: List of message dictionaries to summarize (all except last)

        Returns:
            None. Updates self.context_summaries and optionally flushes to long-term memory.

        Side Effects:
            - Summarizes messages[:-1] using LLM
            - Parses structured metadata from summary
            - Appends Document to self.context_summaries
            - Flushes to long-term memory if context_summaries exceeds long_context_order
            - Logs debug/error messages

        Raises:
            Logs errors but does not raise exception.
        """
        try:
            summary_start = time.time()
            # Use async version to avoid blocking the event loop
            summary: str = await self.llm_long_context.generate_response_async(SUMMARY_PROMPT.format(messages=messages[:-1]))
            context_summary_modified=False
            if summary is not None:
                # Parse structured metadata from the summary
                metadata = self._parse_summary_metadata(summary)
                summary_text = metadata.pop("summary_text", summary)

                doc = Document(page_content=summary_text, metadata=metadata if metadata else {})
                async with self._context_lock:
                    self.context_summaries.append(doc)

                    if len(self.context_summaries) > self.long_context_order:
                        # Flush overflowed summaries to long-term memory (if configured)
                        if self.long_term_memory and self.persist_memory:
                            pass_summary = self.context_summaries[:-1]

                            await self._flush_to_long_term_memory(pass_summary)
                            self.context_summaries=self.context_summaries[-1:]
                            context_summary_modified=True
                        if not context_summary_modified:
                            self.context_summaries = self.context_summaries[-self.long_context_order:]
                        if self.context_summaries is None:
                            self.context_summaries = []
                    self.chat_history = self.chat_history[-1:]

            if self.logger:
                self.logger.debug(f"Background long-context summary completed in {time.time()-summary_start:.2f}s")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Background long-context summary failed: {e}")

    async def _update_context_callable(self, query, role, node=None):
        """Update context information using the context_callable function.

        Calls the context_callable (if provided) with the user query to fetch
        dynamic context. Updates self.info with the result.

        Args:
            query: The user query string to pass to context_callable
            role: The role of the message sender ('user' or other)

        Returns:
            None. Updates self.info in-place.

        Behavior:
            - If role is 'user' and context_callable is set, calls it with query
            - If result is dict, merges it into self.info
            - If result is non-dict, stores it under 'USEFUL DATA' key
            - If role is not 'user', removes 'USEFUL DATA' from self.info
        """
        if node is not None:
            if self.callable_config:
                callable_func = self.callable_config.get(node)
                if callable_func:
                    context_result = await callable_func(query) if asyncio.iscoroutinefunction(callable_func) else callable_func(query)
        
            elif role.lower()=="user":
                if self.context_callable:
                    if not context_result:
                        context_result = await self.context_callable(query) if asyncio.iscoroutinefunction(self.context_callable) else self.context_callable(query)
            elif 'USEFUL DATA' in self.info.keys():
                self.info.pop('USEFUL DATA', None)

            if isinstance(context_result, dict):
                self.info.update(**context_result)
            else:
                self.info.update({'USEFUL DATA': context_result})
            

    async def generate_response_mas(
        self,
        prompt: str,
        output_structure: Type[BaseModel],
        agent_context: Optional[dict] = None,
        agent_name: Optional[str] = "assistant",
        component_context: list = [],
        **kwargs
    ):
        """
        MAS-specific response generation with agent context support.

        Generates structured responses with support for:
        - Long context (summaries + long-term memory retrieval)
        - Component context deduplication
        - Dynamic context via context_callable
        - Streaming callbacks

        Args:
            prompt: The input prompt
            output_structure: Pydantic model for structured output (AnswerFormat with _cached_schema attribute)
            agent_context: Dictionary containing information about other agents
            agent_name: Name of the current agent
            component_context: Additional context messages from previous components
            **kwargs: Additional arguments (k for long-term memory retrieval, passed_from for role tracking)

        Returns:
            dict: Structured response matching output_structure schema, or error dict on failure

        Raises:
            Logs errors but returns error dict instead of raising exception.
        """

        # start=time.time()

        # Handle context extension
        if self.long_context and len(self.chat_history) > self.memory_order:
            await self._update_long_context_background(self.chat_history)
            # await self._update_chat_history()
        elif len(self.chat_history)> self.memory_order:
            await self._update_chat_history()

        role = await self._update_role(agent_name,kwargs)
        await self._update_context_callable(query=kwargs.get('query', prompt), role=role, node=kwargs.get('node', None))

        await self._update_component_context(component_context=component_context, role=role, prompt=prompt)

        # Build long context docs: context_summaries + long-term memory retrieval (if configured)
        long_context_docs = await self._load_long_context_docs(prompt, **kwargs)

        mas_inputs = {
            "useful_info": self._format_info_for_llm(self.info) if self.info else "None",
            "current_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            "question": prompt,
            "long_context": long_context_docs,
            "history": self.chat_history,
            "schema": getattr(output_structure, '_cached_schema', None) or output_structure.model_json_schema(),  # From output_structure._cached_schema
            "coworking_agents_info": agent_context if agent_context is not None else "NO AGENTS PRESENT"
        }

        try:
            print("\n\n\n\n",self.prompt.format(**mas_inputs),"\n\n")
            # print("END FORMAT TIME", time.time()-start_formating_time)
            # Use the prompt template with MAS-specific inputs
            if self.logger:
                startapi=time.time()
                self.logger.debug(f"LLM Api request time {time.time()-startapi}")

            # Use json_mode for OpenAI and Gemini for better reliability
            structured_model = self._return_structured_model(prompt, output_structure)

            response = (await structured_model.ainvoke(
                self.prompt.format(**mas_inputs)
            )).model_dump()

            if self.logger:


                self.logger.debug(f"LLM Api response time {time.time()-startapi}")
                self.logger.info(f"Returned Response : {response}")



            self.chat_history.append({'role': role, 'content': " ".join(prompt.split()[:self.content_capping_limit])})

            if isinstance(response, dict) and 'answer' in response and response['answer'] is not None:
                self.chat_history.append({'role': agent_name, 'content': " ".join(response['answer'].split()[:self.content_capping_limit])})
            elif isinstance(response, dict) and 'reasoning' in response and response['reasoning'] is not None:
                self.chat_history.append({'role': agent_name, 'content': response['reasoning']})

            return response

        except Exception as e:
            print(f'LLM Response Error Occurred: {e}')
            return str(e)

    def get_category(self) -> str:
        """Returns the category of the llm."""
        return self.category


    async def _update_chat_history(self):
        """Trim chat history to memory_order when it exceeds the limit.

        When chat_history exceeds memory_order, saves older messages to chat_log
        (if configured) and keeps only the most recent messages.

        When long_context is enabled, keeps fewer messages (memory_order//2) since
        older messages are already summarized into context_summaries.

        Returns:
            None. Updates self.chat_history in-place.

        Side Effects:
            - Saves messages beyond memory_order to chat_log file (if configured)
            - Truncates chat_history to memory_order (or memory_order//2 if long_context) most recent messages
        """
        if len(self.chat_history) > self.memory_order:
            if self.chat_log:
                await self._save_chat_history(chat_history=self.chat_history[:-self.memory_order])
            # When long_context is enabled, keep fewer messages since older ones are summarized
            keep_count = self.memory_order // 2 if self.long_context else self.memory_order
            self.chat_history=self.chat_history[-keep_count:]


    async def _update_component_context(self, component_context, role=None, prompt=None):
        """
        CENTRALIZED DEDUPLICATION & TRUNCATION

        Uses deduplicate_and_truncate_chat_history() to:
        1. Extend chat_history with component_context
        2. Extract tool output from current prompt
        3. Deduplicate messages (exact + similarity-based)
        4. Truncate similar tool outputs to reduce token usage
        5. Prevent accumulation of duplicate data

        This ensures:
        - Same tool output doesn't appear multiple times
        - Similar content with minor variations is caught
        - Chat history stays clean and token-efficient
        """
        # Step 1: Use centralized deduplication function
        self.chat_history = deduplicate_and_truncate_chat_history(
            chat_history=self.chat_history,
            component_context=component_context,
            current_prompt=prompt,
            similarity_threshold=0.75  # 75% similarity = duplicate
        )


    async def _load_long_context_docs(self, prompt_str, **kwargs):
        """Load long context docs from context_summaries and long-term memory.

        Args:
            prompt_str: Full formatted prompt string containing <QUESTION>:...
            **kwargs: Additional arguments (k for retrieval count)

        Returns:
            List of document strings (page_content)
        """
        # Extract the actual user question from the formatted prompt
        import re
        user_question = prompt_str

        # Try to extract from <QUESTION>:...pattern
        # First try to extract ORIGINAL QUESTION from context block
        if kwargs.get('query'):
            user_question = kwargs['query']

        # Load from context_summaries (extract page_content strings)
        docs = [context.page_content for context in self.context_summaries if len(self.context_summaries) > 0]
        # print("CONTEXT SUMMARIES",docs)
        # print("long term memory obj",self.long_term_memory, self.persist_memory)
        # print(f"DEBUG: context_summaries_count={len(self.context_summaries)}, long_context_order={self.long_context_order}, memory_order={self.memory_order}, chat_history_len={len(self.chat_history)}")

        # Load from long-term memory ONLY if context summaries have overflowed
        # This prevents unnecessary embedding API calls on every query
        # Embeddings should only be computed when actually retrieving from persistent storage
        if self.long_term_memory and self.persist_memory:
            long_term_memories = await self._retrieve_from_long_term_memory(user_question, k=kwargs.get('k', 5))
            if long_term_memories:
                # print("LONG_TERM_MEMORIES",long_term_memories)
                # Extract page_content from Document objects to maintain consistent return type
                docs.extend([doc.page_content for doc in long_term_memories])

        return docs




    async def _update_role(self, agent_name, kwargs):
        """Determine the role for the current message based on kwargs.

        Checks if 'passed_from' is in kwargs to determine the message role.
        This is used for tracking which agent/component originated the message.

        Args:
            agent_name: The default agent name to use as role
            kwargs: Dictionary that may contain 'passed_from' key

        Returns:
            str: The role to use for the message. Returns 'passed_from' value if
                 present and not None, otherwise returns agent_name.
        """
        if 'passed_from' in kwargs:
            if kwargs['passed_from'] is not None:
                return kwargs['passed_from']
            else:
                return agent_name
        else:
            return agent_name


    async def _save_chat_history(self, chat_history):
            """Saves the chat history to the chat_log file, if provided."""
            if self.chat_log:
                try:
                    import json
                    import aiofiles
                    file_extension = self.chat_log.split('.')[-1].lower()

                    if file_extension == 'json':
                        try:
                            async with aiofiles.open(self.chat_log, 'r') as f:
                                content = await f.read()
                                try:
                                    existing_data = json.loads(content)
                                except json.JSONDecodeError:
                                    existing_data = []

                            if isinstance(existing_data, list):
                                combined_data = existing_data + chat_history
                            else:
                                combined_data = [existing_data] + chat_history

                            async with aiofiles.open(self.chat_log, 'w') as f:
                                await f.write(json.dumps(combined_data, indent=4))

                        except FileNotFoundError:
                            async with aiofiles.open(self.chat_log, 'w') as f:
                                await f.write(json.dumps(chat_history, indent=4))
                    else:
                        async with aiofiles.open(self.chat_log, 'a') as f:
                            await f.write(str(chat_history) + '\n')

                except Exception as e:
                    print(f"Error saving chat history: {e}")

    async def _flush_to_long_term_memory(self, documents: List[Document]) -> None:
        """Flush overflowed context summaries to persistent long-term memory storage.

        Called when context_summaries exceeds long_context_order. Saves documents
        to long-term memory for persistent storage across sessions.

        Args:
            documents: List of Document objects to save to long-term memory

        Returns:
            None. Saves documents asynchronously to long-term memory.

        Raises:
            Logs error if save fails, but does not raise exception.

        Preconditions:
            - self.long_term_memory must be initialized
            - self.user_id must be set for user isolation
            - self.persist_memory must be True
        """
        if self.long_term_memory and self.user_id:
            try:
                await self.long_term_memory.save(user_id=self.user_id, documents=documents)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to flush to long-term memory: {e}")

    def _parse_summary_metadata(self, summary: str) -> dict:
        """Parse structured metadata from LLM summary response.

        Extracts: summary_text, entities, relationships, preferences, facts, actions,
        keywords, memory_type, confidence, importance, categories.

        Returns dict with all extracted fields, with summary_text as the main content.
        """
        import re, json

        metadata = {
            "entities": [],
            "relationships": [],
            "preferences": [],
            "facts": [],
            "actions": [],
            "keywords": [],
            "memory_type": "other",
            "confidence": 0.5,
            "importance": 0.5,
            "categories": [],
        }

        try:
            # Extract summary text (everything before first structured field)
            summary_text = summary
            first_field_match = re.search(r"^(ENTITIES|RELATIONSHIPS|PREFERENCES|FACTS|ACTIONS|KEYWORDS|MEMORY_TYPE|CONFIDENCE|IMPORTANCE|CATEGORIES):", summary, re.MULTILINE)
            if first_field_match:
                summary_text = summary[:first_field_match.start()].strip()

            metadata["summary_text"] = summary_text

            # Extract ENTITIES
            m = re.search(r"^ENTITIES:\s*(.+?)(?=^[A-Z_]+:|$)", summary, re.MULTILINE | re.DOTALL)
            if m:
                entities_str = m.group(1).strip()
                metadata["entities"] = [e.strip() for e in entities_str.split(',') if e.strip()]

            # Extract RELATIONSHIPS
            m = re.search(r"^RELATIONSHIPS:\s*(.+?)(?=^[A-Z_]+:|$)", summary, re.MULTILINE | re.DOTALL)
            if m:
                rels_str = m.group(1).strip()
                metadata["relationships"] = [r.strip() for r in rels_str.split('\n') if r.strip()]

            # Extract PREFERENCES
            m = re.search(r"^PREFERENCES:\s*(.+?)(?=^[A-Z_]+:|$)", summary, re.MULTILINE | re.DOTALL)
            if m:
                prefs_str = m.group(1).strip()
                metadata["preferences"] = [p.strip() for p in prefs_str.split('\n') if p.strip()]

            # Extract FACTS
            m = re.search(r"^FACTS:\s*(.+?)(?=^[A-Z_]+:|$)", summary, re.MULTILINE | re.DOTALL)
            if m:
                facts_str = m.group(1).strip()
                metadata["facts"] = [f.strip() for f in facts_str.split('\n') if f.strip()]

            # Extract ACTIONS
            m = re.search(r"^ACTIONS:\s*(.+?)(?=^[A-Z_]+:|$)", summary, re.MULTILINE | re.DOTALL)
            if m:
                actions_str = m.group(1).strip()
                metadata["actions"] = [a.strip() for a in actions_str.split('\n') if a.strip()]

            # Extract KEYWORDS
            m = re.search(r"^KEYWORDS:\s*(.+?)(?=^[A-Z_]+:|$)", summary, re.MULTILINE | re.DOTALL)
            if m:
                keywords_str = m.group(1).strip()
                metadata["keywords"] = [k.strip() for k in keywords_str.split(',') if k.strip()]

            # Extract MEMORY_TYPE
            m = re.search(r"^MEMORY_TYPE:\s*(.+?)(?=^[A-Z_]+:|$)", summary, re.MULTILINE | re.DOTALL)
            if m:
                mem_type = m.group(1).strip().lower()
                valid_types = ["preference", "fact", "action", "recommendation", "question", "other"]
                metadata["memory_type"] = mem_type if mem_type in valid_types else "other"

            # Extract CONFIDENCE (0.0-1.0)
            m = re.search(r"^CONFIDENCE:\s*([\d.]+)", summary, re.MULTILINE)
            if m:
                try:
                    conf = float(m.group(1))
                    metadata["confidence"] = max(0.0, min(1.0, conf))
                except ValueError:
                    pass

            # Extract IMPORTANCE (0.0-1.0)
            m = re.search(r"^IMPORTANCE:\s*([\d.]+)", summary, re.MULTILINE)
            if m:
                try:
                    imp = float(m.group(1))
                    metadata["importance"] = max(0.0, min(1.0, imp))
                except ValueError:
                    pass

            # Extract CATEGORIES
            m = re.search(r"^CATEGORIES:\s*(\[.*?\])", summary, re.MULTILINE | re.DOTALL)
            if m:
                list_str = m.group(1)
                try:
                    categories = json.loads(list_str)
                    if isinstance(categories, list):
                        metadata["categories"] = [str(c).strip() for c in categories if c]
                except Exception:
                    # Fallback: parse manually
                    categories = [c.strip(" '\"") for c in list_str.strip('[]').split(',') if c.strip()]
                    metadata["categories"] = categories

        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error parsing summary metadata: {e}")

        return metadata

    async def _retrieve_from_long_term_memory(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant memories from long-term memory.

        Optionally filters by categories from the latest context summary.
        """
        if not self.long_term_memory or not self.user_id:
            return []

        try:
            # Optionally filter by categories from latest summary
            categories = None
            # Protect context_summaries read with lock if long_context is enabled
            if self.long_context and self._context_lock:
                async with self._context_lock:
                    if self.context_summaries:
                        categories = self.context_summaries[-1].metadata.get("categories")

            return await self.long_term_memory.search(
                user_id=self.user_id,
                query=query,
                k=k,
                categories=categories,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to retrieve from long-term memory: {e}")
            return []





    async def astream_response_mas(
        self,
        prompt: str,
        output_structure: Type[BaseModel],
        agent_context: Optional[dict] = None,
        agent_name: Optional[str] = "assistant",
        component_context: list = [],
        **kwargs
    ):
        """
        MAS-specific response generation with streaming support.

        Generates structured responses with streaming capability. Streams chunks
        via streaming_callback if configured.

        Args:
            prompt: The input prompt
            output_structure: Pydantic model for structured output (AnswerFormat with _cached_schema attribute)
            agent_context: Dictionary containing information about other agents
            agent_name: Name of the current agent
            component_context: Additional context messages from previous components
            **kwargs: Additional arguments (k for long-term memory retrieval, passed_from for role tracking)

        Returns:
            AsyncGenerator: Async generator yielding streaming chunks, or final dict on completion

        Raises:
            Logs errors but yields error dict instead of raising exception.

        Side Effects:
            - Calls streaming_callback with each chunk if configured
            - Updates chat_history with user and assistant messages
        """
        # Handle context extension
        if self.long_context and len(self.chat_history) > self.memory_order:
            await self._update_long_context_background(self.chat_history)
            # await self._update_chat_history()
        elif len(self.chat_history)> self.memory_order:
            await self._update_chat_history()

        role = await self._update_role(agent_name,kwargs)
        await self._update_context_callable(query=kwargs.get('query', prompt), role=role, node=kwargs.get('node', None))

        await self._update_component_context(component_context=component_context, role=role, prompt=prompt)

        # Build long context docs: context_summaries + long-term memory retrieval (if configured)
        long_context_docs = await self._load_long_context_docs(prompt, **kwargs)

        mas_inputs = {
            "useful_info": self._format_info_for_llm(self.info) if self.info else "None",
            "current_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            "question": prompt,
            "long_context": long_context_docs,
            "history": self.chat_history,
            "schema": getattr(output_structure, '_cached_schema', None) or output_structure.model_json_schema(),  # From output_structure._cached_schema
            "coworking_agents_info": agent_context if agent_context is not None else "No agents present"
        }

        try:
            # Create structured output model for streaming
            # Use json_mode for OpenAI and Gemini for better reliability
            structured_llm= self._return_structured_model(prompt=prompt,output_structure=output_structure)

            # Final response to be returned
            final_response = None

            # Stream the response chunks
            async for chunk in structured_llm.astream(
                self.prompt.format(**mas_inputs)
            ):
                # Send chunk to callback if provided
                if self.streaming_callback:
                    # print('streaming tokens...................', chunk)
                    if hasattr(chunk, 'model_dump'):
                        await self.streaming_callback(chunk.model_dump())
                    else:
                        await self.streaming_callback(chunk)

                # Update final response
                final_response = chunk

            # Convert final response to dict if it's a model
            if hasattr(final_response, 'model_dump'):
                response_dict = final_response.model_dump()
            elif hasattr(final_response, 'content'):
                # AIMessage object - streaming failed to produce Pydantic model
                raise ValueError(f"Streaming failed to produce structured output. Got AIMessage instead: {final_response.content[:200]}")
            else:
                response_dict = final_response

            # Update chat history with the complete response or last question/prompt
            # Apply word capping to prevent token bloat
            self.chat_history.append({'role': role, 'content': " ".join(prompt.split()[:self.content_capping_limit])})
            if isinstance(response_dict, dict) and 'answer' in response_dict and response_dict['answer'] is not None:
                self.chat_history.append({'role': agent_name, 'content': " ".join(response_dict['answer'].split()[:self.content_capping_limit])})
            elif isinstance(response_dict, dict) and 'reasoning' in response_dict and response_dict['reasoning'] is not None:
                self.chat_history.append({'role': agent_name, 'content': response_dict['reasoning']})

            return response_dict

        except Exception as e:
            print(f'LLM Response Error Occurred: {e}')
            return {'error': str(e)}
