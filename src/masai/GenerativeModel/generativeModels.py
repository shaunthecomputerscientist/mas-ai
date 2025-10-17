from pydantic import BaseModel
from typing import Dict, List, Literal, Type, Tuple, Callable
from typing import Optional
from ..schema import Document
from datetime import datetime
from ..Memory.InMemoryStore import InMemoryDocStore
from ..GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel
from ..Tools.logging_setup.logger import setup_logger
from ..prompts.prompt_templates import SUMMARY_PROMPT
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
            context: Additional context messages
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
            "useful_info": str([self.info.items() if self.info else 'Nothing Provided'][0]),
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
            Kwargs:
                memory_store (InMemoryDocStore) : from masai.Memory.InMemoryStore import InMemoryDocStore and use it.
                k (int, optional) : returns top k elements from memory store matching the query
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            category=category,
            memory_order=memory_order,
            prompt_template=prompt_template,
            info=extra_context if extra_context else {'USEFUL DATA': 'None'}
        )

        self.category = category
        self.logger=setup_logger()
        self.streaming = streaming
        self.streaming_callback = streaming_callback
        self.context_callable=context_callable


        self.long_context = long_context
        if self.long_context:
            self.llm_long_context = GenerativeModel(model_name=self.model_name,category=self.category,temperature=0.5,memory=False)
            self.context_summaries: List= []
        else:
            self.llm_long_context, self.context_summaries = None, None


        self.long_context_order = long_context_order

        self.LTIMStore : InMemoryDocStore= kwargs.get('memory_store')

        if self.LTIMStore and self.long_context:
            if not kwargs['memory_store']:
                raise ValueError('InMemoryDocStore instance must be Provided.')
            else:
                self.LTIMStore = kwargs['memory_store']
        else:
            self.LTIMStore=None
        self.chat_log = chat_log

        # Cache for performance optimization (only for info, not schema)
        self._cached_formatted_info = None  # Will be set on first use
        
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
            summary: str= self.llm_long_context.generate_response(SUMMARY_PROMPT.format(messages=messages))
            if summary is not None:
                
                self.context_summaries.append(Document(page_content=summary))
                # print(self.context_summaries)
            
            # print(len(self.context_summaries),self.long_context_order)
            if len(self.context_summaries) > self.long_context_order:
                if self.LTIMStore:
                    pass_summary = self.context_summaries[:-self.long_context_order]
                    await self._save_in_memory(pass_summary)
                
                # print('reducing context summaries')
                self.context_summaries: list = self.context_summaries[-self.long_context_order:]
                # print('size after reduction', len(self.context_summaries))
                if self.context_summaries is None:
                   self.context_summaries = []
            
            if self.logger:         
                self.logger.debug(f"Summarized For LSTM {time.time()-summary_start}")
            
            return self.context_summaries
            
        except Exception as e:
            
            self.logger.error(f"Error in long context summarization: {e}")
            raise e
    

    async def _update_context_callable(self, query, role):
        if role.lower()=="user":
            if self.context_callable:
                context_result = await self.context_callable(query) if asyncio.iscoroutinefunction(self.context_callable) else self.context_callable(query)
                
                if isinstance(context_result, dict):
                    self.info.update(**context_result)
                else:
                    self.info.update({'USEFUL DATA': context_result})
        elif 'USEFUL DATA' in self.info.keys():
            self.info.pop('USEFUL DATA', None)


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

        Args:
            prompt: The input prompt
            output_structure: Pydantic model for structured output (AnswerFormat with _cached_schema attribute)
            agent_context: Dictionary containing information about other agents
            agent_name: Name of the current agent
            component_context: Additional context messages from previous components
        """
        
        # start=time.time()
        
        # Handle context extension
        
            
        if self.long_context and len(self.chat_history) > self.memory_order:
            self.context_summaries: list = await self._update_long_context(self.chat_history)
            await self._update_chat_history()
        elif len(self.chat_history)> self.memory_order:
            await self._update_chat_history()

        role = await self._update_role(agent_name,kwargs)
        await self._update_context_callable(query=prompt, role=role)
        in_memory_store_data : Optional[List] = await self._handle_in_memory_store_search(k=kwargs.get('k'), prompt=prompt)
        await self._update_component_context(component_context=component_context, role=role, prompt=prompt)                        
        
        

        # # Prepare MAS-specific inputs
        # start_formating_time = time.time()


        # Cache formatted info if it hasn't changed
        if self._cached_formatted_info is None or self._cached_formatted_info != str(list(self.info.items())):
            self._cached_formatted_info = str(list(self.info.items())) if self.info else "None"

        mas_inputs = {
            "useful_info": self._cached_formatted_info,
            "current_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            "question": prompt,
            "long_context": self.context_summaries if not in_memory_store_data else in_memory_store_data.extend(self.context_summaries),
            "history": self.chat_history,
            "schema": getattr(output_structure, '_cached_schema', None) or output_structure.model_json_schema(),  # From output_structure._cached_schema
            "coworking_agents_info": agent_context if agent_context is not None else "NO AGENTS PRESENT"
        }
        
        try:
            # print("\n\n\n\n",self.prompt.format(**mas_inputs),"\n\n")
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
            
            # Update chat history with structured response and last question/prompt
            self.chat_history.append({'role': role, 'content': prompt})
            if isinstance(response, dict) and 'answer' in response and response['answer'] is not None:
                self.chat_history.append({'role': agent_name, 'content': response['answer']})
            elif isinstance(response, dict) and 'reasoning' in response and response['reasoning'] is not None:
                self.chat_history.append({'role': agent_name, 'content': response['reasoning']})
            
            # print("returning response", time.time()-start)
            
            return response
            
        except Exception as e:
            print(f'LLM Response Error Occurred: {e}')
            return str(e)

    def get_category(self) -> str:
        """Returns the category of the llm."""
        return self.category
    
    
    async def _update_chat_history(self):
        if len(self.chat_history) > self.memory_order:
            if self.chat_log:
                await self._save_chat_history(chat_history=self.chat_history[:-self.memory_order//2])
            self.chat_history=self.chat_history[-self.memory_order//2:]

    
    async def _update_component_context(self, component_context, role, prompt):
        # import time
        # start_time = time.time()

        # Extract tool output from current prompt once
        tool_output_in_prompt = self._extract_tool_output_from_prompt(prompt)

        if tool_output_in_prompt:
            # Truncate overlapping tool outputs in component context messages
            if component_context:
                truncated_context = []
                for message in component_context:
                    if not isinstance(message, dict) or 'content' not in message:
                        truncated_context.append(message)
                        continue

                    content = message.get('content', '')
                    if content is None:
                        content = ''

                    truncated_message = message.copy()
                    truncated_message['content'] = self._truncate_overlapping_tool_output(content, tool_output_in_prompt)
                    truncated_context.append(truncated_message)
                component_context = truncated_context

            # Also truncate tool outputs in existing chat_history to avoid duplication
            # This handles both tagged tool outputs AND untagged content that overlaps
            # history_truncate_start = time.time()
            self.chat_history = [
                {'role': msg.get('role', ''), 'content': self._truncate_overlapping_tool_output(msg.get('content', '') or '', tool_output_in_prompt)}
                if isinstance(msg, dict) else msg
                for msg in self.chat_history
            ]

            # print(f"⏱️ CONTEXT FILTERING TIME: Total={time.time()-start_time:.4f}s | Component={len(component_context or [])} msgs | History={len(self.chat_history)} msgs | Time={time.time()-history_truncate_start:.4f}s")

        if component_context:
            self.chat_history.extend(component_context)
        # else:
        #     self.chat_history.append({'role': role, 'content': prompt})







    async def _update_role(self, agent_name, kwargs):
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
                    
    async def _save_in_memory(self, documents: List[Document|str]):
        await self.LTIMStore.add_documents(documents=documents)

    async def _handle_in_memory_store_search(self, k, prompt):
        if self.LTIMStore:
            k = k or 1
            content: list = await self.LTIMStore.asearch(query=prompt, k=k)
            if len(content)>=1:
                return [Document(page_content='\n'.join([data['page_content'] for data in content]))]
            else:
                return None
        else:
            return None
        
    def _extract_tool_output_from_prompt(self, prompt: str) -> str:
        """
        Extract tool output content from prompt's <PREVIOUS TOOL OUTPUT START>...<END> tags.
        Args:
            prompt: The prompt that may contain tool output tags
        Returns:
            Extracted tool output content, or empty string if not found
        """
        import re
        pattern = r'<PREVIOUS TOOL OUTPUT START>\n(.*?)\n<PREVIOUS TOOL OUTPUT END>'
        match = re.search(pattern, prompt, re.DOTALL)
        return match.group(1).strip() if match else ''

    def _check_overlap(self, text1: str, text2: str, min_chunk_words: int = 20) -> bool:
        """
        Check if two texts have significant overlap by finding common word chunks.
        Returns True if at least min_chunk_words consecutive words match.
        """
        if not text1 or not text2:
            return False

        words1 = text1.split()
        words2 = text2.split()

        if len(words1) < min_chunk_words or len(words2) < min_chunk_words:
            # If either text is too short, check if shorter one is substring of longer
            shorter = text1 if len(words1) < len(words2) else text2
            longer = text2 if len(words1) < len(words2) else text1
            return shorter in longer

        # Check if any chunk of min_chunk_words from text1 appears in text2
        for i in range(len(words1) - min_chunk_words + 1):
            chunk = ' '.join(words1[i:i + min_chunk_words])
            if chunk in text2:
                return True

        return False

    def _truncate_overlapping_tool_output(self, content: str, tool_output_reference: str, max_words: int = 30) -> str:
        """
        Truncate tool output in content if it overlaps with reference.
        Handles both tagged tool outputs and untagged content that contains tool output.

        Args:
            content: Message content that may contain tool output
            tool_output_reference: The reference tool output from current prompt
            max_words: Maximum number of words to keep in tool output
        Returns:
            Content with truncated tool output if overlap found, otherwise unchanged
        """
        if not content or not isinstance(content, str) or not tool_output_reference:
            return content or ''

        import re

        # Clean reference (remove "..." for comparison)
        reference_clean = tool_output_reference.rstrip('.')

        # Pattern for tagged tool outputs
        pattern = r'<PREVIOUS TOOL OUTPUT START>\n(.*?)\n<PREVIOUS TOOL OUTPUT END>'

        def truncate_if_overlapping(match):
            tool_output = match.group(1).strip()

            # Check if already truncated
            if tool_output.endswith('...'):
                return match.group(0)

            # Clean tool output
            tool_output_clean = tool_output.rstrip('.')

            # Check overlap in BOTH directions (handles truncated vs full)
            has_overlap = (self._check_overlap(tool_output_clean, reference_clean) or
                          self._check_overlap(reference_clean, tool_output_clean))

            if has_overlap:
                words = tool_output.split()
                if len(words) <= max_words:
                    return match.group(0)  # Already short enough

                # Truncate
                truncated = ' '.join(words[:max_words])
                return f'<PREVIOUS TOOL OUTPUT START>\n{truncated}...\n<PREVIOUS TOOL OUTPUT END>'

            return match.group(0)  # No overlap

        # First, handle tagged tool outputs
        modified_content = re.sub(pattern, truncate_if_overlapping, content, flags=re.DOTALL)

        # Second, check if untagged content contains large chunks of tool output
        # (e.g., LLM directly outputting tool results in its response)
        if modified_content == content and '<PREVIOUS TOOL OUTPUT' not in content:  # No tagged outputs
            # Check if content itself (without tags) overlaps with reference
            content_clean = content.rstrip('.')
            if self._check_overlap(content_clean, reference_clean, min_chunk_words=30):
                # Large overlap found in untagged content - truncate the entire content
                words = content.split()
                if len(words) > max_words:
                    truncated = ' '.join(words[:max_words])
                    return f'{truncated}... [truncated - overlaps with tool output]'

        return modified_content


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

        Args:
            prompt: The input prompt
            output_structure: Pydantic model for structured output (AnswerFormat with _cached_schema attribute)
            agent_context: Dictionary containing information about other agents
            agent_name: Name of the current agent
            component_context: Additional context messages from previous components
            stream_callback: Function to receive streaming chunks for UI updates
        """
        # Handle context extension
        if self.long_context and len(self.chat_history) > self.memory_order:
            self.context_summaries: list = await self._update_long_context(self.chat_history)
            await self._update_chat_history()
        elif len(self.chat_history)> self.memory_order:
            await self._update_chat_history()

        role = await self._update_role(agent_name,kwargs)
        await self._update_context_callable(query=prompt)
        in_memory_store_data : Optional[List] = await self._handle_in_memory_store_search(k=kwargs.get('k'), prompt=prompt)
        await self._update_component_context(component_context=component_context, role=role, prompt=prompt)

        # Prepare MAS-specific inputs
        # Extract cached schema from output_structure (attached at agent creation)

        if self._cached_formatted_info is None or self._cached_formatted_info != str(list(self.info.items())):
            self._cached_formatted_info = str(list(self.info.items())) if self.info else "None"

        mas_inputs = {
            "useful_info": self._cached_formatted_info,
            "current_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            "question": prompt,
            "long_context": self.context_summaries if not in_memory_store_data else in_memory_store_data.extend(self.context_summaries),
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
            else:
                response_dict = final_response
                
            # Update chat history with the complete response or last question/prompt
            self.chat_history.append({'role': role, 'content': prompt})
            if isinstance(response_dict, dict) and 'answer' in response_dict and response_dict['answer'] is not None:
                self.chat_history.append({'role': agent_name, 'content': response_dict['answer']})
            elif isinstance(response_dict, dict) and 'reasoning' in response_dict and response_dict['reasoning'] is not None:
                self.chat_history.append({'role': agent_name, 'content': response_dict['reasoning']})
            
            return response_dict
            
        except Exception as e:
            print(f'LLM Response Error Occurred: {e}')
            return {'error': str(e)}
