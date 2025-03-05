import os
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.anthropic import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from pydantic import BaseModel
from typing import Dict, List, Literal, Type, Tuple
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain.schema import Document
from datetime import datetime, timezone
from ..GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel
from ..prompts.prompt_templates import SUMMARY_PROMPT

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
            response = structured_llm.invoke(
                self.prompt.format(**input_data)
            ).model_dump()
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
        long_context: bool = True,
        long_context_order: int = 10,
        chat_log: str = None
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            category=category,
            memory_order=memory_order,
            prompt_template=prompt_template,
            info=extra_context
        )
        """
        G
        """
        self.category = category
        
        self.long_context = long_context
        if self.long_context:
            self.llm_long_context = GenerativeModel(model_name=self.model_name,category=self.category,temperature=0.5,memory=False)
            self.context_summaries = []
            self.long_context_order = long_context_order
            self.chat_log = chat_log
        
    def _update_long_context(self, messages: List[Dict[str, str]]) -> Tuple[List[Document], List[Dict[str, str]]]:
        """
        Updates the long-term context by summarizing messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Tuple of (updated context summaries, truncated messages)
        """
        
        try:
            summary = self.llm_long_context.generate_response(SUMMARY_PROMPT.format(messages=messages))
            # print("Summary Added")
            self.context_summaries.append(Document(page_content=summary))
            
            if len(self.context_summaries) > self.long_context_order:
                self.context_summaries = self.context_summaries[-self.long_context_order:]
            
            if self.chat_log:
                self._save_chat_history(chat_history=messages[:-self.memory_order//2])
            truncated_messages = messages[-self.memory_order//2:]
            return self.context_summaries, truncated_messages
            
        except Exception as e:
            self.logger.error(f"Error in long context summarization: {e}")
            return self.context_summaries, messages

    def generate_response_mas(
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
            output_structure: Pydantic model for structured output (AnswerFormat)
            agent_context: Dictionary containing information about other agents
            agent_name: Name of the current agent
            component_context: Additional context messages from previous components
        """
        # Handle context extension
        if self.long_context and len(self.chat_history) > self.memory_order:
            self.context_summaries, truncated_messages = self._update_long_context(self.chat_history)
            self.chat_history=truncated_messages

        role = self._update_role(agent_name,kwargs)
        
        self._update_component_context(component_context=component_context, role=role, prompt=prompt)
        
        if len(self.chat_history)>self.memory_order:
            self._save_chat_history(self.chat_history[:-self.memory_order])
            self.chat_history=self.chat_history[-self.memory_order:]
                
        # print(self.chat_history)

        # Prepare MAS-specific inputs
        mas_inputs = {
            "useful_info": str([self.info.items() if self.info else 'None']),
            "current_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            "question": prompt,
            "long_context": self.context_summaries,
            "history": self.chat_history,
            "schema": output_structure.model_json_schema(),
            "coworking_agents_info": agent_context if agent_context is not None else "No agents present"
        }

        try:
            # print("\n\n\n\n",self.prompt.format(**mas_inputs),"\n\n")
            # Use the prompt template with MAS-specific inputs
            response = self.model.with_structured_output(output_structure).invoke(
                self.prompt.format(**mas_inputs)
            ).model_dump()
            
            # Update chat history with structured response
            if isinstance(response, dict) and 'answer' in response:
                self.chat_history.append({'role': role, 'content': response['answer']})
            
            return response
            
        except Exception as e:
            print(f'LLM Response Error Occurred: {e}')
            return str(e)

    def get_category(self) -> str:
        """Returns the category of the llm."""
        return self.category
    
    def _update_component_context(self, component_context, role, prompt):
        if component_context:
            self.chat_history.extend(component_context)
            self.chat_history.append({'role': role, 'content': prompt})
        else:
            self.chat_history.append({'role': role, 'content': prompt})
    def _update_role(self, agent_name, kwargs):
        if 'passed_from' in kwargs:
            if kwargs['passed_from'] is not None:
                return kwargs['passed_from']
            else:
                return agent_name
        else:
            return agent_name        
        
    
    def _save_chat_history(self, chat_history):
            """Saves the chat history to the chat_log file, if provided."""
            if self.chat_log:
                try:
                    import json
                    file_extension = self.chat_log.split('.')[-1].lower()
                    
                    if file_extension == 'json':
                        try:
                            with open(self.chat_log, 'r+') as f:
                                try:
                                    existing_data = json.load(f)
                                except json.JSONDecodeError:
                                    existing_data = []  # Handle empty or invalid JSON file
                                
                                if isinstance(existing_data, list):
                                    combined_data = existing_data + chat_history
                                else:
                                    combined_data = [existing_data] + chat_history  # Handle case where existing data is not a list
                                
                                f.seek(0)  # Rewind to the beginning of the file
                                json.dump(combined_data, f, indent=4)
                                f.truncate()  # Remove any remaining old data
                        except FileNotFoundError:
                            # File doesn't exist, create it and save the chat history
                            with open(self.chat_log, 'w') as f:
                                json.dump(chat_history, f, indent=4)
                    else:
                        with open(self.chat_log, 'a') as f:
                            f.write(str(chat_history) + '\n')  # Append as string with newline
                    
                    print(f"Chat history saved to {self.chat_log}")
                except Exception as e:
                    print(f"Error saving chat history: {e}")