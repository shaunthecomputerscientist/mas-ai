

#SAMPLE SCRIPT SHOWING EXAMPLE OF USING MAS-AI
#---------------------------------AgentManager---------------------------------
import os
import sys
from src.masai.AgentManager.AgentManager import AgentManager, AgentDetails
from src.masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig
#---------------------------------Tools---------------------------------
from src.masai.Tools.tools.baseTools import human_in_loop_input
from src.masai.Tools.tools.searchTools import search_tool, youtube_transcript
from src.masai.Tools.tools.InputOutputTools import file_handler_IO, files_checker, save_long_term_details
from src.masai.Tools.tools.calendarTools import fetch_calendar_events, manage_calendar_event
from src.masai.Tools.tools.emailTools import email_handler
from src.masai.Tools.tools.visiontools import Vision_Model
from src.masai.Tools.utilities.tokenGenerationTool import token_stream,MarkupProcessor
from dotenv import load_dotenv
load_dotenv()
import threading
from threading import Thread
import time
import concurrent.futures
import asyncio



# User provides path to their model config
model_config_path = os.path.join(os.getcwd(), 'model_config.json')

# Define Agent Manager
manager = AgentManager(
    context={"HUMAN NAME": "SHAUN"},
    logging=False,
    model_config_path=model_config_path,
    chat_log='MAS/WORKSPACE/chat_log.json' # ensure it exixts
)

# Define tools
tools_for_researcher = [human_in_loop_input, youtube_transcript, file_handler_IO, files_checker, search_tool, Vision_Model]
tools_for_personal = [file_handler_IO, human_in_loop_input, files_checker, save_long_term_details, Vision_Model]
tools_for_productivity = [file_handler_IO, email_handler, human_in_loop_input, files_checker, fetch_calendar_events, manage_calendar_event]

# Define agent details for each agent
research_agent_details = AgentDetails(
    capabilities=[
        "reasoning",
        "coding",
        "science",
        "research",
        "mathematics",
        "searching capabilities"
    ],
    description="Make best use of the tools by analyzing entire context and history, question and deciding next best step.",
    style="You give very elaborate answers"
)

personal_agent_details = AgentDetails(
    capabilities=[
        "general answering",
        "personalized conversation",
        "complex reasoning",
        "creative tasks",
    ],
    description="""Have access to current human's personal data to give personalized answers.
    You can save interesting facts about user's life, personal details, etc in long term memory.
    Assign appropriate tasks to other agents (if available)""",
    style="acts as a personal assistant focusing on personalized interactions. Do not share details about other agents."
)

productivity_agent_details = AgentDetails(
    capabilities=[
        "email management",
        "calendar management",
        "meeting scheduling",
        "productivity optimization",
        "time management"
    ],
    description="Specializing in productivity work for user like send/read emails, read meetings, set calendar events/schedules",
    style="focuses on efficient task execution and organization"
)
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Create agents with their respective details
manager.create_agent(
    agent_name="research_agent",
    tools=tools_for_researcher,
    agent_details=research_agent_details,
    plan=True
)

manager.create_agent(
    agent_name="general_personal_agent",
    tools=tools_for_personal,
    agent_details=personal_agent_details,
    plan=True
)

manager.create_agent(
    agent_name="productivity_agent",
    tools=tools_for_productivity,
    agent_details=productivity_agent_details,
    plan=True
    
)
#---------------------------------------------------------------------------------------------------------

# Create Supervisor

supervisor_config = SupervisorConfig(
    model_name="gemini-2.0-flash-exp",
    temperature=0.7,  # Standard temperature for balanced creativity/consistency
    model_category="gemini",
    memory_order=20,  # Keep last 20 messages in context
    memory=True,      # Enable memory/context tracking
    extra_context={"user name": "shaun"},  # Additional context for supervisor role
    supervisor_system_prompt=None
)

# Simplified main execution flow

# Assuming manager and supervisor_config are defined
def handle_task_result(task):
    """Callback function to handle completed task results."""
    print("--------------------------------")
    token_stream(task['answer'], delay=0.05, color='blue', token_type='word')
    print("--------------------------------")

# Function to run hierarchical MAS
async def run_hierarchical_mas():
    mas_hierarchical = MultiAgentSystem(
        agentManager=manager,
        supervisor_config=supervisor_config,
        heirarchical_mas_result_callback=handle_task_result,
        agent_return_direct=True
    )

    while True:
        try:
            query = input("Enter a query for hierarchical MAS:\n")
            if query.lower() == 'exit':
                break
            result = await mas_hierarchical.initiate_hierarchical_mas(query)
            token_stream(result['answer'] or 'No answer', delay=0.05, color='blue', token_type='word')
        except KeyboardInterrupt:
            print("\nExiting hierarchical MAS...")
            break

# Function to run decentralized MAS
def run_decentralized_mas():
    mas_decentralized = MultiAgentSystem(agentManager=manager)

    while True:
        try:
            query = input("Enter your query for decentralized MAS: ")
            if query.lower() == 'exit':
                break
            result = mas_decentralized.initiate_decentralized_mas(
                query=query,
                set_entry_agent=manager.get_agent(agent_name="general_personal_agent")
            )
            token_stream(
                result['answer'],
                delay=0.05,
                color='blue',
                token_type='word'
            )
        except KeyboardInterrupt:
            print("\nExiting decentralized MAS...")
            break


def run_sequential_mas():
    agent_sequence = input("Enter agent sequence (comma-separated, e.g., research_agent,general_personal_agent):\n").split(',')
    agent_sequence = [agent.strip() for agent in agent_sequence]  # Clean up whitespace
    
    while True:
        try:
            query = input("Enter your query for sequential MAS: ")
            if query.lower() == 'exit':
                break
            
            # Initialize MultiAgentSystem (without supervisor config)
            mas_sequential = MultiAgentSystem(agentManager=manager)
            
            result = mas_sequential.initiate_sequential_mas(
                query=query,
                agent_sequence=agent_sequence,
                memory_order=3  # You can adjust the memory order
            )
            token_stream(
                result,  # The result is a string in sequential MAS
                delay=0.05,
                color='blue',
                token_type='word'
            )
        except KeyboardInterrupt:
            print("\nExiting sequential MAS...")
            break
        except Exception as e:
            print(f"Error in sequential MAS: {e}")
            break
        
        

if __name__ == "__main__":
    if len(sys.argv) > 1:
        execution_type = sys.argv[1].lower()
        if execution_type == 'hierarchical':
            asyncio.run(run_hierarchical_mas())
        elif execution_type == 'decentralized':
            run_decentralized_mas()
        elif execution_type == 'sequential':
            run_sequential_mas()
        else:
            print("Invalid execution type. Please specify 'hierarchical', 'decentralized', or 'sequential'.")
    else:
        print("Please specify the execution type ('hierarchical', 'decentralized', or 'sequential') as a command-line argument.")