#---------------------------------AgentManager---------------------------------
import os
from mas_ai.AgentManager.AgentManager import AgentManager, AgentDetails
from mas_ai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig
#---------------------------------Tools---------------------------------
from src.mas_ai.Tools.baseTools import human_in_loop_input
from src.mas_ai.Tools.searchTools import search_tool, youtube_transcript
from src.mas_ai.Tools.InputOutputTools import file_handler_IO, files_checker, save_long_term_details
from src.mas_ai.Tools.calendarTools import fetch_calendar_events, manage_calendar_event
from src.mas_ai.Tools.emailTools import email_handler
from src.mas_ai.Tools.visiontools import Vision_Model
from src.mas_ai.Tools.utilities.tokenGenerationTool import token_stream,MarkupProcessor

from dotenv import load_dotenv
load_dotenv()
import threading
from threading import Thread
import time
import concurrent.futures

# User provides path to their model config
model_config_path = os.path.join(os.getcwd(), 'model_config.json')

# Initialize manager with user's config
manager = AgentManager(
    context={"HUMAN NAME": "SHAUN"},
    logging=True,
    model_config_path=model_config_path
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

# Create agents with their respective details
manager.create_agent(
    agent_name="research_agent",
    tools=tools_for_researcher,
    agent_details=research_agent_details,
    plan=False
)

manager.create_agent(
    agent_name="general_personal_agent",
    tools=tools_for_personal,
    agent_details=personal_agent_details,
    plan=False
)

manager.create_agent(
    agent_name="productivity_agent",
    tools=tools_for_productivity,
    agent_details=productivity_agent_details,
    plan=False
    
)

manager.get_agent('research_agent').display()

supervisor_config = SupervisorConfig(
    model_name="gemini-2.0-flash-exp",
    temperature=0.7,  # Standard temperature for balanced creativity/consistency
    model_category="gemini",
    memory_order=20,  # Keep last 20 messages in context
    memory=True,      # Enable memory/context tracking
    extra_context={"user name": "shaun"}  # Additional context for supervisor role
)

# Simplified main execution flow
# Assuming manager and supervisor_config are defined
def handle_task_result(task):
    """Callback function to handle completed task results."""
    print("--------------------------------")
    token_stream(task['answer'], delay=0.05, color='blue', token_type='word')
    print("--------------------------------")

# Initialize MultiAgentSystem with result callback
mas_hierarchical = MultiAgentSystem(
    agentManager=manager,
    isVision=False,
    supervisor_config=supervisor_config,
    result_callback=handle_task_result
)

# Main loop for continuous querying
while True:
    try:
        query = input("Enter a query :\n")
        if query.lower() == 'exit':
            break
        result = mas_hierarchical.initiate_hierarchical_mas(query, callback=handle_task_result)
        
        token_stream(result['answer'] or 'No answer', delay=0.05, color='blue', token_type='word')
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    
# mas_decentralized = MultiAgentSystem(agentManager=manager, isVision=False)

# while True:
#     query=input("Enter your query: ")
#     result = mas_decentralized.initiate_decentralized_mas(
#             query=query,
#             set_entry_agent=manager.get_agent(agent_name="general_personal_agent")
#         )
#     # result = manager.get_agent(agent_name='research_agent').initiate_agent(query)
#     # print(result)
#     token_stream(
#         result['answer'],
#         delay=0.05,
#         color='blue',
#         token_type='word'
#     )

# process_vision_task(model="gemini-2.0-flash-001",query="Search for youtube and play hare hare ya. You can use start/win button to open a search bar or use searchbar")

# print(search_tool.invoke({'query':'todays tech news about Elon','source_categories':['tech']}))