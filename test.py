# # api.py
# from fastapi import FastAPI, Depends, HTTPException, Header
# from fastapi.security import OAuth2PasswordBearer
# from pydantic import BaseModel
# from typing import Dict, Optional
# import asyncio
# from datetime import datetime, timedelta
# from sqlalchemy.orm import Session
# from jose import JWTError, jwt
# from mas_ai.AgentManager.AgentManager import AgentManager, AgentDetails
# from mas_ai.MultiAgents.MultiAgent import MultiAgentSystem
# from mas_ai.Tools.searchTools import search_tool
# from mas_ai.Tools.baseTools import human_in_loop_input
# from database import get_db, load_user_context

# app = FastAPI(title="MAS AI API")

# # JWT Configuration
# SECRET_KEY = "your-secret-key"  # Use a secure key in production (e.g., from environment variables)
# ALGORITHM = "HS256"
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # For Swagger UI; replace with your auth endpoint

# # User instances store with timestamps
# user_instances: Dict[str, Dict] = {}
# TIMEOUT_MINUTES = 30  # Reset after 30 minutes of inactivity

# # JWT Dependency
# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     credentials_exception = HTTPException(
#         status_code=401,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         user_id: str = payload.get("user_id")
#         name: str = payload.get("name")
#         if user_id is None or name is None:
#             raise credentials_exception
#         return {"user_id": user_id, "name": name}
#     except JWTError:
#         raise credentials_exception

# # Pydantic models
# class QueryRequest(BaseModel):
#     query: str
#     reset: Optional[bool] = False  # Optional flag to reset instance

# class QueryResponse(BaseModel):
#     answer: str
#     user_id: str

# def create_agent_manager(user_context: Dict) -> AgentManager:
#     """Create an AgentManager instance with three pre-built agents and JWT context."""
#     manager = AgentManager(
#         logging=True,
#         context={
#             "STUDENT_NAME": user_context["name"],
#             "USER_ID": user_context["user_id"],
#             "PREFERENCES": {}  # Extend with additional JWT fields if available
#         },
#         model_config_path="model_config.json"
#     )

#     # Define three agents for the decentralized MAS
#     # 1. General Student Query Agent: Answers general student questions
#     general_query_details = AgentDetails(
#         capabilities=["general Q&A", "student support", "communication"],
#         style="friendly and concise answers",
#         description="Handles general student queries, providing quick and helpful responses."
#     )
#     manager.create_agent(
#         agent_name="general_query_agent",
#         tools=[human_in_loop_input, search_tool],  # Tools for general queries
#         agent_details=general_query_details,
#         plan=False
#     )

#     # 2. Subject Matter Agent: Assists with lecture Q&A
#     subject_matter_details = AgentDetails(
#         capabilities=["lecture support", "subject expertise", "explanation"],
#         style="detailed and educational responses",
#         description="Provides in-depth answers to lecture-related questions and subject matter queries."
#     )
#     manager.create_agent(
#         agent_name="subject_matter_agent",
#         tools=[search_tool, human_in_loop_input],  # Tools for academic content
#         agent_details=subject_matter_details,
#         plan=False
#     )

#     # 3. Assignment Helper Agent: Supports assignment-related tasks
#     assignment_helper_details = AgentDetails(
#         capabilities=["assignment help", "problem-solving", "guidance"],
#         style="structured and actionable advice",
#         description="Assists with assignment questions, problem-solving, and task guidance."
#     )
#     manager.create_agent(
#         agent_name="assignment_helper_agent",
#         tools=[human_in_loop_input, search_tool],  # Tools for assignment support
#         agent_details=assignment_helper_details,
#         plan=False
#     )

#     return manager

# def get_or_create_mas_instance(user_id: str, user_context: Dict, db: Session) -> MultiAgentSystem:
#     """Get or create a decentralized MAS instance for the user."""
#     current_time = datetime.now()

#     if user_id in user_instances:
#         instance_data = user_instances[user_id]
#         last_used = instance_data["last_used"]
#         if current_time - last_used < timedelta(minutes=TIMEOUT_MINUTES):
#             instance_data["last_used"] = current_time
#             return instance_data["mas_instance"]
#         else:
#             del user_instances[user_id]  # Timeout exceeded, reset instance

#     # Fallback to DB if JWT context is incomplete
#     db_context = load_user_context(user_id, db)
#     full_context = {**db_context, **user_context}  # JWT takes precedence

#     # Create new instance
#     manager = create_agent_manager(full_context)
#     mas = MultiAgentSystem(agentManager=manager, isVision=False)  # Decentralized mode
#     user_instances[user_id] = {
#         "mas_instance": mas,
#         "last_used": current_time
#     }
#     return mas

# async def cleanup_instances():
#     """Periodically clean up expired instances."""
#     while True:
#         current_time = datetime.now()
#         expired = [
#             user_id for user_id, data in user_instances.items()
#             if current_time - data["last_used"] >= timedelta(minutes=TIMEOUT_MINUTES)
#         ]
#         for user_id in expired:
#             del user_instances[user_id]
#             print(f"Cleaned up instance for user_id: {user_id}")
#         await asyncio.sleep(60)  # Check every minute

# @app.on_event("startup")
# async def startup_event():
#     """Start the cleanup task on server startup."""
#     asyncio.create_task(cleanup_instances())

# @app.post("/query", response_model=QueryResponse)
# async def process_query(
#     request: QueryRequest,
#     user: Dict = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Process a user query using the decentralized MAS workflow."""
#     user_id = user["user_id"]

#     # Reset instance if requested
#     if request.reset and user_id in user_instances:
#         del user_instances[user_id]

#     # Get or create MAS instance for the user
#     mas_instance = get_or_create_mas_instance(user_id, user, db)

#     # Run decentralized workflow with general_query_agent as the entry point
#     try:
#         result = await mas_instance.initiate_decentralized_mas_async(
#             query=request.query,
#             set_entry_agent=mas_instance.agentManager.get_agent("general_query_agent")
#         )
#         answer = result["answer"]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

#     # Update last used timestamp
#     user_instances[user_id]["last_used"] = datetime.now()

#     return QueryResponse(answer=answer, user_id=user_id)

# @app.get("/status")
# async def get_instance_status(user: Dict = Depends(get_current_user)):
#     """Check if an instance exists for the user and its last used time."""
#     user_id = user["user_id"]
#     if user_id in user_instances:
#         last_used = user_instances[user_id]["last_used"]
#         return {"user_id": user_id, "status": "active", "last_used": last_used.isoformat()}
#     return {"user_id": user_id, "status": "inactive"}

# # Example token generation (for testing; replace with your auth system)
# @app.get("/token")
# async def get_token(user_id: str, name: str):
#     """Generate a JWT token for testing."""
#     payload = {"user_id": user_id, "name": name, "exp": datetime.utcnow().timestamp() + 3600}  # 1-hour expiry
#     token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
#     return {"access_token": token, "token_type": "bearer"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)