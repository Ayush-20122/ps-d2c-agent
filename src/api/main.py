from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.agent.graph import ask
from src.ingestion.vector_store import setup_database

# ─────────────────────────────────────────
# 1. LIFESPAN — startup and shutdown logic
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs setup_database() once on startup before any requests are handled.
    This ensures the pgvector extension, table, and HNSW index all exist
    before the first request hits the agent.
    Using lifespan is the modern FastAPI pattern — replaces the deprecated
    @app.on_event("startup") decorator.
    """
    print("Starting up Playstation D2C AI Agent...")
    setup_database()
    print("Database Ready. Agent is live!")
    yield
    # Everything after yield runs after shutdown
    print("Shutting Down Playstation D2C AI Agent.")

# ─────────────────────────────────────────
# 2. APP INITIALIZATION
# ─────────────────────────────────────────

app = FastAPI(
    title="Playstation D2C AI Agent",
    description="A RAG-powered conversational agent for PlayStation game discovery and support.",
    version="0.1.0",
    lifespan=lifespan
)

# ─────────────────────────────────────────
# 3. REQUEST AND RESPONSE MODELS
# ─────────────────────────────────────────

class AskRequest(BaseModel):
    """
    Request body schema for the /ask endpoint.
    Pydantic automatically validates incoming JSON against this model.
    """
    question: str = Field(
        ...,                                                # ... means required, no default allowed
        min_length=3,
        max_length=500,
        description="The player's question about PlayStation games or subscriptions.",
        examples=["What are good PS5 exclusives?"]
    )

class AskResponse(BaseModel):
    """
    Response body schema for the /ask endpoint.
    """
    question: str = Field(description="The original question asked by the player.")
    answer: str = Field(description="The AI Agent's grounded answer")

# ─────────────────────────────────────────
# 4. ROUTES
# ─────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Used by load balancers and monitoring tools to verify the service is alive.
    Returns 200 OK if the service is running.
    """
    return {"status":"healthy", "service":"Playstation D2C AI Agent"}

@app.post("/ask", response_model=AskResponse)
def ask_agent(request: AskRequest):
    """
    Main endpoint — accepts a player's question and returns
    a grounded answer from the RAG-powered LangGraph agent.
    """
    try:
        answer = ask(request.question)
        return AskResponse(
            question=request.question,
            answer=answer
        )
    except Exception as e:
        # Return a 500 with the error message
        # In production this would log to an observability platform
        raise HTTPException(
            status_code=500,
            detail=f"Agent Error: {str(e)}"
        )
    
