from langsmith import traceable
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from src.retrieval.retriever import retrieve, format_context
from src.config import ANTHROPIC_API_KEY

# ─────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared state object that flows through every node in the graph.
    Each node reads from and writes to this state.

    Fields:
        question:   The user's original natural language question
        context:    Retrieved chunks from pgvector formatted as a string
        answer:     Claude's final generated answer
    """
    question: str
    context:  str
    answer:   str

# ─────────────────────────────────────────
# 2. LLM INITIALIZATION
# ─────────────────────────────────────────

# Initialize Claude claude-sonnet-4-6 as our LLM
# temperature=0 means deterministic outputs — no randomness
# ideal for factual Q&A where we want consistent answers
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=ANTHROPIC_API_KEY,
    temperature=0,
    max_tokens=1024
)


# ─────────────────────────────────────────
# 3. NODE DEFINITIONS
# ─────────────────────────────────────────

@traceable(name="retrieve_node")
def retrieve_node(state: AgentState) -> AgentState:
    """
    Node 1: Retrieval
    Takes the user's question from state, embeds it,
    searches pgvector, and formats the results as context.
    Updates state with the retrieved context string.
    """
    print(f"\n[Agent] Retrieving Context for: {state['question']}")

    results = retrieve(
        query=state['question'],
        top_k=5,
        source=None,
        chunk_type=None
    )

    context = format_context(results)
    print(f"[Agent] Retrieved {len(results)} chunks.")

    # Return updated state — LangGraph merges this with existing state
    return {"context": context}

@traceable(name="generate_node")
def generate_node(state: AgentState) -> AgentState:
    """
    Node 2: Generation
    Takes the question and retrieved context from state, builds a prompt, and sends it to Claude for answer generation.
    Updates state with Claude's answer.
    """
    print("[Agent] Generating Answer with Claude...")

    system_prompt = """You are a knowledgeable PlayStation gaming assistant for Sony Interactive Entertainment's Direct to Consumer platform.
                    Your job is to help players discover games, understand PlayStation subscriptions, and answer gaming questions accurately.
                    You answer questions based ONLY on the context provided below. If the context doesn't contain enough information to answer confidently, say so honestly rather than making things up.
                    Always be specific — mention game names, ratings, genres, and developers when available in the context.
                    Keep answers concise, helpful, and enthusiastic about PlayStation gaming."""
    
    user_prompt = f"""Context from Playstation game database:
    {state['context']}

    ---

    Player's question: {state['question']}

    Please answer the player's question based on the context above."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    # response.content contains Claude's text response
    return {"answer": response.content}

# ─────────────────────────────────────────
# 4. GRAPH CONSTRUCTION
# ─────────────────────────────────────────

def build_graph():
    """
    Constructs and compiles the LangGraph agent.
    Returns a compiled graph ready to invoke.
    """
    # Initialize the graph with our state schema
    graph = StateGraph(AgentState)

    # Add nodes — each node is a function that takes and returns state
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # Define edges — the flow between nodes
    # set_entry_point: where the graph starts
    graph.set_entry_point("retrieve")
    
    # After retrieve, always go to generate
    graph.add_edge("retrieve", "generate")

    # After generate, end the graph
    graph.add_edge("generate", END)

    # Compile the graph into an executable object
    return graph.compile()

# ─────────────────────────────────────────
# 5. CONVENIENCE FUNCTION
# ─────────────────────────────────────────

@traceable(name="ask")
def ask(question: str) -> str:
    """
    Main entry point for asking the agent a question.
    Builds the graph, runs it, and returns Claude's answer.
    """
    graph = build_graph()

    final_state = graph.invoke({
        "question":question,
        "context": "",
        "answer": ""
    })

    return final_state["answer"]

# ─────────────────────────────────────────
# 6. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "What are some good action RPGs on PlayStation?",
        "Who developed God of War?",
        "What is The Witcher 3 about?",
        "Recommend me a PS5 exclusive game",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        answer = ask(question)
        print(f"Answer: {answer}")