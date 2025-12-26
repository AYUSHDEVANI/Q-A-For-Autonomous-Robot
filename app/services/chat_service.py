import os
import re
import logging
import numpy as np
from typing import TypedDict
from cachetools import TTLCache
# from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from app.services.pdf_service import get_retriever

logger = logging.getLogger(__name__)

# Cache
question_cache = TTLCache(maxsize=100, ttl=3600)

# Models
GROQ_MODEL = "llama-3.1-8b-instant"
# Initialize these globally or lazily
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') # REMOVED: Local model
llm = None  # Will be initialized in init_chat_model

class State(TypedDict):
    question: str
    context: str
    relevance_score: float
    answer: str
    history: list

workflow_app = None

def init_chat_model():
    global llm, workflow_app
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found. Chat features may fail.")
    
    # Initialize ChatGroq
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name=GROQ_MODEL, 
        temperature=0.5
    )
    
    # Build Workflow
    workflow = StateGraph(State)
    workflow.set_entry_point("pdf_retrieve")
    workflow.add_node("pdf_retrieve", pdf_retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_conditional_edges(
        "pdf_retrieve",
        lambda state: "web_search" if state["relevance_score"] < 0.2 else "generate",
        {"web_search": "web_search", "generate": "generate"}
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    workflow_app = workflow.compile()

# Nodes
def pdf_retrieve_node(state: State) -> State:
    logger.debug(f"pdf_retrieve_node input state: {state}")
    retriever = get_retriever()
    if not retriever:
        state["context"] = ""
        state["relevance_score"] = 0.0
        return state
        
    try:
        # Reformulate question based on history? 
        # For simplicity, we search with the raw question, but ideally we'd condense.
        # Let's keep retrieval simple for now as per user request to just "Add history to prompt".
        docs = retriever.invoke(state["question"])
        logger.info(f"Retrieved {len(docs)} documents.")
        context = "\\n\\n".join(doc.page_content for doc in docs)
        state["context"] = context
        
        # REMOVED: Local CrossEncoder Re-ranking
        # Assuming retrieval is good enough or relying on LLM to filter.
        # Set a default high relevance since we trust the retriever + LLM prompt
        state["relevance_score"] = 1.0 
        logger.info(f"Relevance Score: {state['relevance_score']:.4f}")
    except Exception as e:
        logger.exception("Error in pdf_retrieve_node")
        state["context"] = ""
        state["relevance_score"] = 0.0
    return state

def web_search_node(state: State) -> State:
    logger.debug(f"web_search_node input state: {state}")
    try:
        with DDGS() as ddgs:
            # Maybe include history context in web search too?
            q = state['question']
            web_results = [r for r in ddgs.text(f"brief description of {q} project in robotics or engineering", max_results=1)]
            if web_results:
                logger.info("Web search result found.")
                fallback_context = web_results[0]['body'][:1000]
                state["context"] = f"Note: This project is not currently available in our lab. Here's a brief overview: {fallback_context}"
            else:
                logger.info("No web search results.")
                state["context"] = "Note: This project is not currently available in our lab."
    except Exception as e:
        logger.exception("Error in web_search_node")
        state["context"] = ""
    return state

def generate_node(state: State) -> State:
    logger.debug(f"generate_node input state: {state}")
    try:
        history_text = ""
        if state.get("history"):
            history_text = "\\n".join([f"User: {h[0]}\\nBot: {h[1]}" for h in state["history"]])

        prompt = ChatPromptTemplate.from_template("""
system:
    "You are LabBot, a helpful assistant. Answer based on the context."
    "If the user asks follow-up questions (like 'how much does it weigh?'), use the Chat History to understand what 'it' refers to."

Chat History:
{history}

Context:
{context}

Question: {question}
        """)
        # Note: We replaced the specific bulky prompt with a simpler one that includes history, 
        # or we should merge them. The user wants specific "Definition/Key Points" structure?
        # Let's keep the structure but inject history.
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are LabBot, an advanced and enthusiastic Lab Assistant Robot 🤖.

YOUR MISSION:
Help visitors understand the exciting engineering and robotics projects in our lab using ONLY the provided "Context".
Be engaging, professional, and clear. Imagine you are giving a guided tour.

GUIDELINES:
1. **Context First:** Answer strictly based on the provided 'Context'. If the answer is not there, say: "I don't have details on that specific project in my records, but I can tell you about other available projects!"
2. **Conversation Awareness:** Use 'Chat History' to understand context (e.g., if user asks "How much does it cost?", check history to see what "it" refers to).
3. **Format:** Structure your answers clearly:
   - **🎯 Core Concept:** A one-sentence definition.
   - **🔑 Key Features:** Bullet points of technical specs or main features.
   - **💡 Real-World Application:** How this tech is used in the real world.
4. **Tone:** Helpful, precise, and slightly robotic but friendly (e.g., "Affirmative!", "Here is the data:", "Let me retrieve that...")."""),
            ("human", """Chat History:
{history}

Context: {context}

Question: {question}""")
        ])
        
        formatted_prompt = prompt.format(
            question=state.get("question", ""), 
            context=state.get("context", ""),
            history=history_text
        )
        response = llm.invoke(formatted_prompt)
        state["answer"] = getattr(response, "content", str(response))
    except Exception as e:
        logger.exception("Error in generate_node")
        state["answer"] = f"Error generating answer: {str(e)}"
    return state

async def stream_answer(question: str, history: list = []):
    logger.info(f"Starting streaming answer for: {question}")
    retriever = get_retriever()
    if retriever is None:
        logger.warning("Retriever is None. No PDFs processed.")
        yield "No PDFs processed. Add PDFs to the 'pdfs' folder and restart the app."
        return

    # Cache check (skip if using history?) - ideally cache based on (question, history) or disabled for follow-ups
    # For now, let's DISABLE cache for history-aware queries to ensure freshness
    if not history and question in question_cache:
        logger.info("Serving answer from cache (streaming).")
        yield question_cache[question]
        return

    if not workflow_app:
        logger.info("Initializing chat model on first request...")
        init_chat_model()

    # 1. Retrieval
    # Check history to refine query? (Optional refinement step omitted for brevity)
    try:
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} documents.")
        context = "\\n\\n".join(doc.page_content for doc in docs)
        
        # REMOVED: Local CrossEncoder Re-ranking
        relevance = 1.0
        logger.info(f"Relevance Score: {relevance:.4f}")
        
        if relevance < 0.2:
            logger.info("Low relevance. Attempting web search fallback.")
            try:
                with DDGS() as ddgs:
                    # Provide more context to web search if possible
                    web_results = [r for r in ddgs.text(f"brief description of {question} project in robotics or engineering", max_results=1)]
                    if web_results:
                        logger.info("Web search result found.")
                        fallback_context = web_results[0]['body'][:1000]
                        context = f"Note: This project is not currently available in our lab. Here's a brief overview: {fallback_context}"
                    else:
                        logger.info("No web search results.")
                        context = "Note: This project is not currently available in our lab."
            except Exception as e:
                logger.error(f"Web search error: {e}")
                context = ""     
    except Exception as e:
        logger.error(f"Error in retrieval for stream: {e}")
        context = ""

    # 2. Generation (Async Stream)
    try:
        logger.info("Starting LLM generation (streaming)...")
        
        history_text = ""
        if history:
            history_text = "\\n".join([f"User: {h[0]}\\nBot: {h[1]}" for h in history])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are LabBot, an advanced and enthusiastic Lab Assistant Robot 🤖.

YOUR MISSION:
Help visitors understand the exciting engineering and robotics projects in our lab using ONLY the provided "Context".
Be engaging, professional, and clear. Imagine you are giving a guided tour.

GUIDELINES:
1. **Context First:** Answer strictly based on the provided 'Context'. If the answer is not there, say: "I don't have details on that specific project in my records, but I can tell you about other available projects!"
2. **Conversation Awareness:** Use 'Chat History' to understand context (e.g., if user asks "How much does it cost?", check history to see what "it" refers to).
3. **Format:** Structure your answers clearly:
   - **🎯 Core Concept:** A one-sentence definition.
   - **🔑 Key Features:** Bullet points of technical specs or main features.
   - **💡 Real-World Application:** How this tech is used in the real world.
4. **Tone:** Helpful, precise, and slightly robotic but friendly (e.g., "Affirmative!", "Here is the data:", "Let me retrieve that...")."""),
            ("human", """Chat History:
{history}

Context: {context}

Question: {question}""")
        ])
        formatted_prompt = prompt.format(question=question, context=context, history=history_text)
        
        full_answer = ""
        async for chunk in llm.astream(formatted_prompt):
            content = chunk.content
            if content:
                full_answer += content
                yield content
        
        logger.info(f"Streaming complete. Caching answer ({len(full_answer)} chars).")
        if not history:
             question_cache[question] = full_answer

    except Exception as e:
        logger.error(f"Error generating stream: {str(e)}")
        yield f"Error generating answer: {str(e)}"

def answer_question(question):
    retriever = get_retriever()
    if retriever is None:
        return "No PDFs processed. Add PDFs to the 'pdfs' folder and restart the app."

    if question in question_cache:
        return question_cache[question]

    if not workflow_app:
        init_chat_model()

    try:
        result = workflow_app.invoke({"question": question})
        answer = result["answer"]

        # Clean up output
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer).strip()
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()

        question_cache[question] = answer
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"
