from langchain_groq import ChatGroq
import langchain
from langchain.agents import create_agent

# This will print the raw LLM thoughts, tool requests, and responses to your terminal!
langchain.debug = True
from backend.tools import get_tools, search_tmdb, search_vector_db, get_trending_movies, search_tmdb_tv, get_trending_tv_shows
from dotenv import load_dotenv
import re

load_dotenv()

def extract_reference_title(user_input: str) -> str:
    """Extract a movie title that appears after 'like', 'similar to', or 'reminds me of'."""
    
    # The updated pattern:
    # 1. Finds "like", OR "similar to", OR "reminds me of"
    # 2. Captures the text immediately following it
    # 3. Stops at conjunctions (but, and, etc.) or punctuation
    pattern = r"\b(?:like|similar to|reminds me of)\s+(.+?)(?=\b(?:but|and|or|because|with|then|now)\b|[.,!?]|$)"
    
    matches = re.findall(pattern, user_input, re.IGNORECASE)
    
    if not matches:
        return ""
        
    title = matches[-1].strip()
    return title


def fallback_tool_response(user_input: str) -> str:
    """Return a usable response even when model-level function calling fails."""
    
    # Ignore simple greetings
    lower_input = user_input.strip().lower()
    if lower_input in ["hi", "hello", "hey", "sup", "howdy"]:
        return "Hello! I am your AI Recommender. How can I help you find a movie or TV show today?"

    # Check for trending first
    if "trending" in lower_input or "popular" in lower_input:
        if "tv" in user_input.lower() or "show" in user_input.lower():
            tool_text = get_trending_tv_shows.invoke({"time_window": "day"})
            title_header = "Trending TV Shows Right Now"
        else:
            tool_text = get_trending_movies.invoke({"time_window": "day"})
            title_header = "Trending Movies Right Now"

        return (
            f"### {title_header}\n"
            f"Here are the latest trending titles:\n{tool_text}\n"
        )

    title = extract_reference_title(user_input)
    if title:
        tool_text = search_tmdb.invoke({"query": title})
        if isinstance(tool_text, str) and not tool_text.lower().startswith("tmdb lookup failed") and tool_text.strip() != "No movies found.":
            return (
                "### Movies Related to Your Request\n"
                f"Based on \"{title}\", here are some matches:\n{tool_text}\n"
            )

        vector_text = search_vector_db.invoke({"mood": user_input})
        return (
            "### Recommendations From Local Database\n"
            f"TMDb was unavailable, so I used local matches instead:\n{vector_text}\n"
        )

    tool_text = search_vector_db.invoke({"mood": user_input})
    return (
        "### Recommendations From Local Database\n"
        f"{tool_text}\n"
    )

def run_agent(user_input: str, history: list = None) -> str:
    if history is None:
        history = []
        
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)
    
    system_message = (
        "You are an expert, witty Movie and TV Recommender AI. "
        "A user will give you their mood or request. "
        "CRITICAL: If the user is just greeting you (e.g., 'hi', 'hello', 'hey') or making small talk, simply greet them back conversationally. DO NOT call any tools. "
        "Otherwise, you must use the provided tools to find the best titles. "
        "If the user asks for specific facts, titles, or actors for a movie, use the TMDb tool. If it's a TV show, use the TMDb TV tool. "
        "If the user describes a vibe or mood, use the Vector DB tool. "
        "If the user asks for trending, popular, or new movies, use the trending movies tool. If they ask for trending TV shows, use the trending TV tool. "
        "CRITICAL: If the user explicitly asks for BOTH, or asks generally (e.g. 'what is trending right now' or 'give me a good sci-fi') without specifying movies or TV, you should use BOTH the movie and TV tools to give a mixed recommendation! "
        "After retrieving the information, synthesize a response suggesting up to 3 titles with a short, witty, personalized explanation for each based on the user's input. Format your final response clearly in Markdown."
    )
    
    # Format history as text context to prevent Groq tool-calling errors
    history_context = ""
    if history:
        history_context = "--- Previous Conversation ---\n"
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_context += f"{role}: {msg.get('content', '')}\n"
        history_context += "-----------------------------\n\n"
        history_context += "Now, address the user's latest input below:\n"
        
    final_user_input = history_context + user_input
    messages = [{"role": "user", "content": final_user_input}]
    
    # Create the agent and run it in one go
    agent_executor = create_agent(llm, tools=get_tools, system_prompt=system_message)
    try:
        response = agent_executor.invoke({"messages": messages})
        
        # Return just the text content of the AI's final message
        content = response["messages"][-1].content
        return content if isinstance(content, str) else str(content)
    except Exception as exc:
        print(f"DEBUG EXCEPTION: {exc}")
        error_text = str(exc)
        if "tool_use_failed" in error_text or "Failed to call a function" in error_text:
            return fallback_tool_response(user_input)
        raise