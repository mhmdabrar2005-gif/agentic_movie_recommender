from langchain_groq import ChatGroq
import langchain
from langchain.agents import create_agent

# This will print the raw LLM thoughts, tool requests, and responses to your terminal!
langchain.debug = True
from tools import get_tools, search_tmdb, search_vector_db, get_trending_movies, search_tmdb_tv, get_trending_tv_shows
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
    
    # Check for trending first
    if "trending" in user_input.lower() or "popular" in user_input.lower():
        if "tv" in user_input.lower() or "show" in user_input.lower():
            from tools import get_trending_tv_shows
            tool_text = get_trending_tv_shows.invoke({"time_window": "day"})
            title_header = "Trending TV Shows Right Now"
        else:
            from tools import get_trending_movies
            tool_text = get_trending_movies.invoke({"time_window": "day"})
            title_header = "Trending Movies Right Now"

        return (
            f"### {title_header}\n"
            f"Here are the latest trending titles:\n{tool_text}\n\n"
            "_Note: Fallback mode was used because model tool-calling failed once._"
        )

    title = extract_reference_title(user_input)
    if title:
        tool_text = search_tmdb.invoke({"query": title})
        if isinstance(tool_text, str) and not tool_text.lower().startswith("tmdb lookup failed") and tool_text.strip() != "No movies found.":
            return (
                "### Movies Related to Your Request\n"
                f"Based on \"{title}\", here are some matches:\n{tool_text}\n\n"
                "_Note: Fallback mode was used because model tool-calling failed once._"
            )

        vector_text = search_vector_db.invoke({"mood": user_input})
        return (
            "### Recommendations From Local Database\n"
            f"TMDb was unavailable, so I used local matches instead:\n{vector_text}\n\n"
            "_Note: Fallback mode was used because model tool-calling failed once._"
        )

    tool_text = search_vector_db.invoke({"mood": user_input})
    return (
        "### Recommendations From Local Database\n"
        f"{tool_text}\n\n"
        "_Note: Fallback mode was used because model tool-calling failed once._"
    )

def run_agent(user_input: str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    
    system_message = (
        "You are an expert, witty Movie and TV Recommender AI. "
        "A user will give you their mood or request. "
        "You must use the provided tools to find the best titles. "
        "If the user asks for specific facts, titles, or actors for a movie, use the TMDb tool. If it's a TV show, use the TMDb TV tool. "
        "If the user describes a vibe or mood, use the Vector DB tool. "
        "If the user asks for trending, popular, or new movies, use the trending movies tool. If they ask for trending TV shows, use the trending TV tool. "
        "CRITICAL: If the user explicitly asks for BOTH, or asks generally (e.g. 'what is trending right now' or 'give me a good sci-fi') without specifying movies or TV, you should use BOTH the movie and TV tools to give a mixed recommendation! "
        "After retrieving the information, synthesize a response suggesting up to 3 titles with a short, witty, personalized explanation for each based on the user's input. Format your final response clearly in Markdown."
    )
    
    # Create the agent and run it in one go
    agent_executor = create_agent(llm, tools=get_tools, system_prompt=system_message)
    try:
        response = agent_executor.invoke({"messages": [{"role": "user", "content": user_input}]})
        
        # Return just the text content of the AI's final message
        content = response["messages"][-1].content
        return content if isinstance(content, str) else str(content)
    except Exception as exc:
        error_text = str(exc)
        if "tool_use_failed" in error_text or "Failed to call a function" in error_text:
            return fallback_tool_response(user_input)
        raise