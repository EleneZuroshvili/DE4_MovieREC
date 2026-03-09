"""
Movie Recommendation Multi-Agent System

Agents:
1. Movie Database Agent (RAG) - Searches ChromaDB for movie recommendations
2. Movie Advisor Agent (Orchestrator) - Builds user profile, routes to sub-agents

Capabilities used: RAG, Multi-agent orchestration, Memory (via Chainlit app)
"""

import json
from pathlib import Path

import chromadb
from agents import Agent, FunctionTool, function_tool

import dotenv

dotenv.load_dotenv()

# --- Models ---
ORCHESTRATOR_MODEL = "litellm/bedrock/anthropic.claude-3-haiku-20240307-v1:0"
RAG_MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"


# --- Bedrock tool converter ---
def bedrock_tool(tool: dict) -> FunctionTool:
    """Converts an OpenAI Agents SDK function_tool to a Bedrock-compatible FunctionTool."""
    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )


# --- ChromaDB setup ---
chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
movie_db = chroma_client.get_collection(name="movies_database")


# --- RAG Tool: Movie Lookup ---
@function_tool
def movie_lookup_tool(query: str, max_results: int = 5) -> str:
    """
    Search the movie database for movies matching a description, genre, mood, or title.

    Args:
        query: A natural language search query describing the kind of movie wanted.
               Examples: "funny comedy for family", "dark thriller with a twist",
               "romantic movie set in Paris", "sci-fi space adventure".
        max_results: Number of movie results to return (default 5).

    Returns:
        A formatted string with matching movie details.
    """
    results = movie_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No movies found matching: {query}"

    formatted = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        title = meta["title"]
        genres = json.loads(meta.get("genres_json", "[]"))
        genres_str = ", ".join(genres) if genres else "Unknown"

        entry = f"{i+1}. **{title}**\n"
        entry += f"   Genres: {genres_str}\n"
        entry += f"   Overview: {doc}\n"

        formatted.append(entry)

    return "Movie Search Results:\n\n" + "\n".join(formatted)


# --- Sub-Agent: Profile Builder Agent ---
profile_builder_agent = Agent(
    name="Profile Builder Agent",
    instructions="""
    You are a friendly movie preference profiler. Your job is to understand what
    kind of movie the user is in the mood for RIGHT NOW.

    Based on the user's message, extract or infer:
    - Preferred genre(s) (action, comedy, drama, horror, sci-fi, romance, thriller, animation, etc.)
    - Mood (fun/lighthearted, intense/thrilling, emotional/deep, scary, inspiring, relaxing)
    - Audience (solo, date night, family with kids, friends group)
    - Any other preferences mentioned (specific actors, directors, time period, language, etc.)

    Return a concise profile summary that can be used to search for movies.
    Format it as a clear, searchable description.

    Example output:
    "Looking for a lighthearted comedy suitable for a family movie night,
    preferably something animated or with a PG rating, recent releases preferred."

    If the user hasn't given enough info, create the best profile you can from what's available.
    Do NOT ask follow-up questions - just work with what you have.
    """,
    model=RAG_MODEL,
)


# --- Convert sub-agent to tool for the orchestrator ---
profile_builder_tool = profile_builder_agent.as_tool(
    tool_name="profile-builder",
    tool_description="Analyze the user's message to build a movie preference profile. Pass the user's raw message about what kind of movie they want.",
)


# --- Main Orchestrator Agent: Movie Advisor ---
movie_advisor = Agent(
    name="Movie Advisor",
    instructions="""
    You are a movie advisor chatbot. You help users discover movies.

    ## Workflow (ALWAYS follow these steps)

    1. Use the profile-builder tool to understand what the user wants.
    2. Use the movie_lookup_tool to search the database with the profile.
    3. List ALL movies from the search results as a numbered list.

    For follow-ups ("something similar", "more like that", "find X"):
    - Use movie_lookup_tool with a refined query. ALWAYS list the results.

    ## OUTPUT FORMAT (MANDATORY)

    EVERY response MUST contain a numbered list of movies in this exact format:

    1. **Movie Title** - Genre - Why it's a good match.
    2. **Movie Title** - Genre - Why it's a good match.
    3. **Movie Title** - Genre - Why it's a good match.

    NEVER respond without listing specific movie titles from the search results.
    NEVER summarize the results without showing individual movie names.
    If the search returned movies, you MUST list them by name.
    Present 3-5 movies per response.

    Remember the user's preferences across the conversation.
    Be concise and friendly.
    """,
    model=ORCHESTRATOR_MODEL,
    tools=[profile_builder_tool, bedrock_tool(movie_lookup_tool.__dict__)],
)
