"""
Movie Recommendation Multi-Agent System — MOCK VERSION (no ChromaDB needed)

Uses hardcoded movie data for testing agent logic, routing, and memory.
Swap back to movie_agent.py for the real ChromaDB-backed version.
"""

from agents import Agent, FunctionTool, function_tool
import dotenv

dotenv.load_dotenv()

# --- Models ---
ORCHESTRATOR_MODEL = "litellm/bedrock/anthropic.claude-3-haiku-20240307-v1:0"
RAG_MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"


# --- Bedrock tool converter ---
def bedrock_tool(tool: dict) -> FunctionTool:
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


# --- Mock movie database ---
MOCK_MOVIES = [
    {"title": "inception", "year": "2010", "genres": "action, science fiction, adventure",
     "vote_average": 8.4, "runtime": 148, "tagline": "Your mind is the scene of the crime.",
     "overview": "Cobb steals information from people's minds by entering their dreams. Offered a chance to have his criminal history erased, he must pull off the ultimate heist: planting an idea in someone's mind."},
    {"title": "the dark knight", "year": "2008", "genres": "drama, action, crime, thriller",
     "vote_average": 8.5, "runtime": 152, "tagline": "Why So Serious?",
     "overview": "Batman raises the stakes in his war on crime, facing the Joker, a criminal mastermind who wants to plunge Gotham City into anarchy."},
    {"title": "interstellar", "year": "2014", "genres": "adventure, drama, science fiction",
     "vote_average": 8.3, "runtime": 169, "tagline": "Mankind was born on Earth. It was never meant to die here.",
     "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival as Earth becomes uninhabitable."},
    {"title": "the shawshank redemption", "year": "1994", "genres": "drama, crime",
     "vote_average": 8.7, "runtime": 142, "tagline": "Fear can hold you prisoner. Hope can set you free.",
     "overview": "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison."},
    {"title": "pulp fiction", "year": "1994", "genres": "thriller, crime",
     "vote_average": 8.5, "runtime": 154, "tagline": "Just because you are a character doesn't mean you have character.",
     "overview": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
    {"title": "toy story", "year": "1995", "genres": "animation, comedy, family",
     "vote_average": 7.7, "runtime": 81, "tagline": "The adventure takes off!",
     "overview": "Woody, a cowboy doll, is Andy's favorite toy. When a new spaceman figure, Buzz Lightyear, arrives, Woody feels threatened."},
    {"title": "finding nemo", "year": "2003", "genres": "animation, family, comedy",
     "vote_average": 7.9, "runtime": 100, "tagline": "There are 3.7 trillion fish in the ocean. They're looking for one.",
     "overview": "A clownfish named Marlin, along with forgetful Dory, embarks on a journey to find his abducted son Nemo."},
    {"title": "the conjuring", "year": "2013", "genres": "horror, thriller",
     "vote_average": 7.5, "runtime": 112, "tagline": "Based on the true case files of the Warrens.",
     "overview": "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse."},
    {"title": "get out", "year": "2017", "genres": "horror, thriller, mystery",
     "vote_average": 7.6, "runtime": 104, "tagline": "Just because you're invited, doesn't mean you're welcome.",
     "overview": "A young African-American visits his white girlfriend's parents for the weekend, where his growing uneasiness leads to a shocking discovery."},
    {"title": "the notebook", "year": "2004", "genres": "romance, drama",
     "vote_average": 7.9, "runtime": 123, "tagline": "Behind every great love is a great story.",
     "overview": "An epic love story centered around an older man who reads aloud to a woman with Alzheimer's, transporting her to the past."},
    {"title": "la la land", "year": "2016", "genres": "comedy, drama, romance, music",
     "vote_average": 7.9, "runtime": 128, "tagline": "Here's to the fools who dream.",
     "overview": "Mia, an aspiring actress, and Sebastian, a jazz musician, fall in love while pursuing their dreams in Los Angeles."},
    {"title": "mad max: fury road", "year": "2015", "genres": "action, adventure, science fiction",
     "vote_average": 7.6, "runtime": 120, "tagline": "What a lovely day.",
     "overview": "In a post-apocalyptic wasteland, Max teams up with Furiosa to flee from a tyrannical warlord and his army in an armored tanker truck."},
    {"title": "the grand budapest hotel", "year": "2014", "genres": "comedy, drama",
     "vote_average": 8.0, "runtime": 100, "tagline": "A perfect holiday without the hassle.",
     "overview": "A legendary concierge at a famous European hotel and his trusted lobby boy become entangled in the theft of a priceless painting."},
    {"title": "parasite", "year": "2019", "genres": "comedy, thriller, drama",
     "vote_average": 8.5, "runtime": 132, "tagline": "Act like you own the place.",
     "overview": "The struggling Kim family sees an opportunity when the son starts tutoring for the wealthy Park family, leading to an unexpected entanglement."},
    {"title": "coco", "year": "2017", "genres": "animation, family, comedy, fantasy",
     "vote_average": 8.2, "runtime": 105, "tagline": "The celebration of a lifetime.",
     "overview": "Despite his family's generations-old ban on music, young Miguel dreams of becoming a musician and finds himself in the Land of the Dead."},
]


def _search_mock(query: str, max_results: int = 5) -> list[dict]:
    """Simple keyword matching against mock movies."""
    query_lower = query.lower()
    keywords = query_lower.replace(",", " ").split()

    scored = []
    for movie in MOCK_MOVIES:
        searchable = f"{movie['title']} {movie['genres']} {movie['overview']} {movie['tagline']}".lower()
        score = sum(1 for kw in keywords if kw in searchable)
        if score > 0:
            scored.append((score, movie))

    scored.sort(key=lambda x: (-x[0], -x[1]["vote_average"]))

    if not scored:
        # Return top-rated as fallback
        fallback = sorted(MOCK_MOVIES, key=lambda m: -m["vote_average"])
        return fallback[:max_results]

    return [m for _, m in scored[:max_results]]


# --- Mock RAG Tool: Movie Lookup ---
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
    results = _search_mock(query, max_results)

    if not results:
        return f"No movies found matching: {query}"

    formatted = []
    for i, meta in enumerate(results):
        title = meta["title"].title()
        entry = f"{i+1}. **{title}** ({meta['year']})\n"
        entry += f"   Genres: {meta['genres'].title()}\n"
        entry += f"   Rating: {meta['vote_average']}/10\n"
        entry += f"   Runtime: {meta['runtime']} min\n"
        if meta.get("tagline"):
            entry += f"   Tagline: {meta['tagline']}\n"
        entry += f"   Overview: {meta['overview']}\n"
        formatted.append(entry)

    return "🎬 Movie Search Results:\n\n" + "\n".join(formatted)


# --- Sub-Agent 1: Movie Database Agent (RAG) ---
movie_db_agent = Agent(
    name="Movie Database Agent",
    instructions="""
    You are a movie database specialist. Your job is to search the movie database
    and return relevant movie recommendations based on the query you receive.

    When you receive a request:
    1. Use the movie_lookup_tool to search for matching movies.
    2. Present the results clearly with title, year, genre, rating, and a brief description.
    3. If the first search doesn't yield good results, try rephrasing the query and searching again.
    4. You can search multiple times with different queries to find the best matches.
    5. Don't use the movie_lookup_tool more than 3 times per request.
    """,
    model=RAG_MODEL,
    tools=[bedrock_tool(movie_lookup_tool.__dict__)],
)


# --- Sub-Agent 2: Profile Builder Agent ---
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


# --- Convert sub-agents to tools for the orchestrator ---
movie_search_tool = movie_db_agent.as_tool(
    tool_name="movie-database-search",
    tool_description="Search the movie database for recommendations based on a description of what the user wants. Pass a detailed description of the user's preferences.",
)

profile_builder_tool = profile_builder_agent.as_tool(
    tool_name="profile-builder",
    tool_description="Analyze the user's message to build a movie preference profile. Pass the user's raw message about what kind of movie they want.",
)


# --- Main Orchestrator Agent: Movie Advisor ---
movie_advisor = Agent(
    name="Movie Advisor",
    instructions="""
    You are a friendly and knowledgeable movie advisor chatbot. You help users
    discover movies they'll love.

    ## Your Workflow

    **For new recommendation requests:**
    1. First, use the profile-builder tool to understand the user's preferences
       from their message.
    2. Then, use the movie-database-search tool with the profile to find matching movies.
    3. Present the recommendations in a friendly, conversational way with brief
       reasons why each movie fits what they're looking for.

    **For follow-up questions (e.g. "tell me more", "something similar", "another one"):**
    - Use the conversation history (memory) to recall what was previously discussed.
    - Use the movie-database-search tool with a refined query based on context.

    **For general movie chat:**
    - If the user asks about a specific movie, search for it and share what you know.
    - Be conversational and enthusiastic about movies!

    ## Guidelines
    - Always present 3-5 recommendations unless the user asks for more or fewer.
    - Include the movie title, year, genre, rating, and a brief reason it's a good match.
    - Use emojis sparingly to keep it fun (🎬 🍿 ⭐).
    - Remember the user's preferences across the conversation for better recommendations.
    - If the user mentions they liked/disliked a previous suggestion, factor that into
      future recommendations.
    - Be concise but helpful. Don't write walls of text.
    """,
    model=ORCHESTRATOR_MODEL,
    tools=[profile_builder_tool, movie_search_tool],
)
