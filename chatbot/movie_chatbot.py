"""
Movie Recommendation Chatbot — Chainlit App

Capabilities:
  - RAG: Movie database search via ChromaDB
  - Multi-agent: Orchestrator → Profile Builder + Movie DB Agent
  - Memory: SQLiteSession for multi-turn conversation history

Run with:
    chainlit run chatbot/movie_chatbot.py --port 10000
"""

import chainlit as cl
import dotenv

from openai.types.responses import ResponseTextDeltaEvent
from agents import Runner, SQLiteSession
from agents.stream_events import RunItemStreamEvent, AgentUpdatedStreamEvent

from movie_agent import movie_advisor
from poster_utils import insert_posters_into_text

dotenv.load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    # Initialize memory session for this user
    session = SQLiteSession("movie_conversation_history")
    cl.user_session.set("agent_session", session)

    # Welcome message
    await cl.Message(
        content=(
            "🎬 **Welcome to Movie Advisor!**\n\n"
            "I can help you find the perfect movie. Just tell me:\n"
            "- What genre or mood you're in the mood for\n"
            "- Who you're watching with (solo, date night, family, friends)\n"
            "- Any preferences (actors, time period, language, etc.)\n\n"
            "Or just say something like *\"I want a funny movie for tonight\"* "
            "and I'll take it from there! 🍿"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")

    result = Runner.run_streamed(
        movie_advisor,
        message.content,
        session=session,
    )

    msg = cl.Message(content="")
    current_agent_name = movie_advisor.name

    async for event in result.stream_events():

        # Track agent handoffs
        if isinstance(event, AgentUpdatedStreamEvent):
            current_agent_name = event.new_agent.name

        # Stream final message text to screen
        elif event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)

        # Show tool calls as Chainlit steps
        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and len(event.data.item.arguments) > 0
        ):
            tool_name = event.data.item.name
            with cl.Step(name=f"🔧 {tool_name}", type="tool") as step:
                step.input = event.data.item.arguments

        # Show tool outputs
        elif isinstance(event, RunItemStreamEvent):
            if event.name == "tool_output":
                output = str(event.item.output)
                preview = output[:500] + "…" if len(output) > 500 else output
                with cl.Step(name="📋 Tool Result", type="tool") as step:
                    step.output = preview

    await msg.update()

    # Insert posters inline next to each recommendation
    updated_content = await insert_posters_into_text(msg.content)
    if updated_content != msg.content:
        msg.content = updated_content
        await msg.update()
