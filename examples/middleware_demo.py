# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Middleware demo — shows middleware hooks in action with Gemini."""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from autogen.beta import Agent, tool
from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.events import ModelResponse
from autogen.beta.middleware.builtin import HistoryLimiter, LoggingMiddleware


# ---------------------------------------------------------------------------
# A simple tool for the agent
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "Cloudy, 15°C",
        "tokyo": "Sunny, 28°C",
        "new york": "Rainy, 12°C",
    }
    return weather_data.get(city.lower(), f"No data for {city}")


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", max_output_tokens=512)

    agent = Agent(
        "weather_bot",
        prompt="You are a helpful weather assistant. Use the get_weather tool to answer weather questions. Be concise.",
        config=config,
        tools=[get_weather],
        middleware=[
            LoggingMiddleware(),  # outermost — sees everything first/last
            HistoryLimiter(max_events=20),
        ],
    )

    # --- Observer demo: @agent.on() ---
    @agent.on(ModelResponse)
    async def on_response(event: ModelResponse) -> None:
        if event.message and event.message.content:
            print(f"  [Observer] Saw response: {event.message.content[:50]}...")
        elif event.tool_calls:
            print(f"  [Observer] Saw tool call response: {[c.name for c in event.tool_calls.calls]}")

    # Ask something that triggers a tool call
    print("\n*** Demo: Tool call with middleware ***\n")
    conv = await agent.ask("What's the weather in Tokyo?")
    print(f"\nFinal answer: {conv.message.message.content}")

    # Multi-turn
    print("\n\n*** Demo: Multi-turn conversation ***\n")
    conv = await conv.ask("And what about London?")
    print(f"\nFinal answer: {conv.message.message.content}")


if __name__ == "__main__":
    asyncio.run(main())
