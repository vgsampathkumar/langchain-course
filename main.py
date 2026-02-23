import os
from dotenv import load_dotenv
load_dotenv()

import langchainhub as hub
from langchain_classic.agents import AgentExecutor, create_react_agent # Both in classic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Use TavilySearchResults for the current version
tools = [TavilySearchResults()]

def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
