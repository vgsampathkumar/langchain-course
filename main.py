import os
from dotenv import load_dotenv

# The 2026 Hub path
from langchain_classic.hub import pull 

# The Agent logic
from langchain_classic.agents import AgentExecutor, create_react_agent 

# Partner packages
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

#load the env variables from the .env file
load_dotenv()

# Initialize
tools = [TavilySearch()] 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
print("🚀 ReAct Agent Starting...")
    
# This pulls from the Hub correctly
prompt = pull("hwchase17/react")
    
agent = create_react_agent(llm, tools, prompt)

##### ReAct framework follows a strict loop of Thought, Action, Observation until it reaches a final answer.#####

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=5
)

chain=agent_executor

def main():
   
    result=chain.invoke(
        input={
            "input": "search for the job postinsg for the Azure data engineering technical program manager or technical product owner or technical chapter lead with more focus on data and AI Engineering in the zip code of Raleigh, Apex ,Cary and Morrisville North Carolina"
            }
        )
    print("✅ Agent ready! Testing query...")
    # agent_executor.invoke({"input": "What is the population of Apex, NC?"})

if __name__ == "__main__":
    main()