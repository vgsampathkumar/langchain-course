import os
from dotenv import load_dotenv

# The 2026 Hub path
from langchain_classic.hub import pull 

# The Agent logic
from langchain_classic.agents import AgentExecutor, create_react_agent 
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import ReActSingleInputOutputParser

# Partner packages
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

#load the env variables from the .env file
load_dotenv()

# Initialize
tools = [TavilySearch()] 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
print("🚀 ReAct Agent Starting...")


    
# This pulls from the Hub correctly
prompt = pull("hwchase17/react")
structured_llm=llm.with_structured_output(AgentResponse)
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions=PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools","tool_names"]
    ).partial(format_instructions=output_parser.get_format_instructions())
    
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)

##### ReAct framework follows a strict loop of Thought, Action, Observation until it reaches a final answer.#####

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=5
)

extract_output=RunnableLambda(lambda x: x["output"])
parse_output=RunnableLambda(lambda x: output_parser.parse(x))

chain=agent_executor|extract_output|parse_output

def main():
   
    result=chain.invoke(
        input={
            "input": "search for the job postinsg for the Azure data engineering technical program manager or technical product owner or technical chapter lead with more focus on data and AI Engineering in the zip code of Raleigh, Apex ,Cary and Morrisville North Carolina"
            }
        )
    print("✅ Agent ready! Testing query...")
    print(result)
    # agent_executor.invoke({"input": "What is the population of Apex, NC?"})

if __name__ == "__main__":
    main()