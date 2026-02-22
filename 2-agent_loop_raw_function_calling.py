from dotenv import load_dotenv

load_dotenv()
import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL ="qwen3:1.7b"

print(f"Initializing model {MODEL}...")

#--------------Tools(Langchain @tool decorator)-----------------
@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """Lookup the price of a product from the catalogue."""
    print(f"Looking up price for product: {product_name}")
    # In a real implementation, this would query a database or an API
    prices = {
        "laptop": "$999",
        "smartphone": "$499",
        "headphones": "$199"
    }
    return prices.get(product_name.lower(), 0.0)

@traceable(run_type="tool")
def get_product_discount(price: float, discount_tier: str) -> float:
    """Lookup and apply the discount for a product from the catalogue. Available tiers are: bronze, silver ,gold """
    print(f" >> Executong applty dicsount (price={price},discount_tier='{discount_tier}')")
    discount_percentages={"bronze": 5,"silver": 10,"gold": 15}    
    discount=discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)

    tools_for_llm=[
        {
            "type": "function",
            "name": "get_product_price",
            "description": "Lookup the price of a product from the catalogue.",
            "parameters": {
                "type": "string",
                "properties":{"product_name":{"type":"string"}
                },
                "required": ["product_name"]
            }
        },
        {
            "type": "function",
            "name": "get_product_discount",
            "description": "Lookup and apply the discount for a product from the catalogue. Available tiers are: bronze, silver ,gold ",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "discount_tier": {"type": "string"}
                },
                "required": ["price", "discount_tier"]
            }
        }
    ]

# -------------Agent Loop-----------------
@traceable(name="Langchain agent_loop")
def run_agent(question:str):
   tools=[get_product_price,get_product_discount]
   tool_dict={tool.name: tool for tool in tools}
   llm=init_chat_model(f"openai:gpt-5",temperature=0.0)
   llm_with_tools=llm.bind_tools(tools)
   print(f"Question: {question}")
   print("=" * 50)
   messages = [
       SystemMessage(
           content=("You are a helpful shopping assistant ."
                            "You have access to product catalog tool "
                            "and a disccount tool. \n\n "
                            "STRICT RUKES - you must follow these 10 commandments exactly \n"
                            "1. Always use the tools to get the price and apply discount. Never guess or make up prices.\n"
                            "2. Always call the get_product_price tool first to get the price before applying any discount.\n"
                            "3. When applying discount, use the get_product_discount tool and provide the price after the calling the get_product_price and discount tier (bronze, silver, gold) as arguments.\n"
                            "4. If the question does not specify a discount tier, apply the bronze discount by default.\n"
                            "5. Always provide the final answer in the format: 'The price of {product} after applying {discount_tier} discount is ${final_price}'\n"
                            "6. If you cannot find the product in the catalog, respond with 'Sorry, I couldn't find the price for {product}.'\n" \
                            "7. If the question is not related to product price or discount, respond with 'Sorry, I can only help with product prices and discounts.'\n" \
                            "8. Always think step by step and use the tools iteratively to arrive at the final answer.\n" \
                            "9. You have a maximum of 10 iterations to find the answer. If you cannot find the answer within 10 iterations, respond with 'Sorry, I couldn't find the answer within the iteration limit.'\n  " \
                            "10. You MUST call the discount tool even if you think you know the price. Never provide a final answer until both tools have been called for this request."
                            "11. Always remember to follow the rules and use the tools correctly to provide accurate answers. PLEASE DO NOT VIOLATE THE RULES OR MAKE UP ANSWERS. ALWAYS USE THE TOOLS AND FOLLOW THE STEPS CAREFULLY TO ARRIVE AT THE FINAL ANSWER. THANK YOU!"
                    )
                ),
                HumanMessage(content= question),
   ]

   for iteration in range(MAX_ITERATIONS):
           print(f"Iteration {iteration + 1}:")
           ai_message = llm_with_tools.invoke(messages)
           tool_calls=ai_message.tool_calls
           print(f"Agent response: {ai_message.content}")
          
           if not tool_calls:
               # If there are no tool calls, we assume it's the final answer
               print("Final answer from agent:")
               return ai_message.content
           
         
       
          # process only the fisrt tool call - force one tool call per iteration
           tool_call=tool_calls[0]        # if isinstance(response, ToolMessage):
           tool_name = tool_call.get("name")
           tool_args = tool_call.get("args",{})
           tool_call_id = tool_call.get("id")
           
           print(f"Tool call detected: {tool_name} with args {tool_args}")
           
           tool_to_use=tool_dict.get(tool_name)
           if tool_to_use is None:
                raise ValueError(f"Unknown tool: {tool_name}")
           
           observation = tool_to_use.invoke (tool_args)
           
           print(f"Observation from tool: {observation}")

           messages.append(ai_message)  # Add the agent's message with tool call to the conversation    
           messages.append(
                ToolMessage(content=observation,tool_call_id=tool_call_id)
                )  # Add the tool's observation to the conversation    
       

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    print("This is a simple agent loop example with tool calling using Langchain.")
    print("The agent will try to find the price of a product and apply a discount to it.")
    print("Available products: laptop, smartphone, headphones") 
    result = run_agent("What is the price of laptop after applying the gold discount?")