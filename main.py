import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def main():
    print("Hello from langchain-course!\n")
    load_dotenv()
    information = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.

    Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

    In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.

    Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in early 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). After a public feud with Trump, Musk left the Trump administration and returned to managing his companies. Musk is a supporter of global far-right figures, causes, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE. The emails he sent to Jeffrey Epstein are included in the Epstein files, which were published between 2025–26 and became a topic of worldwide debate.

   """
    summary_prompt = """ given the informtion {information}, summarize the key points in a concise manner. Focus on the most important details and main ideas, and avoid unnecessary information. The summary should be clear and easy to understand, providing a brief overview of the content while retaining the essential information. Aim for a summary that captures the essence of the original text without losing its meaning or context. Also please 2 good and important things 1.Short summary 2. 2 key intesrting facts that are not widely known about the topic. The summary should be no more than 3 sentences long, and the interesting facts should be concise and informative, providing additional insights into the topic that may not be commonly known. The goal is to provide a comprehensive yet concise overview of the topic, highlighting its most important aspects while also offering some unique and intriguing information that may not be widely known. Please provide the summary and interesting facts in a clear and organized manner, making it easy for readers to understand and appreciate the key points of the topic."""

    summary_prompt_template = PromptTemplate.from_template(template=summary_prompt)

    llm = ChatOpenAI(model="gpt-5", temperature=0.3)
    # llm = ChatOllama(model="gemma3:270m", temperature=0.3)
    chain = summary_prompt_template | llm
    response =chain.invoke(input={"information": information})
    print("Summary and interesting facts about Elon Musk:\n")
    print(response)

    # print("============================\n")
    # print("printing environment variables...openai_api_key\n")
    # print(os.environ.get("OPENAI_API_KEY")+"\n")
    # print("printing environment variables...google_api_key\n")
    # print(os.environ.get("GOOGLE_API_KEY"))


if __name__ == "__main__":
    main()
