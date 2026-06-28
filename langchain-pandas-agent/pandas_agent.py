import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv(override=True)

df = pd.read_csv("data/superstore.csv", encoding="latin-1")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True,  # required: this agent generates and executes real Python code
)

if __name__ == "__main__":
    response = agent.invoke("What is the total Sales across the entire dataset?")
    print("\nFINAL ANSWER:", response["output"])