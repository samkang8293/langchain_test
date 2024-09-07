from dotenv import load_dotenv, find_dotenv
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI

load_dotenv(find_dotenv())

chat = ChatOpenAI(model_name="gpt-4", temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data")
]
response = chat(messages)

print(response.content, end='\n')