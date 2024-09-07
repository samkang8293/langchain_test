from dotenv import load_dotenv, find_dotenv
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate

load_dotenv(find_dotenv())

chat = ChatOpenAI(model_name="gpt-4", temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data")
]
response = chat(messages)

print(response.content, end='\n')

template = """
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines
"""

prompt = PromptTemplate(
    input_variables = ["concept"],
    template = template
)

llm(prompt.format(concept="autoencoder"))
llm(prompt.format(concept="regularization"))