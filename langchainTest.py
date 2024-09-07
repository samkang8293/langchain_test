from dotenv import load_dotenv, find_dotenv
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import (
    LLMChain,
    SimpleSequentialChain
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
import pinecone
from langchain.vectorstores import Pinecone

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

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("autoencoder"))

second_prompt = PromptTemplate(
    input_variables=["ml_concept"],
    template="Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

explanation = overall_chain.run("autoencoder")
print(explanation)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

texts = text_splitter.create_documents([explanation])

embeddings = OpenAIEmbeddings(model_name="ada")

query_result = embeddings.embed_query(texts[0].page_content)
print(query_result)

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)

index_name = "langchain-quickstart"
search = Pinecone.from_documents(texts, embeddings, index_name=index_name)

query = "What is magical about an encoder"
result = search.similarity_search(query)

print(result)