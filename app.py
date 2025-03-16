from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import time
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

embeddings = download_hugging_face_embeddings()

memory = ConversationBufferMemory(k=5, memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{prompt_template}"),
    ]
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

docsearch = PineconeVectorStore(index=index, embedding=embeddings)

# Setup Prompt
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# LLM Initialization
# llm = ChatGroq(
#     temperature=0.8, 
#     groq_api_key=GROQ_API_KEY, 
#     model_name="llama-3.3-70b-versatile"
# )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs=chain_type_kwargs,
    memory=memory,
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])  # Only POST is needed for chat
def chat():
    data = request.json  # Receiving JSON data from fetch
    msg = data.get("msg")  # Get user message
    print("User input:", msg)
    result = qa({"query": msg})  # Query the AI chain
    print("Response:", result["result"])
    return result["result"]  # Return only the clean response as plain text

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
