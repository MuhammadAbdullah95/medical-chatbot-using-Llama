from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv
import time
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Initializing the Pinecone
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

# Creating Embeddings for Each of The Text Chunks & storing
for i, doc in tqdm(
    enumerate(text_chunks), total=len(text_chunks)
):  # Add tqdm for progress bar
    vector = embeddings.embed_query(doc.page_content)
    # Pass metadata as a dictionary, use f-string for ID
    index.upsert([(f"doc_{i}", vector, {"text": doc.page_content, **doc.metadata})])
