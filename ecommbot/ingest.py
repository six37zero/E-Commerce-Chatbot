import sys

# Force Python to use UTF-8 encoding for standard output (fix UnicodeEncodeError)
sys.stdout.reconfigure(encoding='utf-8')
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
  # Updated import
from dotenv import load_dotenv
import os
import pandas as pd
from ecommbot.data_converter import dataconveter

# Load environment variables
load_dotenv()

# Fetch credentials from .env
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_KEYSPACE = os.getenv('ASTRA_DB_KEYSPACE')

# Use Hugging Face Embeddings without API key
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Updated model path
    model_kwargs={"device": "cpu"}  # Set device to "cuda" for GPU acceleration
)

def ingestdata(status):
    vstore = AstraDBVectorStore(
        embedding=embedding, 
        collection_name='chatbotecomm',
        api_endpoint=ASTRA_DB_API_ENDPOINT, 
        token=ASTRA_DB_APPLICATION_TOKEN, 
        namespace=ASTRA_DB_KEYSPACE
    )

    storage = status
    if storage is None:
        docs = dataconveter()
        inserted_ids = vstore.add_documents(docs)
        return (vstore, inserted_ids)
    
    return vstore

if __name__ == '__main__':
    vstore, inserted_ids = ingestdata(None)
    print(f'\nInserted {len(inserted_ids)} documents.')
    
    results = vstore.similarity_search('can you tell me the low budget sound basshead.')
    for res in results:
           print(f'* {res.page_content} [{res.metadata}]'.encode('utf-8', 'ignore').decode('utf-8'))
