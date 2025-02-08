import os
import time
import requests  # ✅ Ensure requests module is imported
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from ecommbot.ingest import ingestdata

# ✅ Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')  # ✅ Ensure API key is loaded

# ✅ Choose a Working Hugging Face Model
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ✅ API Headers for Authentication
HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY.strip()}",
    "Content-Type": "application/json",
}

def huggingface_chat(query, context=None, max_context_tokens=100):
    """
    Sends a structured query to Hugging Face API with retrieval-augmented context,
    ensuring only **relevant product recommendations** are returned.
    """

    # ✅ Ensure `query` is a string
    if not isinstance(query, str):
        query = str(query)

    structured_query = f"📌 **User Query:** {query}\n"

    if context:
        structured_query = f"🛒 **Product Context (User Reviews):** {context}\n\n{structured_query}"

    # ✅ Ensure input does not exceed token limit
    structured_query = " ".join(structured_query.split()[:max_context_tokens])

    payload = {
        "inputs": structured_query,
        "parameters": {
            "max_new_tokens": 100,   # ✅ Ensures short and relevant response
            "temperature": 0.3,      # ✅ Low temp for factual responses
            "top_p": 0.85,           # ✅ Balanced randomness
            "repetition_penalty": 1.5,  # ✅ Avoids repetitive text
            "truncation": "only_first",  # ✅ CORRECT parameter to truncate input
        },
    }

    max_retries = 5
    retry_delay = 20  # ✅ Increased model loading time

    for attempt in range(max_retries):
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

        print("\n📡 Hugging Face API Response:", response.status_code, response.text)

        if response.status_code == 200:
            result = response.json()
            generated_text = result[0].get("generated_text", "").strip()

            # ✅ Extract only the last part of the response (most relevant)
            relevant_part = generated_text.split("\n\n")[-1]  

            if not relevant_part or len(relevant_part) < 10:
                print("⚠️ Short response received! Retrying...")
                time.sleep(5)
                continue

            return relevant_part  # ✅ Returns **only the relevant product answer**

        elif response.status_code == 503:
            print(f"⚠️ Model is still loading. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

        elif response.status_code == 429:
            print("⚠️ Rate limit reached. Retrying...")
            time.sleep(10)

        else:
            raise ValueError(f"API Error {response.status_code}: {response.text}")

    raise ValueError("❌ ERROR: Maximum retry attempts reached!")


# ✅ Function to Generate AI Response with Retrieval
def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={'k': 5})

    def chat_function(query):
        context_documents = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in context_documents[:5]])
        response = huggingface_chat(query, context)
        return response

    return chat_function

# ✅ Main Function for Testing
if __name__ == '__main__':
    vstore = ingestdata('done')
    chain = generation(vstore)

    query = "tell me the best earbuds for the bass ?"
    print(f"\n🛒 Query: {query}")
    
    response = chain(query)
    print(f"\n🤖 AI Response: {response}")
