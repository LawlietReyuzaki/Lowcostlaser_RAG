import os
import json
import time
import pinecone
import openai
from tqdm import tqdm
from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

index_name = "rag-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=1536, metric="cosine", spec=spec)
index = pc.Index(index_name)

# Initialize OpenAI API
openai.api_key = openai_api_key
embed_model = "text-embedding-ada-002"
chunk_file_path = "document_chunks.json"

# Functions for indexing and retrieval
def get_embedding_from_text(text):
    response = openai.Embedding.create(input=text, engine=embed_model)
    return response['data'][0]['embedding']

def save_chunks_to_local_storage(chunks):
    with open(chunk_file_path, 'w') as f:
        json.dump(chunks, f)

def load_chunks_from_local_storage():
    if os.path.exists(chunk_file_path):
        with open(chunk_file_path, 'r') as f:
            return json.load(f)
    return None

def index_document(txt_file):
    stored_chunks = load_chunks_from_local_storage()
    if stored_chunks:
        for chunk_id, data in tqdm(stored_chunks.items()):
            index.upsert(vectors=[(chunk_id, data["embedding"], {"text": data["text"]})])
        return

    with open(txt_file, 'r', encoding='unicode_escape') as file:
        document = file.read()
    chunks = [document[i:i+512] for i in range(0, len(document), 512)]

    chunk_data = {}
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        embedding = get_embedding_from_text(chunk)
        chunk_id = f"{txt_file}_chunk_{i}"
        chunk_data[chunk_id] = {"embedding": embedding, "text": chunk}
        index.upsert(vectors=[(chunk_id, embedding, {"text": chunk})])

    save_chunks_to_local_storage(chunk_data)

def retrieve(query, top_k=3, limit=3750):
    res = openai.Embedding.create(input=[query], engine=embed_model)
    xq = res['data'][0]['embedding']

    contexts = []
    time_waited = 0
    while len(contexts) < top_k and time_waited < 60 * 12:
        res = index.query(vector=xq, top_k=top_k, include_metadata=True)
        contexts.extend([x['metadata']['text'] for x in res['matches']])
        if len(contexts) >= top_k:
            break
        time.sleep(15)
        time_waited += 15

    if time_waited >= 60 * 12:
        contexts = ["No contexts retrieved. Try to answer the question yourself!"]

    prompt_start = "Answer the question based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    
    for i in range(1, len(contexts) + 1):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = prompt_start + "\n\n---\n\n".join(contexts[:i-1]) + prompt_end
            break
    else:
        prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end

    return prompt

def complete(prompt):
    sys_prompt ='''You are a customer service chatbot for LowCostLasers. Your primary role is to assist customers by answering questions related to products, services, shipping, returns, and payment options based on the provided data. When asked about product details, shipping times, or return policies, respond accurately using the available information.
Persona and Interaction Guidelines
Identity: You are a helpful, interactive assistant. You greet users with varied, friendly introductions such as:

“Hello! How can I assist you today?”
“Hi there! Ask me anything.”
“Welcome! How can I help you today?”
Conversational Engagement:

Use personalized greetings to make the interaction feel warm.
Incorporate different ways of saying goodbye at the end of conversations, such as:
“Thank you for reaching out! Have a great day!”
“It was a pleasure assisting you. Goodbye!”
“Take care! Feel free to reach out anytime.”

Data Reliance: Use only the provided data to answer questions. Do not explicitly mention that you are relying on this data.
Stay Focused: If the user attempts to divert the conversation to unrelated topics, politely redirect them to customer service or sales queries.
Fallback Response: If a question cannot be answered based on the available data, use the fallback response provided above.
Role Limitation: You are not permitted to answer queries outside of your assigned role or unrelated subjects like coding or personal advice.

Final Note:
Continue to enhance the experience by ensuring smooth transitions and a helpful tone throughout each interaction.

 If the details are not covered, respond with:
"Apologies, I do not have that information. Please contact our support team at Michael@lowcostlasers.com or call 786-357-1224 for further assistance."

.'''
    res = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return res['choices'][0]['message']['content'].strip()


# Flask routes
def index_document_route():
    #data = request.get_json()
    #txt_file = data.get('txt_file')
    #index_document(txt_file)
    index_document('lowcostlasers_data.txt')
    #return jsonify({"status": "Document indexed successfully"})

@app.route('/query', methods=['POST'])
def query_route():
    data = request.get_json()
    query = data.get('query')
    prompt = retrieve(query)
    response = complete(prompt)
    return jsonify({"response": response})

# Run the app
if __name__ == '__main__':
    index_document_route()
    app.run(host='0.0.0.0', port=5000)
