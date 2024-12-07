import json
from flask import Flask, request
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.schema import Document
from typing import Iterable, List, Optional

from langchain_groq import ChatGroq

cached_llm =ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")



app = Flask(__name__)

chat_history = []

folder_path = "db"

embedding = FastEmbedEmbeddings()

# Prompt tailored for the Insights Assistant
raw_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an Insights Assistant specialized in analyzing and retrieving 
         mathematical and contextual insights from structured data sources like invoices. 
         Use the structured format to answer queries with precise insights. 
         If the query is irrelevant or lacks sufficient data, respond accordingly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Query: {input}\nContext: {context}"),
    ]
)


def flatten_json(json_obj, parent_key='', sep='.'):
    """Flatten a nested JSON object, retaining its structure in a key-value format."""
    items = []
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(flatten_json(item, f"{new_key}[{i}]", sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            items.extend(flatten_json(item, f"{parent_key}[{i}]", sep=sep).items())
    else:
        items.append((parent_key, json_obj))
    return dict(items)

def process_json_to_documents(file_path):
    """Process a JSON file into structured chunks as Document objects."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    documents = []
    for entry in data:
        flattened_entry = flatten_json(entry)  # Flatten the JSON for structured embedding
        content = "\n".join([f"{key}: {value}" for key, value in flattened_entry.items()])
        documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

class CustomTextSplitter(RecursiveCharacterTextSplitter):
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents into chunks."""
        texts, metadatas = [], []
        for doc in documents:
            if isinstance(doc, Document):
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")
        return self.create_documents(texts, metadatas=metadatas)

@app.route("/upload_json", methods=["POST"])
def upload_json():
    file = request.files["file"]
    file_name = file.filename
    save_path = f"json/{file_name}"
    file.save(save_path)
    print(f"File saved: {file_name}")

    # Process the JSON file
    docs = process_json_to_documents(save_path)
    print(f"Processed {len(docs)} documents from the JSON file.")

    # Chunk documents logically
    text_splitter = CustomTextSplitter(chunk_size=1024, chunk_overlap=80)
    chunks = text_splitter.split_documents(docs)

    # Embed and persist
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response

@app.route("/ask_json", methods=["POST"])
def ask_json():
    json_content = request.json
    query = json_content.get("query")
    print(f"Query received: {query}")

    # Load vector store
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.1})

    # History-aware retrieval chain
    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Based on the context, generate a search query for retrieving relevant information."
            ),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=cached_llm, retriever=retriever, prompt=retriever_prompt
    )

    # Create the document chain with updated prompt
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Prepare input for the chain
    input_data = {
        "input": query,
        "context": "",  # Placeholder for now; will be populated by the retriever
        "chat_history": chat_history,  # Include the chat history
    }

    # Execute the chain
    result = retrieval_chain.invoke(input_data)
    print(result["answer"])

    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))

    # Return result
    response = {"answer": result["answer"]}
    return response



def start_app():
    app.run(host="0.0.0.0", port=80, debug=False)

if __name__ == "__main__":
    start_app()
