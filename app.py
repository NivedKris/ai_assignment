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
from typing import Iterable, List
from PyPDF2 import PdfReader

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


load_dotenv()


# Fetch the API key and model name from environment variables
key = os.getenv("API_KEY")
sanskrit_llm = os.getenv("LLM_NAME")


os.makedirs("pdfs", exist_ok=True)


cached_llm = ChatGroq(temperature=0, groq_api_key=key, model_name=sanskrit_llm, max_tokens=1000)


app = Flask(__name__)

chat_history = []

folder_path = "db"

embedding = FastEmbedEmbeddings()

# Prompt tailored for the Insights Assistant in Sanskrit
# Prompt tailored for the Insights Assistant (in Sanskrit)
raw_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are conversing with an AI assistant who only knows Sanskrit. The assistant is here to help you with your queries, but the user only knows Sanskrit. The use of any other language is strictly prohibited, so only respond in Sanskrit."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "प्रश्नः: {input}\nप्रसङ्गः: {context}"),
    ]
)



def process_pdf_to_documents(file_path):
    """Process a PDF file into structured chunks as Document objects."""
    reader = PdfReader(file_path)
    documents = []

    for page in reader.pages:
        content = page.extract_text()
        if content:
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

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files["file"]
    file_name = file.filename
    save_path = f"pdfs/{file_name}"
    file.save(save_path)
    print(f"File saved: {file_name}")

    # Process the PDF file
    docs = process_pdf_to_documents(save_path)
    print(f"Processed {len(docs)} documents from the PDF file.")

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

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
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
                "सन्दर्भाधारितं सम्बन्धितसूचनां प्राप्तुं अन्वेषणप्रश्नं उत्पादय।"
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

app.run(port=5000)
