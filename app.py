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

from langchain_groq import ChatGroq

import os
os.makedirs("json", exist_ok=True)
cached_llm = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

app = Flask(__name__)

chat_history = []

folder_path = "db"

embedding = FastEmbedEmbeddings()

# Sanskrit Prompt tailored for the Insights Assistant
raw_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """त्वं एकः सहायकः असि यः संस्कृतभाषायाम् विशेषज्ञः। त्वम् केवलं संस्कृतभाषायामेव उत्तरं ददासि। 
         यदि प्रश्नः असंगतं वा अपर्याप्तं स्यात्, तर्हि "प्रश्नः स्पष्टः नास्ति" इत्येव उत्तरं ददासि।"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "प्रश्नः: {input}\nप्रसङ्गः: {context}"),
    ]
)


@app.route("/ask_json", methods=["POST"])
def ask_json():
    json_content = request.json
    question = json_content.get("question", "")
    if not question:
        return {"answer": "प्रश्नः स्पष्टः नास्ति।"}, 400

    print(f"प्रश्नः प्राप्तः: {question}")

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
                "प्रसङ्गं अन्वेष्टुं कृपया प्रसङ्गानुसारं शोधप्रश्नं उत्पादय।"
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
        "input": question,
        "context": "",  # Placeholder for now; will be populated by the retriever
        "chat_history": chat_history,  # Include the chat history
    }

    # Execute the chain
    result = retrieval_chain.invoke(input_data)
    print(f"उत्तरम्: {result['answer']}")

    # Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))

    # Return result
    response = {"answer": result["answer"]}
    return response


def start_app():
    app.run(host="0.0.0.0", port=80, debug=False)


if __name__ == "__main__":
    start_app()
