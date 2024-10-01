from django.shortcuts import render 
import ollama
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from django.core.files.storage import default_storage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests
from django.http import JsonResponse

LLM_Model = 'gemma2:2b'

def LLM(prompt):
    response = ollama.chat(model=LLM_Model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def helloFunction(request):
    userInput = request.GET.get('userInput')
    response = LLM(userInput)
    response = {"response": response}
    return render(request, 'chatbot.html', response)

def customChatbot(request):
    userInput = request.GET.get('userInput')
    
    # Check if the FAISS index exists
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index", "index.faiss")  # Add 'index.faiss' to path

    if os.path.exists(FAISS_PATH):
        # If the FAISS index exists, proceed with the RAG function
        response = RAG(userInput)
    else:
        # If the FAISS index does not exist, prompt the user to train the chatbot first
        response = "Please train the chatbot first."

    response = {"response": response}
    return render(request, 'customChatbot.html', response)

def upload_files(request):
    if request.method == 'POST' and request.FILES.getlist('files[]'):
        # Get the list of uploaded files
        uploaded_files = request.FILES.getlist('files[]')
        # Define the base directory where files will be saved
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(BASE_DIR, 'data')
        # Ensure the 'data' directory exists
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        # Process each file
        success_messages = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            # Save the file to the 'data' directory
            with default_storage.open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            success_messages.append(f'File "{uploaded_file.name}" uploaded successfully!')
        # Return a response indicating success for all files
        alert = {'response': success_messages}
        return render(request, 'customChatbot.html', alert)
    else:
        # If no files were uploaded, return an error
        alert = {'error': 'No files uploaded.'}
        return render(request, 'customChatbot.html', alert)

def RAG(user_input):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
    # Set up the local model
    llm = Ollama(model=LLM_Model)
    # Load the FAISS index with allow_dangerous_deserialization set to True
    vector_store = FAISS.load_local(
        FAISS_PATH, 
        OllamaEmbeddings(model=LLM_Model), 
        allow_dangerous_deserialization=True
    )
    # Set up the query prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question only based on the given context:
        <context>
        {context}
        </context>
        Question: {input}
        """)
    # Retrieve context from vector store
    docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, docs_chain)
    # Perform the query
    response = retrieval_chain.invoke({"input": user_input})
    print(":::QUERY RESPONSE:::")
    print(response["answer"])
    return response["answer"]

def train(request):
    def generate_data_store():
        documents = load_documents()
        chunks = split_text(documents)
        save_to_faiss(chunks)
        return JsonResponse({'status': 'success'})

    def load_documents():
        # Load documents from a local directory
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        return documents

    def split_text(documents):
        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def save_to_faiss(chunks):
        # Clear out the database first
        if os.path.exists(FAISS_PATH):
            shutil.rmtree(FAISS_PATH)
        embeddings = OllamaEmbeddings(model=LLM_Model)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_PATH)
        print(f"Saved {len(chunks)} chunks to {FAISS_PATH} using FAISS.")

    # Dynamically point to the 'data' folder in the same directory as this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
    DATA_PATH = os.path.join(BASE_DIR, "data")

    generate_data_store()
    response = {'response': "Chatbot trained successfully!"}
    return render(request, 'customChatbot.html', response)
