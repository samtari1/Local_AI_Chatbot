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
    response = RAG(userInput)
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
            add_start_index=True,)
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    def save_to_faiss(chunks):
        # Clear out the database first
        if os.path.exists(FAISS_PATH):
            shutil.rmtree(FAISS_PATH)
        # Create embeddings using Ollama's local model
        embeddings = OllamaEmbeddings(model=LLM_Model)
        # Initialize FAISS vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        # Save FAISS index
        vector_store.save_local(FAISS_PATH)
        print(f"Saved {len(chunks)} chunks to {FAISS_PATH} using FAISS.")

    # Dynamically point to the 'data' folder in the same directory as this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
    DATA_PATH = os.path.join(BASE_DIR, "data")

    generate_data_store()    
    response = {'response': "Chatbot trained successfully!"}  # Pass response to template
    return render(request, 'customChatbot.html', response)  # Render the HTML page with response  



def weather(request):
    try:
        ip_address = request.GET.get('ip_address')  # Default IP address if not provided
        tone = request.GET.get('tone', 'normal')  # Default tone if not provided
        
        # Get coordinates based on IP address
        coordinate = get_coordinate_from_ip(ip_address)
        
        # Fetch weather forecast
        foreCast = requests.get(f"https://api.weather.gov/points/{coordinate[0]},{coordinate[1]}")
        
        # Fetch location details (city, state, country)
        location_info = get_ip_geolocation(ip_address)
        city = location_info.get('city', 'Unknown City')
        state = location_info.get('region', 'Unknown State')
        country = location_info.get('country', 'Unknown Country')

        location_str = f"{city}, {state}, {country}"

        # Fetch weather forecast details
        forecastUrl = foreCast.json()["properties"]["forecast"]
        forecastContent = requests.get(f"{forecastUrl}").json()
        first_period_detailed_forecast = forecastContent["properties"]["periods"][0]["detailedForecast"]
        
        # Construct query for LLM
        query = f"""
        You are a helpful assistant. Given the following weather conditions:
        {first_period_detailed_forecast},
        please provide a suggestion on what to wear.
        Note: say it in a {tone} way.
        """

        # Generate response from LLM
        response_content = LLM(query)
        
        # Pass the response and location details to the template
        context = {
            'response': response_content,
            'location': location_str
        }
        # Render the HTML page with the response and location information
        return render(request, 'weather.html', context)
    except Exception as e:
        print(f"An error occurred: {e}")
        return render(request, 'weather.html', {'error': 'An error occurred. Please try again later.'})


def get_coordinate_from_ip(ip_address):
    try:
        response = requests.get(f"https://ipinfo.io/{ip_address}/json")
        data = response.json()
        
        if "loc" in data:
            loc = data["loc"]
            latitude, longitude = loc.split(",")
            return float(latitude), float(longitude)
        else:
            print("Location information is not available for this IP address.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_ip_geolocation(ip_address):
    url = f'http://ip-api.com/json/{ip_address}'
    
    response = requests.get(url)
    data = response.json()
    
    if data['status'] == 'success':
        return {
            'ip': data.get('query'),
            'city': data.get('city'),
            'region': data.get('regionName'),
            'country': data.get('country'),
            'location': f"{data.get('lat')}, {data.get('lon')}",
            'postal': data.get('zip'),
            'timezone': data.get('timezone')
        }
    else:
        return {"error": "Could not retrieve data"}
