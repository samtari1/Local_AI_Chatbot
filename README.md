
# AI Chatbot Project

## Overview
This project demonstrates how to build a web-based AI-powered chatbot using Django, Ollama, and a large language model (LLM). The chatbot interacts with users and can be customized to respond to various prompts, including using files for chatbot training.

## Features
- Create a Django-based web application.
- Integrate Ollama to power the chatbot using a large language model (LLM).
- Allow users to interact with the chatbot via a web interface.
- Upload and use documents to train the chatbot.
- Generate responses based on user input and LLM.

## Setup Instructions

### 1. Prerequisites
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) for managing environments.
- Install Ollama from [Ollama](https://ollama.com/download).

### 2. Environment Setup

#### Using `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate LLM
```

#### Using `requirements.txt` file:
```bash
conda create --name LLM
conda activate LLM
pip install -r requirements.txt
```

### 3. Project Setup

#### Create Django Project
```bash
mkdir LLM
cd LLM
django-admin startproject ChatbotProject .
```

#### Verify the Project
```bash
python manage.py runserver
```
Access `http://127.0.0.1:8000/` to verify the project is running.

### 4. App Setup
Create a Django app:
```bash
python manage.py startapp ChatbotApp
```

Add `ChatbotApp` to the `INSTALLED_APPS` in `settings.py`:
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'ChatbotApp',
]
```

### 5. Chatbot Integration
Create a basic chatbot using Ollama:
- In `views.py` of `ChatbotApp`:
```python
from django.shortcuts import render
import ollama

LLM_Model = 'gemma2:2b'

def LLM(prompt):
    response = ollama.chat(model=LLM_Model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def helloFunction(request):
    userInput = request.GET.get('userInput')
    response = LLM(userInput)
    return render(request, 'chatbot.html', {"response": response})
```

### 6. Customizing the Chatbot
You can upload documents and train the chatbot using these:
- Create a template `customChatbot.html` for user interaction with the chatbot.
- Use FAISS indexing to improve document search and response.

### 7. Running the Chatbot
Run the Django server and test the chatbot at:
```bash
python manage.py runserver
```

### 8. Training the Chatbot
Train the chatbot by uploading documents:
1. Go to the custom chatbot page.
2. Upload your documents.
3. Train the chatbot to respond using the uploaded data.

## License
This project is licensed under the MIT License.
