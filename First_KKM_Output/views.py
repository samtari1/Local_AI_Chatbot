from django.shortcuts import render
import ollama

LLM_Model = 'gemma2:2b'

def LLM(prompt):
    response = ollama.chat(model=LLM_Model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def helloFunction(request):
    response = LLM("Hello world!")
    response = {"response": response}
    return render(request, 'chatbot.html', response)
