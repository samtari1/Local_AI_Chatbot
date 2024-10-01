from django.shortcuts import render

def helloFunction(request):
    response = {"response": "Hello World!"}
    return render(request, 'chatbot.html', response)
