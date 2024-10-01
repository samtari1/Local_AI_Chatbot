from django.urls import path
from . import views

urlpatterns = [
    path('', views.helloFunction, name='helloFunction'),
    path('customChatbot', views.customChatbot, name='customChatbot'),
    path('upload', views.upload_files, name='upload_files'),
    path('train', views.train, name='train'),
]
