from django.urls import re_path as url
from blog.busers import views

urlpatterns = [
    url(r'login', views.login)
]