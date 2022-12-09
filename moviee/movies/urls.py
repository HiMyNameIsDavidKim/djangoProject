from django.urls import re_path as url
from moviee.movies import views

urlpatterns = [
    url(r'fake-faces', views.faces)
]