from django.urls import re_path as url, path
from users import views

urlpatterns = [
    url(r'user-list', views.users),
    url(r'login', views.login),
]