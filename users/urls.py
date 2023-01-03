from django.urls import re_path as url, path
from users import views

urlpatterns = [
    url(r'list', views.list),
    url(r'login', views.login),
    # url(r'combo', views.combo),
]