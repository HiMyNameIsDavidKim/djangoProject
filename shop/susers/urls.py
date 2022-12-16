from django.urls import re_path as url, path
from shop.susers import views

urlpatterns = [
    url(r'iris', views.iris),
    # url(r'fashion/(?P<id>)$', views.fashion),
    url(r'fashion', views.fashion)
]