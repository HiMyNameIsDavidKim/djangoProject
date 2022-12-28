import json
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf

from users.services import UserService


@api_view(["GET"])
@parser_classes([JSONParser])
def users(request):
    us = UserService()
    resp = us.get_users()
    print(resp)
    return JsonResponse({'users': resp})


def login(request):
    pass