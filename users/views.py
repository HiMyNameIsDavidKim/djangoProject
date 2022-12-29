import json
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf
from rest_framework.response import Response
from users.models import Users
from users.serializers import UserSerializer


@api_view(["GET"])
@parser_classes([JSONParser])
def users(request):
    if request.method == 'GET':
        serializer = UserSerializer(Users.objects.all(), many=True)
        return Response(serializer.data)


def login(request):
    pass