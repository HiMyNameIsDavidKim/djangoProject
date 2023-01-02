import json
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf
from rest_framework.response import Response
from users.models import Users
from users.repositories import UserRepository
from users.serializers import UserSerializer


@api_view(["GET"])
@parser_classes([JSONParser])
def list(request): return UserRepository().users()


@api_view(['POST'])
@parser_classes([JSONParser])
def login(request): return UserRepository().login(request.data)


######
@api_view(['POST', 'PUT', 'PATCH', 'DELETE', 'GET'])
@parser_classes([JSONParser])
def combo(request):
    if request.method == "POST":
        new_user = request.data
        print(f'new user from react: {new_user}')
        serializer = UserSerializer(data=new_user)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse({"result": "success"})
        return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == "GET":
        return Response(UserRepository().find_by_username(request.data))

    elif request.method == "PATCH":
        return None

    elif request.method == "PUT":
        modify_user = UserRepository().find_by_username(request.data['username'])
        db_user = UserRepository().find_by_id(modify_user.id)
        serializer = UserSerializer(data=db_user)
        if serializer.is_valid():
            serializer.update(modify_user, db_user)
            return JsonResponse({"result": "SUCCESS"})

    elif request.method == "DELETE":
        delete_user = UserRepository().find_by_username(request.data['username'])
        db_user = UserRepository().find_by_id(delete_user.id)
        db_user.delete()
        return JsonResponse({"result": "success"})