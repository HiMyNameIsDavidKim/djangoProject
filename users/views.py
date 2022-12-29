import json
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.authtoken.models import Token
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


@api_view(['POST'])
@parser_classes([JSONParser])
def login(request):
    try:
        print(f"로그인 정보: {request.data}")
        loginInfo = request.data
        loginUser = Users.objects.get(username=loginInfo['username'])
        print(f"회원 ID: {loginUser.id}")
        if loginUser.password == loginInfo["password"]:
            dbUser = Users.objects.all().filter(id=loginUser.id).values()[0]
            print(f'DB 정보: {dbUser}')
            serializer = UserSerializer(loginUser, many=False)
            return JsonResponse(data=serializer.data, safe=False)
        else:
            return Response("로그인 실패")
    except:
        return Response("로그인 실패")