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

class UserRepository(object):
    def users(self):
        return Response(UserSerializer(Users.objects.all(), many=True).data)

    def login(self, kwargs):
        print(f'로그인 정보 : {kwargs}')
        loginUser = Users.objects.get(username=kwargs['username'])
        print(f"회원 ID: {loginUser.id}")
        if loginUser.password == kwargs["password"]:
            dbUser = Users.objects.all().filter(id=loginUser.id).values()[0]
            print(f'DB 정보: {dbUser}')
            serializer = UserSerializer(loginUser, many=False)
            return JsonResponse(data=serializer.data, safe=False)
        else:
            return Response("로그인 실패")

    def get_all(self):
        return Response(UserSerializer(Users.objects.all(), many=True).data)

    def find_by_id(self, id):
        return Users.objects.all().filter(id=id).values()[0]

    def find_by_username(self, username):
        return Users.objects.all().filter(username=username).values()[0]