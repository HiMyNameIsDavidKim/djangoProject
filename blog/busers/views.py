from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

@api_view(['POST'])
@parser_classes([JSONParser])
def login(request):
    user_info = request.data
    username = user_info['username']
    password = user_info['password']
    print(f'Data from react {user_info}')
    print(f'Username from react {username}')
    print(f'Password from react {password}')
    return JsonResponse({'Response Test': 'SUCCESS'})