# import json
# from django.shortcuts import render
# from django.http import JsonResponse
# from rest_framework.decorators import api_view, parser_classes
# from rest_framework.parsers import JSONParser
# import tensorflow as tf
#
# @api_view(["GET"])
# @parser_classes([JSONParser])
# def users(request):
#     resp = ''
#     return JsonResponse({'result': resp})