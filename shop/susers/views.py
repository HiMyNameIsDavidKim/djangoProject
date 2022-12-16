import json
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf

from shop.susers.fashion_service import FashionService
from shop.susers.iris_service import IrisService


@api_view(['POST'])
@parser_classes([JSONParser])
def iris(request):
    iris_data = request.data
    petal_width_cm = iris_data['petal_width_cm']
    petal_length_cm = iris_data['petal_length_cm']
    sepal_width_cm = iris_data['sepal_width_cm']
    sepal_length_cm = iris_data['sepal_length_cm']
    print(f'Data from react {iris_data}')
    print(f'PetalWidthCm from react {petal_width_cm}')
    print(f'PetalLengthCm from react {petal_length_cm}')
    print(f'SepalWidthCm from react {sepal_width_cm}')
    print(f'SepalLengthCm from react {sepal_length_cm}')
    pwc = tf.constant(float(iris_data['petal_width_cm']))
    plc = tf.constant(float(iris_data['petal_length_cm']))
    swc = tf.constant(float(iris_data['sepal_width_cm']))
    slc = tf.constant(float(iris_data['sepal_length_cm']))
    features = [pwc, plc, swc, slc]
    resp = IrisService().service_model(features)
    return JsonResponse({'result': resp})

@api_view(["GET"])
def fashion(request, id):
    print(f"### GET  ###")
    print(f"React ID is {request.GET['id']}")
    resp = FashionService().service_model(int(request.GET['id']))
    return JsonResponse({'result': resp})