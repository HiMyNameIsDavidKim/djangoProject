from rest_framework.response import Response
from rest_framework.decorators import api_view

@api_view(['GET'])
def hello(request):
    print('######## 0 ########')
    return Response({'message': 'server is started.'})