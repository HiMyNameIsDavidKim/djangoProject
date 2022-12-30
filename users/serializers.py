from rest_framework import serializers
from .models import Users

class UserSerializer(serializers.ModelSerializer):
    username = serializers.CharField()
    password = serializers.CharField()
    created_at = serializers.CharField()
    rank = serializers.CharField()
    point = serializers.CharField()
    token = serializers.CharField()

    class Meta:
        model = Users
        fields = '__all__'

    def create(self, validated_data):
        return Users.objects.create(**validated_data)

    def update(self, instance, validated_data):
        Users.objects.filter(pk=instance.id).update(**validated_data)

    def delete(self, validated_data):
        pass