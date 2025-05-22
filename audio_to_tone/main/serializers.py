from rest_framework import serializers

class AudioFileSerializer(serializers.Serializer):
    file = serializers.FileField()