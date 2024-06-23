from rest_framework import serializers


class ImageUrlsSerializer(serializers.Serializer):
    image_urls = serializers.ListField(
        child=serializers.CharField()
    )
    class_code = serializers.CharField()
    lesson_time = serializers.CharField()


class SyncPersonalDataSerializer(serializers.Serializer):
    student_code = serializers.CharField()


class FaceRecognitionInLesson(serializers.Serializer):
    class_code = serializers.CharField()
    class_id = serializers.CharField()
    lesson_id = serializers.CharField()
