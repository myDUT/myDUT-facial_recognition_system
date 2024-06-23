import os.path
import shutil

import boto3
from minio import Minio
from botocore.exceptions import ClientError
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import FaceRecognitionInLesson, SyncPersonalDataSerializer
from .services import recognize_faces_from_urls, call_to_check_in


class FaceRecognitionView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = FaceRecognitionInLesson(data=request.data)
        if serializer.is_valid():
            class_id = serializer.validated_data['class_id']
            class_code = serializer.validated_data['class_code']
            lesson_id = serializer.validated_data['lesson_id']

            bucket_name = settings.MINIO_STORAGE_BUCKET_NAME
            key_prefix = 'recognition-data/' + class_code + '_' + class_id + '/' + lesson_id + '/' + 'raw'
            upload_path = 'recognition-data/' + class_code + '_' + class_id + '/' + lesson_id + '/' + 'checked'

            # Initialize S3 client
            s3_client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY_ID,
                secret_key=settings.MINIO_SECRET_ACCESS_KEY,
                secure=settings.MINIO_USE_SSL,
            )

            try:
                # List objects in s3 bucket with prefix
                objects = s3_client.list_objects(bucket_name=bucket_name, prefix=key_prefix, recursive=True)

                raw_images = []

                for obj in objects:
                    key = obj.object_name

                    response = s3_client.get_object(bucket_name, key)

                    raw_image = response.read()
                    raw_images.append(raw_image)

                recognized_student = set(recognize_faces_from_urls(raw_images, upload_path))

                # Call to Backend service to check in for recognized student
                call_to_check_in(recognized_student=recognized_student, class_id=class_id, class_code=class_code, lesson_id=lesson_id)

                return Response({'recognized_student': recognized_student}, status=status.HTTP_200_OK)

            except ClientError as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SyncPersonalDataView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = SyncPersonalDataSerializer(data=request.data)
        if serializer.is_valid():
            student_code = serializer.validated_data['student_code']
            bucket_name = settings.MINIO_STORAGE_BUCKET_NAME
            key_prefix = 'user-data/' + student_code + '/'

            # Initialize S3 client
            s3_client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY_ID,
                secret_key=settings.MINIO_SECRET_ACCESS_KEY,
                secure=settings.MINIO_USE_SSL,
            )

            try:
                # List objects in s3 bucket with prefix
                objects = s3_client.list_objects(bucket_name=bucket_name, prefix=key_prefix, recursive=True)

                # Check exist folder to download, if exist remove
                folder_download = f'./personal_data/{student_code}'

                # Create the folder if not exist
                os.makedirs(folder_download, exist_ok=True)

                for filename in os.listdir(folder_download):
                    file_path = os.path.join(folder_download, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):  # uncomment if you want to remove directories as well
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")

                # if os.path.exists(folder_download):
                #     os.remove(folder_download)

                for obj in objects:
                    key = obj.object_name

                    # Download each file to local
                    download_path = f'./personal_data/{student_code}/{key.split("/")[-1]}'

                    s3_client.fget_object(bucket_name, key, download_path)

                return Response({"message": "Personal Data Download Complete"}, status=status.HTTP_200_OK)
            except ClientError as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
