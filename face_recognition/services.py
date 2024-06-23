import os
import cv2
import io
import uuid
import requests
from minio import Minio
from minio.error import S3Error
from django.conf import settings

from ultralytics import YOLO
from ultralytics.engine.results import Results
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

import numpy as np


def recognize_faces_from_urls(images, upload_path):
    recognized_student = []
    for raw_image in images:
        # Doc anh su dung Pillow
        # image = Image.open(io.BytesIO(img_data))
        image = Image.open(io.BytesIO(raw_image))
        detect_face(image, recognized_student, upload_path)

    return recognized_student


def detect_face(input_image, recognized_student, upload_path):
    model = YOLO(f'./model/best.pt')
    results: Results = model.predict(input_image)[0]

    return extract_face(input_image, results, recognized_student, upload_path)


def extract_face(input_image, results, recognized_student, upload_path):
    draw = ImageDraw.Draw(input_image)
    detected_objects = []

    if hasattr(results, 'boxes') and hasattr(results, 'names'):
        for box in results.boxes.xyxy:
            object_id = int(box[-1])
            object_name = results.names.get(object_id)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Crop the detected face
            face_image = input_image.crop((x1, y1, x2, y2))

            # Convert PIL Image to numpy array
            face_np = np.array(face_image)

            file_to_delete = "./personal_data/representations_facenet512.pkl"
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)

            model = DeepFace.find(img_path=face_np, db_path="./personal_data", enforce_detection=False,
                                  model_name="Facenet512", detector_backend="yolov8",
                                  silent=True)

            # Check if a face was recognized in the image
            if model and len(model[0]['identity']) > 0:
                # Extract the name and append it to the list
                name = model[0]['identity'][0].split('\\')[1]

            else:
                # If no face is recognized, set name to 'unknown'
                name = 'unknown'

            detected_objects.append((object_name, (x1, y1, x2, y2)))
            if name != 'unknown':
                recognized_student.append(name)
                draw_rectangle(draw, (x1, y1, x2, y2), name, color='red', width=4)

    # Save the image with rectangles drawn around the objects
    upload_image_to_minio(input_image=input_image, upload_path=upload_path)

    return 0


def upload_image_to_minio(input_image, upload_path):
    bucket_name = settings.MINIO_STORAGE_BUCKET_NAME

    # Init MinIO client
    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY_ID,
        secret_key=settings.MINIO_SECRET_ACCESS_KEY,
        secure=settings.MINIO_USE_SSL,
    )

    # Create uuid key
    image_filename = upload_path + f"/{uuid.uuid4()}.jpg"

    # Chuyển đổi hình ảnh thành byte
    byte_array = io.BytesIO()
    input_image.save(byte_array, format="JPEG")
    image_data = byte_array.getvalue()

    try:
        # Upload images to MinIO
        minio_client.put_object(
            bucket_name=bucket_name,
            object_name=image_filename,
            data=io.BytesIO(image_data),
            length=len(image_data),
            content_type='image/jpeg'
        )
        print(f"Successfully uploaded {image_filename}")
    except S3Error as err:
        print(f"Failed to upload {image_filename} to MinIO: {err}")


def call_to_check_in(recognized_student, class_id, class_code, lesson_id):
    url = 'http://' + settings.CHECKIN_SERVICE_HOST + ':' + settings.CHECKIN_SERVICE_PORT + '/attendance-records/facial-check-in'
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'studentCodes': list(recognized_student),
        'classId': class_id,
        'classCode': class_code,
        'lessonId': lesson_id,
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')


def draw_rectangle(draw, coordinates, name_face, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle([rect_start, rect_end], outline=color)
        draw.text((coordinates[0], coordinates[1] - 10), name_face, spacing=8)


def recognize_faces(input_image, cnt):
    faces = DeepFace.extract_faces(input_image, detector_backend='yolov8', enforce_detection=False, align=False)

    for face in faces:
        face_img, x, y, w, h = face['face'], face['facial_area']['x'], face['facial_area']['y'], face['facial_area'][
            'w'], face['facial_area']['h']
        # face_img = input_image[y:y + h, x:x + w]

        # Face Recognition
        # try:
        #
        #
        # except Exception as e:
        model = DeepFace.find(img_path=face_img, db_path="facial_database", enforce_detection=False,
                              model_name="Facenet512", detector_backend="yolov8", distance_metric="cosine",
                              silent=True)[0]
        if model and len(model[0]['identity']) > 0:
            name = model[0]['identity'][0].split('\\')[1]

        # Vẽ bounding box và tên lên ảnh
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if name != '':
            cv2.putText(input_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the image with rectangles drawn around the objects
    cv2.imwrite(f"./recognized_face/detected_faced_{cnt}.jpg", input_image)

    return 0
