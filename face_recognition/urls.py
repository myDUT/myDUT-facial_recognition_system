from django.urls import path
from .views import FaceRecognitionView, SyncPersonalDataView

urlpatterns = [
    path('recognize', FaceRecognitionView.as_view(), name='face-recognition'),
    path('sync-personal-data', SyncPersonalDataView.as_view(), name='sync-personal-data')
]