from django.urls import path
from .views import AudioAnalysisAPI

urlpatterns = [
    path('analyze/', AudioAnalysisAPI.as_view(), name='analyze-audio'),
]