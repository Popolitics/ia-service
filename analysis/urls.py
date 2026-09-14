from django.urls import path

from analysis.views import health

urlpatterns = [
    path("health/", health, name="health"),
]
