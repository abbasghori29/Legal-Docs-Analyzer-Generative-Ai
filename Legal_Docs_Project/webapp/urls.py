from django.urls import path,include
from . import views

urlpatterns = [
    path("",views.index),
    path("upload_pdf",views.uploadPdf,name="upload_pdf")
]

