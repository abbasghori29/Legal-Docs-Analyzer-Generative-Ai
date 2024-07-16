from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os
# Create your views here.

def index(request):
    return render(request,"index.html")


@csrf_exempt
@require_http_methods(["POST"])

def uploadPdf(request):
    print("request received")
    
    # Define the path where you want to save the uploaded files
    upload_dir = '../uploads'  # Replace with your desired path
    # os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists
    
    # Check if the request has a file
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    uploaded_file = request.FILES['file']
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    # Save the uploaded file to the specified path without chunking
    with open(file_path, 'wb') as destination:
        destination.write(uploaded_file.read())
    return JsonResponse({"Success ": True},status=200)
    

