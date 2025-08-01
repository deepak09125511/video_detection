from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from pathlib import Path
from detection_app.detector_runner import process_video
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import HttpResponseBadRequest 
# Create your views here.

from django.http import HttpResponseBadRequest  # add this import

def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video')  # safer access

        if not video_file:
            return HttpResponseBadRequest("⚠️ No video file uploaded or field name mismatch.")

        input_path = os.path.join(settings.MEDIA_ROOT, video_file.name)
        with open(input_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        output_filename = f"processed_{video_file.name}"
        output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

        # Run processing
        process_video(input_path, output_path)

        return render(request, 'result.html', {
            'output_video': output_filename
        })

    return render(request, 'upload.html')
