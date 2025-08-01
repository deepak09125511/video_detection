import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # You can use yolov8s.pt for better accuracy

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Video input/output
input_path = r"C:\Users\ajlan\OneDrive\Desktop\Tinker\video_detection\vedios\test_vid1.mp4"
output_path = r"C:\Users\ajlan\OneDrive\Desktop\Tinker\video_detection\outputprocessed_video.mp4"
cap = cv2.VideoCapture(input_path)

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)[0]
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0:  # person class
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw boxes
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0,255,0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Detection + Tracking completed.")
