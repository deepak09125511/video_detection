import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import mediapipe as mp
from captioning.caption_engine import get_funny_caption

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # Small, fast model

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Persistent captions
person_captions = {}

# Video input/output
input_path = r"D:\video_detection\vedios\test_vid1.mp4"
output_path = r"D:\video_detection\src\Output\processed_video.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
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

    # For each tracked person
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        # Extract region of interest for pose estimation
        person_roi = frame[int(t):int(t+h), int(l):int(l+w)]
        if person_roi.size == 0:
            continue

        # Run MediaPipe Pose on person
        person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        results = pose.process(person_rgb)

        # Get or assign funny caption
        if track_id not in person_captions:
            if results.pose_landmarks:
                person_captions[track_id] = get_funny_caption(results.pose_landmarks.landmark)
            else:
                person_captions[track_id] = "🤔"

        caption = person_captions[track_id]

        # Draw bounding box
        cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw caption
        cv2.putText(frame, caption, (int(l), int(t) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()
print("✅ Detection + Tracking + Persistent Captioning completed.")


