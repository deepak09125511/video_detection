import os
import cv2
import sys
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe_silicon as mp

# Add path to captioning folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from detection_app.captioning.caption_engine import get_funny_caption

# Initialize models globally (safe)
model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=30)

# Persistent caption dictionary
person_captions = {}

def process_video(input_path, output_path):
    print("🎬 Input Path:", input_path)
    print("💾 Output Path:", output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception(f"❌ Failed to open input video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("📽️ FPS from input:", fps)
    if fps == 0:
        fps = 30
        print("⚠️ FPS was zero, set default to 30")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("🖼️ Frame size:", width, height)

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("❌ VideoWriter failed to open. Check codec or output path.")
        return
    else:
        print("✅ VideoWriter initialized successfully")

    # ✅ Create a NEW pose instance per call
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_written = False  # Track if at least one frame is written

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame or end of video")
            break
        print("✅ Frame read")

        # YOLO detection
        results = model(frame)[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            if int(cls) == 0:  # person
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        print(f"🧍 Detected {len(detections)} people")

        # DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)
        print(f"🎯 Tracking {len(tracks)} objects")

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = track.to_ltrb()

            # Clip bounding box to stay within frame
            l = max(0, int(l))
            t = max(0, int(t))
            r = min(frame.shape[1], int(l + w))
            b = min(frame.shape[0], int(t + h))

            person_roi = frame[t:b, l:r]
            if person_roi.shape[0] == 0 or person_roi.shape[1] == 0:
                print(f"⚠️ Empty ROI for ID {track_id}, skipping...")
                continue

            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_rgb)

            if track_id not in person_captions:
                if results_pose.pose_landmarks:
                    person_captions[track_id] = get_funny_caption(results_pose.pose_landmarks.landmark)
                else:
                    person_captions[track_id] = "🤔"

            caption = person_captions[track_id]
            print(f"📝 Caption for ID {track_id}: {caption}")

            # Draw results on frame
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, caption, (l, t - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        out.write(frame)
        frame_written = True

    cap.release()
    out.release()
    pose.close()
    cv2.destroyAllWindows()

    if not frame_written:
        print("⚠️ No frames were written! Output video will be empty or corrupted.")
    else:
        print("✅ Detection + Tracking + Captioning completed and video saved.")
