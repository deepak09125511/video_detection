# Human Motion Detection & Captioning System

This project is a **Python-based backend system** for detecting humans in a video, tracking them with unique IDs, estimating their body pose using MediaPipe, and generating funny captions based on their posture.

## 🎯 Features

- Human detection using **YOLOv8**.
- Multi-object tracking using **DeepSORT**.
- Pose estimation using **MediaPipe**.
- Funny or descriptive caption generation for each person using a custom `caption_engine`.
- Processed output video with bounding boxes, IDs, and captions.

---

## 🛠️ Backend Technologies Used

| Component         | Library/Tool                | Description |
|------------------|-----------------------------|-------------|
| Detection         | [YOLOv8](https://github.com/ultralytics/ultralytics) | Detects persons in each frame. |
| Tracking          | [DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) | Tracks each detected person with a unique ID. |
| Pose Estimation   | [MediaPipe Pose](https://developers.google.com/mediapipe) | Estimates body landmarks and posture. |
| Captioning Engine | Custom `caption_engine.py` | Converts pose data into humorous or meaningful text. |
| Video Processing  | `cv2` (OpenCV)              | Reads and writes video frames. |

---

## 📂 Directory Structure

```
video_detection/
│
├── backend/
│   └── src/
│       └── detector/
│           └── yolo_deepsort_detect.py     # Main detection pipeline
│
├── captioning/
│   └── caption_engine.py                  # Caption logic using pose keypoints
│
├── vedios/                                 # Input videos here
│   └── test_vid1.mp4
│
├── output/                                 # Processed videos will be saved here
│   └── processed_video.mp4
```

---

## ▶️ How to Run

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd video_detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the input video** in the `vedios/` folder.

4. **Run the detection script**:
   ```bash
   python backend/src/detector/yolo_deepsort_detect.py
   ```

5. **View the processed video** in the `output/` folder.

---

## 📬 Contact

For queries or collaboration, feel free to contact the author.
