# Human Motion Detection & Captioning System

## Basic Details
### Team Name: 404 Squad

### Team Members
- Team Lead: Deepak - TKM College of Engineering
- Member 2: Ajlan - TKM College of Engineering

### Project Description
A video processing web app that detects human motion and automatically generates funny captions based on pose estimation and tracking.

### The Problem (that doesn't exist)
People watching CCTV or surveillance footage don’t get enough *laughs*. Videos are serious — where are the memes?

### The Solution (that nobody asked for)
We turn boring video footage into entertainment by automatically identifying what each person might be doing and slapping a funny caption on them.

---

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
