import json
import random
from mediapipe.python.solutions.pose import PoseLandmark
import os

# ✅ Fixed path construction
json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pose', 'funny_labels.json'))

with open(json_path, "r") as f:
    captions = json.load(f)

def get_funny_caption(landmarks):
    if not landmarks:
        return random.choice(captions["default"])

    def y(part): return landmarks[part].y
    def x(part): return landmarks[part].x

    if abs(x(PoseLandmark.LEFT_WRIST) - x(PoseLandmark.RIGHT_WRIST)) < 0.1:
        return random.choice(captions["hugging"])

    if y(PoseLandmark.LEFT_WRIST) < y(PoseLandmark.NOSE) and y(PoseLandmark.RIGHT_WRIST) < y(PoseLandmark.NOSE):
        return random.choice(captions["hands_up"])

    leg_up = (
        y(PoseLandmark.LEFT_KNEE) < y(PoseLandmark.LEFT_HIP) or
        y(PoseLandmark.RIGHT_KNEE) < y(PoseLandmark.RIGHT_HIP)
    )
    if leg_up:
        return random.choice(captions["one_leg_up"])

    bent_over = (
        y(PoseLandmark.LEFT_SHOULDER) > y(PoseLandmark.LEFT_KNEE) and
        y(PoseLandmark.RIGHT_SHOULDER) > y(PoseLandmark.RIGHT_KNEE)
    )
    if bent_over:
        return random.choice(captions["bent_over"])

    hands_up = (
        y(PoseLandmark.LEFT_WRIST) < y(PoseLandmark.NOSE) or
        y(PoseLandmark.RIGHT_WRIST) < y(PoseLandmark.NOSE)
    )
    if hands_up and leg_up:
        return random.choice(captions["combo_pose"])

    if abs(x(PoseLandmark.LEFT_SHOULDER) - x(PoseLandmark.LEFT_HIP)) > 0.2:
        return random.choice(captions["running"])

    return random.choice(captions["default"])

