import json
import random
from mediapipe.python.solutions.pose import PoseLandmark

# Load the short funny captions
with open("funny_labels.json", "r") as f:
    captions = json.load(f)

def get_funny_caption(landmarks):
    if not landmarks:
        return random.choice(captions["default"])

    def y(part): return landmarks[part].y
    def x(part): return landmarks[part].x

    # Hugging (hands close together)
    if abs(x(PoseLandmark.LEFT_WRIST) - x(PoseLandmark.RIGHT_WRIST)) < 0.1:
        return random.choice(captions["hugging"])

    # Hands up
    if y(PoseLandmark.LEFT_WRIST) < y(PoseLandmark.NOSE) and y(PoseLandmark.RIGHT_WRIST) < y(PoseLandmark.NOSE):
        return random.choice(captions["hands_up"])

    # One leg up
    leg_up = (
        y(PoseLandmark.LEFT_KNEE) < y(PoseLandmark.LEFT_HIP) or
        y(PoseLandmark.RIGHT_KNEE) < y(PoseLandmark.RIGHT_HIP)
    )
    if leg_up:
        return random.choice(captions["one_leg_up"])

    # Bent over (shoulders close to knees)
    bent_over = (
        y(PoseLandmark.LEFT_SHOULDER) > y(PoseLandmark.LEFT_KNEE) and
        y(PoseLandmark.RIGHT_SHOULDER) > y(PoseLandmark.RIGHT_KNEE)
    )
    if bent_over:
        return random.choice(captions["bent_over"])

    # Combo pose: hands up + leg up
    hands_up = (
        y(PoseLandmark.LEFT_WRIST) < y(PoseLandmark.NOSE) or
        y(PoseLandmark.RIGHT_WRIST) < y(PoseLandmark.NOSE)
    )
    if hands_up and leg_up:
        return random.choice(captions["combo_pose"])

    # Running logic
    if abs(x(PoseLandmark.LEFT_SHOULDER) - x(PoseLandmark.LEFT_HIP)) > 0.2:
        return random.choice(captions["running"])

    return random.choice(captions["default"])
