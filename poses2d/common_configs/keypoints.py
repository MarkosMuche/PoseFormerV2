

# Indices of landmarks in MediaPipe's output corresponding to PoseNet's keypoints
COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

MEDIAPIPE_KEYPOINT_INDEXES = {
    0: "nose",                11: "left_shoulder",       22: "right_thumb",
    1: "left_eye_inner",      12: "right_shoulder",      23: "left_hip",
    2: "left_eye",            13: "left_elbow",          24: "right_hip",
    3: "left_eye_outer",      14: "right_elbow",         25: "left_knee",
    4: "right_eye_inner",     15: "left_wrist",          26: "right_knee",
    5: "right_eye",           16: "right_wrist",         27: "left_ankle",
    6: "right_eye_outer",     17: "left_pinky",          28: "right_ankle",
    7: "left_ear",            18: "right_pinky",         29: "left_heel",
    8: "right_ear",           19: "left_index",          30: "right_heel",
    9: "mouth_left",          20: "right_index",         31: "left_foot_index",
    10: "mouth_right",        21: "left_thumb",          32: "right_foot_index"
}


KEYPOINTS_13 = {
    'NOSE_TIP': 0,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
}