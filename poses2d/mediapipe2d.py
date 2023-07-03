
import sys
sys.path.append('..')

import cv2
import mediapipe as mp

from helpers import mediapipe_to_coco
import numpy as np


def mediapipe_extract_2d_poses_17(video_path):
    '''
    This returns the coco order of 17 keypoints.
    inorder to change to h36m format, you should use the function to convert it.
    '''
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    sequence_2d_poses = []
    sequence_2d_poses_coco = [] # coco ones
    sequence_visibilities = []
    # Get the dimensions of the video frames
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame from the video")

    original_height, original_width = frame.shape[:2]
    
    # Start capturing again
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()

        # If the frame was read successfully
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            if results.pose_landmarks:
                pose_landmarks = []
                pose_visibilities = []

                for landmark in results.pose_landmarks.landmark:
                    pose_landmarks.append([landmark.x, landmark.y])
                    pose_visibilities.append([landmark.visibility])
                sequence_2d_poses.append(pose_landmarks)
                # get posenet list
                coco_landmarks, coco_visibilities = mediapipe_to_coco(pose_landmarks, pose_visibilities)
                sequence_2d_poses_coco.append(coco_landmarks)
                sequence_visibilities.append(coco_visibilities)
        else:
            # If the frame was not read successfully, break the loop
            break

    cap.release()
    sequence_2d_poses_coco = np.array(sequence_2d_poses_coco)
    sequence_visibilities = np.array(sequence_visibilities).squeeze()

    # original_height, original_width are the height and width of the original images
    sequence_2d_poses_coco_unnorm = np.zeros_like(sequence_2d_poses_coco)

    sequence_2d_poses_coco_unnorm[:, :, 0] = sequence_2d_poses_coco[:, :, 0] * original_width
    sequence_2d_poses_coco_unnorm[:, :, 1] = sequence_2d_poses_coco[:, :, 1] * original_height


    return sequence_2d_poses_coco_unnorm, sequence_visibilities


if __name__ == "__main__":

    video_path = '/media/mark/New Volume/Ariel/codes/PoseFormerV2/demo/video/sample_video.mp4'

    coco_kpts, coco_vises = mediapipe_extract_2d_poses_17(video_path=video_path)

    print(coco_kpts)



