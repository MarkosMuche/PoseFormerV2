
import sys
sys.path.append('..')

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import procrustes
from fastdtw import fastdtw
from common_configs.keypoints import MEDIAPIPE_KEYPOINT_INDEXES, COCO_KEYPOINT_INDEXES


def mediapipe_to_coco(mediapipe_landmarks, mediapipe_visibilities = None):
    '''Converts from mediapipe to coco. 
    Mediapipe has 33 keypoints and coco has 17'''

    # Invert the COCO_KEYPOINT_INDEXES dictionary
    COCO_KEYPOINT_INDEXES_INV = {v: k for k, v in COCO_KEYPOINT_INDEXES.items()}

    # Create the mapping from MediaPipe to COCO
    MEDIATO_COCO_MAPPING = {k: COCO_KEYPOINT_INDEXES_INV[v] 
                            for k, v in MEDIAPIPE_KEYPOINT_INDEXES.items() 
                            if v in COCO_KEYPOINT_INDEXES_INV}

    coco_keypoints = [mediapipe_landmarks[MEDIATO_COCO_MAPPING[i]] for 
                      i in range(33) if i in MEDIATO_COCO_MAPPING]
    
    if mediapipe_visibilities:
        coco_visibilities = [mediapipe_visibilities[MEDIATO_COCO_MAPPING[i]] for 
                            i in range(33) if i in MEDIATO_COCO_MAPPING]

        return coco_keypoints, coco_visibilities
    else:
        return coco_keypoints


def calculate_dtw_distance(sequence_1, sequence_2):
    ''' This uses dtw distance.'''
    num_joints = sequence_1.shape[1]
    num_dims = sequence_1.shape[2]

    total_distance = 0

    # Define a custom distance function that can handle 2-D input arrays
    def custom_distance(u, v):
        return np.sqrt(np.sum((u - v) ** 2))

    # Calculate the DTW distance separately for each joint and each dimension
    for joint in range(num_joints):
        for dim in range(num_dims):
            # Flatten the arrays into true 1-D arrays
            sequence_1_flat = sequence_1[:, joint, dim].flatten()
            sequence_2_flat = sequence_2[:, joint, dim].flatten()

            distance, _ = fastdtw(sequence_1_flat, sequence_2_flat, dist=custom_distance)
            total_distance += distance

    # Average the distances
    average_distance = total_distance / (num_joints * num_dims)

    return average_distance


def compare_poses(poses_3d1, poses_3d2):
    '''
    This function uses procrustes analysis
    '''
    num_frames_to_compare = min(len(poses_3d1), len(poses_3d2))
    # Ensure the number of frames to compare is less than or equal to the number of frames in both videos
    assert num_frames_to_compare <= min(len(poses_3d1), len(poses_3d2)), "Number of frames to compare is greater than the number of frames in one or both videos"

    # Resample the 3D poses to have the same number of frames
    poses_3d1_resampled = resample_poses(poses_3d1, num_frames_to_compare)
    poses_3d2_resampled = resample_poses(poses_3d2, num_frames_to_compare)

    # Perform Procrustes analysis on each pair of corresponding frames
    disparities = []
    for pose_3d1, pose_3d2 in zip(poses_3d1_resampled, poses_3d2_resampled):
        _, _, disparity = procrustes(pose_3d1, pose_3d2)
        disparities.append(disparity)

    # Calculate the average disparity
    average_disparity = np.mean(disparities)

    return average_disparity


def transform_poses(poses_3d1, poses_3d2):
    '''
    This function uses procrustes analysis to transform the poses in poses_3d2 to match the poses in poses_3d1
    '''
    num_frames_to_compare = min(len(poses_3d1), len(poses_3d2))
    # Ensure the number of frames to compare is less than or equal to the number of frames in both videos
    assert num_frames_to_compare <= min(len(poses_3d1), len(poses_3d2)), "Number of frames to compare is greater than the number of frames in one or both videos"

    # Resample the 3D poses to have the same number of frames
    poses_3d1_resampled = resample_poses(poses_3d1, num_frames_to_compare)
    poses_3d2_resampled = resample_poses(poses_3d2, num_frames_to_compare)

    # Perform Procrustes analysis on each pair of corresponding frames and transform the poses in poses_3d2
    transformed_poses_3d2 = []
    for pose_3d1, pose_3d2 in zip(poses_3d1_resampled, poses_3d2_resampled):
        matrix, translation, scaling = procrustes(pose_3d1[:, :2], pose_3d2[:, :2])  # Apply Procrustes analysis to x, y coordinates only
        transformed_pose_3d2 = np.zeros_like(pose_3d2)
        transformed_pose_3d2[:, :2] = scaling * (pose_3d2[:, :2] @ matrix.T) + translation  # Apply transformation to x, y coordinates only
        transformed_pose_3d2[:, 2] = pose_3d2[:, 2]  # Leave z coordinate unchanged
        transformed_poses_3d2.append(transformed_pose_3d2)

    return poses_3d1_resampled, np.array(transformed_poses_3d2)


def resample_poses(poses_3d, num_frames):
    num_joints = poses_3d.shape[1]
    num_dims = poses_3d.shape[2]

    # Create an array to hold the resampled poses
    poses_3d_resampled = np.zeros((num_frames, num_joints, num_dims))

    # Create an interpolation function for each joint and dimension
    for joint in range(num_joints):
        for dim in range(num_dims):
            f = interp1d(np.linspace(0, 1, len(poses_3d)), poses_3d[:, joint, dim])
            poses_3d_resampled[:, joint, dim] = f(np.linspace(0, 1, num_frames))

    return poses_3d_resampled


