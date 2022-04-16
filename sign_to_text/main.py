from .landmark_detection import get_body_pose
from .model_inference import infer


def get_label(frames):
    body_poses = get_body_pose(frames)
    label = infer(body_poses)
    return label
