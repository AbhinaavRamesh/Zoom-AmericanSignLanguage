import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
import torch
from google.protobuf.json_format import MessageToDict

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)



pose_key_point_map = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "rightEye": mp_pose.PoseLandmark.RIGHT_EYE,
    "leftEye": mp_pose.PoseLandmark.LEFT_EYE,
    "rightEar": mp_pose.PoseLandmark.RIGHT_EAR,
    "leftEar": mp_pose.PoseLandmark.LEFT_EAR,
    "rightShoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "leftShoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "rightElbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "leftElbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "rightWrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "leftWrist": mp_pose.PoseLandmark.LEFT_WRIST,
}

hand_key_point_map = {
    "wrist": mp_hands.HandLandmark.WRIST,
    "indexTip": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "indexDIP": mp_hands.HandLandmark.INDEX_FINGER_DIP,
    "indexPIP": mp_hands.HandLandmark.INDEX_FINGER_PIP,
    "indexMCP": mp_hands.HandLandmark.INDEX_FINGER_MCP,
    "middleTip": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "middleDIP": mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
    "middlePIP": mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    "middleMCP": mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    "ringTip": mp_hands.HandLandmark.RING_FINGER_TIP,
    "ringDIP": mp_hands.HandLandmark.RING_FINGER_DIP,
    "ringPIP": mp_hands.HandLandmark.RING_FINGER_PIP,
    "ringMCP": mp_hands.HandLandmark.WRIST.RING_FINGER_MCP,
    "littleTip": mp_hands.HandLandmark.PINKY_TIP,
    "littleDIP": mp_hands.HandLandmark.PINKY_DIP,
    "littlePIP": mp_hands.HandLandmark.PINKY_PIP,
    "littleMCP": mp_hands.HandLandmark.PINKY_MCP,
    "thumbTip": mp_hands.HandLandmark.THUMB_TIP,
    "thumbIP": mp_hands.HandLandmark.THUMB_IP,
    "thumbMP": mp_hands.HandLandmark.THUMB_MCP,
    "thumbCMC": mp_hands.HandLandmark.THUMB_CMC,
}


def convert_to_dict(source, dictionary):
    result = {k : [] for k in  dictionary.keys()}
    for vec in source:
        for k,v in dictionary.items():
            if vec is not None:
                result[k].append(torch.FloatTensor([vec.landmark[v].x , vec.landmark[v].y]))
            else:
                result[k].append(torch.FloatTensor([0.0,0.0]))
    final_result = {}
    for k in result:
        if len(result[k]) > 0:
            final_result[k] = torch.stack(result[k]).numpy()
    return final_result



def get_body_pose(frames):
    left_hand_results = []
    right_hand_results = []
    pose_results = []
    left_hand_key_point_map = { f"{k}_0": v for (k,v) in hand_key_point_map.items()}
    right_hand_key_point_map = { f"{k}_1": v for (k,v) in hand_key_point_map.items()}
    for frame in frames:
        image = cv2.flip(frame, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(image)
        left_hand_landmarks = hand_result.multi_hand_landmarks[0]
        right_hand_landmarks = None
        if len(hand_result.multi_hand_landmarks) > 1:
            right_hand_landmarks = hand_result.multi_hand_landmarks[1]
        if MessageToDict(hand_result.multi_handedness[0])["classification"][0]["label"] == "Right":
            left_hand_landmarks,right_hand_landmarks = right_hand_landmarks,left_hand_landmarks
        left_hand_results.append(left_hand_landmarks)
        right_hand_results.append(right_hand_landmarks)
        pose_results.append(pose.process(image).pose_landmarks)
    final_dict = {}
    
    pose_dict = convert_to_dict(pose_results,pose_key_point_map)
    final_dict.update(pose_dict)
    left_hand_dict = convert_to_dict(left_hand_results, left_hand_key_point_map)
    final_dict.update(left_hand_dict)
    right_hand_dict = convert_to_dict(right_hand_results, right_hand_key_point_map)
    final_dict.update(right_hand_dict)
    return final_dict
    

if __name__ == "__main__":
    print(get_body_pose([cv2.imread("image3.png"), cv2.imread("image3.png")]))
    

