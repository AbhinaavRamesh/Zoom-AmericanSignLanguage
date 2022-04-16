from .normalization.body_normalization import BODY_IDENTIFIERS
from .normalization.hand_normalization import HAND_IDENTIFIERS
from .normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from .normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict
import torch
import numpy as np

model = torch.load("/home/imdaredevil/mindspark-14-team/sign_to_text/checkpoints/best_model.pth", map_location="cpu")
model = model.to("cuda")
model.train(False)

label_file = open("/home/imdaredevil/mindspark-14-team/sign_to_text/labels.txt") 
labels = []
for label in label_file.readlines():
    labels.append(label.split()[-1])


HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]

def dictionary_to_tensor(landmarks_dict):

    output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)

def preprocess(depth_map):
    depth_map_shape = list(depth_map.values())[0].shape
    if "neck" not in depth_map:
        depth_map["neck"] = torch.zeros(depth_map_shape)
    depth_map = normalize_single_body_dict(depth_map)
    depth_map = normalize_single_hand_dict(depth_map)
    depth_map = dictionary_to_tensor(depth_map)
    depth_map = depth_map - 0.5
    return depth_map.squeeze(0).to("cuda")


def get_output(model_input):
    with torch.no_grad():
        model_output = model(model_input).expand(1, -1, -1)
        softmax = torch.nn.Softmax().to("cuda")
        softmax_output = softmax(model_output[0])
        output = np.argmax(softmax_output.cpu().detach().numpy(), axis=-1)
        return labels[output[0]]



def infer(body_poses):
    model_input = preprocess(body_poses)
    return get_output(model_input)

    
