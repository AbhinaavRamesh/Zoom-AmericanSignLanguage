#Inference Main

from torchvision import transforms
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.utils.data as data
import os
import numpy as np
import cv2
import argparse

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

class BaseModel(nn.Module):

    def __init__(self, in_channels, out_classes, flatten_size, fuse_early=0):
        super(BaseModel, self).__init__()

        layers_conv = []

        # BLOCK 1
        layers_conv.append(nn.Conv2d(in_channels=in_channels, out_channels=96,
                                     kernel_size=7, padding=0, stride=2))
        layers_conv.append(nn.ReLU(inplace=True))
        layers_conv.append(nn.BatchNorm2d(96))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # BLOCK 2
        layers_conv.append(nn.Conv2d(in_channels=96, out_channels=256,
                                     kernel_size=5, padding=1, stride=2))
        layers_conv.append(nn.ReLU(inplace=True))
        layers_conv.append(nn.BatchNorm2d(256))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # BLOCK 3
        layers_conv.append(nn.Conv2d(in_channels=256, out_channels=512,
                                     kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU(inplace=True))

        # BLOCK 4
        layers_conv.append(nn.Conv2d(in_channels=512, out_channels=512,
                                     kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU(inplace=True))

        # BLOCK 5
        layers_conv.append(nn.Conv2d(in_channels=512, out_channels=512,
                                     kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU(inplace=True))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv_net = nn.Sequential(*layers_conv)

        if fuse_early:
            self.classifier = nn.Sequential(
                # BLOCK 6
                nn.Linear(flatten_size, 4096),
                nn.ReLU(True),
                nn.Dropout(),

                # BLOCK 7
                nn.Linear(4096, 2048),
                nn.ReLU(True),
                nn.Dropout(),
            )
        else:
            self.classifier = nn.Sequential(
                # BLOCK 6
                nn.Linear(flatten_size, 4096),
                nn.ReLU(True),
                nn.Dropout(),

                # BLOCK 7
                nn.Linear(4096, 2048),
                nn.ReLU(True),
                nn.Dropout(),

                # BLOCK 8
                nn.Linear(2048, out_classes),
            )

        # init weights
        self._initialize_weights()

        print(self.conv_net)
        print(self.classifier)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)
        return out

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().float()
    return Variable(x, volatile=volatile)

def load_model(ModelPath):
    base_model = BaseModel(20, 6, 32768).eval()  # eval mode (batchnorm uses moving mean/variance)

    # use GPU if available.
    if torch.cuda.is_available():
        base_model.cuda()

    # Load the trained model parameters
    base_model.load_state_dict(torch.load(os.path.join(ModelPath, 'FinalModel.pkl'), map_location=lambda storage, loc: storage))
    return base_model

class TemporalDataset(data.Dataset):
    """Temporal Model"""
    def __init__(self, flow_name_list, transform=None):
        self.flow_name_list = flow_name_list
        self.transform = transform

    def __getitem__(self, index):

        try:
            optical_flow = []
            for i in range(index, index + 10):
                cur_flow = np.load(self.flow_name_list[index])
                if i == index:
                    optical_flow = cur_flow.f.arr_0
                else:
                    optical_flow = np.concatenate((optical_flow, cur_flow.f.arr_0), axis=2)

            optical_flow = self.crop_center(optical_flow, 256, 256)
            

        except Exception as e:
            print("Error Loading Flow File")
            pass

        return np.einsum('ijk->kij', optical_flow)

    def __len__(self):
        return len(self.flow_name_list)

    def crop_center(self, arr, cropx, cropy):
        y, x, z = arr.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return arr[starty:starty + cropy, startx:startx + cropx]

def get_temporal_loader(flow_name_list, batch_size, shuffle, transform, num_workers):
    print(flow_name_list)
    dataset = TemporalDataset(flow_name_list=flow_name_list, transform=transform)


    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    return data_loader

def create_dir(path):
    try:
        os.makedirs(path,exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def calculate_flow(prev_frame, next_frame):
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prev_frame, next_frame, None)
    return flow

def minimal_flow(rgb_dir, flow_dir):

    files = [i for i in os.listdir(rgb_dir) if i[-4:]==".png"]
    if len(files) == 0:
        return

    prev_frame = cv2.imread(rgb_dir + "0.png")
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    for file_name in range(1, len(files)):
        next_frame = cv2.imread(rgb_dir + "{}.png".format(file_name))
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if file_name == len(files) - 1:
            flow = calculate_flow(next_frame, next_frame)
            np.savez(os.path.join(flow_dir, "{}.npz".format(file_name - 1)), flow)
            break

        flow = calculate_flow(prev_frame, next_frame)
        np.savez(os.path.join(flow_dir + "{}.npz".format(file_name - 1)), flow)
        prev_frame = next_frame

def save_optical_flow(path):
    create_dir(path+"/Temp/")
    minimal_flow(path,path+"Temp/")




def load_dataset(path):
    save_optical_flow(path)
    path_list=[]
    video_files = os.listdir(path+"/Temp/")
    for i, video_file in enumerate(video_files):
        video_number = video_file.split('.')[0]
        if int(video_number) < len(video_files) - 10:
            path_list.append(os.path.join(path+"/Temp/", video_file))
            

    return path_list

def test(model, validation_loader, criterion):
    test_results = []
    model.eval()
    with torch.no_grad():
        for i, images in enumerate(validation_loader):
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            outputs = model(images)
            outputs = F.softmax(outputs,dim=images.shape[0])
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for iii in pred.tolist():
                test_results.append(iii[0])
    majority_vote=[]
    for i in test_results:
        majority_vote.append((i,test_results.count(i)))

    return sorted(majority_vote,key =lambda x:x[1])[0][0]
def run_test(model_path,img_path):
    
    test_path_list = load_dataset(img_path)
    data_loader_test = get_temporal_loader(test_path_list, 1, shuffle=False,transform=VAL_TRANSFORM, num_workers=0)
    model = load_model(model_path)
    criterion = nn.CrossEntropyLoss()
    return test(model, data_loader_test, criterion)

labels={"Hello":0,"Happy":1,"No":2,"Eat":3,"Nice":4,"Want":5}

parser = argparse.ArgumentParser()
parser.add_argument("--modelPath",help="Directory Path to Trained Model")
parser.add_argument("--imgDir",help="Path to Directory containing atleast 10 frames for the model to predict upon")
args=parser.parse_args()

pred_label=run_test(args.modelPath,args.imgDir)
invert_label = {v:k for (k,v) in labels.items()}
print(invert_label.get(pred_label))
