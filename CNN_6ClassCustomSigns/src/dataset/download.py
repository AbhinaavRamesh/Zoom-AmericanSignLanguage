from globals import create_dir
import json
import config as cfg
import subprocess
import cv2
import os
import shutil

def download_dataset():
    create_dir(cfg.MSASL_RGB_PATH)
    save_videos("train")#, cfg.TRAIN_JSON_PATH)
    save_videos("val")#, cfg.VAL_JSON_PATH)
    save_videos("test")#, cfg.TEST_JSON_PATH)


def resize_frame(frame):
    h, w, _ = frame.shape
    if h < w:
        resize_ratio = round(cfg.MIN_RESIZE / h, 2)
        h = cfg.MIN_RESIZE
        w = w * resize_ratio
    else:
        resize_ratio = round(cfg.MIN_RESIZE / w, 2)
        w = cfg.MIN_RESIZE
        h = h * resize_ratio

    return int(h), int(w)


def save_frames(file_name, dir_name, file_count=0):
    video = cv2.VideoCapture(file_name)
    h, w = 0, 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if file_count == 0:
            h, w = resize_frame(frame)

        resized_image = cv2.resize(frame.copy(), (w, h))
        cv2.imwrite(dir_name + '/' + str(file_count) + '.png', resized_image)
        file_count += 1

    video.release()
    cv2.destroyAllWindows()
    return h, w


def save_videos(folder_name, video_id=0):
    json_data=[]
    labels={"Hello":0,"Happy":1,"No":2,"Eat":3,"Nice":4,"Want":5}
    create_dir(cfg.MSASL_RGB_PATH + "/" + folder_name)        # create train, val or test directory
    path="/Users/abhinaavramesh/Downloads/ManualData/"
    if folder_name=="train":
        cut_off=[str(i) for i in range(4)]
    elif folder_name=="test":
        cut_off=['4']    
    else:
        cut_off=['5']

    ls=[i for i in os.listdir(path) if i[-4:]==".mp4"]
    for it in ls:
        if it[-5] in cut_off:

            dir_name = cfg.MSASL_RGB_PATH + "/" + folder_name + "/" + str(video_id)
            file_name = cfg.MSASL_RGB_PATH + "/" + folder_name + '/%s_%d.mp4' % (cfg.DATASET_NAME, video_id)

            create_dir(dir_name)       # create video subdir
            shutil.copy(path+it,file_name)
            
            # open video
            h, w = save_frames(file_name, dir_name)

            # save json
            json_data.append({
                "videoId": '%d' % video_id,
                "cleanText": it.split('_')[0],
                "label": labels[it.split('_')[0]],
                "width": w,
                "height": h
            })

            video_id += 1

    with open(cfg.MSASL_RGB_PATH + "/%s_%s_rgb.json" % (cfg.DATASET_NAME, folder_name), "w") as outfile:
        json.dump(json_data, outfile)
