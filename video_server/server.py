import pickle
import socket
import struct
from sign_to_text import get_label
import traceback
import os

import cv2

HOST = "0.0.0.0"
PORT = 8090

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)




conn, addr = s.accept()

data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED

frames = []
while True:

    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    # Extract frame
    frame = pickle.loads(frame_data)
    frames.append(frame)
    i = 0
    if len(frames) >= 10:
        os.makedirs(f"test{i}", exist_ok=True)
        i += 1
        try:
            label = get_label(frames)
            print("\n", label)
        except:
            print("",end="")
        for j,frame in enumerate(frames):
            cv2.imwrite(f"test{i-1}/img{j}.jpg", frame)
        frames = frames[8:]
        
    # Display

#from tqdm import tqdm
#import requests

#url = "http://download.thinkbroadband.com/10MB.zip"
#response = requests.get(url, stream=True)

#with open("10MB", "wb") as handle:
 #   for data in tqdm(response.iter_content()):
  #      handle.write(data)
