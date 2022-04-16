from turtle import delay
import cv2
# import numpy as np
import socket
# import sys
import pickle
import struct
import time
import random
from threading import Thread
# cap = cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8089))

# recieved_data = b'' ### CHANGED
# # while True:
#     # ret,frame = cap.read()
#     # Serialize frame
#     data = "I am ASL "+str(random.randint(0,10)) #pickle.dumps(frame)

#     # Send message length first
#     # message_size = struct.pack("L", len(data)) ### CHANGED

#     # Then data
#     print("Sending: "+data)
#     clientsocket.send(data.encode())
#     recieved_data=clientsocket.recv(1024)
#     print(b"Got from server: "+recieved_data)
#     time.sleep(5)


def listenForServer():
    should_exit=False
    while not should_exit:
        recieved_data=clientsocket.recv(1024)
        if not recieved_data:
            should_exit = True
        print(b"Got from server: "+recieved_data)


def sendText(text):
    print("Sending to server"+text)
    clientsocket.send(text.encode())

def sampleSend():
    while True:
        data = "I am ASL "+str(random.randint(0,10)) #pickle.dumps(frame)
        sendText(data)
        time.sleep(5)

Thread(target=listenForServer).start()
Thread(target=sampleSend).start()