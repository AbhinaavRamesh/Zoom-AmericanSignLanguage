
import socket
from threading import Thread
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8090))

def useData(a):
    print(a)


def listenForServer(useDataFromServer):
    def serverListener():
        should_exit=False
        while not should_exit:
            recieved_data=clientsocket.recv(1024)
            if not recieved_data:
                should_exit = True
            useDataFromServer(recieved_data.decode())
    return serverListener
    


def sendText(text):
    print("Sending to server"+text)
    clientsocket.send(text.encode())

Thread(target=listenForServer(useData)).start()
