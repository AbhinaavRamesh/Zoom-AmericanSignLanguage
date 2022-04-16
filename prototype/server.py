import socket
from _thread import *

ServerSideSocket = socket.socket()
host = '127.0.0.1'
port = 8090
ThreadCount = 0
try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))

print('Socket is listening..')
ServerSideSocket.listen(5)


all_connections=[]

def multi_threaded_client(connection):
    all_connections.append(connection)
    while True:
        data = connection.recv(2048)
        response = data.decode('utf-8')
        print(response)
        if not data:
            break
        for a_connection in all_connections:
            if a_connection != connection:
                a_connection.sendall(str.encode(response))
    connection.close()



while True:
    Client, address = ServerSideSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(multi_threaded_client, (Client, ))
    ThreadCount += 1
ServerSideSocket.close()