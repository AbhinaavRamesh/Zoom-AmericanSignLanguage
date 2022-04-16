import socket
import selectors
import types


HOST = ''
PORT = 8089


active_socks=[]

def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    print(f"Accepted connection from {addr}")
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)
    active_socks.append(addr)

def service_connection(key, mask):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)  # Should be ready to read
        print(b"Server recieved "+recv_data)
        if recv_data:
            data.outb += recv_data
        else:
            print(f"Closing connection to {data.addr}")
            sel.unregister(sock)
            sock.close()
    if mask & selectors.EVENT_WRITE:
        if data.outb:
            print("Mask is write "+str(data.outb))
            print(active_socks)
            for address in active_socks:
                sent=sock.sendto(data.outb, address)
                print(f"Echoing {data.outb!r} to {address}")
            data.outb = data.outb[sent:]






sel = selectors.DefaultSelector()


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
print('Server now listening on port ',PORT)
s.setblocking(False)
sel.register(s, selectors.EVENT_READ, data=None)


try:
    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                accept_wrapper(key.fileobj)
            else:
                service_connection(key, mask)
except KeyboardInterrupt:
    print("Caught keyboard interrupt, exiting")
finally:
    sel.close()













# conn1, addr1 = s.accept()

# conn2, addr2 = s.accept()

# data = b'' ### CHANGED
# # payload_size = struct.calcsize("L") ### CHANGED

# while True:

#     data1 = conn1.recv(1024)
#     data2 = conn2.recv(1024)
#     # Retrieve message size
#     # while len(data) < payload_size:
#     #     data += conn.recv(4096)
#     print(data1)
#     print(data2)

#     conn1.send("Some data".encode())
#     conn2.send("Some data2".encode())
#     # packed_msg_size = data[:payload_size]
#     # data = data[payload_size:]
#     # msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

#     # # Retrieve all data based on message size
#     # while len(data) < msg_size:
#     #     data += conn.recv(4096)

#     # frame_data = data[:msg_size]
#     # data = data[msg_size:]

#     # # Extract frame
#     # frame = pickle.loads(frame_data)

#     # # Display
#     # # cv2.imshow('frame', frame)
#     # print("TEXT GG")
#     # cv2.waitKey(113)






