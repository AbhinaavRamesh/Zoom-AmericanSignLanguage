import socketio

# create a Socket.IO server
sio = socketio.Server()

# wrap with a WSGI application
app = socketio.WSGIApp(sio)


@sio.event
async def connect(sid):
    sio.save_session(sid, {'username': sid})
    print("connected")
