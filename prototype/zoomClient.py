import socket
from threading import Thread

from __future__ import division
from .micro_stream import MicrophoneStream
from google.cloud import speech
import sys
import re

# cap = cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8089))




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








RATE = 16000
CHUNK = int(RATE / 10)

def listen_print_loop(responses, callback):
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            callback(transcript + overwrite_chars + "\r")
            
            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            callback(transcript + overwrite_chars)
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                callback("Cutting the call")
                break

            num_chars_printed = 0


def handle_microphone(callback):
    language_code = "en-US"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses, sendText)


if __name__ == "__main__":
    handle_microphone(print)