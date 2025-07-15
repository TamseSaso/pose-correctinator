#!/usr/bin/env python3
import socket
import simpleaudio as sa

# Listen on all interfaces, same port you configured in main.py
LISTEN_IP   = "0.0.0.0"
LISTEN_PORT = 5005

# Load your alert sound (must be a .wav file)
wave_obj = sa.WaveObject.from_wave_file("alert.wav")

# Set up the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))
print(f"Listening for posture alerts on port {LISTEN_PORT}…")

try:
    while True:
        data, addr = sock.recvfrom(1024)     # buffer size
        if data == b"bad_posture":
            print(f"⚠️  Bad posture alert from {addr}")
            wave_obj.play()                  # fire-and-forget
except KeyboardInterrupt:
    print("\nShutting down listener.")
    sock.close()