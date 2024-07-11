import threading
import socket as st
import cv2
import numpy as np
from PyQt5 import QtCore
import time

class VideoSignal(QtCore.QObject):
    frame_received = QtCore.pyqtSignal(np.ndarray, int)

class VideoServer:
    def __init__(self, max_clients=3, host='localhost', port=5000):
        self.max_clients = max_clients
        self.host = host
        self.port = port
        self.running = threading.Event()
        self.client_sockets = []
        self.client_threads = []
        self.video_signal = VideoSignal()

    def start_server(self):
        self.running.set()
        self.server_thread = threading.Thread(target=self.server_loop)
        self.server_thread.start()

    def stop_server(self):
        self.running.clear()
        if self.server_thread:
            self.server_thread.join()
        for thread in self.client_threads:
            thread.join()
        for sock in self.client_sockets:
            sock.close()

    def server_loop(self):
        server = st.socket(st.AF_INET, st.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(self.max_clients)
        print(f"서버 시작: {self.host}:{self.port}에서 대기 중")

        while self.running.is_set():
            if len(self.client_sockets) < self.max_clients:
                try:
                    server.settimeout(1.0)
                    client_socket, addr = server.accept()
                    print(f"클라이언트 연결됨: {addr}")

                    self.client_sockets.append(client_socket)
                    client_index = len(self.client_sockets) - 1

                    client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_index))
                    self.client_threads.append(client_thread)
                    client_thread.start()
                except st.timeout:
                    continue

        server.close()
        print("서버 종료")

    def handle_client(self, client_socket, client_index):
        data = b''
        payload_size = 8

        try:
            while self.running.is_set():
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        return
                    data += packet

                frame_size = int(data[:payload_size].decode())
                data = data[payload_size:]

                while len(data) < frame_size:
                    data += client_socket.recv(4096)

                frame_data = data[:frame_size]
                data = data[frame_size:]

                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                self.video_signal.frame_received.emit(frame, client_index)
                time.sleep(0.08)
        finally:
            client_socket.close()
            if client_socket in self.client_sockets:
                self.client_sockets.remove(client_socket)
                self.client_threads[client_index].join()
                self.client_threads.pop(client_index)
