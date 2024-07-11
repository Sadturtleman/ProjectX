import cv2
import socket
import threading
import time

def send_video(host='localhost', port=5000):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    cap = cv2.VideoCapture(r'C:\programming\sharing\firedetect\resources\fire2.mp4')

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            encoded, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            frame_size = str(len(frame_data)).zfill(8).encode()
            client_socket.sendall(frame_size + frame_data)
            time.sleep(0.08)
    finally:
        cap.release()
        client_socket.close()

if __name__ == "__main__":
    video_thread = threading.Thread(target=send_video)
    video_thread.start()
