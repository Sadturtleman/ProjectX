import sys
from PyQt5 import QtWidgets
from app import VideoReceiver
from server import VideoServer

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    receiver = VideoReceiver(max_clients=3)
    server = VideoServer(max_clients=3)

    server.video_signal.frame_received.connect(receiver.update_frame)

    server.start_server()
    receiver.show()

    receiver.server = server  # 서버 인스턴스 참조 추가

    sys.exit(app.exec_())
