from keras.api.models import load_model
import numpy
import cv2

def model():
    return load_model(r'C:\programming\sharing\firedetect\submodel.keras')

def predict(img: numpy.ndarray, models) -> bool:
    # fire -> True, non fire -> False

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = numpy.float32(img) / 255.0
    img = numpy.expand_dims(img, axis=0)
    
    result = models.predict(img, batch_size=32)
    
    return result[0][0] < 0.5

# video_receiver.py
import ai
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from server import VideoSignal

class VideoReceiver(QtWidgets.QMainWindow):
    def __init__(self, max_clients=3):
        super().__init__()
        self.max_clients = max_clients
        self.model = ai.model()  # AI 모델 로드
        self.initUI()
        
        self.video_signal = VideoSignal()
        self.video_signal.frame_received.connect(self.update_frame)

    def initUI(self):
        self.setWindowTitle('방과후 야외 화재 감지기')
        self.setGeometry(100, 100, 640, 480)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        vlayout = QtWidgets.QVBoxLayout()

        hlayout = QtWidgets.QHBoxLayout()
        vlayout.addLayout(hlayout)

        self.labels = []
        for i in range(self.max_clients):
            label = QtWidgets.QLabel(self)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setText(f'Client {i + 1} not connected')
            hlayout.addWidget(label)
            self.labels.append(label)

        hlayout2 = QtWidgets.QHBoxLayout()
        vlayout.addLayout(hlayout2)

        self.checklabels = [] 
        for i in range(self.max_clients):

            # fire detected, no fire 출력되는 라벨들 여기만 디자인 수정필요
            label = QtWidgets.QLabel(self)
            label.setAlignment(QtCore.Qt.AlignCenter)
            hlayout2.addWidget(label)
            self.checklabels.append(label)
            
        self.central_widget.setLayout(vlayout)

    @QtCore.pyqtSlot(np.ndarray, int)
    def update_frame(self, frame, client_index):
        is_fire = ai.predict(frame, self.model)  # 화재 여부 예측

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        height, width, channel = frame.shape
        bytes_per_line = channel * width

        qimg = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        self.labels[client_index].setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.labels[client_index].setText('')

        if is_fire:
            self.checklabels[client_index].setText("Fire detected!")
            self.checklabels[client_index].setStyleSheet("color: red; font-size: 30px; font-weight: bold;") #글씨 빨간색으로 크고 굵게
            self.labels[client_index].setStyleSheet("border: 3px solid red;") #cctv 테두리
            self.checklabels[client_index].setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)  # 중앙 상단에 배치
        else:
            self.checklabels[client_index].setText("No fire.")
            self.checklabels[client_index].setStyleSheet("color: green; font-size: 30px; font-weight: bold;")
            self.labels[client_index].setStyleSheet("border: none;")
            self.checklabels[client_index].setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)  # 중앙 상단에 배치
            
    def closeEvent(self, event):
        self.server.stop_server()
        super().closeEvent(event)
