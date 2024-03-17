import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class Canvas(QWidget):
    def __init__(self, radio_buttons, send_button):
        super().__init__()
        self.setFixedSize(660, 640)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.last_point = QPoint()
        self.radio_buttons = radio_buttons
        self.send_button = send_button

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
        elif event.button() == Qt.RightButton:
            self.image.fill(Qt.white)
            self.update()
            for radio_button in self.radio_buttons:
                radio_button.setChecked(True)
            self.send_button.setDisabled(True)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 5, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
            self.send_button.setDisabled(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Программа с фотографией и радиокнопками")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Создаем список для хранения радиокнопок
        radio_buttons = []
        for i in range(10):
            radio_buttons.append(QRadioButton(f"Цифра {i+1}"))
        send_button = QPushButton("Отправить")

        # Добавляем Canvas() слева
        canvas = Canvas(radio_buttons, send_button)
        layout.addWidget(canvas)

        # Добавляем вертикальные радиокнопки справа
        radio_layout = QVBoxLayout()
        for i in range(10):
            radio_layout.addWidget(radio_buttons[i])

        radio_layout.addWidget(send_button)
        layout.addLayout(radio_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
