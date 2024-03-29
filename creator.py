from PIL import Image
import numpy as np
import sys
import os
import csv
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


def smooth(I):
    J = I.copy()
    J[1:-1] = J[1:-1] // 2 + J[:-2] // 4 + J[2:] // 4
    J[:, 1:-1] = J[:, 1:-1] // 2 + J[:, :-2] // 4 + J[:, 2:] // 4
    return J


class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.last_point = QPoint()
        self.image = QImage(400, 400, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.pen_color = Qt.black
        self.pen_width = 30

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.image)
            painter.setPen(
                QPen(
                    self.pen_color,
                    self.pen_width,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()


class ButtonsWidget(QWidget):
    def __init__(self, drawing_widget):
        super().__init__()
        self.drawing_widget = drawing_widget

        layout = QVBoxLayout()

        button_layout = QGridLayout()
        for i in range(3):
            for j in range(3):
                number = 3 * i + j + 1
                button = QPushButton(str(number))
                button.setFixedSize(50, 50)
                button_layout.addWidget(button, i, j)
                button.clicked.connect(lambda _, num=number: self.save_image(num))

        # Добавляем кнопку "Load"
        load_button = QPushButton("Load")
        load_button.setFixedSize(50, 50)
        load_button.clicked.connect(self.load_data)
        button_layout.addWidget(load_button, 3, 0)

        # Добавляем кнопку "0"
        zero_button = QPushButton("0")
        zero_button.setFixedSize(50, 50)
        zero_button.clicked.connect(lambda: self.save_image(0))
        button_layout.addWidget(zero_button, 3, 1)

        # Добавляем кнопку "Clear"
        clear_button = QPushButton("Clear")
        clear_button.setFixedSize(50, 50)
        clear_button.clicked.connect(self.on_reset_click)
        button_layout.addWidget(clear_button, 3, 2)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def save_image(self, number):
        import cv2  # pip install opencv-python

        tmp = "tmp"
        if not os.path.exists(tmp):
            os.mkdir(tmp)

        tmpfilename = os.path.join(tmp, f"image_{number}.png")
        image = self.drawing_widget.image
        image.save(tmpfilename)

        resize = Image.open(tmpfilename)
        new_size = (28, 28)
        resize.thumbnail(new_size)
        resizefilename = os.path.join(tmp, f"image_{number}_resize.png")
        resize.save(resizefilename)
        grey_img = cv2.imread(resizefilename, cv2.IMREAD_GRAYSCALE)

        greyfilename = os.path.join(tmp, f"image_{number}_grey.png")
        cv2.imwrite(greyfilename, grey_img)

        denoise = smooth(grey_img)
        denoisefilename = os.path.join(tmp, f"image_{number}_denoise.png")
        cv2.imwrite(denoisefilename, denoise)

        a = np.array([number], dtype=int)
        arr = np.hstack((a.reshape(1, a.shape[0]), np.reshape(denoise, (1, 784))))

        csvname = os.path.join("data", f"train.csv")
        arr_str = ",".join(map(str, arr[0]))
        with open(csvname, "a") as f:
            f.write(arr_str + "\n")

        self.on_reset_click()

    def on_reset_click(self):
        self.drawing_widget.image.fill(Qt.white)
        self.drawing_widget.update()

    def load_data(self):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Программа для рисования")
        self.setFixedSize(400, 700)

        self.drawing_widget = DrawingWidget()
        self.buttons_widget = ButtonsWidget(self.drawing_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.drawing_widget)
        main_layout.addWidget(self.buttons_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
