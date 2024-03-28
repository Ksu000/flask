import os
import sys
import csv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


# Функция для сохранения массива чисел в csv файл
def save_to_csv(data):
    with open("data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


# Функция для загрузки последней строки из csv файла
def load_from_csv():
    with open("data.csv", "r") as file:
        reader = csv.reader(file)
        data = None
        for row in reader:
            data = row
        return data


class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.last_point = QPoint()
        self.image = QImage(400, 400, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.pen_color = Qt.black
        self.pen_width = 10

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
        buttons = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
        for i in range(3):
            for j in range(3):
                number = 3 * i + j + 1
                button = QPushButton(str(number))
                button.setFixedSize(50, 50)
                button_layout.addWidget(button, i, j)
                button.clicked.connect(
                    lambda _, num=number: self.save_image(num)
                )  # Подключаем обработчик сохранения для каждой кнопки

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
        file_name = os.path.join("data", f"image_{number}.png")
        self.drawing_widget.image.save(file_name)
        print(f"Saved image for number {number}, {file_name=}")
        self.on_reset_click()

    # def save_image(self, number):
    #     img = self.drawing_widget.image
    #     if not img.isNull():
    #         img = img.convertToFormat(QImage.Format_RGB32)
    #         width = img.width()
    #         height = img.height()
    #         print(width)
    #         print(height)
            
    #         ptr = img.constBits()
    #         ptr.setsize(img.byteCount())
            
    #         arr = np.array(ptr).reshape(height, width, 1)
            
    #         # Now you can access the pixel values in the NumPy array 'arr'
    #         print("Image array shape:", arr.shape)
    #         print(arr)
    #     else:
    #         print("Error: QImage is null.")
    #     return 

    #     img_array = self.drawing_widget.image.get_image_array()
    #     img_array_resized = np.zeros((28, 28))
    #     for i in range(28):
    #         for j in range(28):
    #             arr = img_array[i * 16 : i * 16 + 16, j * 16 : j * 16 + 16]
    #             arr_sum = np.sum(arr)
    #             arr_value = 255 - int(arr_sum / 4294967040 * 255)
    #             if arr_value != 0:
    #                 img_array_resized[i, j] = arr_value

    #     user_signal = np.reshape(img_array_resized, (1, img_array_resized.size))
    #     ia = np.array(user_signal)
    #     print("Processed image array:", ia)

    def on_reset_click(self):
        self.drawing_widget.image.fill(Qt.white)
        self.drawing_widget.update()

    def load_data(self):
        data = load_from_csv()
        if data:
            print("Loaded data:", data)
        else:
            print("No data found in the CSV file")


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
