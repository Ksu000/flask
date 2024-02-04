import os
import sys
import json
import pygame
import numpy as np

# Инициализация Pygame
pygame.init()

# Установка размеров окна
WIDTH, HEIGHT = 448, 448
CELL_SIZE = 1
LINE_WIDTH = 20

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Создание экрана
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drawing Program")

# Создание поверхности для рисования
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(WHITE)


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix / norm
    return matrix


def sigmoid(x, der=False):
    """
    Сигмоида для опредления значения весов
    """
    if der:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def smooth(I):
    J = I.copy()
    J[1:-1] = J[1:-1] // 2 + J[:-2] // 4 + J[2:] // 4
    J[:, 1:-1] = J[:, 1:-1] // 2 + J[:, :-2] // 4 + J[:, 2:] // 4
    return J


def calc(canvas):
    # Разбиение рисунка на массив 28x28 по 8x8 точек
    img_array = pygame.surfarray.array2d(canvas).T
    img_array_resized = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            arr = img_array[i * 16 : i * 16 + 16, j * 16 : j * 16 + 16]
            arr = np.sum(arr)
            arr = 255 - int(arr / 4294967040 * 255)
            if arr != 0:
                img_array_resized[i, j] = arr

    # np.savetxt(
    #     "data/img_array_resized.csv",
    #     img_array_resized.astype(int),
    #     delimiter=",",
    #     fmt="%s",
    # )
    user_signal = np.reshape(img_array_resized, (1, img_array_resized.size))
    ia = np.array(user_signal)

    filename = os.path.join("data", "genom.json")
    with open(filename, encoding="utf-8") as f:
        genom_dct = json.load(f)
    for key in genom_dct:
        genom = genom_dct[key]
        break

    PIXELS_PER_IMAGE = 784
    HIDDEN_SIZE = 40
    NUM_LABELS = 10

    layer1_out = PIXELS_PER_IMAGE * HIDDEN_SIZE
    second = HIDDEN_SIZE * NUM_LABELS
    layer2_out = layer1_out + second
    layer1 = np.reshape(genom[:layer1_out], (PIXELS_PER_IMAGE, HIDDEN_SIZE))
    layer2 = np.reshape(genom[layer1_out:layer2_out], (HIDDEN_SIZE, NUM_LABELS))

    # На выходе первого скрытого слоя
    l1 = sigmoid(np.dot(ia, layer1))
    # На выходе второго скрытого слоя
    l2 = sigmoid(np.dot(l1, layer2))
    l3 = normalize_2d(l2)

    for num, val in enumerate(l3[0]):
        print(num, val, "*" * int(50 * val))
    print()


# Основной цикл программы
drawing = False
while True:
    screen.fill(WHITE)
    screen.blit(canvas, (0, 0))

    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Левая кнопка мыши
                drawing = True
            elif event.button == 3:  # Правая кнопка мыши
                canvas.fill(WHITE)
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            calc(canvas)

    # Рисование на холсте
    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(canvas, BLACK, mouse_pos, LINE_WIDTH)

    pygame.display.flip()
