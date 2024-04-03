import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла train.csv без заголовков
data = pd.read_csv('data/train.csv', header=None)

# Фильтрация строк, где значение первой колонки равно 0
filtered_data = data[data[0] == 7]

# Создание пустой матрицы для объединенного изображения
combined_image = np.zeros((28, 28))
for i, row in enumerate(filtered_data.iterrows()):
    image = np.array(row[1][1:]).reshape(28, 28)
    for x in range(28):
        for y in range(28):
            if image[x, y] != 0:
                combined_image[x, y] += 1

# Отображение объединенного изображения
plt.imshow(combined_image, cmap='gray')
plt.axis('off')
# Сохранение изображения в файл
plt.savefig('combined_image.png')
plt.show()
