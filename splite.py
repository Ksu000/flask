import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.regularizers import l2, l1
from keras.initializers import Constant


# Загрузка данных
data = np.genfromtxt("data/train.csv", delimiter=",")
np.random.shuffle(data)

# Разделение на входные данные и метки
X = data[:, 1:]
y = data[:, 0]

# One-hot encoding меток
y = to_categorical(y)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(1, input_shape=(X.shape[1],), activation="sigmoid"))
model.add(Dense(40, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))


# Компиляция модели
model.compile(
    loss="poisson",
    # optimizer='adam',
    optimizer=RMSprop(0.001),
    metrics=["accuracy"],
)

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=30, validation_data=(X_test, y_test))

# Оценка модели на тестовом наборе
loss, accuracy = model.evaluate(X_test, y_test)

print("Test loss:", loss)
print("Test accuracy:", accuracy)

model.summary()

# import ipdb; ipdb.set_trace()
model.save_weights('my_model.weights.h5')
