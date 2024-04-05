import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

# Загрузка данных из файла
data = np.genfromtxt('data/train.csv', delimiter=',')
X = data[:, 1:]  # Битовый вектор изображений
y = data[:, 0]   # Целевая метка

# Рандомизация массива
# np.random.seed(4)
shuffle_index = np.random.permutation(len(X))
X_shuffled = X[shuffle_index]
y_shuffled = y[shuffle_index]

# Разбиение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    X_shuffled, y_shuffled, test_size=0.2, random_state=42)

# Создание модели SVM
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)

# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(solver='saga')
# clf.fit(X_train, y_train)

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(algorithm='kd_tree')
# clf.fit(X_train, y_train)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10000, max_depth=10)
# clf.fit(X_train, y_train)

# Оценка точности модели
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Сохранение модели в файл
joblib.dump(clf, 'svm_model.pkl')
print("Модель сохранена успешно!")

# Загрузим сохраненную модель
loaded_model = joblib.load('svm_model.pkl')

# Применим модель к выбранной строке
predicted_class = loaded_model.predict(X_test[:10])

print(f"{predicted_class} Предсказанная метка класса")
print(f"{y_test[:10]} Истинная метка класса")
