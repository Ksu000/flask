import numpy as np
import random
import time
import os
import json
import random


def load_json(folder_name, file_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    filename = os.path.join(folder_name, file_name)
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump(dict(), f, ensure_ascii=True, sort_keys=True)
    with open(filename, encoding="utf-8") as f:
        load_dct = json.load(f)
    return load_dct


def save_json(folder_name, file_name, save_dct):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    filename = os.path.join(folder_name, file_name)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_dct, f, ensure_ascii=False, indent=4, sort_keys=True)


# Сигмоида необходима для опредления значения весов
def sigmoid(x, der=False):
	if der:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))


def get_data(filename):
    # Открываем файл
    data = np.genfromtxt(filename, delimiter=',')
    input_array = data[1:, :-1]  # Первый массив - все строки из файла кроме первой. Входные данные
    output_data = data[1:, -1]  # Второй массив - первая строка из файла. Выходные данные
    # Создадим вектор для проверки
    output_lst = list()
    for mnum in output_data:
        rres = [1 if x == int(mnum) else 0 for x in range(10)]
        output_lst.append(rres)
    output_array = np.array(output_lst)  # Выходной массив
    return input_array, output_array


def recombination(dad, mom, combination=0.8, mutations=0.2, diff=2):
    assert len(dad) == len(mom), 'len(dad) != len(mom)'
    child1 = []
    child2 = []
    copies = False
    for n in range(len(dad)):            
        if combination < random.random():
            copies = not copies
        if mutations > random.random():
            child1.append(mom[n] + (diff * (random.random() - 0.5)))
            child2.append(dad[n] - (diff * (random.random() - 0.5)))
        elif copies:
            child1.append(dad[n])
            child2.append(mom[n])
        else:
            child1.append(mom[n])
            child2.append(dad[n])
    return child1, child2


def get_best_parents(best_json, diff=2):
    parents = list()
    for _, val in best_json.items():
        parents.append(val)
    while len(parents) < 2:
        this = list()
        for _ in range((784 * 10) + (10 * 10)):
            this.append(diff * (random.random() - 0.5))
        this.append(0)
        this.append(0)
        parents.append(this)
    return parents[0], parents[1], 


def check_genom(input_array, genom):
    layer1 = np.reshape(genom[:(784 * 10)], (784, 10))
    layer2 = np.reshape(genom[(784 * 10):(784 * 10) + (10 * 10)], (10, 10))
    bias1 = genom[-1]
    bias2 = genom[-2]

    total_error = 0
    # Перебираем все наборы данных, которые подают на вход
    for num, ia in enumerate(input_array):
        # На выходе первого скрытого слоя
        l1 = sigmoid(np.dot(ia, layer1) + bias1)
        # На выходе второго скрытого слоя
        l2 = sigmoid(np.dot(l1, layer2) + bias2)
        # Насколько мы ошиблись?
        error = np.sum(output_array[num] - l2)
        total_error += abs(error)
    return total_error


if __name__ == '__main__':
    start_time = time.time()
    input_array, output_array = get_data('data/mnist_test.csv')
    # input_array, output_array = get_data('/content/sample_data/mnist_test.csv')
    fdj = load_json('data', 'filedata.json')

    for epoch in range(5):
        num = 0
        filedata_json = dict()
        for error_key, child in fdj.items():
            filedata_json.setdefault(error_key, child)
            if num > 10:
                break
            num += 1

        keys_lst = list(set(filedata_json.keys()))
        best_dct = dict()
        if len(keys_lst) >= 2:
            keys_lst.sort()
            print(epoch, keys_lst[0], time.time() - start_time, sep='\t')
            for k in keys_lst[:2]:
                best_dct[k] = filedata_json[k]

        params = {
            'combination': random.random(),
            'mutations': random.random() / 2,
            'diff': random.randint(-10, 10),
        }
        for train in range(33):
            print(train, end=' ')
            dad, mom = get_best_parents(best_dct)
            c, d = recombination(dad, mom, **params)
            for child in [c, d]:
                child_error = check_genom(input_array, child)
                error_key = f'{child_error:020.10f}'
                filedata_json.setdefault(error_key, child)
        print()
        save_json('data', 'filedata.json', filedata_json)
