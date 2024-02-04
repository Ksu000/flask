import numpy as np
import random
import os
import json

# входной слой состоит из 784 нейронов,
# скрытый слой состоит из 40 нейронов
# выходной слой из 10 нейронов
PIXELS_PER_IMAGE = 784
HIDDEN_SIZE = 40
NUM_LABELS = 10


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


def sigmoid(x, der=False):
    """
    Сигмоида для опредления значения весов
    """
    if der:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def hexabin(x):
    """
    Приведение восьмибитного значения к уровню 0..1
    """
    return x / 255


def recombination(dad, mom, combination=0.8, mutations=0.2, diff=2):
    assert len(dad) == len(mom), "len(dad) != len(mom)"
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


def check_genom(input_array, genom, show_output=False):
    layer1_out = PIXELS_PER_IMAGE * HIDDEN_SIZE
    second = HIDDEN_SIZE * NUM_LABELS
    layer2_out = layer1_out + second

    layer1 = np.reshape(genom[:layer1_out], (PIXELS_PER_IMAGE, HIDDEN_SIZE))
    layer2 = np.reshape(genom[layer1_out:layer2_out], (HIDDEN_SIZE, NUM_LABELS))

    total_error = 0
    # Перебираем все наборы данных, которые подают на вход
    for num, ia in enumerate(input_array):
        # На выходе первого скрытого слоя
        l1 = sigmoid(np.dot(ia, layer1))
        # На выходе второго скрытого слоя
        l2 = sigmoid(np.dot(l1, layer2))
        # Насколько мы ошиблись?
        # output_array[num] * 10 + 1 * np.square(output_array[num] - l2)
        error = np.sum((11 - (output_array[num] * 10)) * (np.square(output_array[num] - l2)))
        # error = np.sum(np.square(output_array[num] - l2))
        total_error += error
        # total_error += np.sum(np.square(output_array[num] - l2))
        # total_error += np.sum(np.multiply(output_array[num], l2))
        
        # import ipdb; ipdb.set_trace()
        if show_output:
            print(l2)
            print(output_array[num])
            print(error)
            show_output = False
    return total_error


def get_data(filename):
    data = np.genfromtxt(filename, delimiter=",")
    input_array = data[1:, 1:]
    output_data = data[1:, 0]
    # Создадим вектор для проверки
    output_lst = list()
    for od in output_data:
        output_lst.append([1 if x == int(od) else 0 for x in range(10)])
    output_array = np.array(output_lst)
    return hexabin(input_array), output_array


if __name__ == "__main__":

    input_array, output_array = get_data("data/mnist_test.csv")
    # ('/content/sample_data/mnist_test.csv')
    # input_array, output_array = get_data('data/mnist_train.csv')
    
    # Инициализация весовых коэффицентов
    dad = list()
    for _ in range((PIXELS_PER_IMAGE * HIDDEN_SIZE) + (HIDDEN_SIZE * NUM_LABELS)):
        dad.append(random.random() - 0.5)
    dad.append(0)
    dad.append(0)

    mom = list()
    for _ in range((PIXELS_PER_IMAGE * HIDDEN_SIZE) + (HIDDEN_SIZE * NUM_LABELS)):
        mom.append(random.random() - 0.5)
    mom.append(0)
    mom.append(0)

    while True:
        genom_dct = dict()
        son, daughter = recombination(dad, mom)

        error_dad = check_genom(input_array, dad, show_output=(random.random() < 0.1))
        genom_dct.setdefault(error_dad, dad)

        error_mom = check_genom(input_array, mom)
        genom_dct.setdefault(error_mom, mom)

        error_son = check_genom(input_array, son)
        genom_dct.setdefault(error_son, son)

        error_daughter = check_genom(input_array, daughter)
        genom_dct.setdefault(error_daughter, daughter)

        save_json('data', 'genom.json', genom_dct)
        print(genom_dct.keys())

        dad_key, mom_key, *_ = sorted(genom_dct)

        dad = genom_dct[dad_key]
        mom = genom_dct[mom_key]

        # import ipdb; ipdb.sset_trace()
