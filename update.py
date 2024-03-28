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
    """
    Сохранить json файл
    """
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


def check_one_gen(ia, oa, layer1, layer2, bias1, bias2):
    # На выходе первого скрытого слоя
    l1 = sigmoid(np.dot(ia, layer1) + bias1)
    # На выходе второго скрытого слоя
    l2 = sigmoid(np.dot(l1, layer2) + bias2)
    # Насколько мы ошиблись?
    error = np.sum((11 - (oa * 10)) * (np.square(oa - l2)))
    return error


def get_perceptron(input_array, output_array, genom):
    layer1_out = PIXELS_PER_IMAGE * HIDDEN_SIZE
    second = HIDDEN_SIZE * NUM_LABELS
    layer2_out = layer1_out + second
    bias1_out = layer2_out + HIDDEN_SIZE

    layer1 = np.reshape(genom[:layer1_out], (PIXELS_PER_IMAGE, HIDDEN_SIZE))
    layer2 = np.reshape(genom[layer1_out:layer2_out], (HIDDEN_SIZE, NUM_LABELS))
    bias1 = np.reshape(genom[layer2_out:bias1_out], (HIDDEN_SIZE,))
    bias2 = np.reshape(genom[bias1_out:], (NUM_LABELS,))

    return {"layer1": layer1, "layer2": layer2, "bias1": bias1, "bias2": bias2}


def check_genom(input_array, output_array, genom):
    perceptron_dct = get_perceptron(input_array, output_array, genom)
    total_error = 0
    # Перебираем все наборы данных, которые подают на вход
    for num, ia in enumerate(input_array):
        oa = output_array[num]
        total_error += check_one_gen(ia, oa, **perceptron_dct)
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

    # input_array, output_array = get_data("/content/sample_data/mnist_train_small.csv")
    # ('/content/sample_data/mnist_test.csv')
    input_array, output_array = get_data('data/mnist_train.csv')

    shear = sorted(
        [random.randint(0, len(output_array)), random.randint(0, len(output_array))]
    )
    input_array = input_array[shear[0] : shear[1]]
    output_array = output_array[shear[0] : shear[1]]

    # genom_dct = load_json("/content/drive/MyDrive/Colab Notebooks", "genom.json")
    genom_dct = load_json("data", "genom.json")
    if genom_dct:
        dad_key, mom_key, *_ = sorted(genom_dct)
        dad = genom_dct[dad_key]
        mom = genom_dct[mom_key]
    else:
        # Инициализация весовых коэффицентов
        dad = list()
        for _ in range((PIXELS_PER_IMAGE * HIDDEN_SIZE) + (HIDDEN_SIZE * NUM_LABELS)):
            dad.append(random.random() - 0.5)
        dad.extend([0 for _ in range(HIDDEN_SIZE)])
        dad.extend([0 for _ in range(NUM_LABELS)])

        mom = list()
        for _ in range((PIXELS_PER_IMAGE * HIDDEN_SIZE) + (HIDDEN_SIZE * NUM_LABELS)):
            mom.append(random.random() - 0.5)
        mom.extend([0 for _ in range(HIDDEN_SIZE)])
        mom.extend([0 for _ in range(NUM_LABELS)])

    while True:
        genom_dct = dict()
        son, daughter = recombination(dad, mom)

        error_dad = check_genom(input_array, output_array, dad)
        genom_dct.setdefault(error_dad, dad)

        error_mom = check_genom(input_array, output_array, mom)
        genom_dct.setdefault(error_mom, mom)

        error_son = check_genom(input_array, output_array, son)
        genom_dct.setdefault(error_son, son)

        error_daughter = check_genom(input_array, output_array, daughter)
        genom_dct.setdefault(error_daughter, daughter)

        save_json("data", "genom.json", genom_dct)
        # save_json("/content/drive/MyDrive/Colab Notebooks", "genom.json", genom_dct)
        print(genom_dct.keys())

        dad_key, mom_key, *_ = sorted(genom_dct)

        dad = genom_dct[dad_key]
        mom = genom_dct[mom_key]

        perceptron_dct = get_perceptron(input_array, output_array, dad)
        ia = input_array[0]
        oa = output_array[0]
        error = check_one_gen(ia, oa, **perceptron_dct)
        print(error)
        if error < 1:
            break
