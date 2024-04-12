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


def sigmoid(x):
    """
    Сигмоида для опредления значения весов
    """
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    """
    Rectified Linear Activation
    """
    return x * (x > 0)


def LeakyReLU(x):
    """
    Leaky Rectified Linear Activation
    """
    return np.where(x > 0, x, x * 0.01)


def ELU(x):
    """
    Exponential Linear Unit
    """
<<<<<<< HEAD
    alpha = 0.9
=======
    alpha = 1.5
>>>>>>> main
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def activation(x):
    return ReLU(x)


def hexabin(x):
    """
    Приведение восьмибитного значения к уровню 0..1
    """
    return x / 255


def recombination(dad, mom, combination=0.6, mutations=0.7, diff=1):
    assert len(dad) == len(mom), "len(dad) != len(mom)"
    child1 = []
    child2 = []
    copies = False
    for n in range(len(dad)):
        if combination < random.random():
            copies = not copies
        if mutations > random.random():
            delta = random.randint(-1 * diff, 1 * diff)
            child1.append(mom[n] + delta)
            child2.append(dad[n] - delta)
        elif copies:
            child1.append(dad[n])
            child2.append(mom[n])
        else:
            child1.append(mom[n])
            child2.append(dad[n])
    return child1, child2


def check_one_gen(ia, layer1, layer2, bias1, bias2):
    # На выходе первого скрытого слоя
    l1 = activation(np.dot(ia, layer1) + bias1)
    # На выходе второго скрытого слоя
    l2 = activation(np.dot(l1, layer2) + bias2)
    return l2


def calc_error_check_one_gen(oa, l2):
    # Насколько мы ошиблись?
    # Если ячейка с максимальным значением в слое вывода совпадает
    # с ячейкой с максимальным значением в одномерной матрице
    # l - labels - то счётчик правильных ответов увеличивается
    error = int(np.argmax(oa) != np.argmax(l2))
    # error = np.sum(np.square(oa - l2))
    return error


def cut_genom(genom):
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
    perceptron_dct = cut_genom(genom)
    total_error = 0
    # Перебираем все наборы данных, которые подают на вход
    for num, ia in enumerate(input_array):
        l2 = check_one_gen(ia, **perceptron_dct)
        oa = output_array[num]
        total_error += calc_error_check_one_gen(oa, l2)
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


def create_random_genom():
    dad = list()
    for _ in range((PIXELS_PER_IMAGE * HIDDEN_SIZE) + (HIDDEN_SIZE * NUM_LABELS)):
        diff = 1
        dad.append(random.randint(-1 * diff, 1 * diff))
    dad.extend([0 for _ in range(HIDDEN_SIZE)])
    dad.extend([0 for _ in range(NUM_LABELS)])
    return dad


if __name__ == "__main__":
    data_folder = "data"
    csvname = os.path.join(data_folder, "train.csv")
    input_array, output_array = get_data(csvname)
    genom_dct = load_json(data_folder, "genom.json")

    if genom_dct and len(genom_dct) >= 2:
        dad_key, mom_key, *_ = sorted(genom_dct)
        dad = genom_dct[dad_key]
        mom = genom_dct[mom_key]
    else:
        # Инициализация весовых коэффицентов
        dad = create_random_genom()
        mom = create_random_genom()

    genesis = 0
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

        dad_key, mom_key, *_ = sorted(genom_dct)

        dad = genom_dct[dad_key]
        mom = genom_dct[mom_key]

        if dad_key < 2:
            print(genesis, genom_dct.keys())
            save_json(data_folder, "genom.json", genom_dct)
            break

        if genesis > 1000:
            print(genesis, genom_dct.keys())
            save_json(data_folder, "genom.json", genom_dct)
            break

        if genesis % 25 == 0:
            print(genesis, genom_dct.keys())

        genesis += 1
