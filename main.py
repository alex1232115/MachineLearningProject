import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, concatenate
import numpy as np
import math
import random
import time
import datetime


# Утилитарные функции ----------------------------------------------------------------------------------------


def func(x, y) -> float:
    return (np.sin(x) + np.sin(y) - 0.5 * y)


def range_with_number(start, stop, number):
    if number <= 1:
        return []
    step = (stop - start) / (number - 1)
    return [start + i * step for i in range(number)]


def get_func_results(function, start, stop, number, rand: bool) -> dict:
    res_dict = {}
    step = (stop - start) / number
    if rand:
        for i in range(number):
            x = random.uniform(start, stop)
            y = random.uniform(start, stop)
            z = function(x, y)
            res_dict[(x, y)] = z
    else:
        for i in range_with_number(start, stop, int(np.sqrt(number))):
            for j in range_with_number(start, stop, int(np.sqrt(number))):
                res_dict[(i, j)] = function(i, j)

    return res_dict


def plot_graf(cord_tuple):
    xgrid, ygrid, zgrid = cord_tuple

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xgrid, ygrid, zgrid)
    plt.show()


def generate_fn_results_for_plot(function, start, stop, number):
    x = np.linspace(start, stop, number)
    y = np.linspace(start, stop, number)

    xgrid, ygrid = np.meshgrid(x, y)

    z = function(xgrid, ygrid)
    return xgrid, ygrid, z


def print_model_results(hidden_layer_num, neuron_layer_num, score, iter_time, rand):
    print(' ')
    print('--------------------------------')
    print('Параметры модели: ')
    print('Число скрытых слоев: ', hidden_layer_num)
    print('Число нейронов в каждом слое: ', neuron_layer_num)
    print('Cредняя квадратичная ошибка: ', score)
    print('Время обучения: ', iter_time)

    if rand:
        file = 'scores_rand.txt'
    else:
        file = 'scores.txt'

    with open(file, 'a') as f:
        f.write('\n'
                '--------------------------------\n'
                'Параметры модели: \n'
                f'Число скрытых слоев: {hidden_layer_num}\n'
                f'Число нейронов в каждом слое: {neuron_layer_num}\n'
                f'Cредняя квадратичная ошибка: {score}\n'
                f'Время обучения: {iter_time}\n')


# Функции для создания, тренировки и тестирования моделей -------------------------------------------------------------


def create_model(hidden_layer_number, neuron_number, activation='sigmoid', input_number=2) -> Model:
    input_layer = Input(shape=(input_number,))

    hidden_layers = []
    for i in range(hidden_layer_number):
        if i == 0:
            hidden_layers.append(Dense(units=neuron_number, activation=activation)(input_layer))
        else:
            hidden_layers.append(Dense(units=neuron_number, activation=activation)(hidden_layers[-1]))

    output_layer = Dense(units=1, activation='linear')(hidden_layers[-1])
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

    return model


def train_model(model, func, start, stop, number, rand_data: bool, epoch, batch_size, verbose=2) -> Model:
    test_dict = get_func_results(func, start, stop, number, rand_data)
    input_list = np.array(list(test_dict.keys()))
    z_list = np.array(list(test_dict.values()))

    model.fit(input_list, z_list, epochs=epoch, batch_size=batch_size, verbose=verbose)
    return model


def test_model(model, start, stop, number, func):
    test_data = np.random.randint(start, stop, size=(number, 2)).tolist()
    real_results = []

    for sublist in test_data:
        real_results.append(func(sublist[0], sublist[1]))

    score = model.evaluate(test_data, real_results, verbose=0)
    return score


# Выбор оптимальной конфигурации --------------------------------------------------------------------------------------


def variate_layers_and_neurons(start, stop, train_number, test_number, rand: bool):
    start_val = start
    stop_val = stop

    print('Рандомные значения: ', rand)
    start_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print('Время начала сравнения: ', start_time)

    score_list = []

    for hidden_layer_num in range(1, 10):
        for neuron_layer_num in range(2, 20):
            start_time_iter = time.time()

            score = test_model(train_model(create_model(
                hidden_layer_number=hidden_layer_num,
                neuron_number=neuron_layer_num
            ),
                func=func,
                start=start_val,
                stop=stop_val,
                number=train_number,
                rand_data=rand,
                epoch=50,
                batch_size=1,
                verbose=0
            ),
                start=start_val,
                stop=stop_val,
                number=test_number,
                func=func
            )

            end_time_iter = time.time()

            iter_time = time.strftime('%H:%M:%S', time.gmtime(end_time_iter - start_time_iter))

            score_list.append({'Число скрытых слоев': hidden_layer_num,
                               'Число нейронов в каждом слое': neuron_layer_num,
                               'Cредняя квадратичная ошибка': score,
                               'Время обучения': iter_time})
            print_model_results(hidden_layer_num, neuron_layer_num, score, iter_time, rand)

    print('Рандомные значения: ', rand)
    print(score_list)
    end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print('Время конца сравнения: ', end_time)


# Подбор оптимального размера обучения -----------------------------------------------------------------------------


def different_training_set(start, stop, test_number, rand: bool):
    print('Рандомные значения: ', rand)
    start_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print('Время начала сравнения: ', start_time)

    score_list = []

    for train_number in range(100, 2000, 100):
        for epoch_number in range(1, 52, 5):
            start_time_iter = time.time()

            score = test_model(train_model(create_model(
                hidden_layer_number=5,
                neuron_number=5
            ),
                func=func,
                start=start,
                stop=stop,
                number=train_number,
                rand_data=rand,
                epoch=epoch_number,
                batch_size=1,
                verbose=0
            ),
                start=start,
                stop=stop,
                number=test_number,
                func=func
            )

            end_time_iter = time.time()
            iter_time = time.strftime('%H:%M:%S', time.gmtime(end_time_iter - start_time_iter))

            score_list.append({'Размер дадасета': train_number,
                               'Количество эпох': epoch_number,
                               'Cредняя квадратичная ошибка': score,
                               'Время обучения': iter_time})

    print('Рандомные значения: ', rand)
    print(score_list)
    end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print('Время конца сравнения: ', end_time)


# print(create_model(5, 2, "sigmoid", 2).summary())

def train_once(start, stop, train_number, test_number, rand):
    start_val = start
    stop_val = stop

    start_time_iter = time.time()

    score = test_model(train_model(create_model(hidden_layer_number=5,
                                                neuron_number=5
                                                ),
                                   func=func,
                                   start=start_val,
                                   stop=stop_val,
                                   number=train_number,
                                   rand_data=rand,
                                   epoch=40,
                                   batch_size=1,
                                   verbose=2
                                   ),
                       start=start_val,
                       stop=stop_val,
                       number=test_number,
                       func=func
                       )
    end_time_iter = time.time()

    iter_time = time.strftime('%H:%M:%S', time.gmtime(end_time_iter - start_time_iter))

    print('Время обучения: ', iter_time)
    print('Среднее квадратичная ошибка: ', score)


#train_once(1, 30, 500, 10, False)


