import decimal
import os
import keras
import self as self
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import random

os.environ['TF_CPP_MIN_LOG-LEVEL'] = '2'


def func(x, y) -> float:
    # return -5 * y ** 2 * np.sin(x + y) - 0.001 * x + 0.55 * y + - 0.001*x**2 - y**2

    # return ((np.exp((-x/3 ** 2) - (y/3 ** 2))) * 2)
    return 0.5 * np.sin(2 * x) + np.cos(x) + np.exp(-x ** 2 - y ** 2) * 8.5 + np.cos(x + y) * 5 + x + y + 10


def create_my_plot(function):
    x = np.linspace(-4, 4, 100)  # returns an array of evenly spaced values within a specified range
    y = np.linspace(-4, 4, 100)

    x_grid, y_grid = np.meshgrid(x, y)  # create grid of coordinates

    z_grid = function(x_grid, y_grid)

    fig = plt.figure(
        dpi=100)  # creates a blank canvas where various elements of the plot can be added using other methods

    ax = fig.add_subplot(111, projection='3d')  # добавления нового графика или подграфика к существующей фигуре.
    # Он принимает такие параметры, как количество строк, количество столбцов и индекс графика или подграфика.

    ax.plot_surface(x_grid, y_grid, z_grid)  # Для создания 3D-графиков поверхности. Это позволяет пользователям
    # визуализировать данные в трех измерениях путем построения поверхности, определяемой координатами x, y и z.

    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


create_my_plot(func)


def generate_random_results(function, number_of_values) -> dict:
    result_dictionary = {}

    count = 0

    while count != number_of_values:
        x_value = random.uniform(-4, 4)  # This function returns a random floating-point number
        # in the range from a to b inclusive
        y_value = random.uniform(-4, 4)  # This function returns a random floating-point number
        # in the range from a to b inclusive
        result_function_z = function(x_value, y_value)

        result_dictionary[(x_value, y_value)] = result_function_z

        count = count + 1

    return result_dictionary


def function_result(function, number_of_values) -> dict:
    result_dictionary = {}

    number_of_values = int(np.sqrt(number_of_values))

    range_number_x = []
    range_number_y = []

    range_number_x = np.linspace(-4, 4, number_of_values)
    range_number_y = np.linspace(-4, 4,
                                 number_of_values)  # np.linspace is a function in the NumPy library in Python that
    # returns evenly spaced numbers over a specified interval. It takes three arguments: start, stop, and num.

    for i in range_number_x:
        for j in range_number_y:
            result_dictionary[(i, j)] = function(i, j)

    return result_dictionary


# Функции для создания, тренировки и тестирования моделей -------------------------------------------------------------

def create_network_model(hidden_layers, neurons, input_number=2) -> Model:
    model_test = keras.Sequential()

    model_test.add(Dense(units=neurons, activation='relu', input_shape=(2,)))

    for count in range(2, hidden_layers):
        neuron_number = neurons * count
        model_test.add(Dense(units=neuron_number, activation='relu'))

    model_test.add(Dense(units=1, activation='linear', name="output_layer"))

    model_ex = keras.Model(inputs=model_test.inputs, outputs=model_test.layers[-1].output)

    model_ex.compile(optimizer='adam', loss='mean_squared_error')

    return model_ex


# -----------------------------------------------------------------------------------------------------------------------


def train_model(model, function, number_of_values, rand_data: bool, epoch, batch_size, verbose=2) -> Model:
    if rand_data:
        training_dict = generate_random_results(function, number_of_values)
    else:
        training_dict = function_result(function, number_of_values)

    input_list = np.array(list(training_dict.keys()))  # значения x,y
    z_list = np.array(list(training_dict.values()))  # значение функции в этих координатах

    model.fit(input_list, z_list, epochs=epoch, batch_size=batch_size, verbose=verbose)

    return model


def test_model(model, number_of_values, function):
    test_data = np.random.randint(-4, 4, size=(number_of_values, 2)).tolist()
    real_results = []

    for sublist in test_data:
        real_results.append(function(sublist[0], sublist[1]))

    score = model.evaluate(test_data, real_results, verbose=0)  # Returns the loss value & metrics values
    # for the model in test mode.

    return score


# ---------------------------------------------------------------------------------------------------------------------------------


def neuron_dependency(rand: bool, train_num):
    result_dict_neuron = {}

    for layers in range(2, 8):
        neuron_num = layers * 2

        model = create_network_model(hidden_layers=layers, neurons=neuron_num)

        training_model = train_model(model=model, function=func, number_of_values=train_num, rand_data=rand,
                                     epoch=50, batch_size=1, verbose=2)

        result = test_model(training_model, train_num, func)

        a = 0

        if layers > 2:
            for i in range(1, layers, +1):
                a = a + neuron_num * i

        neuron_sum = neuron_num + 1 + a

        result_dict_neuron[neuron_sum] = result

    my_list = result_dict_neuron.items()
    x, y = zip(*my_list)

    plt.plot(x, y)
    plt.xlabel('Количество нейронов')
    plt.ylabel('Ошибка')
    plt.title('Зависимость кол-ва нейронов на точность')
    plt.show()


# neuron_dependency(False, 200)


def layers_dependency(rand: bool, train_num):
    result_dict_layers = {}

    for layers in range(2, 8):
        neuron_num = layers * 2

        model = create_network_model(hidden_layers=layers, neurons=neuron_num)

        training_model = train_model(model=model, function=func, number_of_values=train_num, rand_data=rand,
                                     epoch=50, batch_size=1, verbose=2)

        result = test_model(training_model, train_num, func)

        result_dict_layers[layers] = result

    my_list = result_dict_layers.items()
    x, y = zip(*my_list)

    plt.plot(x, y)
    plt.xlabel('Количество слоёв')
    plt.ylabel('Ошибка')
    plt.title('Зависимость кол-ва слоёв на точность')
    plt.show()


# layers_dependency(False, 200)


def search_optimal_training_size(rand: bool):
    result_dict = {}

    sizes = [100, 250, 500, 1000, 2000]

    for size in sizes:
        model = create_network_model(hidden_layers=6, neurons=12)

        training_model = train_model(model=model, function=func, number_of_values=size, rand_data=rand,
                                     epoch=50, batch_size=1, verbose=2)

        result = test_model(training_model, size, func)

        result_dict[size] = result

    my_list = result_dict.items()
    x, y = zip(*my_list)

    plt.plot(x, y)
    plt.xlabel('Training size')
    plt.ylabel('Loss')
    plt.title('Search optimal training size')
    plt.show()


# search_optimal_training_size(False)


# 1 модель с равномерным распределением

def model_uniform(train_number=1000):
    model_test = keras.Sequential()

    model_test.add(Dense(units=30, activation='relu', input_shape=(2,)))

    model_test.add(Dense(units=30, activation='relu'))
    model_test.add(Dense(units=30, activation='relu'))
    model_test.add(Dense(units=30, activation='relu'))
    model_test.add(Dense(units=30, activation='relu'))

    model_test.add(Dense(units=30, activation='linear', name="output_layer"))

    model = keras.Model(inputs=model_test.inputs, outputs=model_test.layers[-1].output)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return test_model(train_model(model, func, 1000, False, 50, 1, 0), train_number, func)


# 2 модель с последовательным распределением
def model_sequential(train_number=1000):
    model_test = keras.Sequential()

    model_test.add(Dense(units=10, activation='relu', input_shape=(2,)))

    model_test.add(Dense(units=12, activation='relu'))
    model_test.add(Dense(units=24, activation='relu'))
    model_test.add(Dense(units=36, activation='relu'))
    model_test.add(Dense(units=48, activation='relu'))

    model_test.add(Dense(units=60, activation='linear', name="output_layer"))

    model = keras.Model(inputs=model_test.inputs, outputs=model_test.layers[-1].output)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return test_model(train_model(model, func, 1000, False, 50, 1, 2), train_number, func)


# 3 модель с обратным распределение
def model_reverse(train_number=1000):
    model_test = keras.Sequential()

    model_test.add(Dense(units=60, activation='relu', input_shape=(2,)))

    model_test.add(Dense(units=48, activation='relu'))
    model_test.add(Dense(units=36, activation='relu'))
    model_test.add(Dense(units=24, activation='relu'))
    model_test.add(Dense(units=12, activation='relu'))

    model_test.add(Dense(units=10, activation='linear', name="output_layer"))

    model = keras.Model(inputs=model_test.inputs, outputs=model_test.layers[-1].output)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return test_model(train_model(model, func, 1000, False, 50, 1, 2), train_number, func)


# result_1 = model_uniform(500)
# result_2 = model_sequential(500)
# result_3 = model_reverse(500)
#
# print("Ошибка на 1 моделе (42,42,42,42,42): ", result_1)
# print("Ошибка на 2 моделе (4,8,32,64,102): ", result_2)
# print("Ошибка на 3 моделе (102,64,32,8,4): ", result_3)
#
# plt.bar(['uniform', 'sequential', 'reverse'], [result_1, result_2, result_3], color='orange')


def epoch_search(rand: bool, number_of_values):
    result_dict = {}

    epochs = [10, 20, 40, 60, 100]

    model_test = keras.Sequential()

    model_test.add(Dense(units=10, activation='relu', input_shape=(2,)))

    model_test.add(Dense(units=36, activation='relu'))
    model_test.add(Dense(units=36, activation='relu'))
    model_test.add(Dense(units=36, activation='relu'))
    model_test.add(Dense(units=36, activation='relu'))

    model_test.add(Dense(units=36, activation='linear', name="output_layer"))

    model = keras.Model(inputs=model_test.inputs, outputs=model_test.layers[-1].output)

    model.compile(optimizer='adam', loss='mean_squared_error')

    min_result = decimal.MAX_EMAX
    optimal_epoch = 0

    for epoch in epochs:

        training_model = train_model(model=model, function=func, number_of_values=number_of_values, rand_data=rand,
                                     epoch=epoch, batch_size=1, verbose=0)

        result = test_model(training_model, number_of_values, func)

        result_dict[epoch] = result

        if result < min_result:
            min_result = result
            optimal_epoch = epoch

    my_list = result_dict.items()
    x, y = zip(*my_list)

    plt.plot(x, y)
    plt.xlabel('Epoch size')
    plt.ylabel('Loss')
    plt.title('Search optimal epoch number')
    plt.show()

    print("Оптимальное количество эпох: ", optimal_epoch)


# epoch_search(False, 500)


def optimal_model():
    # модель со случайными данными
    model_test_random = keras.Sequential()

    model_test_random.add(Dense(units=10, activation='relu', input_shape=(2,)))

    model_test_random.add(Dense(units=36, activation='relu'))
    model_test_random.add(Dense(units=36, activation='relu'))
    model_test_random.add(Dense(units=36, activation='relu'))
    model_test_random.add(Dense(units=36, activation='relu'))

    model_test_random.add(Dense(units=36, activation='linear', name="output_layer"))

    model_random = keras.Model(inputs=model_test_random.inputs, outputs=model_test_random.layers[-1].output)

    model_random.compile(optimizer='adam', loss='mean_squared_error')

    training_model_random = train_model(model=model_random, function=func, number_of_values=1000, rand_data=True,
                                        epoch=100, batch_size=1, verbose=0)

    result_random = test_model(training_model_random, 1000, func)

    # модель с регулярными данными
    model_test_regular = keras.Sequential()

    model_test_regular.add(Dense(units=10, activation='relu', input_shape=(2,)))

    model_test_regular.add(Dense(units=36, activation='relu'))
    model_test_regular.add(Dense(units=36, activation='relu'))
    model_test_regular.add(Dense(units=36, activation='relu'))
    model_test_regular.add(Dense(units=36, activation='relu'))

    model_test_regular.add(Dense(units=36, activation='linear', name="output_layer_2"))

    model_regular = keras.Model(inputs=model_test_regular.inputs, outputs=model_test_regular.layers[-1].output)

    model_regular.compile(optimizer='adam', loss='mean_squared_error')

    training_model_regular = train_model(model=model_random, function=func, number_of_values=1000, rand_data=False,
                                         epoch=100, batch_size=1, verbose=0)

    result_regular = test_model(training_model_regular, 1000, func)

    print("Оптимальная случайная выборка: ", result_random)
    print("Оптимальная регулярная выборка: ", result_regular)


#optimal_model()


def create_optimal_model():
    model = keras.Sequential()

    model.add(Dense(units=10, activation='relu', input_shape=(2,)))

    model.add(Dense(units=36, activation='relu'))
    model.add(Dense(units=36, activation='relu'))
    model.add(Dense(units=36, activation='relu'))
    model.add(Dense(units=36, activation='relu'))

    model.add(Dense(units=36, activation='linear', name="output_layer"))

    model_optimal = keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)

    model_optimal.compile(optimizer='adam', loss='mean_squared_error')

    training_model_random = train_model(model=model_optimal, function=func, number_of_values=1000, rand_data=False,
                                        epoch=100, batch_size=1, verbose=0)

    result = test_model(training_model_random, 1000, func)

    print("Ошибка на оптимальной модели : ", result)


model = create_optimal_model

x = np.linspace(-4, 4, 1000)
y = np.linspace(-4, 4, 1000)

xgrid, ygrid = np.meshgrid(x, y)

z = func(xgrid, ygrid)

Z_model = self.model.predict(np.column_stack((xgrid.ravel(), ygrid.ravel())))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xgrid, ygrid, Z_model.reshape(xgrid.shape))

plt.show()
