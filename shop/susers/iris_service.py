import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np


class IrisService(object):
    def __init__(self):
        global model, graph, target_names
        model = keras.models.load_model('/Users/davidkim/PycharmProjects/djangoProject/shop/susers/save/iris_model.h5')
        # graph = tf.get_default_graph()
        target_names = datasets.load_iris().target_names

    def execute(self):
        self.service_model()

    def service_model(self, features):
        features = np.reshape(features, (1, 4))
        Y_prob = model.predict(features, verbose=0)
        predicted = Y_prob.argmax(axis=-1)
        result = predicted[0]
        resp = ''
        if result == 0:
            resp = 'setosa, 부채붓꽃'
        elif result == 1:
            resp = 'versicolor, 버시칼라'
        elif result == 2:
            resp = 'virginica, 버지니카'
        print(f'species: {resp}')
        return resp


iris_menus = ["Exit", # 0
              "Spec", # 1
              "Execute", # 2
]
iris_lambda = {
    "1": lambda t: t.spec(),
    "2": lambda t: t.execute(),
    "3": lambda t: print(" ** No Function ** "),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}


if __name__ == '__main__':
    iris = IrisService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("Exit")
            break
        else:
            try:
                iris_lambda[menu](iris)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")