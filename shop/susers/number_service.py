import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import os
import keras.datasets.mnist



class NumberService(object):
    def __init__(self):
        global class_names
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # self, i, predictions_array, true_label, img
    def service_model(self, i) -> '':
        # i = int(input('input test number : '))
        model = keras.models.load_model(r"/Users/davidkim/PycharmProjects/djangoProject/shop/susers/save/number_model.h5")
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        predictions = model.predict(test_images)
        predictions_array, true_label, img = predictions[i], test_labels[i], test_images[i]

        result = np.argmax(predictions_array)
        print(f"예측한 답 : {str(result)}")
        return str(result)



number_menus = ["Exit", # 0
                "Service", # 1
]
number_lambda = {
    "1": lambda t: t.service_model(),
    "2": lambda t: print(" ** No Function ** "),
    "3": lambda t: print(" ** No Function ** "),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}


if __name__ == '__main__':
    number = NumberService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(number_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("Exit")
            break
        else:
            try:
                number_lambda[menu](number)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")