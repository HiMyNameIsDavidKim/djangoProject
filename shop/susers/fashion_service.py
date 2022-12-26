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



class FashionService(object):
    def __init__(self):
        global class_names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # self, i, predictions_array, true_label, img
    def service_model(self, i) -> '':
        model = keras.models.load_model(r"/Users/davidkim/PycharmProjects/djangoProject/shop/susers/save/fashion_model2.h5")
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        predictions = model.predict(test_images)
        predictions_array, true_label, img = predictions[i], test_labels[i], test_images[i]

        result = np.argmax(predictions_array)
        print(f"예측한 답 : {result}")

        if result == 0:
            resp = 'T-shirt/top'
        elif result == 1:
            resp = 'Trouser'
        elif result == 2:
            resp = 'Pullover'
        elif result == 3:
            resp = 'Dress'
        elif result == 4:
            resp = 'Coat'
        elif result == 5:
            resp = 'Sandal'
        elif result == 6:
            resp = 'Shirt'
        elif result == 7:
            resp = 'Sneaker'
        elif result == 8:
            resp = 'Bag'
        elif result == 9:
            resp = 'Ankle boot'
        return resp


fashion_menus = ["Exit", # 0
                "Service", # 1
]
fashion_lambda = {
    "1": lambda t: t.service_model(3),
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
    fashion = FashionService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(fashion_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("Exit")
            break
        else:
            try:
                fashion_lambda[menu](fashion)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")