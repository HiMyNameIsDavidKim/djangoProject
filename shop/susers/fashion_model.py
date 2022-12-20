import os

import keras.datasets.fashion_mnist
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense

class FashionModel(object):

    def __init__(self):
        pass

    def create_model(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        plt.figure()
        plt.imshow(train_images[10])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        model = Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test Accuracy is {test_acc}')
        file_name = os.path.join(os.path.abspath("save"), "fashion_model2.h5")
        print(f"저장경로: {file_name}")
        model.save(file_name)


fashion_menus = ["Exit", # 0
               "Create Model", # 1
]
fashion_lambda = {
    "1": lambda t: t.create_model(),
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
    fashion = FashionModel()
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