import os

import keras.datasets.mnist
import tensorflow as tf


class NumberModel(object):
    def __init__(self):
        pass

    def create_model(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print('테스트 정확도:', test_acc)

        file_name = os.path.join(os.path.abspath("save"), "number_model.h5")
        print(f"저장경로: {file_name}")
        model.save(file_name)


number_menus = ["Exit", # 0
               "Create Model", # 1
]
number_lambda = {
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
    number = NumberModel()
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