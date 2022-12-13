import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder


class Iris(object):
    def __init__(self):
        self.iris = datasets.load_iris()
        self.my_iris = None
        self._X = self.iris.data
        self._Y = self.iris.target

    def execute(self):
        self.create_model()

    def spec(self):
        print(f'{self.iris.feature_names}')

    def create_model(self):
        X = self._X
        Y = self._Y
        enc = OneHotEncoder()
        Y_1hot = enc.fit_transform(Y.reshape(-1, 1)).toarray()
        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, Y_1hot, epochs=300, batch_size=10)
        print('Model Training is completed.')

        file_name = './save/iris_model.h5'
        model.save(file_name)
        print(f'Model Saved on "{file_name}"')


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
    iris = Iris()
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