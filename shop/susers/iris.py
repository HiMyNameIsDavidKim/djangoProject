import pandas as pd


class Iris(object):
    def __init__(self):
        self.iris = pd.read_csv('./data/Iris.csv')
        self.my_iris = None

    def process(self):
        self.spec()

    def spec(self):
        print(" --- 1.Shape ---")
        print(self.iris.shape)
        print(" --- 2.Features ---")
        print(self.iris.columns)
        print(" --- 3.Info ---")
        print(self.iris.info())
        print(" --- 4.Case Top1 ---")
        print(self.iris.head(1))
        print(" --- 5.Case Bottom1 ---")
        print(self.iris.tail(3))
        print(" --- 6.Describe ---")
        print(self.iris.describe())
        print(" --- 7.Describe All ---")
        print(self.iris.describe(include='all'))


def menu_show(ls):
    [print(f"{i}.{j}") for i, j in enumerate(ls)]
    return input("Choose menu : ")


iris_menus = ["Exit", # 0
                "Spec", # 1
]

iris_lambda = {
    "1": lambda t: t.spec(),
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
    iris = Iris()
    while True:
        menu = menu_show(iris_menus)
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