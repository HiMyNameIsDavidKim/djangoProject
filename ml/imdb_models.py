import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class ImdbModels(object):
    def __init__(self):
        global train_input, train_target, test_input, test_target, val_input, val_target
        (train_input, train_target), (test_input, test_target) = tf.keras.datasets.imdb.load_data(num_words=500)
        train_input, val_input, train_target, val_target = train_test_split(train_input, train_target,
                                                                            test_size=0.2, random_state=42)

    def process(self):
        self.seed()
        self.dataset_info()
        self.create_model()

    def seed(self):
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()

    def dataset_info(self):
        print(train_input.shape, test_input.shape)
        print(len(train_input[0]))
        print(len(train_input[1]))
        lengths = np.array([len(x) for x in train_input])
        print(np.mean(lengths), np.median(lengths))
        plt.hist(lengths)
        plt.xlabel('length')
        plt.ylabel('frequency')
        plt.show()

    def create_model(self):
        model = keras.Sequential()
        model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
        model.add(keras.layers.Dense(1, activation='sigmoid'))






if __name__ == '__main__':
    imdb = ImdbModels()
    imdb.process()