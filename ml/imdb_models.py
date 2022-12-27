import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class ImdbModels(object):
    def __init__(self):
        global train_input, train_target, test_input, test_target, \
            val_input, val_target, train_seq, val_seq, sim_model_path, imb_model_path
        (train_input, train_target), (test_input, test_target) = tf.keras.datasets.imdb.load_data(num_words=500)
        train_input, val_input, train_target, val_target = train_test_split(train_input, train_target,
                                                                            test_size=0.2, random_state=42)
        train_seq = tf.keras.preprocessing.sequence.pad_sequences(train_input, maxlen=100)
        val_seq = tf.keras.preprocessing.sequence.pad_sequences(val_input, maxlen=100)
        sim_model_path = './save/best-simplernn-model.h5'
        imb_model_path = './save/best-embedding-model.h5'

    def process(self):
        self.seed()
        self.dataset_info()
        self.create_model()
        self.create_imbeded()

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
        train_oh = keras.utils.to_categorical(train_seq)
        print(train_oh.shape)
        val_oh = keras.utils.to_categorical(val_seq)
        print(val_oh.shape)
        model.summary()

        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model.compile(optimizer=rmsprop, loss='binary_crossentropy',
                      metrics=['accuracy'])

        checkpoint_cb = keras.callbacks.ModelCheckpoint(sim_model_path,
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                          restore_best_weights=True)

        history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                            validation_data=(val_oh, val_target),
                            callbacks=[checkpoint_cb, early_stopping_cb])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()

    def create_imbeded(self):
        model2 = keras.Sequential()
        model2.add(keras.layers.Embedding(500, 16, input_length=100))
        model2.add(keras.layers.SimpleRNN(8))
        model2.add(keras.layers.Dense(1, activation='sigmoid'))
        model2.summary()

        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
                       metrics=['accuracy'])

        checkpoint_cb = keras.callbacks.ModelCheckpoint(imb_model_path,
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                          restore_best_weights=True)

        history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                             validation_data=(val_seq, val_target),
                             callbacks=[checkpoint_cb, early_stopping_cb])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()


if __name__ == '__main__':
    imdb = ImdbModels()
    imdb.process()