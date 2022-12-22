import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.saving.save import load_model


class FruitServices(object):
    def __init__(self):
        global test_data_dir, train_data_dir, apple_name, batch_size, img_height, img_width, model_path
        test_data_dir = f'/Users/davidkim/PycharmProjects/djangoProject/ml/fruits-360-5/Test'
        train_data_dir = f'/Users/davidkim/PycharmProjects/djangoProject/ml/fruits-360-5/Training'
        apple_name = ['/Apple Braeburn', '/Apple Crimson Snow', '/Apple Golden 1', '/Apple Golden 2', '/Apple Golden 3']
        batch_size = 32
        img_height = 100
        img_width = 100
        model_path = '/Users/davidkim/PycharmProjects/djangoProject/ml/save/CNNClassifier.h5'

    def data_set(self):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_data_dir,
            validation_split=0.3,
            subset="training",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_data_dir,
            validation_split=0.3,
            subset="validation",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_data_dir,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        self.test_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
            test_data_dir,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False)

    def show_class_names(self):
        self.class_names = self.train_ds.class_names
        print(self.class_names)

    def execute(self):
        self.data_set()
        self.show_class_names()
        # self.show_apple()
        # self.show_ds()
        # self.create_model()
        # self.show_graph()
        self.eval_model()

    def show_apple(self):
        img = tf.keras.preprocessing.image.load_img(train_data_dir + apple_name[4] + '/0_100.jpg')
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def show_ds(self):
        y = np.concatenate([y for x, y in self.test_ds1], axis=0)
        x = np.concatenate([x for x, y in self.test_ds1], axis=0)
        plt.figure(figsize=(3, 3))
        plt.imshow(x[0].astype("uint8"))
        plt.title(self.class_names[y[0]])
        plt.axis("off")
        plt.show()

    def create_model(self):
        BUFFER_SIZE = 10000
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        print(type(self.train_ds))

        num_classes = 5
        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(.50),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(.50),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dropout(.50),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.summary()

        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        checkpointer = ModelCheckpoint(model_path, save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy',
                                                          restore_best_weights=True)
        epochs = 20
        self.history = model.fit(
            self.train_ds,
            batch_size=batch_size,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[checkpointer, early_stopping_cb]
        )
        print(len(self.history.history['val_accuracy']))

    def show_graph(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = range(1, len(self.history.history['val_accuracy']) + 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def eval_model(self):
        model = load_model(model_path)

        test_loss, test_acc = model.evaluate(self.test_ds)
        print("test loss: ", test_loss)
        print("test accuracy: ", test_acc)

        predictions = model.predict(self.test_ds1)
        score = tf.nn.softmax(predictions[0])
        print(f"This is {self.class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} % confidence.")
        score = tf.nn.softmax(predictions[-1])
        print(f"This is {self.class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} % confidence.")


if __name__ == '__main__':
    FruitServices().execute()