import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SamPredServices(object):
    def __init__(self):
        self.df1 = pd.read_csv('./data/kospi_data.csv', index_col=0, header=0, encoding='utf-8', sep=',')
        self.df2 = pd.read_csv('./data/samsung_data.csv', index_col=0, header=0, encoding='utf-8', sep=',')
        self.kospi200 = None
        self.samsung = None
        self.x = None
        self.y = None
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.y_train = None
        self.y_test = None

    def process(self):
        # self.df_modify()
        self.np_read()
        self.split_xy(self.samsung, 4, 1)

        self.dataset_dnn()
        # self.modeling_dnn()
        self.evaluate('dnn')

        self.dataset_lstm()
        # self.modeling_lstm()
        self.evaluate('lstm')


    def df_modify(self):
        df1 = self.df1
        df2 = self.df2
        print(df1, df1.shape)
        print(df2, df2.shape)

        df1 = df1.drop(['변동 %'], axis=1).dropna(axis=0)
        df2 = df2.drop(['변동 %'], axis=1).dropna(axis=0)
        for i in range(len(df1.index)):
            if type(df2.iloc[i,4]) == float:
                pass
            if 'K' in df1.iloc[i,4]:
                df1.iloc[i,4] = int(float(df1.iloc[i,4].replace('K', ''))*1000)
            elif 'M' in df1.iloc[i,4]:
                df1.iloc[i,4] = int(float(df1.iloc[i,4].replace('M', ''))*1000*1000)
        for i in range(len(df2.index)):
            if type(df2.iloc[i,4]) == float:
                pass
            elif 'K' in df2.iloc[i,4]:
                df2.iloc[i,4] = int(float(df2.iloc[i,4].replace('K', ''))*1000)
            elif 'M' in df2.iloc[i,4]:
                df2.iloc[i,4] = int(float(df2.iloc[i,4].replace('M', ''))*1000*1000)
        for i in range(len(df2.index)):
            for j in range(len(df2.iloc[i])-1):
                df2.iloc[i,j] = int(df2.iloc[i,j].replace(',', ''))
        df1 = df1.sort_values(['날짜'], ascending=[True])
        df2 = df2.sort_values(['날짜'], ascending=[True])
        print(df1, df1.shape)
        print(df2, df2.shape)

        df1 = df1.values
        df2 = df2.values
        np.save('./data/kospi_data.npy', arr=df1)
        np.save('./data/samsung_data.npy', arr=df2)

    def np_read(self):
        kospi200 = np.load('./data/kospi_data.npy', allow_pickle=True)
        samsung = np.load('./data/samsung_data.npy', allow_pickle=True)
        print(kospi200, kospi200.shape)
        print(samsung, samsung.shape)
        self.kospi200 = kospi200
        self.samsung = samsung

    def split_xy(self, dataset, time_steps, y_column):
        x, y = list(), list()
        for i in range(len(dataset)):
            x_end_number = i + time_steps
            y_end_number = x_end_number + y_column

            if y_end_number > len(dataset):
                break
            tmp_x = dataset[i:x_end_number, :]
            tmp_y = dataset[x_end_number:y_end_number, 3]
            x.append(tmp_x)
            y.append(tmp_y)
        self.x = np.array(x)
        self.y = np.array(y)

    def dataset_dnn(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, random_state=1, test_size=0.3)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        print(x_train.shape)
        print(x_test.shape)

        scaler = StandardScaler()
        scaler.fit(x_train)
        self.x_train_scaled = scaler.transform(x_train).astype(float)
        self.x_test_scaled = scaler.transform(x_test).astype(float)
        self.y_train = y_train.astype(float)
        self.y_test = y_test.astype(float)

    def modeling_dnn(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(20, )))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit(self.x_train_scaled, self.y_train, validation_split=0.2, verbose=1,
                  batch_size=1, epochs=100, callbacks=[early_stopping])
        model.save('./save/samsung_predict_dnn.h5')

        loss, mse = model.evaluate(self.x_test_scaled, self.y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def evaluate(self, arg):
        model = load_model('./save/samsung_predict_'+arg+'.h5')
        y_pred = model.predict(self.x_test_scaled)
        print(f'-----evaluate by {arg}-----')
        for i in range(5):
            print('close: ', self.y_test[i], ' / ', 'predict: ', y_pred[i])

    def dataset_lstm(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, random_state=1, test_size=0.3)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        print(x_train.shape)
        print(x_test.shape)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train).astype(float)
        x_test_scaled = scaler.transform(x_test).astype(float)
        self.x_train_scaled = np.reshape(x_train_scaled, (x_train_scaled.shape[0], 4, 5)).astype(float)
        self.x_test_scaled = np.reshape(x_test_scaled, (x_test_scaled.shape[0], 4, 5)).astype(float)
        self.y_train = y_train.astype(float)
        self.y_test = y_test.astype(float)

    def modeling_lstm(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(4, 5)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit(self.x_train_scaled, self.y_train, validation_split=0.2, verbose=1,
                  batch_size=1, epochs=100, callbacks=[early_stopping])
        model.save('./save/samsung_predict_lstm.h5')

        loss, mse = model.evaluate(self.x_test_scaled, self.y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)





if __name__ == '__main__':
    SamPredServices().process()