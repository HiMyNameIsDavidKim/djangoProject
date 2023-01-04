from enum import Enum

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, concatenate
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from abc import *


class StockPredModels(object):
    def __init__(self):
        global kospi_csv, sam_csv, kospi_npy, sam_npy, \
            dnn_model, dnn_ensemble, lstm_model, lstm_ensemble
        origin_path = '/Users/davidkim/PycharmProjects/djangoProject/ml'
        kospi_csv = origin_path + '/data/kospi_data.csv'
        sam_csv = origin_path + '/data/samsung_data.csv'
        kospi_npy = origin_path + '/data/kospi_data.npy'
        sam_npy = origin_path + '/data/samsung_data.npy'
        dnn_model = origin_path + '/save/samsung_predict_dnn_sam.h5'
        dnn_ensemble = origin_path + '/save/samsung_predict_dnn_ensemble.h5'
        lstm_model = origin_path + '/save/samsung_predict_lstm_sam.h5'
        lstm_ensemble = origin_path + '/save/samsung_predict_lstm_ensemble.h5'
        self.df_kospi = pd.read_csv(kospi_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.df_sam = pd.read_csv(sam_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.kospi200 = None
        self.samsung = None

    def process(self):
        # self.df_modify(self.df_kospi, self.df_sam)
        # self.np_read()
        # self.process_dnn(fit_refresh=False)
        # self.process_lstm(fit_refresh=False)
        # self.process_dnn_ensemble(fit_refresh=False)
        # self.process_lstm_ensemble(fit_refresh=False)
        # test_dnn = self.pred_test(dnn_model, x_test_dnn, y_test_dnn)
        # test_lstm = self.pred_test(lstm_model, x_test_lstm, y_test_lstm)
        # test_dnn_en = self.pred_test(dnn_ensemble, x_test_dnn_en, y_test_dnn_en)
        # test_lstm_en = self.pred_test(lstm_ensemble, x_test_lstm_en, y_test_lstm_en)

        req_kospi, req_sam = self.df_req('2022- 12- 23')
        req_kospi, req_sam = self.df_modify(req_kospi, req_sam)
        req_x, req_y = self.split_xy(req_sam, 4, 1)
        req_x, req_y = self.dataset_req(lstm_model, req_x, req_y)
        result = self.pred_req(lstm_model, req_x, req_y)

    def process_dnn(self, fit_refresh):
        x, y = self.split_xy(self.samsung, 4, 1)
        x_train_scaled, x_test_scaled, y_train, y_test = self.dataset_dnn(x, y)
        if fit_refresh:
            self.modeling_dnn(dnn_model, x_train_scaled, x_test_scaled, y_train, y_test)
        global x_test_dnn, y_test_dnn
        x_test_dnn, y_test_dnn = x_test_scaled, y_test

    def process_lstm(self, fit_refresh):
        x, y = self.split_xy(self.samsung, 4, 1)
        x_train_scaled, x_test_scaled, y_train, y_test = self.dataset_lstm(x, y)
        if fit_refresh:
            self.modeling_lstm(lstm_model, x_train_scaled, x_test_scaled, y_train, y_test)
        global x_test_lstm, y_test_lstm
        x_test_lstm, y_test_lstm = x_test_scaled, y_test

    def process_dnn_ensemble(self, fit_refresh):
        x1, y1 = self.split_xy(self.samsung, 4, 1)
        x1_train_scaled, x1_test_scaled, y1_train, y1_test = self.dataset_dnn(x1, y1)
        x2, y2 = self.split_xy(self.kospi200, 4, 1)
        x2_train_scaled, x2_test_scaled, y2_train, y2_test = self.dataset_dnn(x2, y2)
        if fit_refresh:
            self.modeling_dnn_ensemble(x1_train_scaled, x2_train_scaled, y1_train,
                                       x1_test_scaled, x2_test_scaled, y1_test)
        global x_test_dnn_en, y_test_dnn_en
        x_test_dnn_en, y_test_dnn_en = [x1_test_scaled, x2_test_scaled], y1_test

    def process_lstm_ensemble(self, fit_refresh):
        x1, y1 = self.split_xy(self.samsung, 4, 1)
        x1_train_scaled, x1_test_scaled, y1_train, y1_test = self.dataset_lstm(x1, y1)
        x2, y2 = self.split_xy(self.samsung, 4, 1)
        x2_train_scaled, x2_test_scaled, y2_train, y2_test = self.dataset_lstm(x2, y2)
        if fit_refresh:
            self.modeling_lstm_ensemble(x1_train_scaled, x2_train_scaled, y1_train,
                                       x1_test_scaled, x2_test_scaled, y1_test)
        global x_test_lstm_en, y_test_lstm_en
        x_test_lstm_en, y_test_lstm_en = [x1_test_scaled, x2_test_scaled], y1_test

    def df_modify(self, df1, df2):
        print(df1, df1.shape)
        print(df2, df2.shape)

        df1 = df1.drop(['변동 %'], axis=1).dropna(axis=0)
        df2 = df2.drop(['변동 %'], axis=1).dropna(axis=0)
        for i in range(len(df1.index)):
            if type(df2.iloc[i, 4]) == float:
                pass
            if 'K' in df1.iloc[i, 4]:
                df1.iloc[i, 4] = int(float(df1.iloc[i, 4].replace('K', '')) * 1000)
            elif 'M' in df1.iloc[i, 4]:
                df1.iloc[i, 4] = int(float(df1.iloc[i, 4].replace('M', '')) * 1000 * 1000)
        for i in range(len(df2.index)):
            if type(df2.iloc[i, 4]) == float:
                pass
            elif 'K' in df2.iloc[i, 4]:
                df2.iloc[i, 4] = int(float(df2.iloc[i, 4].replace('K', '')) * 1000)
            elif 'M' in df2.iloc[i, 4]:
                df2.iloc[i, 4] = int(float(df2.iloc[i, 4].replace('M', '')) * 1000 * 1000)
        for i in range(len(df2.index)):
            for j in range(len(df2.iloc[i]) - 1):
                df2.iloc[i, j] = int(df2.iloc[i, j].replace(',', ''))
        df1 = df1.sort_values(['날짜'], ascending=[True])
        df2 = df2.sort_values(['날짜'], ascending=[True])
        print(df1, df1.shape)
        print(df2, df2.shape)

        df1 = df1.values
        df2 = df2.values
        np.save(kospi_npy, arr=df1)
        np.save(sam_npy, arr=df2)
        np_df1 = np.array(df1)
        np_df2 = np.array(df2)
        return np_df1, np_df2

    def np_read(self):
        kospi200 = np.load(kospi_npy, allow_pickle=True)
        samsung = np.load(sam_npy, allow_pickle=True)
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
        return np.array(x), np.array(y)

    def dataset_dnn(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train).astype(float)
        x_test_scaled = scaler.transform(x_test).astype(float)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        return x_train_scaled, x_test_scaled, y_train, y_test

    def modeling_dnn(self, name, x_train_scaled, x_test_scaled, y_train, y_test):
        model = Sequential()
        model.add(Dense(64, input_shape=(20,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
                  batch_size=1, epochs=100, callbacks=[early_stopping])
        model.save(name)

        loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def dataset_lstm(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train).astype(float)
        x_test_scaled = scaler.transform(x_test).astype(float)
        x_train_scaled = np.reshape(x_train_scaled, (x_train_scaled.shape[0], 4, 5)).astype(float)
        x_test_scaled = np.reshape(x_test_scaled, (x_test_scaled.shape[0], 4, 5)).astype(float)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        return x_train_scaled, x_test_scaled, y_train, y_test

    def modeling_lstm(self, name, x_train_scaled, x_test_scaled, y_train, y_test):
        model = Sequential()
        model.add(LSTM(64, input_shape=(4, 5)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
                  batch_size=1, epochs=100, callbacks=[early_stopping])
        model.save(name)

        loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def modeling_dnn_ensemble(self,
                              x1_train_scaled, x2_train_scaled, y1_train,
                              x1_test_scaled, x2_test_scaled, y1_test):
        input1 = Input(shape=(20,))
        dense1 = Dense(64)(input1)
        dense1 = Dense(32)(dense1)
        dense1 = Dense(32)(dense1)
        output1 = Dense(32)(dense1)

        input2 = Input(shape=(20,))
        dense2 = Dense(64)(input2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        output2 = Dense(32)(dense2)

        merge = concatenate([output1, output2])
        output3 = Dense(1)(merge)

        model = Model(inputs=[input1, input2], outputs=output3)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit([x1_train_scaled, x2_train_scaled], y1_train,
                  validation_split=0.2, verbose=1, batch_size=1, epochs=100,
                  callbacks=[early_stopping])
        model.save(dnn_ensemble)

        loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def modeling_lstm_ensemble(self,
                               x1_train_scaled, x2_train_scaled, y1_train,
                               x1_test_scaled, x2_test_scaled, y1_test):
        input1 = Input(shape=(4, 5))
        dense1 = LSTM(64)(input1)
        dense1 = Dense(32)(dense1)
        dense1 = Dense(32)(dense1)
        output1 = Dense(32)(dense1)

        input2 = Input(shape=(4, 5))
        dense2 = LSTM(64)(input2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        output2 = Dense(32)(dense2)

        merge = concatenate([output1, output2])
        output3 = Dense(1)(merge)

        model = Model(inputs=[input1, input2], outputs=output3)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit([x1_train_scaled, x2_train_scaled], y1_train,
                  validation_split=0.2, verbose=1, batch_size=1, epochs=100,
                  callbacks=[early_stopping])
        model.save(lstm_ensemble)

        loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def pred_test(self, name, x_test_scaled, y_test):
        model = load_model(name)
        y_pred = model.predict(x_test_scaled)
        print(f'-----predict test by {name[70:]}-----')
        for i in range(5):
            print('close: ', int(y_test[i]), ' / ', 'predict: ', int(y_pred[i]))
        return str(int(y_test[0]))

    def df_req(self, date):
        df_kospi = pd.read_csv(kospi_csv)
        df_sam = pd.read_csv(sam_csv)
        i_kospi = df_kospi.index[(df_kospi["날짜"] == date)].to_list()[0]
        i_sam = df_sam.index[(df_sam["날짜"] == date)].to_list()[0]
        req_kospi = self.df_kospi.iloc[i_kospi:i_kospi+5]
        req_sam = self.df_sam.iloc[i_sam:i_sam+5]
        return req_kospi, req_sam

    def dataset_req(self, name, x, y):
        x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x).astype(float)
        if name == lstm_model:
            x = np.reshape(x, (x.shape[0], 4, 5)).astype(float)
        y = y.astype(float)
        return x, y

    def pred_req(self, name, x, y):
        model = load_model(name)
        y_pred = model.predict(x)
        print(f'-----predict test by {name[70:]}-----')
        print('close: ', int(y), ' / ', 'predict: ', int(y_pred))
        return str(int(y_pred))


if __name__ == '__main__':
    StockPredModels().process()
