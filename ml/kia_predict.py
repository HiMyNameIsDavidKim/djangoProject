import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import data
from matplotlib import font_manager, rc, pyplot as plt
import yfinance as yf
yf.pdr_override()
import platform
path = "c:/Windows/Fonts/malgun.ttf"
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')
plt.rcParams['axes.unicode_minus'] = False


class KiaPredServices(object):
    def __init__(self):
        global start_date, end_date, item_code, save_path
        start_date = '2018-1-4'
        end_date = '2021-12-31'
        item_code = '000270.KS'
        save_path = './save/kia_close.png'

    def process(self):
        self.item_predict()

    def item_predict(self):
        item = data.get_data_yahoo(item_code, start_date, end_date)
        print(f'item head: {item.head(3)}')
        print(f'item tail: {item.tail(3)}')
        item['Close'].plot(figsize=(12, 6), grid=True)
        item_trunc = item[:'2021-12-31']
        df = pd.DataFrame({'ds': item_trunc.index, 'y': item_trunc['Close']})
        df.reset_index(inplace=True)
        del df['Date']
        prophet = Prophet(daily_seasonality=True)
        prophet.fit(df)
        future = prophet.make_future_dataframe(periods=61)
        forecast = prophet.predict(future)
        prophet.plot(forecast)
        plt.figure(figsize=(12, 6))
        plt.plot(item.index, item['Close'], label='real')
        plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
        plt.grid()
        plt.legend()
        plt.savefig(save_path)


if __name__ == '__main__':
    KiaPredServices().process()