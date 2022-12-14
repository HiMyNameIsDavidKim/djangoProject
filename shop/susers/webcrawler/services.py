import csv
import pandas as pd

from shop.susers.webcrawler.models import ScrapVO
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


class ScrapService(ScrapVO):
    def __init__(self):
        global driverpath, naver_url, savepath
        naver_url = f'https://movie.naver.com/movie/sdb/rank/rmovie.naver'
        savepath = f'/Users/davidkim/PycharmProjects/djangoProject/shop/susers/webcrawler/save/result.csv'

    def musicChart(self, arg): # 기본 크롤링
        soup = BeautifulSoup(urlopen(arg.domain + arg.query_string), 'lxml')
        title = {'class': arg.class_names[0]}
        artist = {'class': arg.class_names[1]}
        titles = soup.find_all(name=arg.tag_name, attrs=title)
        artists = soup.find_all(name=arg.tag_name, attrs=artist)
        titles = [i.find('a').text for i in titles]
        artists = [i.find('a').text for i in artists]

        [print(f'{i + 1}위.{j} - {k}')
         for i, j, k in zip(range(len(titles)), titles, artists)]

        diction = {}
        for i, j in enumerate(titles):
            diction[j] = artists[i]

        arg.diction = diction
        arg.dict_to_dataframe()
        arg.dataframe_to_csv()

    def naver_movie_review(self):
        driver = webdriver.Chrome('/Users/davidkim/chromedriver')
        driver.get(naver_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        all_divs = soup.find_all('div', attrs={'class', 'tit3'})
        products = [[div.a.string for div in all_divs]]
        with open(savepath, 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerows(products)
        driver.close()
        df = pd.read_csv(savepath)
        resp = [{'rank': f"{i + 1}", 'title': f"{j}"} for i, j in enumerate(df)]
        return resp


if __name__ == '__main__':
    ss = ScrapService()
    print(ss.naver_movie_review())
