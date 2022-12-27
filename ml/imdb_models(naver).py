import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


class NImdbModels(object):
    def __init__(self):
        global url, driver, save_path, dff
        url = 'https://movie.naver.com/movie/point/af/list.naver?&page='
        driver = webdriver.Chrome(r'/Users/davidkim/chromedriver')
        save_path = './save/naver_movie_review_corpus.csv'
        dff = [[]]

    def process(self):
        self.crawling()

    def crawling(self):
        movie_names = []
        reviews = []
        stars = []

        for page in range(1, 6):
            driver.get(url + str(page))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            title_key = soup.find_all('td', attrs={'class', 'title'})
            movie_name = [i.a.text for i in title_key]
            review = [i.br.next_element.strip() for i in title_key]
            star = [f'{i.div.em.text}/10점' for i in title_key]
            movie_names += movie_name
            reviews += review
            stars += star
            time.sleep(1)
        driver.quit()

        df = pd.DataFrame({'movie_names': movie_names, 'stars': stars, 'reviews': reviews})

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(df)
        df.to_csv(save_path)


if __name__ == '__main__':
    nimdb = NImdbModels()
    nimdb.process()