import time
from collections import defaultdict
from math import exp, log

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import numpy as np


class NImdbModels(object):
    def __init__(self):
        global url, save_path, review_train, k
        url = 'https://movie.naver.com/movie/point/af/list.naver?&page='
        save_path = './save/naver_movie_review_corpus.csv'
        review_train = './save/naver_movie_review_corpus.csv'
        self.word_probs = []
        k = 0.5

    def process(self):
        # self.crawling()
        self.model_fit()
        result = self.classify(input('댓글을 작성해 주세요! : '))
        print(f'positive: {result}')

    def crawling(self):
        driver = webdriver.Chrome(r'/Users/davidkim/chromedriver')
        reviews = []
        stars = []

        for page in range(1, 6):
            driver.get(url + str(page))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            title_key = soup.find_all('td', attrs={'class', 'title'})
            review = [i.br.next_element.strip() for i in title_key]
            star = [i.div.em.text for i in title_key]
            reviews += review
            stars += star
            time.sleep(0.1)
        driver.quit()

        df = pd.DataFrame({'reviews': reviews, 'stars': stars})
        df.to_csv(save_path, sep=',', encoding='utf-8', index=False)

        print(df)

    def load_corpus(self):
        corpus = pd.read_csv(review_train, sep=',', encoding='utf-8')
        corpus = np.array(corpus)
        return corpus

    def count_words(self, train_X):
        counts = defaultdict(lambda : [0,0])
        for doc, point in train_X:
            if self.isNumber(doc) is False:
                words = doc.split()
                for word in words:
                    counts[word][0 if point > 3.5 else 1] += 1
        return counts

    def isNumber(self, arg):
        try:
            float(arg)
            return True
        except ValueError:
            return False

    def probability(self, word_probs, doc):
        docwords = doc.split()
        log_prob_if_class0 = log_prob_if_class1 = 0.0
        for word, prob_if_class0, prob_if_class1 in word_probs:
            if word in docwords:
                log_prob_if_class0 += log(prob_if_class0)
                log_prob_if_class1 += log(prob_if_class1)
            else:
                log_prob_if_class0 += log(1.0 - prob_if_class0)
                log_prob_if_class1 += log(1.0 - prob_if_class1)
        prob_if_class0 = exp(log_prob_if_class0)
        prob_if_class1 = exp(log_prob_if_class1)
        return prob_if_class0 / (prob_if_class0 + prob_if_class1)

    def word_probabilities(self, counts, n_class0, n_class1, k):
        return [(w,
                (class0 + k) / (n_class0 + 2 * k),
                (class1 + k) / (n_class1 + 2 * k))
                for w, (class0, class1) in counts.items()]

    def classify(self, doc):
        return self.probability(word_probs=self.word_probs, doc=doc)

    def model_fit(self):
        train_X = self.load_corpus()
        '''
        '재밌어요': [1,0]
        '재미 없어요': [0,1]
        '''
        num_class0 = len([1 for _, point in train_X if point > 3.5])
        num_class1 = len(train_X) - num_class0
        word_counts = self.count_words(train_X)
        print(f" ************  word_counts is {word_counts}")
        self.word_probs = self.word_probabilities(word_counts, num_class0, num_class1, k)




if __name__ == '__main__':
    nimdb = NImdbModels()
    nimdb.process()