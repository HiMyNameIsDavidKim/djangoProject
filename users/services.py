import random
import string

import pandas as pd
from sqlalchemy import create_engine


class UserService(object):
    def __init__(self):
        global username, password, created_at, rank, point
        username = []
        password = []
        created_at = []
        rank = []
        point = []

    def execute(self):
        df = self.create_dummy_df()
        self.sql_insert(df)

    def create_dummy_df(self):
        name_set = string.ascii_letters

        while len(username) != 100:
            rand_name = ''.join(random.sample(name_set, 5))
            if (rand_name in username):
                pass
            else:
                username.append(rand_name)
                password.append(str(123))
                created_at.append(["2022-12-22"])
                rank.append([1])
                point.append([0])

        df = pd.DataFrame({'username': username, 'password': password,
                           'created_at': created_at, 'rank': rank, 'point': point})
        print(df)
        return df

    def sql_insert(self, df):
        engine = create_engine(
            "mysql+pymysql://root:root@localhost:3306/mydb",
            encoding='utf-8')
        df.to_sql(name='users',
                  if_exists='append',
                  con=engine,
                  index=False)


if __name__ == '__main__':
    UserService().execute()
