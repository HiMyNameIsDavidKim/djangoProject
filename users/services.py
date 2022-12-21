import random
import string


class UserService(object):
    def __init__(self):
        global id, username, password
        id = []
        username = []
        password = []

    def dummy_df(self):
        id_set = string.ascii_letters + string.digits
        name_set = string.ascii_letters

        while len(id) != 100:
            rand_id = ''.join(random.sample(id_set, 10))
            rand_name = ''.join(random.sample(name_set, 5))
            if (rand_id in id) or (rand_name in username):
                pass
            else:
                id.append(rand_id)
                username.append(rand_name)
                password.append(str(123))
        df = [id, username, password]
        return df

if __name__ == '__main__':
    us = UserService()
    df = us.dummy_df()
    print(df)