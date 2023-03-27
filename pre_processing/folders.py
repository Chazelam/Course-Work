import os

n = 23 #23
for i in range(2, n + 1):
    os.mkdir("../DATA/new_data/{0}/".format(i))