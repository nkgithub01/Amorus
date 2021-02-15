# will resize populations

import pandas as pd
from timeit import timeit
a = pd.read_csv("population.csv")
new_num_users = 1000
b = a.iloc[:new_num_users, :]
b.to_csv("populations.csv", index=False)