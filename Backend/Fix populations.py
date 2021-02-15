# will resize populations

import pandas as pd

a = pd.read_csv("population.csv")
new_num_users = 59945
b = a.iloc[:new_num_users+1, 1:]
b.to_csv("population.csv", index=False)