import pandas as pd

a = pd.read_csv("population.csv")
b = a.iloc[:59946,1:]
b.to_csv("population.csv", index=False)