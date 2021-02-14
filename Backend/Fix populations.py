import pandas as pd

a = pd.read_csv("population.csv")
b = a.iloc[:1000,1:]
b.to_csv("populations.csv", index=False)