setupa = '''
import pandas as pd
'''
setupb = '''
import pandas as pd
a = pd.read_csv("population.csv")
'''
a = '''
a = pd.read_csv("population.csv")
'''
b = '''
a.to_csv("test.csv")
'''


print(timeit(a, setupa, number =1))
print(timeit(b, setupb, number = 1))