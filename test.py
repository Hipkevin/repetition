import pandas as pd

data = pd.read_excel('data/data.xlsx')
print(data)

label = pd.read_csv('data/label.csv', delimiter=',')
print(label)
print(label['questionID'])