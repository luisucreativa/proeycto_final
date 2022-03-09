import pandas as pd
from palmerpenguins import load_penguins
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


penguins = load_penguins()

penguins = penguins.dropna()

y = penguins.loc[:, "sex"]

X = penguins.iloc[:, 3:6]

x_train, x_test, y_train, y_test = train_test_split(X, y)


log_reg = LogisticRegression()

modelo = log_reg.fit(x_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(modelo, open(filename, 'wb'))


