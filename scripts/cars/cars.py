import pandas as pd
from sklearn.linear_model import LinearRegression

auto = pd.read_csv('../../data/auto-mpg.csv')
print(auto.head())
print(len(auto))

X = auto.iloc[:, 1:-1]
X.drop('horsepower', axis=1, inplace=True)
y = auto.loc[:, 'mpg']

print(X.head())
print(y.head())

lr = LinearRegression()
lr.fit(X, y)
print(lr.score(X, y))

my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]

cars = [my_car1, my_car2]

cars_predict = lr.predict(cars)
print(cars_predict)