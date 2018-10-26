#! /bin/python3.6
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

data = load_boston()
print(dir(data))
print("Data Features")
print(data.feature_names)

# no need to stratify here
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.2,
                                                    random_state=42)

regressor = KNeighborsRegressor()
regressor.fit(x_train, y_train)
print(f'Score:\n{regressor.score(x_test, y_test)}\n')

print("now for the linear regressor")
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(f'Score:\n{regressor.score(x_test, y_test)}\n')

print("both seem awful")
