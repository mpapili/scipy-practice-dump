#! /bin/python3.6
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

print("you can also use k-nearest-neighbors algorithm as a regressor")
print("it will either take the nearest point's output if k=1, or average the nearest k points if k>1")

# 1000 evenly spaced numbers between -5 and 5
x = np.linspace(-5, 5, 1000)

# seed our random
rng = np.random.RandomState(42)

# set our target array 
# sin function + linear function + some random noise!
y = np.sin(4 * x) + rng.uniform(size=len(x))

#plt.show()

# we need to make it 2-dimmesional. (1000,1)
x = x[:, np.newaxis]

# split out test data, no need to stratify here
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.25,
                                                    random_state=42)

regressor = KNeighborsRegressor() 
regressor.fit(x_train, y_train)

print("score")
print(regressor.score(x_test, y_test))
