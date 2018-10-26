#! /bin/python3.6
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

x, y = make_blobs(centers=2, random_state=0)
print(f'X shape: {x.shape}')
print(f'Y shape: {y.shape}')

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.25,
                                                    random_state=1234,
                                                    stratify=y)


classifier = LogisticRegression()
print(x_train.shape)
print(y_train.shape)

classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)

print(np.sum(prediction != y_test), 'errors')

print(f'{prediction}\n{y_test}')

accuracy = np.mean(prediction == y_test)
print(f'accuracy:  {accuracy}%')

humanAccuracy = classifier.score(x_test, y_test)
print(f'human-readable accuracy:  {humanAccuracy}')

print(f'classifier coef:  {classifier.coef_}')
print(f'classifier intercept:   {classifier.intercept_}')


print("we can also try with k nearest neighbors")

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

prediction = classifier.predict(x_test)

print(np.sum(prediction != y_test), ' errors')
print(knn.score(x_test, y_test) * 100, '% accuracy wow')
