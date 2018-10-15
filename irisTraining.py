#! /bin/python3.6
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

iris = load_iris()
x, y = iris.data, iris.target
'''
    x = [1, 2, 3, 4 ... ]   y = [ 1 ]
        [3, 4, 5, 6 ... ]       [ 2 ]
        [. . . . . . . .]       [ 3 ]
        [n  n  n  n  n  ]       [ 4 ]
'''

classifier = KNeighborsClassifier()

print(f'data set is: {x.shape}\nanswer set is: {y.shape}')
print('Now we need to split the data into a training set and test set\n')

train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    train_size=0.5,
                                                    random_state=123
                                                    )
print(f'Shape of train data is {train_x.shape}')
print(f'Shape of test data is {test_x.shape}\n')

# look at the data we split up
print('For the three options in "y":')
print(f'Full Dataset:   {(np.bincount(y) / float (len (y)) * 100.0)}%')
print(f'Training Dataset:   {(np.bincount(train_y) / float (len (train_y)) * 100.0)}%')
print(f'Test Dataset:   {(np.bincount(test_y) / float (len (test_y)) * 100.0)}%\n')

# Stratify the data
print('Wow thats horrible we need to STRATIFY this')
print('We can just pass in our target-array, "y", in as the "stratify" arg this time!\n')

train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    train_size=0.5,
                                                    random_state=123,
                                                    stratify=y,
                                                    )

print('Now lets see how it looks:')
print('For the three options in "y":')
print(f'Full Dataset:   {(np.bincount(y) / float (len (y)) * 100.0)}%')
print(f'Training Dataset:   {(np.bincount(train_y) / float (len (train_y)) * 100.0)}%')
print(f'Test Dataset:   {(np.bincount(test_y) / float (len (test_y)) * 100.0)}%\n')

print('now THAT is beautiful data, our training and test sets have the same ratios as the real-world full dataset!\n')

print('now lets use this classifier')
print(f'the docstring and default args are k = 5, this can be modified by parameters passed in when making our classifier!')

print('LETS DO THIS\n\n')

classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)

print("Fraction Correct [Accuracy]:")
print(np.sum(pred_y == test_y) / float(len(test_y)))


print('\nThat is amazing!')


print('\nNow lets graph this')

correct_idx = np.where(pred_y == test_y)
incorrect_idx = np.where(pred_y != test_y)

print(f'Correct indices: {correct_idx}\n')
print(f'Incorrect indices: {incorrect_idx}\n')

### let's make a graph!
colors = ['blue', 'green', 'gray']
for n, color in enumerate(colors):
    idx = np.where(test_y == n)[0]
    plt.scatter(test_x[idx, 1], test_x[idx, 2], color=color, label="Class %s" % str(n))
plt.scatter(test_x[incorrect_idx, 1], test_x[incorrect_idx, 2], color='darkred')
plt.show()

print("woah, a graph of our data!")
print("we can see where the incorrect ones would be wrong with a k=5 knearestneighbors")
