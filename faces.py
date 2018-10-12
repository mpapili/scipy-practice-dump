#! /bin/python3.6
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

print(type(fig))
print("data size is:")
print(faces.data.size)
print("so each image is")
print(faces.data.size ** (1/2), 'squared')


print("adding subplots")
for i in range(64):
    ax = fig.add_subplot(8,8,i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.bone, interpolation='nearest')
plt.show()
