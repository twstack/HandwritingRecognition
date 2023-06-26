import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load digits dataset
digits = datasets.load_digits()

# Print the data description
print(digits.DESCR, "\n")

# Print the data
print(digits.data, "\n")

# Print target values
print(digits.target, "\n")

# Visualize the image at index 100
plt.gray() 
plt.matshow(digits.images[100])
plt.show()

# Print the target label at index 100
print(digits.target[100], "\n")

# Create KMeans model
model = KMeans(n_clusters=10, random_state=42)

# Fit the data to the model
model.fit(digits.data)

# Visualize the centroids
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)

    # Display images
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

# Let's assume this 2D array is obtained from the `test.html` file
new_samples = np.array(
    [
[0.00,0.00,0.00,0.53,0.00,0.00,0.00,0.00,0.00,0.00,0.46,7.40,2.44,0.00,0.00,0.00,0.00,0.00,0.77,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.77,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.77,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.77,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.15,5.65,1.53,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,6.10,3.20,0.00,0.00,0.00,0.00,0.00,0.00,6.86,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.09,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.62,3.28,0.00,0.00,0.00,0.00,0.00,0.00,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.00,7.40,2.97,0.00,0.00,0.00,0.00,0.00,0.00,0.99,0.15,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,5.04,4.50,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.34,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.34,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.34,0.00,0.00,0.00,0.00,0.00,0.00,5.80,5.72,0.00,0.00,0.00,0.00,0.00,0.00,3.97,6.11,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.38,0.23,0.00,0.00,0.00,0.00,0.00,0.00,5.79,4.35,0.00,0.00,0.00,0.00,0.00,0.00,6.10,4.57,0.00,0.00,0.00,0.00,0.00,0.00,6.10,4.57,0.00,0.00,0.00,0.00,0.00,0.00,6.10,4.57,0.00,0.00,0.00,0.00,0.00,0.00,6.10,4.57,0.00,0.00,0.00,0.00,0.00,0.00,6.02,4.50,0.00,0.00,0.00,0.00,0.00,0.00,1.07,0.61,0.00,0.00,0.00]
]
)

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(0, end='')
    elif new_labels[i] == 1:
        print(9, end='')
    elif new_labels[i] == 2:
        print(2, end='')
    elif new_labels[i] == 3:
        print(1, end='')
    elif new_labels[i] == 4:
        print(6, end='')
    elif new_labels[i] == 5:
        print(8, end='')
    elif new_labels[i] == 6:
        print(4, end='')
    elif new_labels[i] == 7:
        print(5, end='')
    elif new_labels[i] == 8:
        print(7, end='')
    elif new_labels[i] == 9:
        print(3, end='')
