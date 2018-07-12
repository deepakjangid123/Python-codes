import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


# STEP 1
# Fetch DataSet
# Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres.
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

print(iris_data[13], iris_data[25], iris_data[100])
print(iris_labels[13], iris_labels[25], iris_labels[100])


# STEP 2
# We create a learnset from the sets above. We use permutation from np.random to split the data randomly.
np.random.seed(42)
indices = np.random.permutation(len(iris_data))

# Leave 12 samples for testing purpose
n_training_samples = 12
learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]

print(learnset_data[:4], learnset_labels[:4])
print(testset_data[:4], testset_labels[:4])


# STEP 3
# Data Visualization
# Our data consists of four values per iris item, so we will reduce the data to three values by summing up the third
# and fourth value. This way, we are capable of depicting the data in 3-dimensional space.
X = []
for iclass in range(3):
    X.append([[], [], []])
    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(sum(learnset_data[i][2:]))
colours = ("r", "g", "y")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(3):
    ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
# plt.show()


# STEP 4
# Determining the neighbors
def distance(instance1: list, instance2: list) -> np.float64:
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)

    return np.linalg.norm(instance1 - instance2)


print(distance([3, 5], [1, 1]))
print(distance(learnset_data[3], learnset_data[44]))


# The function 'get_neighbors returns a list with 'k' neighbors, which are closest to the instance 'test_instance'.
def get_neighbors(training_set, labels, test_instance, k, distance=distance):
    """
    get_neighbors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with
    (index, dist, label)
    where
    index    is the index from the training_set,
    dist     is the distance between the test_instance and the
             instance training_set[index]
    distance is a reference to a function used to calculate the
             distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    # Sort 'distances' based on 'dist'
    distances.sort(key=lambda x: x[1])
    # K-nearest neighbors
    neighbors = distances[:k]
    return neighbors


for i in range(5):
    neighbors = get_neighbors(learnset_data, learnset_labels, testset_data[i], 3, distance=distance)
    print(i, testset_data[i], testset_labels[i], neighbors)


# STEP 5
# Voting to get a single result
# We will write a vote function now. This functions uses the class 'Counter' from collections to count the quantity
# of the classes inside of an instance list. This instance list will be the neighbors of course.
# The function 'vote' returns the most common class.
def vote(neighbors):
    """
    :param neighbors: List of neighbors for an instance/entry
    :return: Most common class
    """
    class_counter = Counter()
    for neighbor in neighbors:
        # Class is the 3rd element in 'neighbor' list
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]


for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data, learnset_labels, testset_data[i], 3, distance=distance)
    print("index: ", i, ", result of vote: ", vote(neighbors),
          ", label: ", testset_labels[i], ", data: ", testset_data[i])

# 'vote_prob' is a function like 'vote' but returns the class name and the probability for this class
def vote_prob(neighbors):
    """
    :param neighbors: List of neighbors for an instance/entry
    :return: Most common class name and probability
    """
    class_counter = Counter()
    for neighbor in neighbors:
        # Class is the 3rd element in 'neighbor' list
        class_counter[neighbor[2]] += 1

    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    return winner, votes4winner/sum(votes)


for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data, learnset_labels, testset_data[i], 5, distance=distance)
    print("index: ", i, ", vote_prob: ", vote_prob(neighbors),
          ", label: ", testset_labels[i], ", data: ", testset_data[i])


# STEP 6
# We looked only at k items in the vicinity of an unknown object "UO", and had a majority vote.
# Using the majority vote has shown quite efficient in our previous example, but this didn't take into account
# the following reasoning: The farther a neighbor is, the more it "deviates" from the "real" result.
# Or in other words, we can trust the closest neighbors more than the farther ones.
# Let's assume, we have 11 neighbors of an unknown item UO. The closest five neighbors belong to a class A and
# all the other six, which are farther away belong to a class B.
# What class should be assigned to UO? The previous approach says B, because we have a 6 to 5 vote in favor of B.
# On the other hand the closest 5 are all A and this should count more.
#
# To pursue this strategy, we can assign weights to the neighbors in the following way:
# The nearest neighbor of an instance gets a weight 1/1, the second closest gets a weight of 1/2 and
# then going on up to 1/k for the farthest away neighbor.
def vote_harmonic_weights(neighbors, all_results=True):
    """
    :param neighbors: List of neighbors for an instance/entry
    :param all_results: Set to true if you want all the results for all classes with probability non-zero
    :return: Most common class and probability based on harmonic distribution
    """
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        class_counter[neighbors[index][2]] += 1/(index + 1)
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
            class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)


for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data, learnset_labels, testset_data[i], 6, distance=distance)
    print("index: ", i, ", result of vote: ", vote_harmonic_weights(neighbors, all_results=True))


# The previous approach took only the ranking of the neighbors according to their distance in account.
# We can improve the voting by using the actual distance. To this purpose we will write a new voting function.
def vote_distance_weights(neighbors, all_results=True):
    """
    :param neighbors: List of neighbors for an instance/entry
    :param all_results: Set to true if you want all the results for all classes with probability non-zero
    :return: Most common class and probability based on distance weights distribution
    """
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        dist = neighbors[index][1]
        label = neighbors[index][2]
        class_counter[label] += 1 / (dist**2 + 1)
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)


for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data, learnset_labels, testset_data[i], 6, distance=distance)
    print("index: ", i, ", result of vote: ", vote_distance_weights(neighbors, all_results=True))


# Another Example for Nearest Neighbor Classification
train_set = [(1, 2, 2),
             (-3, -2, 0),
             (1, 1, 3),
             (-3, -3, -1),
             (-3, -2, -0.5),
             (0, 0.3, 0.8),
             (-0.5, 0.6, 0.7),
             (0, 0, 0)]
labels = ['apple',  'banana', 'apple',
          'banana', 'apple', "orange",
          'orange', 'orange']

for test_instance in [(0, 0, 0), (2, 2, 2),
                      (-3, -1, 0), (0, 1, 0.9),
                      (1, 1.5, 1.8), (0.9, 0.8, 1.6)]:
    neighbors = get_neighbors(train_set, labels, test_instance, 2)
    print("vote distance weights: ", vote_distance_weights(neighbors))


# Example with kNN
# We will use the k-nearest neighbor classifier 'KNeighborsClassifier' from 'sklearn.neighbors' on the Iris data set.
# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(learnset_data, learnset_labels)
KNeighborsClassifier(algorithm='auto',
                     leaf_size=30,
                     metric='minkowski',
                     metric_params=None,
                     n_jobs=1,
                     n_neighbors=5,
                     p=2,
                     weights='uniform')
print("Predictions from the classifier:")
print(knn.predict(testset_data))
print("Target values:")
print(testset_labels)