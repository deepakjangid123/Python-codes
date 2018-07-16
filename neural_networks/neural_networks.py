import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.special import expit as activation_function
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


'''
'AND' function:
Input1	Input2	Output
0	      0	      0
0	      1	      0
1	      0	      0
1	      1	      1
'''
class Perceptron:

    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = np.ones(input_length) * 0.5
        else:
            self.weights = weights

    @staticmethod
    def unit_step_function(x):
        if x > 0.5:
            return 1
        return 0

    def __call__(self, in_data):
        weighted_input = self.weights * in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)


p = Perceptron(2, np.array([0.5, 0.5]))
for x in [np.array([0, 0]), np.array([0, 1]),
          np.array([1, 0]), np.array([1, 1])]:
    # Call 'Perceptron' __call__ method
    y = p(x)
    print(x, y)


# Line Separation
class Perceptron:

    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = np.random.random(input_length) * 2 - 1
        self.learning_rate = 0.1

    @staticmethod
    def unit_step_function(x):
        if x < 0:
            return 0
        return 1

    def __call__(self, in_data):
        weighted_input = self.weights * in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)

    def adjust(self, target_result, calculated_result, in_data):
        error = target_result - calculated_result
        for i in range(len(in_data)):
            correction = error * in_data[i] * self.learning_rate
            self.weights[i] += correction


def above_line(point, line_func):
    """
    :param point: (x, y) coordinate of a point
    :param line_func: Line function in y = mx + c form
    :return: Returns whether point if above the line or not
    """
    x, y = point
    if y > line_func(x):
        return 1
    else:
        return 0


points = np.random.randint(1, 100, (100, 2))
p = Perceptron(2)
def lin1(x):
    return x + 4
for point in points:
    p.adjust(above_line(point, lin1), p(point), point)

evaluation = Counter()
for point in points:
    if p(point) == above_line(point, lin1):
        evaluation["correct"] += 1
    else:
        evaluation["wrong"] += 1
print(evaluation.most_common())


# Single-layer Perceptron
npoints = 50
X, Y = [], []
# class 0
X.append(np.random.uniform(low=-2.5, high=2.3, size=(npoints,)))
Y.append(np.random.uniform(low=-1.7, high=2.8, size=(npoints,)))
# class 1
X.append(np.random.uniform(low=-7.2, high=-4.4, size=(npoints,)))
Y.append(np.random.uniform(low=3, high=6.5, size=(npoints,)))
learnset = []
for i in range(2):
    # adding points of class i to learnset
    points = zip(X[i], Y[i])
    for p in points:
        learnset.append((p, i))
colours = ["b", "r"]
for i in range(2):
    plt.scatter(X[i], Y[i], c=colours[i])


class Perceptron:

    def __init__(self, input_length, weights=None):
        if weights == None:
            # input_length + 1 because bias needs a weight as well
            self.weights = np.random.random((input_length + 1)) * 2 - 1
        self.learning_rate = 0.05
        self.bias = 1

    @staticmethod
    def sigmoid_function(x):
        res = 1 / (1 + np.power(np.e, -x))
        return 0 if res < 0.5 else 1

    def __call__(self, in_data):
        weighted_input = self.weights[:-1] * in_data
        weighted_sum = weighted_input.sum() + self.bias * self.weights[-1]
        return Perceptron.sigmoid_function(weighted_sum)

    def adjust(self, target_result, calculated_result, in_data):
        error = target_result - calculated_result
        for i in range(len(in_data)):
            correction = error * in_data[i] * self.learning_rate
            self.weights[i] += correction
            # correct the bias:
        correction = error * self.bias * self.learning_rate
        self.weights[-1] += correction


p = Perceptron(2)
for point, label in learnset:
    p.adjust(label, p(point), point)
evaluation = Counter()
for point, label in learnset:
    if p(point) == label:
        evaluation["correct"] += 1
    else:
        evaluation["wrong"] += 1
print(evaluation.most_common())
colours = ["b", "r"]
for i in range(2):
    plt.scatter(X[i], Y[i], c=colours[i])
XR = np.arange(-8, 4)
m = -p.weights[0] / p.weights[1]
b = -p.weights[-1] / p.weights[1]
print(m, b)
plt.plot(XR, m * XR + b, label="decision boundary")
plt.legend()
# plt.show()


'''
NEURAL NETWORK Implementation
'''
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# Neural network class
class NeuralNetwork:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight matrices of the neural network
        """
        # It is a good idea to choose random values from within the interval
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # Number of input weights in hidden layer = no_of_input_nodes * no_of_hidden_nodes
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # Number of output weights from hidden layer = no_of_output_nodes * no_of_hidden_nodes
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        # Transpose
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network

        # update the weights
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp

        # calculate hidden errors
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)

        # update the weights
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        """
        running the network with an input vector input_vector.
        input_vector can be tuple, list or ndarray
        """

        # turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector


# Simple neural network construction
simple_neural_network = NeuralNetwork(no_of_in_nodes = 2, no_of_out_nodes = 2,
                                      no_of_hidden_nodes = 10, learning_rate = 0.1)
print(simple_neural_network.weights_in_hidden)
print(simple_neural_network.weights_hidden_out)
print(simple_neural_network.run([(3, 4)]))

data1 = [((3, 4), (0.99, 0.01)), ((4.2, 5.3), (0.99, 0.01)),
         ((4, 3), (0.99, 0.01)), ((6, 5), (0.99, 0.01)),
         ((4, 6), (0.99, 0.01)), ((3.7, 5.8), (0.99, 0.01)),
         ((3.2, 4.6), (0.99, 0.01)), ((5.2, 5.9), (0.99, 0.01)),
         ((5, 4), (0.99, 0.01)), ((7, 4), (0.99, 0.01)),
         ((3, 7), (0.99, 0.01)), ((4.3, 4.3), (0.99, 0.01))]
data2 = [((-3, -4), (0.01, 0.99)), ((-2, -3.5), (0.01, 0.99)),
         ((-1, -6), (0.01, 0.99)), ((-3, -4.3), (0.01, 0.99)),
         ((-4, -5.6), (0.01, 0.99)), ((-3.2, -4.8), (0.01, 0.99)),
         ((-2.3, -4.3), (0.01, 0.99)), ((-2.7, -2.6), (0.01, 0.99)),
         ((-1.5, -3.6), (0.01, 0.99)), ((-3.6, -5.6), (0.01, 0.99)),
         ((-4.5, -4.6), (0.01, 0.99)), ((-3.7, -5.8), (0.01, 0.99))]

# To close the existing figure
plt.close()
# Data for training
data = data1 + data2
np.random.shuffle(data)
points1, labels1 = zip(*data1)
X, Y = zip(*points1)
plt.scatter(X, Y, c="r")
points2, labels2 = zip(*data2)
X, Y = zip(*points2)
plt.scatter(X, Y, c="b")
# plt.show()

# Construct Neural-Network
simple_network = NeuralNetwork(no_of_in_nodes=2, no_of_out_nodes=2,
                               no_of_hidden_nodes=2, learning_rate=0.6)
# Take 90% data for training purpose
size_of_learn_sample = int(len(data) * 0.9)
learn_data = data[:size_of_learn_sample]
test_data = data[-size_of_learn_sample:]

for i in range(size_of_learn_sample):
    point, label = learn_data[i][0], learn_data[i][1]
    simple_network.train(point, label)

for i in range(size_of_learn_sample):
    point, label = learn_data[i][0], learn_data[i][1]
    cls1, cls2 = simple_network.run(point)
    print(point, cls1, cls2, end=": ")
    if cls1 > cls2:
        if label == (0.99, 0.01):
            print("class1 correct", label)
        else:
            print("class2 incorrect", label)
    else:
        if label == (0.01, 0.99):
            print("class1 correct", label)
        else:
            print("class2 incorrect", label)


# Neural Network with Bias Nodes
class NeuralNetwork:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate, bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight matrices of the neural
        network with optional bias nodes
        """

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes + bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]  # ???? last element cut off, ???
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [self.bias]))
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector


class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3)]
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8)]
labeled_data = []
for el in class1:
    labeled_data.append([el, [1, 0]])
for el in class2:
    labeled_data.append([el, [0, 1]])

np.random.shuffle(labeled_data)
data, labels = zip(*labeled_data)
labels = np.array(labels)
data = np.array(data)

simple_network = NeuralNetwork(no_of_in_nodes=2, no_of_out_nodes=2,
                               no_of_hidden_nodes=10, learning_rate=0.1,
                               bias=None)

for _ in range(50):
    for i in range(len(data)):
        simple_network.train(data[i], labels[i])
predicted_result = []
for i in range(len(data)):
    print(data[i], labels[i])
    predicted_result.append(simple_network.run(data[i]))
    print(predicted_result)


# Consider only first class for confusion matrix
def unit_step_function(x):
    """
    If 'x' is less than 0.8 return 0 else 1
    """
    if x < 0.8:
        return 0
    return 1


data = [i[0] for i in data]
actual_label = [i[0] for i in labels]
predicted_result = [unit_step_function(i[0]) for i in predicted_result]

confusion_matrix = confusion_matrix(actual_label, predicted_result)
accuracy_score = accuracy_score(actual_label, predicted_result)
classification_report = classification_report(actual_label, predicted_result)
print("Confusion matrix: \n{}\nAccuracy score: {}".format(confusion_matrix, accuracy_score))
print("Classification report: \n{}".format(classification_report))
precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
print("Precision: {}".format(precision))


'''
Neural Network - Testing with MNIST dataset
'''
'''
We will have to load the csv files only once. Because afterwards we are dumping the whole data using pickle.dump
which will be a lot faster than csv file load.
'''
'''
image_size = 28 # width and length
no_of_different_labels = 10 # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist_test.csv", delimiter=",")

# We map the values of the image data into the interval [0.01, 0.99] by dividing the train_data and test_data arrays
# by (255 * 0.99 + 0.01). Because intensity values are in between 0-254.
fac = 255 * 0.99 + 0.01
# Leave first value for each entry, because those are labels
train_imgs = np.asfarray(train_data[:, 1:]) / fac
test_imgs = np.asfarray(test_data[:, 1:]) / fac
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

# We need the labels in our calculations in a one-hot representation. We have 10 digits from 0 to 9,
# i.e. lr = np.arange(10).
# Turning a label into one-hot representation can be achieved with the command: (lr==label).astype(np.int).
lr = np.arange(10)
for label in range(10):
    one_hot = (lr == label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)

# We are ready now to turn our labelled images into one-hot representations.
# Instead of zeroes and one, we create 0.01 and 0.99, which will be better for our calculations
lr = np.arange(no_of_different_labels)
# Transform labels into one hot representation
train_labels_one_hot = (lr == train_labels).astype(np.float)
test_labels_one_hot = (lr == test_labels).astype(np.float)
# We don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 0] = 0.01
test_labels_one_hot[test_labels_one_hot == 1] = 0.99

print(train_labels_one_hot)

# You may have noticed that it is quite slow to read in the data from the csv files.
# We will save the data in binary format with the dump function from the pickle module.
with open ("pickled_mnist.pkl", "bw") as fh:
    data = (train_imgs,
            test_imgs,
            train_labels,
            test_labels,
            train_labels_one_hot,
            test_labels_one_hot)
    pickle.dump(data, fh)
'''

'''
# Before we start using the MNIST data sets with our neural network, we will have a look at same images.
plt.close()
for i in range(10):
    img = train_imgs[i].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()
'''

# Now we are able to read the data by using pickle.load. This is a lot faster than using loadtxt on the csv files.
with open("pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size


# Classifying the Data
class NeuralNetwork:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate, bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight matrices
        of the neural network with optional
        bias nodes"""

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes,
                          self.no_of_in_nodes + bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes,
                          self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can
        be tuple, list or ndarray
        """

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector,
                                           [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih,
                                input_vector)
        output_hidden = activation_function(output_vector1)

        if self.bias:
            output_hidden = np.concatenate((output_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.who += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]
        else:
            x = np.dot(tmp, input_vector.T)
        self.wih += self.learning_rate * x

    def run(self, input_vector):
        """
        input_vector can be tuple, list or ndarray
        """

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[self.bias]]))

        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


ANN = NeuralNetwork(no_of_in_nodes=image_pixels, no_of_out_nodes=10, no_of_hidden_nodes=100, learning_rate=0.1,
                    bias=None)
for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])

for i in range(20):
    res = ANN.run(test_imgs[i])
    print(test_labels[i], np.argmax(res), np.max(res))

corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
print("Accruracy train: ", corrects / (corrects + wrongs))
corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
print("Accruracy test: ", corrects / (corrects + wrongs))
cm = ANN.confusion_matrix(train_imgs, train_labels)
print("Confusion matrix: \n", cm)
for i in range(10):
    print("digit: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))


# Multiple Runs
epochs = 3
NN = NeuralNetwork(no_of_in_nodes=image_pixels,
                   no_of_out_nodes=10,
                   no_of_hidden_nodes=100,
                   learning_rate=0.1,
                   bias=1)
for epoch in range(epochs):
    print("epoch: ", epoch)
    for i in range(len(train_imgs)):
        NN.train(train_imgs[i],
                 train_labels_one_hot[i])

    corrects, wrongs = NN.evaluate(train_imgs, train_labels)
    print("Accruracy train: ", corrects / (corrects + wrongs))
    corrects, wrongs = NN.evaluate(test_imgs, test_labels)
    print("Accruracy test: ", corrects / (corrects + wrongs))


print(np.array(NN.wih).shape)
print(np.array(NN.who).shape)