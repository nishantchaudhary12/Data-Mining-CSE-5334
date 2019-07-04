import numpy as np
import random

#train
mean1 = [1, 0]
cov1 = [[1, 0.75], [0.75, 1]]  # diagonal covariance

x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T
X_temp1 = list(zip(x1, y1))


mean2 = [0, 1.5]
cov2 = [[1, 0.75], [0.75, 1]]  # diagonal covariance

x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T
X_temp2 = list(zip(x2, y2))


X_train = (X_temp1 + X_temp2)
print(X_train)

Y_temp1 = [0 for i in range(1000)]
Y_temp2 = [1 for i in range(1000)]

Y_train = (Y_temp1 + Y_temp2)
print(Y_train)


#test
x_test1, y_test1 = np.random.multivariate_normal(mean1, cov1, 500).T
X_test1 = list(zip(x_test1, y_test1))


x_test2, y_test2 = np.random.multivariate_normal(mean2, cov2, 500).T
X_test2 = list(zip(x_test2, y_test2))


X_test = (X_test1 + X_test2)
print(X_test)

Y_test1 = [0 for i in range(500)]
Y_test2 = [1 for i in range(500)]

Y_test = (Y_test1 + Y_test2)
print(Y_test)

# Shuffle data
c = list(zip(X_train, Y_train))
random.shuffle(c)
X_train, Y_train = zip(*c)

c = list(zip(X_test, Y_test))
random.shuffle(c)
X_test, Y_test = zip(*c)

print()
print("####################################")
print("#####    Prepossessing done    #####")
print("####################################")

class Perceptron(object):

    def __init__(self, lr=0.01, iterations=10):
        self.learning_rate = lr
        self.iterations = iterations
        self.weights = np.ones(3)

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        for q in range(self.iterations):
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
        return self

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_function(self, X):
        return self.sigmoid(np.dot(X, self.weights[1:]) + self.weights[0])

    def predict(self, X):
        return np.where(self.activation_function(X) >= 0.5, 1, 0)

    def test(self, X_test , Y_test):
        count = 0
        for i in range(len(X_test)):
            op = self.predict(X_test[i])
            if op == Y_test[i]:
                count += 1

        accuracy = count / len(X_test)
        return accuracy


alphas = [1.0, 0.1, 0.01]

print("\n### BATCH TRAINING(10 batches) ###")
ln = len(X_train)//10
start = 0
end = ln
count = 1
while start < len(X_train)-1:
    print("##################")
    print("###  BATCH", count, " ###")
    print("##################")

    for a in alphas:
        ppn = Perceptron(lr=a, iterations=10)

        print("Train with learning rate:", a)
        ppn.train(X_train[start:end], Y_train[start:end])
        print('Weights: %s' % ppn.weights)

        accuracy = ppn.test(X_test, Y_test)
        print("Accuracy:", accuracy)
        print("\n###################################\n")

    start = end
    end += ln
    count += 1




print("\n### COMPLETE ONLINE TRAINING ###")
for a in alphas:
    ppn = Perceptron(lr=a, iterations = 10)

    print("Train with learning rate:",a)
    ppn.train(X_train, Y_train)
    print('Weights: %s' % ppn.weights)


    accuracy = ppn.test(X_test, Y_test)
    print("Accuracy:", accuracy)
    print("\n###################################")


