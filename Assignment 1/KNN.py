from math import sqrt
import numpy as np


def standardize_data(data):
	temp = np.array(data, dtype='float64').T
	result = []
	for row in temp:
		ro_std = np.std(row)
		ro_avg = np.average(row)
		t = (row - ro_avg) / ro_std
		result.append(t)

	return np.array(result).T.tolist()


def read_data(filename):
	data = []
	with open(filename) as f:
		for line in f:
			line_data = line[:-1].split(',')
			data.append(line_data)

	return data[1:]


def select_specified_attrs(data, attr_list):
	result = []
	labels = []
	for row in data:
		temp = []
		for j in range(len(row)):
			if j in attr_list:
				temp.append(row[j])
		result.append(temp)
		labels.append(row[1])
	return result, labels


def select_not_specified_attrs(data, attr_list):
	result = []
	labels = []
	for row in data:
		temp = []
		for j in range(len(row)):
			if j not in attr_list:
				temp.append(row[j])
		result.append(temp)
		labels.append(row[1])
	return result, labels


def euclidean_distance(x, y):
	sum = 0
	for i in range(len(x)):
		sum += (float(x[i]) - float(y[i]))**2
	return sqrt(sum)


def split_data(data):
	X_train = data[0][:375]
	X_test = data[0][375:]

	Y_train = data[1][:375]
	Y_test = data[1][375:]

	X_train.insert(0, Y_train)
	X_test.insert(0, Y_test)

	return X_train, X_test


def myknn(X, test, k):
	trainY = X[0]
	trainX = X[1:]
	testY = test[0]
	testX = test[1:]
	testX = np.array(testX).T

	count = 0
	for i in range(len(testX)):
		predict = classify(trainX, trainY, testX[i], k)
		if predict == testY[i]:
			count += 1
	return count / len(testX)


def classify(X, Y, x, k):
	data = np.array(X)
	distance = []
	for i in range(len(data)):
		distance.append(euclidean_distance(data[i], x))

	k_distance = sorted(range(len(distance)), key=lambda j: distance[j])[:int(k)]
	k_lable_list = [Y[i] for i in k_distance]
	majority_label = majority_element(k_lable_list)
	return majority_label


def majority_element(label_list):
	index, counter = 0, 1

	for i in range(1, len(label_list)):
		if label_list[index] == label_list[i]:
			counter += 1
		else:
			counter -= 1
			if counter == 0:
				index = i
				counter = 1

	return label_list[index]


def run():
	filename = "NBAstats.csv"
	data = read_data(filename)
	kk = [1, 5, 10, 30]
	exclude_colm = [0, 1, 3]
	X = select_not_specified_attrs(data, exclude_colm)

	train, test = split_data(X)
	for k in kk:
		result = myknn(train, test, k)
		print("Accuracy for k =", k, " is:", result)

	print("#####################################")
	print("####     use attributes {2P%, 3P%, FT%, TRB, AST, STL, BLK}   ####")
	print("#####################################")
	select_only = [15, 12, 19, 22, 23, 24, 25]
	X = select_specified_attrs(data, select_only)

	train, test = split_data(X)
	for k in kk:
		result = myknn(train, test, k)
		print("Accuracy for k =", k, " is:", result)

if __name__ == "__main__":
	run()
