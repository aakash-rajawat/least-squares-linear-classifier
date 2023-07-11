import numpy as np
import matplotlib.pyplot as plt


def leastSquares(data, label):
    """
    Minimizes the sum of squared errors.
    :param data:    Training inputs  (num_samples x dim)
    :param label:   Training targets (num_samples x 1)
    :return:
        weights:    weights   (dim x 1)
        bias:       bias term (scalar)
    """
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    num_samples = len(data)
    data = np.concatenate((np.ones((num_samples, 1)), data), axis=1)  # before: (38x2), now: (38x3)

    # Take the pseudo inverse
    weight = np.linalg.lstsq((data.T.dot(data)),data.T)[0].dot(label) # inv(A)*b = A\b shape: (3, 1)  # Form the output

    bias = weight[0]  # get bias
    weight = weight[1:]  # get weights

    return weight, bias


def linclass(weight, bias, data):
    """
    Creates the Linear Classifier.
    :param weight:  weights (dim x 1)
    :param bias:    bias term (scalar)
    :param data:    Input to be classified (num_samples x dim)
    :return:
        class_pred: Predicted class (+-1) values  (num_samples x 1)
    """
    # Perform linear classification i.e. class prediction
    class_pred = data.dot(weight) + bias  # Y=X*W+B

    # Discretize classes, make hard decision
    class_pred[class_pred > 0] = 1
    class_pred[class_pred <= 0] = -1

    return class_pred


def plot_(data, labels, weight, bias, window_title):
    """
    Creates the plots.
    :param data:
    :param labels:
    :param weight:
    :param bias:
    :param window_title:
    :return:
    """
    # Define the range
    xmax = 1.5 #max(data(:, 1))
    xmin = -0.5 #min(data(:, 1))
    ymax = 1.5 #max(data(:, 2))
    ymin = -0.5 #min(data(:, 2))

    # Plot the data points and the decision line
    plt.subplot()
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(window_title)
    plt.plot(data[labels==1][:,0], data[labels==1][:,1], c = 'b', marker = 'x', linestyle='none', markersize=5)
    plt.plot(data[labels==-1][:,0], data[labels==-1][:,1], c = 'r', marker = 'o', linestyle='none', markersize=5, fillstyle='none')
    plt.plot([xmin, xmax], [-(weight[0]*xmin+bias)/weight[1], -(weight[0]*xmax+bias)/weight[1]], c = 'k')
    plt.show()


train = {}
test = {}
## Load the data
train.update({'data': np.loadtxt('data\\lc_train_data.dat')})
train.update({'label': np.loadtxt('data\\lc_train_label.dat')})
test.update({'data': np.loadtxt('data\\lc_test_data.dat')})
test.update({'label': np.loadtxt('data\\lc_test_label.dat')})

# Train the classifier using the training dataset
weight, bias = leastSquares(train['data'], train['label'])

# Evaluate the classifier on the training dataset
train.update({'prediction': linclass(weight, bias, train['data'])})

# Print and show the performance of the classifier
train.update({'acc' : sum(train['prediction'] == train['label'])/len(train['label'])})
print('Accuracy on train set: {0}'.format(train['acc']))
plot_(train['data'], train['label'], weight, bias, 'Train Set')

# Test the classifier on the test dataset
test.update({'prediction': linclass(weight, bias, test['data'])})

# Print and show the performance of the classifier
test.update({'acc' : sum(test['prediction'] == test['label'])/len(test['label'])})
plot_(test['data'], test['label'], weight, bias, 'Test Set')
print('Accuracy on test set: \t {0}\n'.format(test['acc']))

# Add outlier to training data, what happens?
print('Adding outliers...')

train['data'] = np.append(train['data'], [[1.5, -0.4],[1.45, -0.35]], axis = 0)
train['label'] = np.append(train['label'], [[-1],[-1]])

# Train the classifier using the training dataset
weight, bias = leastSquares(train['data'], train['label'])

# Evaluate the classifier on the training dataset
train['prediction'] = linclass(weight, bias, train['data'])

# Print and show the performance of the classifier
train['acc'] = sum(train['prediction'] == train['label'])/len(train['label'])
print('Accuracy on train set: {0}'.format(train['acc']))
plot_(train['data'], train['label'], weight, bias, 'Train Set')


# Test the classifier on the test dataset
test['prediction'] = linclass(weight, bias, test['data'])

# Print and show the performance of the classifier
test['acc'] = sum(test['prediction']==test['label'])/len(test['label'])
print('Accuracy on test set: \t {0}\n'.format(test['acc']))
plot_(test['data'], test['label'], weight, bias, 'Test Set')
