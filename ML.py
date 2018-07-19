

from sklearn import neural_network, neighbors, tree, svm, naive_bayes, metrics, utils, model_selection
import warnings


def warn(*args, **kwargs):
    pass


def prep_data(fname):
    """
    Makes the data readable by the the machine learning models
    :param fname: file with dataset
    :return: the split up data and targets of dataset
    """
    content = fname.read()
    content = content.split('\n')[12:-1]
    _data = list()
    _targets = list()
    # get individual data points in each data instance and separate the data from the targets
    for instance in content:
        values = instance.split(',')
        _data.append([float(i) for i in values[:7]])
        _targets.append(int(values[7]))
    return _data, _targets


# Where pipeline will be
warnings.warn = warn
file = open('appendicitis.dat', 'r')
data, targets = prep_data(file)
# Shuffles the data, SHOULD BE USED FOR PIPELINE
# data, targets = utils.shuffle(data, targets)
# Splits the data into train/test, SHOULD BE USED FOR PIPELINE
# x_train, x_test, y_train, y_test = model_selection.train_test_split(data, targets, test_size=.2)


# Analogizers
model = neighbors.KNeighborsClassifier()
model.fit(data, targets)
predictions = model.predict(data)
print(metrics.accuracy_score(predictions, targets))

# ALSO ANALOGIZERS
model = svm.SVC()
model.fit(data, targets)
predictions = model.predict(data)
score = metrics.accuracy_score(predictions, targets)
print(score)

# Connectionists
model = neural_network.MLPClassifier()
model.fit(data, targets)
predictions = model.predict(data)
score = metrics.accuracy_score(predictions, targets)
print(score)

# Symbolists
model = tree.DecisionTreeClassifier()
model.fit(data, targets)
predictions = model.predict(data)
score = metrics.accuracy_score(predictions, targets)
print(score)

# Bayesians
model = naive_bayes.GaussianNB()
model.fit(data, targets)
predictions = model.predict(data)
score = metrics.accuracy_score(predictions, targets)
print(score)
