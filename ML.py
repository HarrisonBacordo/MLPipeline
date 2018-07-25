

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


def run_model(model):
    """
    runs the model
    :param model: model to run
    :return: void
    """
    model.fit(data, targets)
    predictions = model.predict(data)
    score = metrics.accuracy_score(predictions, targets)
    print(score)


# Where pipeline will be
warnings.warn = warn
file = open('appendicitis.dat', 'r')
data, targets = prep_data(file)
# Shuffles the data, SHOULD BE USED FOR PIPELINE
# data, targets = utils.shuffle(data, targets)
# Splits the data into train/test, SHOULD BE USED FOR PIPELINE
# x_train, x_test, y_train, y_test = model_selection.train_test_split(data, targets, test_size=.2)

# Analogizers
run_model(neighbors.KNeighborsClassifier())
# ALSO ANALOGIZERS
run_model(svm.SVC())
# Connectionists
run_model(neural_network.MLPClassifier())
# Symbolists
run_model(tree.DecisionTreeClassifier())
# Bayesians
run_model(naive_bayes.GaussianNB())
