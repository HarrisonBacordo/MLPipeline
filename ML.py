import warnings
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.over_sampling import SMOTE


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
    return np.array(_data), np.array(_targets)


def run_model(model):
    """
    runs the model
    :param model: model to run
    :return: void
    """
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions,
                                                   target_names=["No Appendicitis", "Appendicitis"])
    report_file.write("Accuracy: " + str(accuracy) + '\n')
    report_file.write(report + '\n\n')


# Where pipeline will be
warnings.warn = warn
file = open('appendicitis.dat', 'r')
report_file = open('results.txt', 'w')
data, targets = prep_data(file)

# PIPELINE
# Create the samplers
smote = SMOTE()
enn = RepeatedEditedNearestNeighbours()
# Make shuffle and split in such a way that there are an equal amount of the minority class in each
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
for train_index, test_index in sss.split(data, targets):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = targets[train_index], targets[test_index]

# Symbolists
report_file.write('DT\n')
pipeline = make_pipeline(smote, enn, DecisionTreeClassifier())
run_model(pipeline)
# Connectionists
report_file.write('MLP\n')
pipeline = make_pipeline(smote, enn, MLPClassifier())
run_model(pipeline)
# Bayesians
report_file.write('NB\n')
pipeline = make_pipeline(smote, enn, GaussianNB())
run_model(pipeline)
# Analogizers
report_file.write('KNN\n')
pipeline = make_pipeline(smote, enn, KNeighborsClassifier(1))
run_model(pipeline)
