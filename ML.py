from sklearn import neural_network, neighbors, tree, svm, naive_bayes, metrics, datasets

# Where pipeline will be
# TODO add actual dataset here. iris dataset is just a placeholder
dataset = datasets.load_iris()

# Analogizers
model = neighbors.KNeighborsClassifier()
model.fit(dataset.data, dataset.target)
predictions = model.predict(dataset.data)
print(metrics.accuracy_score(predictions, dataset.target))

# ALSO ANALOGIZERS
model = svm.SVC()
model.fit(dataset.data, dataset.target)
predictions = model.predict(dataset.data)
score = metrics.accuracy_score(predictions, dataset.target)
print(score)

# Connectionists
model = neural_network.MLPClassifier()
model.fit(dataset.data, dataset.target)
predictions = model.predict(dataset.data)
score = metrics.accuracy_score(predictions, dataset.target)
print(score)

# Symbolists
model = tree.DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
predictions = model.predict(dataset.data)
score = metrics.accuracy_score(predictions, dataset.target)
print(score)

# Bayesians
model = naive_bayes.GaussianNB()
model.fit(dataset.data, dataset.target)
predictions = model.predict(dataset.data)
score = metrics.accuracy_score(predictions, dataset.target)
print(score)

