from sklearn import neural_network, neighbors, tree, svm, naive_bayes, metrics, datasets

# Where pipeline will be
iris = datasets.load_iris()

#
model = neighbors.KNeighborsClassifier()
model.fit(iris.data, iris.target)
predictions = model.predict(iris.data)
print(metrics.accuracy_score(predictions, iris.target))


model = neural_network.MLPClassifier()
model.fit(iris.data, iris.target)
predictions = model.predict(iris.data)
score = metrics.accuracy_score(predictions, iris.target)
print(score)

model = tree.DecisionTreeClassifier()
model.fit(iris.data, iris.target)
predictions = model.predict(iris.data)
score = metrics.accuracy_score(predictions, iris.target)
print(score)

model = svm.SVC()
model.fit(iris.data, iris.target)
predictions = model.predict(iris.data)
score = metrics.accuracy_score(predictions, iris.target)
print(score)

model = naive_bayes.GaussianNB()
model.fit(iris.data, iris.target)
predictions = model.predict(iris.data)
score = metrics.accuracy_score(predictions, iris.target)
print(score)

