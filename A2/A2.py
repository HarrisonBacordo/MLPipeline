import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os
import pandas as pd

DATASET_PATH = os.path.join("..", "datasets")


def warn(*args, **kwargs):
    pass


def load_rent_data(mean_rent_path=DATASET_PATH):
    rent_csv = os.path.join(mean_rent_path, "final-dataset.csv")
    return pd.read_csv(rent_csv)


# Where pipeline will be
warnings.filterwarnings("ignore")
report_file = open('classification_report.txt', 'w')
dataset = load_rent_data()
features, targets = dataset[['Months from uni', 'Victoria Num', 'Massey Num', 'GDP', 'Tax', 'Orange Prices', 'Births',
                             'Deaths', 'Tourism Act', 'Tourism Adj']].values, dataset[['Rent']].values

# PIPELINE
# create imputer
imputer = Imputer(strategy="median")
# Create polynomial features
polynomial_features = PolynomialFeatures(degree=10)
# Feature scaling
std_scaler = StandardScaler()
# Create the feature selector
k_best = SelectKBest(f_regression, k=10)

# Make shuffle and split in such a way that there are an equal amount of the minority class in each

# Run pipeline
report_file.write('LINEAR REGRESSION\n')
pipeline = make_pipeline(imputer, polynomial_features, std_scaler, k_best, LinearRegression())
scores_list = np.zeros((10, 10))
for i in range(10, 0, -1):
    for j in range(10, 0, -1):
        polynomial_features = PolynomialFeatures(degree=j)
        k_best = SelectKBest(f_regression, k=i)
        pipeline = make_pipeline(imputer, polynomial_features, std_scaler, k_best, LinearRegression())
        print("WITH i == {} AND DEGREE == {}".format(i, j))
        score = np.average(cross_val_score(pipeline, features, targets.ravel(), cv=5, scoring='explained_variance'))
        scores_list[i-1, j-1] = score
        print(scores_list[i-1, j-1])
max_idx = np.unravel_index(np.argmax(scores_list, axis=None), scores_list.shape)
print(max_idx, scores_list[max_idx])
print(np.max(scores_list))
