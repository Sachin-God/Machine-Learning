#importing Libraries
import numpy as np
import matplotlib.pyplot as plt # library to plot some charts
import pandas as pd  # matrix features and import dataset
from sklearn.impute import SimpleImputer # it is used to handle missing data by replacing it with a specific value, such as the mean, median, most frequent value, or a constant.

# Reading the dataset CSV
dataset = pd.read_csv('Part 1 - Data Preprocessing/Data.csv') # Don't add file name directly add relative file path
X = dataset.iloc[:, :-1].values  # feature Matrix  (iloc -> locate indexes)  [: -> select every col, :-1 -> is selecting every col except last col in selected all columns]
Y = dataset.iloc[:, -1].values  # Dependent variable Vector - which we wanna find out

# Handling Missing data in Data.csv Feature Matrix
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # telling imputer that we wanna replace missing value that are nam with strategy that is mean
imputer.fit(X[:,1:3]) #This method is called to compute the necessary statistics (like mean, median, or most frequent value) from the selected columns, which will be used later to fill in missing values
X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer  # ColumnTransformer allows applying different preprocessing steps to different columns.
from sklearn.preprocessing import OneHotEncoder  # OneHotEncoder transforms categorical features into binary columns (one-hot encoding).
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder  # transforms yes and no into 1 & 0
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)