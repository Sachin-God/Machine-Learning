#importing Libraries
import numpy as np
import matplotlib.pyplot as plt # library to plot some charts
import pandas as pd  # matrix features and import dataset

# Reading the dataset CSV
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # feature Matrix  (iloc -> locate indexes)  [: -> select every col, :-1 -> is selecting every col except last col in selected all columns]
Y = dataset          # Dependent variable Vector