# Machine Learning Assignment - Supervised Learning - yrizvi3
## Datasets

Dataset 1. Wine - loaded using sci-kit learn
Dataset 2. Breast Cancer Wisconsin (Diagnostic) - no data entries made from original
[2] Jupyter Notebooks for each dataset
[1] pdf - yrizvi3-analysis

## Project Structure

winedataset.ipynb - Jupyter Notebook for wine: containing the main code for data loading, preprocessing, model training, evaluation, and visualization. The data for wine was loaded using sci-kit learn. 
# Load the Wine dataset
wine_data = load_wine()

WDBC.ipynb - dataset used without any additional entries. Attached in the folder for ease of access. Jupyter Notebook for WDBC: containing the main code for data loading, preprocessing, model training, evaluation, and visualization.

Environment 

1. Python 3.11.4
2. miniconda3 env
3. Jupyter notebooks

Dependencies 

import numpy as np
import pandas as pd
import time
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, validation_curve, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score




