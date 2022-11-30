import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import sys, time, json, pprint, ast, os, re

# MACHINE LEARNING
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# - MODELS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyRegressor, DummyClassifier

# - METRICS
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from yellowbrick.classifier import ROCAUC

# RECOMMENDER SYSTEM
from scipy.sparse import csr_matrix

# CNN
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers
# from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
