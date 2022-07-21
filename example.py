import numpy
import matplotlib.pyplot as plt
from statistics import *
from sklearn.preprocessing import QuantileTransformer
import scipy
from validation import plot_hist

def load_data():
    D = numpy.load("Train_data.npy")
    L = numpy.load("Train_data_labels.npy")
    DT = numpy.load("Test_data.npy")
    LT = numpy.load("Test_data_labels.npy")
    return D, L , DT, LT

D,L, DT, LR = load_data()
