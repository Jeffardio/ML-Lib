import numpy
from application_parameters import *

def vcol(row_array: numpy.ndarray):
    if len(row_array.shape) != 1:
        return row_array.reshape((row_array.shape[1],1))
    return row_array.reshape((row_array.shape[0],1))
              
def vrow(column_array: numpy.ndarray):
    return column_array.reshape((1,column_array.shape[0]))

def compute_mean(D: numpy.ndarray) -> numpy.ndarray:
    return vcol(D.mean(1))

def compute_std(D: numpy.ndarray) -> numpy.ndarray:
    return vcol(D.std(1))
    
def center_data(D: numpy.ndarray, mu: numpy.ndarray) -> numpy.ndarray:
    return D - mu

def compute_covariance(DC: numpy.ndarray) -> numpy.ndarray:
    """`DC` should be the centered data matrix."""
    return (1/DC.shape[1]) * DC.dot(DC.T)

def compute_classes_mean(D:numpy.ndarray, L: numpy.ndarray) -> numpy.ndarray:
    classes_mean = numpy.zeros((D.shape[0],n_classes)) 
    for i in range(n_classes):
        classes_mean[:,i:i+1] = compute_mean(D[:, L == i])
    return classes_mean

def load(fname):
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
    

