import numpy
from probability import do_nothing
from utils import *
from application_parameters import *
import scipy

def compute_LDA_directions(D: numpy.ndarray, L: numpy.ndarray, mu: numpy.ndarray, m: int) -> numpy.ndarray: 
    mu_cn = compute_classes_mean(D, L)
    Sb = compute_Sb(D, L, mu, mu_cn)
    Sw = compute_Sw(D, L, mu_cn)
    _, U = scipy.linalg.eigh(Sb, Sw)
    P = U[:, ::-1][:, 0:m]
    return P

def compute_LDA_projection(DTR: numpy.ndarray, LTR: numpy.ndarray, DTE: numpy.ndarray,  mu: numpy.ndarray, m: int) -> numpy.ndarray:
    P = compute_LDA_directions(DTR, LTR, mu, m)
    return numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)

def preprocess_LDA(DTR, LTR, DTE, m):
    mu = compute_classes_mean(DTR, LTR)
    train, eval = compute_LDA_projection(DTR, LTR, DTE, mu, m)
    return train, eval, f"LDA_m_{m}"


def preprocess_PCA(DTR: numpy.ndarray, DTE: numpy.ndarray, m: int):
    #WE ASSUME THAT DTR and DTE ARE CENTERED AROUND THE DTR MEAN
    C = compute_covariance(DTR)
    P = compute_PCA_directions(C, m)
    return compute_PCA_projection(DTR, P), compute_PCA_projection(DTE, P), f"PCA_m_{m}"

def compute_PCA_directions(C: numpy.ndarray, m: int) -> numpy.ndarray:
    s, U = numpy.linalg.eigh(C)
    return U[:, ::-1][:, 0:m]

def compute_PCA_projection(D: numpy.ndarray,P :numpy.ndarray) -> numpy.ndarray:
    return numpy.dot(P.T, D)




def compute_Sb(D:numpy.ndarray, L: numpy.ndarray, mu :numpy.ndarray, mu_c: numpy.ndarray):  
    sum = numpy.zeros((D.shape[0],D.shape[0]))
    for i in range(n_classes):
        sum += (L == i).sum()*(mu_c[:,i:i+1] - mu).dot((mu_c[:,i:i+1] - mu).T)     
    return (1/D.shape[1]) * sum

def compute_Sw_c(D_c: numpy.ndarray, mu_c: numpy.ndarray):
    """`DC` is the data of a certain class, `mu_c` is the mean of this class."""
    sum = (D_c - mu_c).dot((D_c - mu_c).T)
    return (1/D_c.shape[1]) * sum 

def compute_Sw(D: numpy.ndarray, L: numpy.ndarray, mu_cn: numpy.ndarray):
    Sw = numpy.zeros((D.shape[0],D.shape[0]))
    for i in range(n_classes):
        Sw += (L == i).sum() * compute_Sw_c(D[:, L == i], mu_cn[:,i:i+1])
    return (1/D.shape[1]) * Sw


           


def load_preprocess_gaussianize(int):
    filename_DTR = "gaussianized/gauss_DTR" + str(int) + ".npy"
    filename_DTE = "gaussianized/gauss_DTE" + str(int) + ".npy"
    DTR = numpy.load(filename_DTR)
    DTE = numpy.load(filename_DTE)
    return DTR, DTE, "Gaussianization"

def preprocess_gaussianization(DTR: numpy.ndarray, DTE: numpy.ndarray):
    gauss_DTR = numpy.zeros(DTR.shape)
    
   
    for f in range(DTR.shape[0]):
        gauss_DTR[f, :] = scipy.stats.norm.ppf(scipy.stats.rankdata(DTR[f, :], method="min")/(DTR.shape[1] + 2))
    gauss_DTE = numpy.zeros(DTE.shape)
    
    
    for f in range(DTR.shape[0]):
        for idx,x in enumerate(DTE[f,:]):
            rank = 0
            for x_i in DTR[f,:]:
                if(x_i < x):
                    rank += 1
            uniform = (rank + 1) /(DTR.shape[1] + 2)
            gauss_DTE[f][idx] = scipy.stats.norm.ppf(uniform)
    
    return gauss_DTR, gauss_DTE,"Gaussianization"
    
def preprocess_Z_score(DTR, DTE):
    mu = compute_mean(DTR)
    std = compute_std(DTR)
    return (DTR - mu) / std, (DTE - mu) / std, "Z-Score"

def do_nothing(DTR, DTE):
    return DTR, DTE

preprocessing_dict = {
    "Z-Score": preprocess_Z_score,
    "Gaussianiation": preprocess_gaussianization,
    "Raw": do_nothing
}