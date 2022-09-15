from cgi import test
import numpy
from Classifiers import *
from statistics import *
import preprocessing as pp
from statistics import *
from validation import *


#TEMPLATE USED FOR EXPERIMENTAL RESULTS
act_model = "RSVM"
act_preproc = "Z-MEAN"

pca_dim = 11

gaussian = ["TMG","MG","TNB","NB"]
logistic = ["QLR"]
support_vm = ["SVM","PSVM","RSVM"]

lambda_values = numpy.logspace(-5,5, num = 51)

gamma_values = [0.001, 0.01, 0.1]
C_values = numpy.logspace(-5, 5, num = 31)

act_lambda = 0
act_K = 0
act_C = 0
act_d = 2
act_gamma = 0
act_iterations = 0
act_modality = ""

def load_data():
    D = numpy.load("data/Train_data.npy")
    L = numpy.load("data/Train_data_labels.npy")
    DT = numpy.load("data/Test_data.npy")
    LT = numpy.load("data/Test_data_labels.npy")
    return D, L , DT, LT


def build_model(DTR, LTR, DTE, model = act_model):
    score, filename = 0,""
    if model in gaussian:        
        score, filename = classifier_model[model](DTR, LTR, DTE)
    if model in logistic:
        score, filename = classifier_model[model](DTR, LTR, DTE, act_lambda)

    if model in support_vm:
        if model == "SVM":
            score, filename = classifier_model[model](DTR, LTR, DTE, 1, act_C)
        if model == "PSVM":
            score, filename = classifier_model[model](DTR, LTR, DTE, 1, act_C,act_d, 1)
        if model == "RSVM":            
            score, filename = classifier_model[model](DTR, LTR, DTE, 1, act_C,act_gamma)

    if model == "GMM":
        score, filename = classifier_model[model](DTR, LTR, DTE, act_iterations, 0.1, 0.01, act_modality)

    return score, filename




def train(model = act_model, preprocessing = act_preproc):
    #DTR, LTR = utils.load("Train.txt")
    #DTE, LTE = utils.load("Test.txt")
    DTR, LTR, DTE, LTE = load_data()
    pp_tec = ""
    if preprocessing == "Z-MEAN":
        DTR, DTE, pp_tec = pp.preprocess_Z_score(DTR, DTE)
    if preprocessing == "Z-MEAN + PCA":
        DTR, DTE,_ = pp.preprocess_Z_score(DTR, DTE)
        DTR, DTE,pp_tec = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
        pp_tec = "Z-Score_" + pp_tec
    if preprocessing == "GAUSSIANIZATION + PCA":
        DTR, DTE,_ = pp.preprocess_gaussianization(DTR,DTE)
        DTR, DTE, pp_tec = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
        pp_tec = "Gaussianization_" + pp_tec
    if preprocessing == "GAUSSIANIZATION":
        DTR, DTE, pp_tec = pp.preprocess_gaussianization(DTR,DTE)
    scores, file_save = build_model(DTR, LTR, DTE, model = act_model)
    file_save = pp_tec + "_" + file_save

    #saving scores for later computations
    numpy.save(f"test_scores/{act_model}/" + file_save, scores)
    print(numpy.around(compute_minimum_NDCF(scores, LTE, 0.5, 1, 1)[0],3))
    


#preprocess  
act_preproc ="Z-NORM"
#model to be trained    
act_model = "LR"
#trying different values of lambda
for l in lambda_values:
    act_lambda = l
    train(act_model, act_preproc)