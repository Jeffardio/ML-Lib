
import numpy
from Classifiers import *
from statistics import *
import preprocessing as pp
from statistics import *
from validation import *

act_model = ""
act_preproc =""

pca_dim = 0

gaussian = []
logistic = []
support_vm = []

act_lambda = 0
act_K = 0
act_C = 0
act_d = 0
act_gamma = 0
act_iterations = 0

def load_data():
    D = numpy.load("Train_data.npy")
    L = numpy.load("Train_data_labels.npy")
    DT = numpy.load("Test_data.npy")
    LT = numpy.load("Test_data_labels.npy")
    return D, L , DT, LT


def build_model(DTR, LTR , DTE, LTE, model = act_model):
    score, filename = 0,""
    if model in gaussian:        
        score, filename = classifier_model[model](DTR, LTR, DTE, LTE)

    if model in logistic:
        score, filename = classifier_model[model](DTR, LTR, DTE, LTE, act_lambda)

    if model in support_vm:

        if model == "SVM":
            score, filename = classifier_model[model](DTR, LTR, DTE, LTE, 1, act_C)

        if model == "PSVM":
            score, filename = classifier_model[model](DTR, LTR, DTE, LTE, 0, act_C,act_d,1)

        if model == "RSVM":            
            score, filename = classifier_model[model](DTR, LTR, DTE, LTE, 1, act_C,act_gamma)

    if model == "GMM":
        score, filename = classifier_model[model](DTR, LTR, DTE, LTE, act_iterations, 0.1, 0.01, act_modality)

    return score, filename


def k_fold_computation(folds, folds_label, N,K = 5, preprocessing = act_preproc):
    Nf = int(N / 5)
    scores = numpy.zeros((N))
    guessed = 0
    print(f"Preprocessing... {preprocessing}")
    for i in range(K):
        DTR, LTR, DTE, LTE = train_and_eval(folds, folds_label)
        """
        PREPROCESSING
        """
        if preprocessing == "Z-MEAN":
            print("Z-MEANING...")
            DTR, DTE = pp.preprocess_Z_mean(DTR, DTE)
        if preprocessing == "Z-MEAN + PCA":
            DTR, DTE = pp.preprocess_Z_mean(DTR, DTE)
            DTR, DTE = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
        if preprocessing == "GAUSSIANIZATION + PCA":
            DTR, DTE = pp.load_preprocess_gaussianize(i)
            DTR, DTE = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
        if preprocessing == "GAUSSIANIZATION":
            print(DTR.sum())
            DTR, DTE = pp.load_preprocess_gaussianize(i)
            print(DTR.sum())
            
            
        """
        TRAINING
        """
        accuracy, _, _, fold_score,file_save = build_model(DTR, LTR, DTE, LTE, model= act_model)
        """
        TESTING
        """
        guessed += accuracy * (Nf)
        scores[i*Nf:i*(Nf) + (Nf)] = fold_score
        folds, folds_label = next_kfold_iter(folds, folds_label)
    return guessed, scores, file_save




def k_fold_approach(K = 5):
    D, L, DT, LT = load_data()
    folds, folds_label = K_fold_split(D, L, K)
    shuff_labels = numpy.concatenate(folds_label[:])
    guessed, scores,file_save = k_fold_computation(folds, folds_label, L.shape[0], preprocessing = act_preproc)
    numpy.save(file_save, scores)
    #accuracy = guessed / (L.shape[0])
    #act_dcf = compute_NDCF(scores, shuff_labels, effective_prior, 1 ,1)
    

DTR, LTR, DTE, LTE = load_data