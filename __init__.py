import numpy
from Classifiers import *
from statistics import *
import preprocessing as pp
from statistics import *
from validation import *

act_model = "RSVM"
act_preproc = "Z-MEAN"

pca_dim = 0

gaussian = ["MG","NB","TNB","TMG"]
logistic = []
support_vm = ["RSVM"]

lambda_values = numpy.logspace(-5,5, num = 51)

gamma_values = [0.001, 0.01, 0.1]
C_values = numpy.logspace(-5, 5, num = 31)

act_lambda = 0
act_K = 0
act_C = 0
act_d = 0
act_gamma = 0
act_iterations = 0
act_modality = ""

def load_data():
    D = numpy.load("data/Train_data.npy")
    L = numpy.load("data/Train_data_labels.npy")
    DT = numpy.load("data/Test_data.npy")
    LT = numpy.load("data/Test_data_labels.npy")
    return D, L , DT, LT


def build_model(DTR, LTR , DTE, LTE, model = act_model):
    score, filename = 0,""
    if model in gaussian:        
        score, filename = classifier_model[model](DTR, LTR, DTE)
    if model in logistic:
        score, filename = classifier_model[model](DTR, LTR, DTE, act_lambda)

    if model in support_vm:

        if model == "SVM":
            score, filename = classifier_model[model](DTR, LTR, DTE, 1, act_C)

        if model == "PSVM":
            score, filename = classifier_model[model](DTR, LTR, DTE, 0, act_C,act_d, 1)

        if model == "RSVM":            
            score, filename = classifier_model[model](DTR, LTR, DTE, 1, act_C,act_gamma)

    if model == "GMM":
        score, filename = classifier_model[model](DTR, LTR, DTE, act_iterations, 0.1, 0.01, act_modality)

    return score, filename


def k_fold_computation(folds, folds_label, N,K = 5, preprocessing = act_preproc):
    Nf = int(N / 5)
    scores = numpy.zeros((N))
    print(f"Preprocessing... {preprocessing}")
    for i in range(K):
        DTR, LTR, DTE, LTE = train_and_eval(folds, folds_label)
        print(LTR.sum(), LTE.sum())
        """
        PREPROCESSING
        """
        pp_tec = ""
        if preprocessing == "Z-MEAN":
            DTR, DTE, pp_tec = pp.preprocess_Z_score(DTR, DTE)
        if preprocessing == "Z-MEAN + PCA":
            DTR, DTE = pp.preprocess_Z_score(DTR, DTE)
            DTR, DTE,pp_tec = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
        if preprocessing == "GAUSSIANIZATION + PCA":
            DTR, DTE = pp.load_preprocess_gaussianize(i)
            DTR, DTE = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
        if preprocessing == "GAUSSIANIZATION":
            DTR, DTE, pp_tec = pp.load_preprocess_gaussianize(i)
            
            
        """
        TRAINING
        """
        fold_score,file_save = build_model(DTR, LTR, DTE, LTE, model= act_model)
        """
        TESTING
        """
        scores[i*Nf:i*(Nf) + (Nf)] = fold_score
        folds, folds_label = next_kfold_iter(folds, folds_label)
    return scores, pp_tec + "_" + file_save




def k_fold_approach(K = 5):
    D, L, DT, LT = load_data()
    folds, folds_label = K_fold_split(D, L, K)
    shuff_labels = numpy.concatenate(folds_label[:])
    scores,file_save = k_fold_computation(folds, folds_label, L.shape[0], preprocessing = act_preproc)
    numpy.save(F"scores/{act_model}/" + file_save, scores)
    print(compute_minimum_NDCF(scores, shuff_labels, 0.5, 1, 1))

for c in C_values:
    act_C = c
    act_gamma = gamma_values[0]
    k_fold_approach()