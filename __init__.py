import numpy
from Classifiers import *
from statistics import *
import preprocessing as pp
from statistics import *
from validation import *

act_model = "TMG"
act_preproc = "Z-SCORE"
pca_dim = 11
gaussian = ["MG","TMG","NB","TNB"]
logistic = ["LR","QLR"]
support_vm = ["SVM","PSVM","RSVM"]

lambda_values = numpy.logspace(-5,5, num = 51)
gamma_values = [0.001, 0.01, 0.1]
C_values = numpy.logspace(-5, 5, num = 31)
act_lambda = 0
act_K = 1
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



def build_model(DTR, LTR , DTE, model):
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


def k_fold_computation(folds, folds_label, N, preprocessing, model, K = 5):
    Nf = int(N / 5)
    scores = numpy.zeros((N))
    print(f"Preprocessing... {preprocessing}")
    for i in range(K):
        DTR, LTR, DTE, LTE = train_and_eval(folds, folds_label)
        """
        PREPROCESSING
        """
        pp_tec = ""
        if preprocessing == "Z-SCORE":
            DTR, DTE, pp_tec = pp.preprocess_Z_score(DTR, DTE)
        if preprocessing == "Z-SCORE + PCA":
            DTR, DTE,_ = pp.preprocess_Z_score(DTR, DTE)
            DTR, DTE,pp_tec = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
            pp_tec += "_Gaussianization"
        if preprocessing == "GAUSSIANIZATION + PCA":
            DTR, DTE,_ = pp.preprocess_gaussianization(DTR, DTE)
            DTR, DTE,pp_tec = pp.preprocess_PCA(DTR, DTE, m = pca_dim)
            pp_tec += "_Gaussianization"
        if preprocessing == "GAUSSIANIZATION":
            DTR, DTE, pp_tec = pp.preprocess_gaussianization(DTR, DTE)
            
        """
        TRAINING
        """
        
        fold_score,file_save = build_model(DTR, LTR, DTE, model)
        
        """
        MERGING SCORES
        """
        scores[i*Nf:i*(Nf) + (Nf)] = fold_score
        folds, folds_label = next_kfold_iter(folds, folds_label)
    return scores, pp_tec + "_" + file_save




def k_fold_approach(preprocessing, model,K):
    #D, L = utils.load("Train.txt")
    D, L, _, _ = load_data()
    folds, folds_label = K_fold_split(D, L, K)
    shuff_labels = numpy.concatenate(folds_label[:])
    scores,file_save = k_fold_computation(folds, folds_label,L.shape[0],preprocessing, model, 5)
    #SAVING SCORES FOR LATER COMPUTATIONS...
    numpy.save(f"scores/{act_model}/" + file_save, scores)
    print(numpy.around(compute_minimum_NDCF(scores, shuff_labels, 0.5, 1, 1)[0],3))


#preprocess  
act_preproc ="Z-SCORE"
#model to be trained    
act_model = "LR"
#trying different values of lambda
for l in lambda_values:
    act_lambda = l
    k_fold_approach(act_preproc, act_model,5)



