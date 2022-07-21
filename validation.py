import chunk
from application_parameters import *
import numpy
import matplotlib.pyplot as plt
import scipy

def compute_accuracy(predicted_class: numpy.ndarray, L: numpy.ndarray):
    return (predicted_class == L).sum() / L.shape[0]

def print_statistics(accuracy: float, min_dcf: float, t_star: float):
    print(f"--- Accuracy of the model: {accuracy} %")
    print(f"---- Min DCF of the model:  {min_dcf}")
    print(f"--- Best treshold: {t_star}")


def plot_bayes_error_plot(score,labels):
    score = score.ravel()
    n_points = 39
    eff_prior_log_odds = numpy.linspace(-4, 4, n_points)
    dcf = numpy.zeros(n_points)
    mindcf = numpy.zeros(n_points)
    for (idx, p) in enumerate(eff_prior_log_odds):
        pi = 1 / (1 + numpy.exp(-p))
        dcf[idx] = compute_NDCF(score,labels, pi, 1, 1)
        mindcf[idx] = compute_minimum_NDCF(score, labels, pi, 1,1)
    plt.plot(eff_prior_log_odds, dcf, label="DCF", color="r")
    plt.plot(eff_prior_log_odds, mindcf , label="min DCF", color="b")
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.show()

def build_conf_mat_uniform(prediction, L):
    conf_mat = numpy.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
                conf_mat[i][j] = (1*numpy.bitwise_and(prediction == i, L == j)).sum()
    

    return conf_mat

def build_conf_mat(llr: numpy.ndarray,L: numpy.ndarray,pi:float, C_fn:float,C_fp:float):
    t = -numpy.log(pi*C_fn/((1-pi)*C_fp))
    predictions = 1*(llr > t)
    return build_conf_mat_uniform(predictions,L)


def compute_DCF(llr: numpy.ndarray, L: numpy.ndarray, pi: float, C_fn: float, C_fp: float):
    conf_mat = build_conf_mat(llr, L, pi, C_fn, C_fp)
    FNR = conf_mat[0][1]/ (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[0][0])
    print(FNR * pi)
    print(FPR * (1-pi))
    return pi * C_fn * FNR + (1-pi) * C_fp * FPR

def compute_NDCF(llr: numpy.ndarray, L: numpy.ndarray, pi: float, C_fn: float, C_fp: float):
    return compute_DCF(llr, L, pi, C_fn, C_fp) / min([pi*C_fn, (1-pi)*C_fp])

def compute_ROC_points(llr, L):
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(llr),numpy.array([numpy.inf])])
    N_label0 = (L == 0).sum()
    N_label1 = (L == 1).sum()
    ROC_points_TPR = numpy.zeros(L.shape[0] +2 )
    ROC_points_FPR = numpy.zeros(L.shape[0] +2 )
    for (idx,t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        TPR = numpy.bitwise_and(pred == 1, L == 1 ).sum() / N_label1
        FPR = numpy.bitwise_and(pred == 1, L == 0).sum() / N_label0
        ROC_points_TPR[idx] = TPR
        ROC_points_FPR[idx] = FPR
    return ROC_points_TPR, ROC_points_FPR

def plot_ROC(llr, L):
    ROC_points_TPR, ROC_points_FPR = compute_ROC_points(llr, L)
    plt.plot(ROC_points_FPR, ROC_points_TPR)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.show()

def compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn):
    FNR = conf_mat[0][1]/ (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[0][0])
    return (pi * C_fn * FNR + (1-pi) * C_fp * FPR)  / min([pi*C_fn, (1-pi)*C_fp])

def compute_minimum_NDCF(llr, L, pi, C_fp, C_fn):
    llr = llr.ravel()
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(llr),numpy.array([numpy.inf])])
    DCF = numpy.zeros(tresholds.shape[0])
    for (idx, t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        conf_mat = build_conf_mat_uniform(pred, L)
        DCF[idx] = compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn)
    argmin = DCF.argmin()
    return DCF[argmin], tresholds[argmin]

def plot_PC_heatmap(D: numpy.ndarray):
    heatmap = numpy.zeros((D.shape[0],D.shape[0]))
    for f1 in range(D.shape[0]):
        for f2 in range(D.shape[0]):
            if f2 <= f1:
                heatmap[f1][f2] = abs(scipy.stats.pearsonr(D[f1,:], D[f2,:])[0])
                heatmap[f2][f1] = heatmap[f1][f2]
    plt.figure()          
    plt.imshow(heatmap, cmap='Greys')
    plt.savefig("figure/heatmap_label_zmean.png")


def plot_hist(D: numpy.ndarray, L: numpy.ndarray):
    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel(features_name[i])
        for j in range(n_classes):
            plt.hist(D[:, L == j][i, :], bins = 70, density = True, alpha = 0.4, label=classes[j], linewidth = 1.0, edgecolor='black' )
        plt.legend()
        plt.tight_layout()
        #plt.savefig('gauss_features/hist_'+ features_name[i]+'.png')     
    plt.show()

def print_statistics(accuracy: float, act_dcf: float, min_dcf: float, t_star: float):
    
    print(f"{accuracy} {act_dcf} {min_dcf}")
    print(f"--- Accuracy of the model: {accuracy} %")
    print(f"--- Act DCF of the model: {act_dcf}")
    print(f"---- Min DCF of the model:  {min_dcf}")
    print(f"--- Best treshold: {t_star}")


classes = {
    0: "Male",
    1: "Female"
}

latex_simbol = {
    "lamba" : "$\ lamba$",
    "C" : "C"
}


def plot_minDCF(min_dcf_05, min_dcf_01, min_dcf_09, pre_proc_tech, ranges, model):
    plt.figure()
    plt.title(pre_proc_tech)
    plt.xlabel('$C$')
    plt.ylabel("minDCF")
    plt.xscale('log')
    
    plt.plot(ranges, min_dcf_01, label = "minDCF($\\tilde{\pi} = 0.1$)")
    plt.plot(ranges, min_dcf_05, label = "minDCF($\\tilde{\pi} = 0.5$)")
    plt.plot(ranges, min_dcf_09, label = "minDCF($\\tilde{\pi} = 0.9$)")
    """
    plt.plot(ranges, min_dcf_01, label = "$log\gamma = -1$")
    plt.plot(ranges, min_dcf_05, label = "$log\gamma = -2$")
    plt.plot(ranges, min_dcf_09, label = "$log\gamma = -3$")
    """
    plt.xlim(ranges[0], ranges[-1])
    plt.legend()
    plt.savefig("plot/" + model + "/new_" + pre_proc_tech + ".png")

def plot_minDCF_GMM(gauss_dcf, zscore_dcf, bounds, mod):
    plt.figure()
    plt.title(mod)
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    x_axis = numpy.arange(len(bounds))
    bounds = numpy.array(bounds)
    plt.bar(x_axis + 0.00 , gauss_dcf, width = 0.25,linewidth = 1.0, edgecolor='black', color="Red", label = "Gaussianization")
    plt.bar(x_axis + 0.25, zscore_dcf, width = 0.25,linewidth = 1.0, edgecolor='black', color="Orange" ,label = "Z-Score")
    plt.xticks([r + 0.125 for r in range(len(bounds))],
        bounds)
    plt.legend()
    plt.savefig("plot/GMM/new_" + mod + ".png")

def K_fold_split(D, L, K, seed = 27):
    """
    Since the dataset is ordered by label we will split the male dataset and the female one in 5 fold. Before we do this we will shuffle the two sub-dataset.
    """
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    chunks = numpy.array_split(idx, K)
    folds = []
    folds_label = []
    for i in range(K):
        folds.append(D[:,chunks[i]])
        folds_label.append(L[chunks[i]])
    return folds, folds_label

def next_kfold_iter(folds, folds_label):
    folds = [folds[1], *folds[2:], folds[0]]
    folds_label = [folds_label[1], *folds_label[2:], folds_label[0]]
    return folds, folds_label

def train_and_eval(folds, folds_label):
    DTR = numpy.concatenate(folds[1:], axis = 1)
    LTR = numpy.concatenate(folds_label[1:])
    DTE = folds[0]
    LTE = folds_label[0]
    return DTR, LTR, DTE, LTE    

